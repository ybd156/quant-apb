"""
状态持久化 —— 引擎与 UI 之间的唯一契约层。

原则：
  1. 原子写（先写 .tmp 再 os.replace），断电不会留下半截 JSON；
  2. UI 只读这些文件，不碰引擎内部状态（审计要求：UI 不做计算）；
  3. 权重器状态用 npz 存（小、快、跨平台）；
  4. 所有写盘都是幂等的，可以放心重复跑。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
LEDGER_DIR = os.path.join(ROOT, "data", "pred_ledger")
for _d in (STATE_DIR, LEDGER_DIR):
    os.makedirs(_d, exist_ok=True)


def _atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _clean(o):
    """递归把 NaN/Inf 转 None —— 浏览器 JSON.parse 不接受裸 NaN。"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, np.floating):
        v = float(o)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    return o


def write_json(name: str, obj: Any, state_dir: str = STATE_DIR) -> str:
    path = os.path.join(state_dir, name if name.endswith(".json") else f"{name}.json")
    _atomic_write(path, json.dumps(_clean(obj), ensure_ascii=False,
                                   default=_default, indent=1, allow_nan=False))
    return path


def read_json(name: str, state_dir: str = STATE_DIR, default: Any = None) -> Any:
    path = os.path.join(state_dir, name if name.endswith(".json") else f"{name}.json")
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")


# ---------------------------------------------------------------- 权重器状态
def weighter_path(market: str = "CN", horizon: int = 5,
                  state_dir: str = STATE_DIR) -> str:
    """权重器状态的实例化文件名。

    旧实现三处（live.rebuild / live.step_once / pipeline）共用
    `weighter.npz`，且 `RunConfig.horizon` 与 `WeightConfig.horizon` 从不互相同步
    —— 于是 `pipeline --horizon 1` 会静默覆盖日更 h=5 循环的状态，
    而 npz 里的 cfg 仍自称 horizon=5（陷阱 13：多实例共用文件名）。
    """
    return os.path.join(state_dir, f"weighter_{str(market).lower()}_h{int(horizon)}.npz")


def save_weighter(w, path: Optional[str] = None, market: str = "CN",
                  horizon: Optional[int] = None) -> str:
    h = int(horizon if horizon is not None
            else getattr(getattr(w, "cfg", None), "horizon", 5) or 5)
    # 顺手把 cfg.horizon 同步成真实周期，避免「文件名 h=1、内容自称 h=5」
    try:
        w.cfg.horizon = h
    except Exception:  # noqa: BLE001
        pass
    path = path or weighter_path(market, h)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             a_g=w.a_g, b_g=w.b_g, E_g=w.E_g, logw_g=w.logw_g, n_g=w.n_g,
             a_r=w.a_r, b_r=w.b_r, E_r=w.E_r, logw_r=w.logw_r, n_r=w.n_r,
             ph_mean=w.ph_mean, ph_sum=w.ph_sum, ph_min=w.ph_min, alarms=w.alarms,
             sids=np.array(w.sids), n_regimes=np.array([w.R]),
             cfg=np.array([json.dumps(w.cfg.__dict__, ensure_ascii=False)]))
    return path


def load_weighter(path: Optional[str] = None, market: str = "CN",
                  horizon: Optional[int] = None, strict: bool = True):
    """读权重器状态。按 (market, horizon) 取实例文件，旧 `weighter.npz` 只读兼容。

    `strict=True`（默认）时，若给了 horizon 而状态内的 cfg.horizon 不一致，
    直接抛错 —— 旧行为是「零报错地拿 h=1 的状态继续跑 h=5」。
    """
    from .learner import AdaptiveWeighter, WeightConfig
    if path is None:
        cands = []
        if horizon is not None:
            cands.append(weighter_path(market, horizon))
        cands.append(os.path.join(STATE_DIR, "weighter.npz"))   # 旧文件，只读兼容
        path = next((c for c in cands if os.path.exists(c)), cands[0])
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    cfg = WeightConfig(**json.loads(str(z["cfg"][0])))
    R = int(z["n_regimes"][0])
    w = AdaptiveWeighter(len(z["sids"]), n_regimes=R, cfg=cfg,
                         sids=[str(s) for s in z["sids"]])
    for k in ("a_g", "b_g", "E_g", "logw_g", "n_g", "ph_mean", "ph_sum", "ph_min", "alarms"):
        setattr(w, k, z[k].astype(np.float64 if k != "alarms" else np.int64))
    for k in ("a_r", "b_r", "E_r", "logw_r", "n_r"):
        setattr(w, k, z[k].astype(np.float64))
    if horizon is not None and int(getattr(cfg, "horizon", -1)) != int(horizon):
        msg = (f"weighter 周期不一致：{os.path.basename(path)} 内 cfg.horizon="
               f"{getattr(cfg, 'horizon', None)}，本次要跑 h={horizon}")
        if os.path.basename(path) == "weighter.npz":
            # 旧的无标识文件：不猜、不混用，当作「该周期尚无状态」，
            # 调用方会走 rebuild 路径（可见的失败，而不是静默拿错状态）。
            print(f"[weighter] {msg} → 视为无状态（请先 rebuild）", file=sys.stderr)
            return None
        if strict:
            raise ValueError(msg + "。拒绝混用（会把 h=1 的记忆喂给 h=5）")
        print(f"[weighter] {msg}（strict=False，继续）", file=sys.stderr)
    return w


# ---------------------------------------------------------------- UI 产物
def export_ui_bundle(res: dict, p3d, family_of: Optional[Dict[str, str]] = None,
                     paper: Optional[dict] = None, state_dir: str = STATE_DIR,
                     market: str = "CN") -> Dict[str, str]:
    """
    把引擎结果切成 UI 需要的 6 个 JSON。所有数值在这里定稿，UI 只负责展示。

    market != "CN" 时，模拟盘写到 paper_us.json / paper_hk.json，
    **不覆盖** A 股的 paper.json —— 三个市场各自独立，成本口径完全不同。
    """
    import numpy as np
    import pandas as pd

    dates = res["dates"]
    sids = res["sids"]
    w = res["weighter"]
    fam = family_of or {}

    # status
    status = {
        "as_of": str(pd.Timestamp(dates[-1]).date()),
        "data_from": str(pd.Timestamp(dates[0]).date()),
        "data_to": str(pd.Timestamp(dates[-1]).date()),
        # res["scores"] 对外是 (N,T)（engine.run_walkforward 已 .T）。
        # 旧写法 shape[1] 取到的是 T（天数），导致 status.json 的 n_stocks
        # 恒等于 n_days，UI 首屏把「112 只」显示成「2633 只」（错 23.5 倍）。
        # 广度是 IR ≈ IC·√BR 的分子，错在这里会直接污染「有多大优势」。
        "n_stocks": int(len(res.get("codes") or [])),
        "n_signals": int(len(sids)),
        "n_days": int(len(dates)),
        "mode": res["meta"].get("mode", "learned"),
        "horizon": res["meta"].get("horizon", 5),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "meta": {k: v for k, v in res["meta"].items() if isinstance(v, (str, int, float))},
    }

    # weights（当日全局权重，按大小排序）
    w_now = w.weights(int(res["REGIME"][-1]))
    diag = res["diagnostics"]
    order = np.argsort(-w_now)
    weights = [{
        "sid": sids[i], "family": fam.get(sids[i], sids[i].split("_")[0]),
        "weight": float(w_now[i]),
        "hit_rate": float(diag["hit_rate"][i]),
        "n_updates": int(diag["n_updates"][i]),
        "mean_payoff": float(diag["mean_payoff"][i]),
        "alarms": int(diag["alarms"][i]),
    } for i in order]

    # regime 最近 500 天
    k = min(500, len(dates))
    regime = [{"date": str(pd.Timestamp(dates[i]).date()), "regime": int(res["REGIME"][i])}
              for i in range(len(dates) - k, len(dates))]

    # ledger 最近 60 个交易日
    led = res.get("ledger")
    ledger = []
    if led is not None and len(led):
        cutoff = dates[-1] - np.timedelta64(60, "D")
        sub = led[led["as_of"] >= cutoff].tail(3000)
        ledger = [
            {"as_of": str(pd.Timestamp(r.as_of).date()), "code": r.code,
             "score": round(float(r.score), 4), "regime": int(r.regime),
             "entry_price": _f(r.entry_price), "exit_price": _f(r.exit_price),
             "ret": _f(r.ret), "hit": None if pd.isna(r.hit) else bool(r.hit),
             "status": r.status}
            for r in sub.itertuples(index=False)
        ]

    # status.json 有 5 个写入方（本函数、live.rebuild、live.step_once、
    # pipeline、daily_append）。本函数过去是整体覆盖，会把别人写的
    # rebuilt_at / last_processed 抹掉 —— 改成读-改-写（AGENTS §7.5 规则二）。
    _old_status = read_json("status", state_dir, default=None) or {}
    if isinstance(_old_status, dict):
        _merged = dict(_old_status)
        _merged.update(status)
        status = _merged

    paths = {
        "status": write_json("status", status, state_dir),
        "weights": write_json("weights", weights, state_dir),
        "regime": write_json("regime", regime, state_dir),
        "ledger": write_json("ledger", ledger, state_dir),
    }
    if paper is not None:
        _m = str(market).upper()
        _name = "paper" if _m == "CN" else f"paper_{_m.lower()}"
        paths["paper"] = write_json(_name, paper, state_dir)
    return paths


def _f(x, nd: int = 6):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, nd)
    except Exception:
        return None


def _ledger_key(r: dict, default_h: int = 5):
    """台账幂等键：(决策日, 代码, 周期)。

    同一决策日对同一只票在同一周期上，只应存在**一个**预测。
    实测旧台账 47,676 行只有 8,150 个唯一键（5.85×重复），且 8,029 个键
    挂着 3–9 个互不相同的分数 —— rebuild 每次把最近 90 天整段再追加一遍。
    没有幂等键，台账无法回答「s 日对 x 股的预测是什么」，命中率也会被
    最近 90 天加权约 5.8 倍。
    """
    as_of = str(r.get("as_of") or "")[:10]
    code = str(r.get("code") or "")
    h = r.get("horizon")
    try:
        h = int(h) if h is not None else int(default_h)
    except Exception:
        h = int(default_h)
    return (as_of, code, h)


def append_ledger(rows: list, year: Optional[int] = None,
                  horizon: Optional[int] = None, market: str = "CN",
                  run_id: Optional[str] = None) -> str:
    """台账只追加：按年存 JSONL，永不回改历史预测。

    相对旧实现的三处修正：
      1. 幂等：以 (as_of, code, horizon) 为键，已落盘的键不再重复写入。
      2. 归属：每行补 horizon / market / run_id（AGENTS §7.5 规则一）。
      3. 裸 NaN 防护：走 _clean + allow_nan=False，与 write_json 一致
         （旧实现只有 default=_default，python float NaN 会写出非法 JSON）。
    """
    year = year or datetime.now().year
    path = os.path.join(LEDGER_DIR, f"{year}.jsonl")
    os.makedirs(LEDGER_DIR, exist_ok=True)
    dh = int(horizon) if horizon is not None else 5
    rid = run_id or (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                     f"-{os.getpid()}")
    mkt = str(market or "CN").upper()

    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(_ledger_key(json.loads(line), dh))
                except Exception:
                    continue

    n_new = n_dup = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            rec = dict(r)
            rec.setdefault("horizon", dh)
            rec.setdefault("market", mkt)
            rec.setdefault("run_id", rid)
            k = _ledger_key(rec, dh)
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            f.write(json.dumps(_clean(rec), ensure_ascii=False,
                               default=_default, allow_nan=False) + "\n")
            n_new += 1
    if n_dup:
        print(f"[ledger] appended {n_new}, skipped {n_dup} duplicate key(s)",
              flush=True)
    return path


def compact_ledger(year: Optional[int] = None, out_suffix: str = ".dedup") -> Dict[str, Any]:
    """生成去重视图，**不修改**原始台账（守 INV-7：历史只追加不改）。

    保留策略：同一键保留**最后一条**（最新一次重算视为当前视图），
    并写 dup_n = 该键历史出现次数，供审计判断重复是否非随机。
    """
    year = year or datetime.now().year
    src = os.path.join(LEDGER_DIR, f"{year}.jsonl")
    if not os.path.exists(src):
        return {"ok": False, "reason": "no_ledger", "path": src}
    last: Dict[Any, dict] = {}
    total = bad = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            k = _ledger_key(r)
            prev = last.get(k)
            r["dup_n"] = (prev["dup_n"] + 1) if prev else 1
            last[k] = r
    dst = os.path.join(LEDGER_DIR, f"{year}{out_suffix}.jsonl")
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for k in sorted(last):
            f.write(json.dumps(_clean(last[k]), ensure_ascii=False,
                               default=_default, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    return {"ok": True, "path": dst, "total_rows": total, "bad_lines": bad,
            "unique_keys": len(last), "duplicate_rows": total - bad - len(last)}
