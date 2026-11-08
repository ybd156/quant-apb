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
def save_weighter(w, path: Optional[str] = None) -> str:
    path = path or os.path.join(STATE_DIR, "weighter.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             a_g=w.a_g, b_g=w.b_g, E_g=w.E_g, logw_g=w.logw_g, n_g=w.n_g,
             a_r=w.a_r, b_r=w.b_r, E_r=w.E_r, logw_r=w.logw_r, n_r=w.n_r,
             ph_mean=w.ph_mean, ph_sum=w.ph_sum, ph_min=w.ph_min, alarms=w.alarms,
             sids=np.array(w.sids), n_regimes=np.array([w.R]),
             cfg=np.array([json.dumps(w.cfg.__dict__, ensure_ascii=False)]))
    return path


def load_weighter(path: Optional[str] = None):
    from .learner import AdaptiveWeighter, WeightConfig
    path = path or os.path.join(STATE_DIR, "weighter.npz")
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
    return w


# ---------------------------------------------------------------- UI 产物
def export_ui_bundle(res: dict, p3d, family_of: Optional[Dict[str, str]] = None,
                     paper: Optional[dict] = None, state_dir: str = STATE_DIR) -> Dict[str, str]:
    """把引擎结果切成 UI 需要的 6 个 JSON。所有数值在这里定稿，UI 只负责展示。"""
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
        "n_stocks": int(res["scores"].shape[1]),
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

    paths = {
        "status": write_json("status", status, state_dir),
        "weights": write_json("weights", weights, state_dir),
        "regime": write_json("regime", regime, state_dir),
        "ledger": write_json("ledger", ledger, state_dir),
    }
    if paper is not None:
        paths["paper"] = write_json("paper", paper, state_dir)
    return paths


def _f(x, nd: int = 6):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, nd)
    except Exception:
        return None


def append_ledger(rows: list, year: Optional[int] = None) -> str:
    """台账只追加：按年存 JSONL，永不回改历史预测。"""
    import pandas as pd
    year = year or datetime.now().year
    path = os.path.join(LEDGER_DIR, f"{year}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
    return path
