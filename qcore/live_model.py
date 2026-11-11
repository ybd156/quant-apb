"""
实时演算 —— 在 NAS 上按需更新数据、训练、预测、盯市。

## N100 的现实约束

实测（本机 16 核）：
  · Alpha158 算 300 只：21 秒
  · 横截面标准化：17 秒
  · GBDT 训练 250 棵 / 615k 样本：**315 秒**

N100 单线程约为桌面 2~2.5× 慢 → 全量重训约 **10-13 分钟**。

所以**不能每次预测都重训**。分级调度：

| 频率 | 任务 | 耗时 |
|---|---|---|
| 交易时段每 3 秒 | 秒级快照（`qcore.tick`） | <1s |
| 交易时段每 60 秒 | 行情/宽度/情绪（`qcore.market_live`） | ~2s |
| 每日收盘后 15:10 | 增量数据 + 打分 + 盯市 | ~30s |
| 每周（可配） | 重训 GBDT | 10-13 分钟（夜间） |
| 每 6 小时 | 数据源探活（`qcore.health`） | ~5s |

## 模型持久化

训练好的 GBDT 用 pickle 存到 `data/model/gbdt.pkl`，附 `meta.json` 记训练时间、
样本数、IC。预测时直接加载，不重训。

## 增量加树（可选）

GBDT 支持 warm-start：保留旧树，在**最近 N 天**的新数据上再加 M 棵树。
这样每日成本从 10 分钟降到 30 秒，代价是对旧规律的适应变慢。
用 `warm_trees` 控制，默认关闭（宁可慢也要稳）。
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "data", "model")
STATE = os.path.join(ROOT, "data", "state")
def _paths(h: int = 1):
    """每个 horizon 一套模型文件，互不覆盖。"""
    if int(h) == 1:
        return (os.path.join(MODEL_DIR, "gbdt.pkl"),
                os.path.join(MODEL_DIR, "meta.json"))
    return (os.path.join(MODEL_DIR, f"gbdt_h{int(h)}.pkl"),
            os.path.join(MODEL_DIR, f"meta_h{int(h)}.json"))


MODEL_PATH = os.path.join(MODEL_DIR, "gbdt.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------------ 模型存取
def save_model(model, meta: Dict, horizon: int = 1) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    mp, metap = _paths(horizon)
    with open(mp, "wb") as f:
        pickle.dump(model, f, protocol=4)
    meta = dict(meta)
    meta["horizon"] = int(horizon)
    meta["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(metap, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return mp


def load_model(horizon: int = 1):
    mp, metap = _paths(horizon)
    if not os.path.isfile(mp):
        return None, {}
    with open(mp, "rb") as f:
        m = pickle.load(f)
    meta = {}
    try:
        with open(metap, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass
    return m, meta


def model_age_days(horizon: int = 1) -> Optional[float]:
    _, meta = load_model(horizon)
    s = meta.get("saved_at")
    if not s:
        return None
    try:
        return (datetime.now().astimezone() - datetime.fromisoformat(s)).total_seconds() / 86400
    except Exception:
        return None


# ------------------------------------------------------------------ 训练
def train(universe: str = "csi300", stocks: int = 300, trees: int = 250,
          train_end: Optional[str] = None, horizon: int = 1,
          verbose: bool = True) -> Dict:
    """
    训练并保存模型。train_end 默认为「今天往前 30 天」，留出验证缓冲。

    horizon 决定标签口径（见 qcore/horizons.py）：
      h=0  日内 close[t]/open[t]-1（仅 T+0 市场）
      h>=1 close[t+1+h]/close[t+1]-1（T 日信号，T+1 收盘成交）
    每个 horizon 存独立的模型文件，互不覆盖。
    """
    from qcore import horizons as HZ
    horizon = int(horizon)
    if horizon not in HZ.HORIZONS:
        return {"ok": False, "reason": f"unsupported_horizon: {horizon}",
                "supported": HZ.all_horizons()}
    import importlib.util
    here = os.path.join(ROOT, "scripts")

    def _load(name, fn):
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, fn))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    bench = _load("bench", "benchmark_alpha158.py")
    if train_end is None:
        train_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    from qcore import qlib_data as QD
    from qcore.gbdt import GBDT

    # 横截面排名归一化（_cs_rank_norm）需要足够多的票才能算出分位。
    # 实测：50 只 -> 归一化后全 NaN -> 展平得到 0 样本 -> 报"样本 0"，
    # 用户完全看不懂。100 只就正常。所以提前拦住并说清原因。
    MIN_STOCKS = 80
    try:
        _st = int(stocks)
    except (TypeError, ValueError):
        _st = 300
    if _st < MIN_STOCKS:
        return {"ok": False, "reason": "stocks_too_few",
                "horizon": horizon, "stocks": _st, "min_stocks": MIN_STOCKS,
                "hint": f"--stocks={_st} 太少。横截面排名归一化需要至少 "
                        f"{MIN_STOCKS} 只票（实测 50 只会使归一化结果全为 NaN，"
                        f"最终样本数 0）。请用 --stocks 100 以上，推荐默认 300"}

    # 训练集口径（评审 R4）：`universe_codes(start=train_end)` 取的是"训练截止那天"
    # 的成分，等于用**未来**的成分去选历史样本（2021 快照选 2008–2019 的票）。
    # `APB_TRAIN_PIT=1` 换成点对点成分（该指数历史上曾入选过的全部票）。
    # 实测（300 只 / 250 树 / h=5）：快照 IC +0.0194(t 5.37) → 点对点 +0.0284(t 9.43)。
    # **默认保持 0**：改训练口径属 AGENTS §10.4，需要一次显式决策。
    if str(os.environ.get("APB_TRAIN_PIT", "0")).lower() in ("1", "true", "yes"):
        try:
            from datetime import date as _date
            _sy = int(str(train_end)[:4]) - 18          # 往前 18 年，覆盖完整牛熊
            _start = f"{max(2000, _sy)}-01-01"
            from .qlib_data import pit_membership as _pm
            _mem = _pm(universe)
            codes = []
            for _c, _spans in _mem.items():
                for (_a, _b) in _spans:
                    if (_b.date() if hasattr(_b, "date") else _b) >= _date.fromisoformat(_start)                        and (_a.date() if hasattr(_a, "date") else _a) <= _date.fromisoformat(str(train_end)[:10]):
                        codes.append(_c)
                        break
            codes = sorted(set(codes))
            if verbose:
                print(f"  [pit] 点对点成分 {len(codes)} 只（{_start} ~ {train_end} 曾入选）",
                      flush=True)
        except Exception as _e:  # noqa: BLE001
            if verbose:
                print(f"  [pit] 不可用（{type(_e).__name__}），回退快照口径", flush=True)
            codes = QD.universe_codes(universe, start=train_end)
    else:
        codes = QD.universe_codes(universe, start=train_end)
    if verbose:
        print(f"  {universe}: {len(codes)} 只，训练截至 {train_end}，"
              f"horizon={horizon}（{HZ.label_for(horizon)}）", flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=horizon,
                             max_stocks=stocks, verbose=verbose)
    if not d.get("ok"):
        return {"ok": False, "reason": "build_features failed"}
    X, y, dates = d["X"], d["y"], d["dates"]

    t0 = time.time()
    Xn = bench._cs_rank_norm(X)
    if verbose:
        print(f"  标准化 {time.time()-t0:.0f}s", flush=True)

    sp = bench._split(dates, "2021-01-01", "2099-12-31", train_end,
                      (datetime.strptime(train_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                      "2099-12-31")
    Xtr, ytr = bench._flatten(Xn, y, sp["train"])
    if Xtr.shape[0] < 5000:
        # 区分两种原因：票太少（归一化失败）vs 时间太短
        _n_fin = int(np.isfinite(Xn).sum())
        _hint = ("横截面归一化后几乎没有有效值，通常是股票数太少"
                 if _n_fin == 0 else
                 f"训练窗口内样本不足。当前 train_end={train_end}，"
                 f"可扩大数据范围或减少 horizon")
        return {"ok": False, "reason": f"样本不足 {Xtr.shape[0]}",
                "horizon": horizon, "n_stocks": len(d.get("codes") or []),
                "hint": _hint}
    t0 = time.time()
    model = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    model.fit(Xtr, ytr)
    dt = time.time() - t0

    # 质量证据（评审 R36）：模型导入要求 manifest 带 qa.holdout_ic，
    # 但没有证据就没法证明"比随机好"。这里算一次**样本内末段**的 Rank IC
    # （最后 60 个交易日），并明确标注 basis，避免被当样本外。
    qa = {"basis": "in_sample_tail_60d"}
    try:
        _tail = int(max(1, Xtr.shape[0] // 20))          # 末尾 5%
        _xt, _yt = Xtr[-_tail:], ytr[-_tail:]
        _p = model.predict(_xt)
        _ok = np.isfinite(_p) & np.isfinite(_yt)
        if _ok.sum() >= 100:
            _r = pd.Series(_p[_ok]).rank().to_numpy()
            _s = pd.Series(_yt[_ok]).rank().to_numpy()
            _ic = float(np.corrcoef(_r, _s)[0, 1])
            qa.update({"holdout_ic": round(_ic, 6), "holdout_n": int(_ok.sum())})
    except Exception as e:  # noqa: BLE001
        qa["error"] = f"{type(e).__name__}: {str(e)[:60]}"

    meta = {"universe": universe, "n_stocks": len(d["codes"]),
            "n_factors": int(X.shape[2]), "n_trees": int(trees),
            "train_end": train_end, "train_n": int(Xtr.shape[0]),
            "horizon": int(horizon),
            "train_seconds": round(dt, 1),
            "qa": qa,
            "trained_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    # horizon 必须传进去，否则 h=5 的模型会覆盖 h=1 的 gbdt.pkl
    save_model(model, meta, horizon=horizon)
    if verbose:
        print(f"  训练完成 {dt:.0f}s，已保存", flush=True)
    return {"ok": True, **meta}


# ------------------------------------------------------------------ 预测
def predict(top_k: Optional[int] = None, universe: str = "csi300",
            stocks: int = 300, horizon: int = 1, verbose: bool = True) -> Dict:
    """
    用已保存的模型给最新一天打分。
    horizon=1 写 picks.json（向后兼容）；其余写 picks_h{h}.json。

    top_k 默认按周期取（见 paper_auto.HORIZON_TOP_K）：
        T+1 -> 2 只，T+3 -> 3，1周 -> 5，2周 -> 8，15天 -> 10
    短线换手极快、成本占比高，列 30 只既没意义也做不过来。
    """
    if top_k is None:
        try:
            from . import paper_auto as _PA
            top_k = _PA.default_top_k(horizon)
        except Exception:
            top_k = {1: 2, 3: 3, 5: 5, 10: 8, 15: 10}.get(int(horizon), 5)
    top_k = int(top_k)
    import importlib.util
    here = os.path.join(ROOT, "scripts")
    spec = importlib.util.spec_from_file_location("bench", os.path.join(here, "benchmark_alpha158.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    from qcore import qlib_data as QD
    from qcore import names as NAMES

    horizon = int(horizon)
    model, meta = load_model(horizon)
    if model is None:
        return {"ok": False, "reason": "model_not_trained", "horizon": horizon,
                "hint": f"先训练该周期：python -m qcore.live_model --train --horizon={horizon}"}

    codes = QD.universe_codes(universe, start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"))
    if not codes:
        codes = QD.universe_codes(universe)
    # 窗口必须够长：build_features 要求每只票 >=300 根日线，alpha158 还要 60 日预热。
    # 之前用 400 个自然日 ≈ 270 个交易日，全部被过滤掉 → features_failed。
    _start = (datetime.now() - timedelta(days=900)).strftime("%Y-%m-%d")
    d = bench.build_features(codes, start=_start,
                             horizon=horizon, max_stocks=stocks, verbose=False)
    if not d.get("ok"):
        return {"ok": False, "reason": "features_failed"}
    X, dates, cl = d["X"], d["dates"], d["codes"]
    Xn = bench._cs_rank_norm(X)

    last = len(dates) - 1
    row = Xn[last]

    # 蒸馏/精简模型只用了部分特征 —— meta.feature_idx 记录是哪几个。
    # 不按它裁剪，模型会因为列数不匹配直接报错（或更糟：静默喂错列）。
    _fidx = (meta or {}).get("feature_idx")
    n_factors_meta = (meta or {}).get("n_factors")
    if _fidx and len(_fidx) != row.shape[1]:
        _idx = np.asarray(_fidx, dtype=int)
        if _idx.max() >= row.shape[1]:
            return {"ok": False, "reason": "feature_idx_out_of_range",
                    "model_features": int(_idx.max()), "built_features": int(row.shape[1]),
                    "hint": "模型是用不同的因子集训练的，需重新训练或换模型"}
        row = row[:, _idx]
    elif n_factors_meta and int(n_factors_meta) != row.shape[1]:
        return {"ok": False, "reason": "feature_count_mismatch",
                "model_features": int(n_factors_meta), "built_features": int(row.shape[1]),
                "hint": "模型特征数与当前代码产出的因子数不一致（可能换了因子集版本）"}

    ok = np.isfinite(row).all(axis=1)
    if ok.sum() < 20:
        return {"ok": False, "reason": "insufficient_valid_features"}
    score = np.full(len(cl), np.nan)
    score[ok] = model.predict(row[ok])

    # ---- 在线校准：把「预测 vs 实际」学到的修正应用到今天的分数上 ----
    # 这是真的在线学习：模型本身不动，但输出会随历史对账结果调整。
    # 如果 a 变成负数，说明模型方向反了 —— 系统自己发现的。
    _cal = None
    try:
        from . import online_calib as OC
        _raw = score[ok]
        _adj, _cal = OC.apply(horizon, _raw)
        if _cal.get("applied"):
            score[ok] = _adj
    except Exception:
        _cal = None

    # ---- 情报叠加（opt-in，默认关闭）----
    # 模型分不动，把"自己找来的新闻 + 历史参照"按排位融合进来。
    # 默认 w=0 = 完全关闭（不改变现有任何行为）；APB_INTEL_W=0.3 才启用。
    _intel = None
    try:
        _w = float(os.environ.get("APB_INTEL_W", "0") or 0)
    except Exception:  # noqa: BLE001
        _w = 0.0
    if _w > 0:
        try:
            from . import intel_features as IF
            _snap = IF.snapshot()
            _b = IF.blend_arrays([str(c) for c in cl], [float(x) for x in score],
                                 _snap.get("by_code") or {}, w=_w)
            score = np.asarray(_b["scores"], dtype=float)
            _intel = {"w": _w, "score_kind": _b["score_kind"],
                      "n_intel": _b["n_intel"], "n_blended": _b["n_blended"],
                      "intel_as_of": _snap.get("as_of")}
        except Exception as e:  # noqa: BLE001
            _intel = {"error": f"{type(e).__name__}: {str(e)[:60]}"}

    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf))[:top_k]
    nm = {}
    for k, v in (NAMES.load() or {}).items():
        nm[str(k)] = v.get("name") if isinstance(v, dict) else str(v)

    def _mk(j, rank, total):
        code = cl[j]
        bare = code[2:] if code[:2] in ("sh", "sz", "bj") else code
        return {"rank": rank, "code": bare, "qlib_code": code,
                "name": nm.get(bare) or nm.get(bare.zfill(6)) or "",
                "score": round(float(score[j]), 6),
                "weight": round(1.0 / max(total, 1), 4)}

    rows = [_mk(j, rank, len(order)) for rank, j in enumerate(order, 1)]

    # 台账对账需要足够多的样本才能算「前 1/3 是否跑赢全体」。
    # 短线只展示 2 只，但评估仍用更多候选 —— 否则每天只有 2 个样本，
    # 分位无从谈起，学习闭环就废了。
    EVAL_TOP = max(len(rows), 30)
    order_all = np.argsort(-np.where(np.isfinite(score), score, -np.inf))[:EVAL_TOP]
    rows_all = [_mk(j, rank, len(order_all)) for rank, j in enumerate(order_all, 1)]

    # ---- 事件调整：用近期事件 + 历史命中率调权重（拿不到就跳过，不影响主流程）----
    ev_info = None
    try:
        from .reward import event_adjust
        from .data import get_pool as _gp
        _pool = _gp(top_n=1200)
        _sec = {str(r.code).zfill(6): str(getattr(r, "industry", "") or "")
                for r in _pool.itertuples(index=False)}
        _w = {str(x["code"]): float(x["weight"]) for x in rows}
        _adj = event_adjust(_w, _sec)
        if _adj.get("applied"):
            for x in rows:
                m = (_adj["multipliers"] or {}).get(str(x["code"]), 1.0)
                x["event_mult"] = m
                x["weight_before_event"] = x["weight"]
                x["weight"] = round(float((_adj["weights"] or {}).get(str(x["code"]), x["weight"])), 4)
            ev_info = {"n_events": _adj.get("n_events_used"),
                       "sector_net": _adj.get("sector_net"),
                       "top_sectors": _adj.get("top_sectors"),
                       "n_changed": len(_adj.get("changed") or []),
                       "has_confidence": _adj.get("has_confidence"),
                       "summary": _adj.get("summary")}
        else:
            ev_info = {"n_events": 0, "summary": _adj.get("summary")}
    except Exception as e:  # noqa: BLE001
        ev_info = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    from qcore import horizons as HZ
    # market 必须写进 picks —— 否则模拟盘无法校验「这份预测是哪个市场的」，
    # 实测导致美股账户拿 A 股 picks 下单
    _mkt = "CN"
    try:
        from . import qlib_data as _QD
        _mkt = str(getattr(_QD, "MARKET", "CN") or "CN").upper()
    except Exception:
        _mkt = "CN"
    # horizon / horizon_label 也写进去。
    # 之前缺这两个字段，UI 只能拿模型的 hold_days（回测参数，常见 20 日）来显示，
    # 于是「选中 T+1 短线」却写着「持有 20 日」—— 用户一眼就看出矛盾。
    _hlab = ""
    try:
        from . import horizons as _HZ
        _hlab = _HZ.label_for(int(horizon))
    except Exception:
        _hlab = f"h{horizon}"
    payload = {"as_of": str(dates[last].date()), "rows": rows, "n": len(rows),
               "rows_all": rows_all, "n_all": len(rows_all),
               "market": _mkt, "horizon": int(horizon),
               "horizon_label": _hlab,
               # 决策来源必须写明（R26）：生产名单由 GBDT 出，
               # 102 信号集成只用于**测量**（回放/UI），不进名单。
               "decision_source": "gbdt",
               "decision_note": ("分数来自 Alpha158+GBDT（+在线校准）；"
                                 "102 信号集成只在 pipeline 回放里跑，不参与下单"),
               "top_k": top_k,
               "top_k_info": (lambda: __import__("qcore.paper_auto", fromlist=["x"]).top_k_info(horizon))(),
               "horizon": horizon,
               "horizon_label": HZ.label_for(horizon),
               "online_calib": _cal,
               "intel": _intel,
               "score_kind": ((_intel or {}).get("score_kind") or "model"),
               "model_age_days": round(model_age_days(horizon) or 0, 1),
               "event_adjust": ev_info,
               "note": "实时打分（模型未重训时用最近一次保存的模型）；"
                       "weight 已含事件调整，weight_before_event 是调整前"}
    os.makedirs(STATE, exist_ok=True)
    # horizon=1 写 picks.json（保持向后兼容），其余写 picks_h{h}.json
    fname = "picks.json" if horizon == 1 else f"picks_h{horizon}.json"
    tmp = os.path.join(STATE, fname + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(STATE, fname))
    if verbose:
        print(f"  预测完成 {payload['as_of']}，{len(rows)} 只，"
              f"horizon={horizon}（{HZ.label_for(horizon)}）-> {fname}", flush=True)

    # 自动登记到该周期的台账 —— 这是自学习的入口。
    # 登记失败不影响预测结果（台账是旁路，不能拖垮主流程）。
    try:
        from . import horizon_ledger as HL
        reg = HL.record(horizon, payload, verbose=False)
        payload["ledger"] = {k: reg.get(k) for k in ("ok", "exit_date", "n")}
    except Exception as e:  # noqa: BLE001
        payload["ledger"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:60]}"}
    return {"ok": True, **payload}


# ------------------------------------------------------------------ 调度
def should_retrain(max_age_days: float = 7.0) -> bool:
    a = model_age_days()
    return a is None or a >= max_age_days


def run_once(retrain_if_due: bool = True, max_age_days: float = 7.0,
             verbose: bool = True) -> Dict:
    """跑一轮：按需重训 → 预测 → 写健康度。"""
    from qcore import health
    out = {"at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if retrain_if_due and should_retrain(max_age_days):
        if verbose:
            print("[live] 模型过期，开始重训 ...", flush=True)
        out["train"] = train(verbose=verbose)
    else:
        out["train"] = {"ok": True, "skipped": True,
                        "age_days": round(model_age_days() or 0, 1)}
    out["predict"] = predict(verbose=verbose)
    try:
        out["health"] = health.check_all().get("summary")
    except Exception as e:  # noqa: BLE001
        out["health"] = {"error": str(e)[:80]}
    return out


def loop(interval_s: int = 3600, max_age_days: float = 7.0,
         verbose: bool = True) -> None:
    """常驻：每小时检查一次，收盘后跑一次完整演算。"""
    last_run_day = None
    while True:
        try:
            now = datetime.now()
            # 15:10 之后跑当天第一次
            if now.weekday() < 5 and (now.hour, now.minute) >= (15, 10):
                if last_run_day != now.date():
                    if verbose:
                        print(f"[live] {now:%H:%M} 收盘演算 ...", flush=True)
                    r = run_once(max_age_days=max_age_days, verbose=verbose)
                    last_run_day = now.date()
                    if verbose:
                        print(f"[live] done: {json.dumps(r, ensure_ascii=False)[:200]}", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[live] ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    import sys
    _H = 1
    for _a in sys.argv:
        if _a.startswith("--horizon="):
            _H = int(_a.split("=", 1)[1])
    if "--train" in sys.argv:
        print(json.dumps(train(horizon=_H), ensure_ascii=False, indent=1))
    elif "--predict" in sys.argv:
        print(json.dumps({k: v for k, v in predict(horizon=_H).items() if k != "rows"},
                         ensure_ascii=False, indent=1))
    elif "--loop" in sys.argv:
        loop()
    else:
        print(json.dumps(run_once(), ensure_ascii=False, indent=1)[:1200])
