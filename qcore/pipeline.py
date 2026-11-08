"""
全流程编排 —— 一条命令跑完：数据 → 信号 → 自学习 → 评估 → 可信度 → 模拟盘 → 报告 → UI 产物。

调用：
    python -m qcore.pipeline --pool 300 --start 2013-01-01 --report reports/apb_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import store
from . import confidence as C
from .backtest import run_baselines, run_long_short
from .context import stock_context
from .data import get_pool, load_kline
from .dataset import build_dataset
from .engine import RunConfig, run_walkforward
from .metrics import evaluate_all, weight_concentration
from .paper import PaperConfig, compare, export_paper, live_nav, load_live_trades, simulate
from .report import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def _san(o):
    """递归把 numpy / pandas 对象转成可 JSON 化的原生类型。"""
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, dict):
        return {str(k): _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def run_full(pool_n: int = 300, start: str = "2013-01-01", horizon: int = 5,
             report_path: Optional[str] = None, market: str = "CN",
             do_context: int = 8) -> Dict:
    t0 = time.time()
    pool = get_pool(top_n=pool_n)
    codes = pool["code"].tolist()
    print(f"[1/7] data: {len(codes)} stocks", flush=True)

    p3d = build_dataset(codes, start=start, horizon=horizon, extra_horizons=[1, 20],
                        verbose=False)
    T, N, S = p3d.shape
    print(f"      dataset: T={T} N={N} S={S} ({p3d.nbytes()/1e6:.0f} MB)", flush=True)

    print("[2/7] walk-forward (learned / equal)", flush=True)
    cfg = RunConfig(horizon=horizon, warmup=120, record_ledger=True)
    res = run_walkforward(p3d, cfg)
    cfg_eq = RunConfig(horizon=horizon, warmup=120, mode="equal", record_ledger=False)
    res_eq = run_walkforward(p3d, cfg_eq)

    print("[3/7] metrics + backtest", flush=True)
    mets = evaluate_all(res, horizons=(1, 5, 20), n_groups=5)
    mets["weight_concentration"] = weight_concentration(res["weights_hist"], sids=res["sids"])
    bt = {"long_short": run_long_short(res, horizon, n_groups=5, cost_bps=10.0),
          "baselines": run_baselines(res, res_eq, horizon, cost_bps=10.0, n_groups=5)}

    print("[4/7] confidence calibration", flush=True)
    cal = C.calibrate(res["scores_TN"], p3d.RET[horizon], horizon, warmup=cfg.warmup)
    p_conf = C.combine(cal["confidence"], res["concord"].T, res["quality"].T)
    ret = p3d.RET[horizon]
    hit = (np.sign(res["scores_TN"]) == np.sign(ret)).astype(float)
    hit[~np.isfinite(ret)] = np.nan
    calrep = C.calibration_report(p_conf, hit)
    print(f"      brier={calrep.get('brier')} ece={calrep.get('ece')} "
          f"skill={calrep.get('brier_skill_vs_base')}", flush=True)

    print("[5/7] paper trading (sim vs live)", flush=True)
    sim = simulate(res["scores_TN"], p3d.OPEN, p3d.CLOSE, p3d.VALID, p3d.dates,
                   PaperConfig(top_k=10, hold=horizon, cost_bps=10.0), start_t=cfg.warmup)
    lt = load_live_trades()
    lv = live_nav(lt, p3d.CLOSE, p3d.dates, codes)
    cmp_ = compare(sim, lv, lt, sim["trades"])
    paper_ui = export_paper(sim, lv, cmp_, p3d.dates)

    print("[6/7] context (now / history)", flush=True)
    top_codes = []
    last = res["scores"][:, -1]
    if np.isfinite(last).any():
        top_codes = [codes[i] for i in np.argsort(-np.nan_to_num(last, nan=-9))[:do_context]]
    ctx = {}
    for c in top_codes:
        try:
            ctx[c] = stock_context(load_kline(c), h=horizon, k=30, name=c)
        except Exception as e:  # noqa: BLE001
            ctx[c] = {"ok": False, "reason": str(e)}

    print("[7/7] export", flush=True)
    from .signals import signal_zoo
    fam = {s.sid: s.family for s in signal_zoo()}
    paths = store.export_ui_bundle(res, p3d, family_of=fam, paper=paper_ui)

    perf = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"T": int(T), "N": int(N), "S": int(S), "start": str(p3d.meta.get("date_from")),
                    "end": str(p3d.meta.get("date_to")), "horizon": horizon},
        "metrics": mets,
        "backtest": {"long_short": {k: v for k, v in bt["long_short"].items()
                                    if not isinstance(v, np.ndarray)},
                     "baselines": bt["baselines"].to_dict("records") if hasattr(bt["baselines"], "to_dict") else bt["baselines"]},
        "confidence": calrep,
        "paper": cmp_,
        "context": ctx,
    }
    store.write_json("performance", _san(perf))

    led = res.get("ledger")
    if led is not None and len(led):
        store.append_ledger(led[led["as_of"] >= p3d.dates[-1] - np.timedelta64(90, "D")].to_dict("records"))

    st = store.read_json("status", default={}) or {}
    st["last_processed"] = str(pd.Timestamp(p3d.dates[-1]).date())
    st["rebuilt_at"] = datetime.now().isoformat(timespec="seconds")
    store.write_json("status", st)
    store.save_weighter(res["weighter"])

    out = {"ok": True, "seconds": round(time.time() - t0, 1),
           "shape": [int(T), int(N), int(S)], "metrics": mets, "backtest": perf["backtest"],
           "confidence": calrep, "paper": cmp_, "paths": paths}

    if report_path:
        try:
            md = render(res, mets, bt, report_path)
            out["report"] = report_path
        except Exception as e:  # noqa: BLE001
            out["report_error"] = f"{type(e).__name__}: {e}"
    print(f"done in {out['seconds']}s", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--report", default=os.path.join(REPORT_DIR, "apb_report.md"))
    a = ap.parse_args(argv)
    out = run_full(a.pool, a.start, a.horizon, a.report)
    print(json.dumps({k: v for k, v in out.items() if k not in ("metrics", "backtest")},
                     ensure_ascii=False, default=_json_default)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
