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
from . import analysis as AN
from . import confidence as C
from . import costs as CO
from . import risk as RK
from . import meta as MT
from . import stats as ST
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
    rt = CO.round_trip_bps(market, notional=1e6)
    cost_bps = float(rt["round_trip_bps"])
    print(f"      cost model [{market}]: round-trip {cost_bps:.2f} bps "
          f"(buy {rt['buy_bps']:.2f} / sell {rt['sell_bps']:.2f})", flush=True)
    bt_ls = run_long_short(res, horizon, n_groups=5, cost_bps=cost_bps)
    try:
        ls_ser = AN.long_short_series(res["scores"], p3d.RET[horizon].T, horizon)
        if ls_ser.size:
            cost_per_period = 0.7 * cost_bps / 1e4
            eq_curve = np.cumprod(1.0 + ls_ser - cost_per_period)
            step = max(1, eq_curve.size // 250)
            bt_ls["equity"] = [round(float(x), 6) for x in eq_curve[::step]]
            bt_ls["equity_note"] = "非重叠 h 日调仓，已扣双边成本，下采样至 ~250 点"
    except Exception as e:  # noqa: BLE001
        bt_ls["equity_error"] = f"{type(e).__name__}: {e}"
    bt = {"long_short": bt_ls,
          "baselines": run_baselines(res, res_eq, horizon, cost_bps=cost_bps, n_groups=5)}

    print("[3.5/7] statistical edge test", flush=True)
    edge = {}
    for hh in (1, horizon, 20):
        edge[str(hh)] = AN.edge_report(res["scores"], p3d.RET[hh].T, res["dates"],
                                       h=hh, n_trials=S)
    eq_cmp = AN.compare_learned_vs_equal(res, res_eq, p3d, horizon)

    meta = MT.learn_meta(res["scores_TN"], p3d.RET[horizon], horizon, warmup=cfg.warmup)
    meta_cmp = MT.compare_raw_vs_meta(res["scores_TN"], p3d.RET[horizon], meta)
    meta_ic = MT.meta_ic(res["scores_TN"], p3d.RET[horizon], meta, horizon)
    _ic_series = AN.daily_ic_series(res["scores"], p3d.RET[horizon].T)
    mon = ST.rolling_edge_monitor(_ic_series[np.isfinite(_ic_series)])
    _hr = meta_cmp.get("raw", {}).get("hit_rate")
    _hm = meta_cmp.get("meta", {}).get("hit_rate")
    _fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
    print(f"      IC h={horizon}: {edge[str(horizon)]['ic']['mean']:+.4f} "
          f"t={edge[str(horizon)]['ic']['t']:+.2f} | meta sign={meta['final_sign']:+d} "
          f"hit {_fmt(_hr)}->{_fmt(_hm)}", flush=True)

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
                   PaperConfig(top_k=10, hold=horizon, cost_bps=cost_bps), start_t=cfg.warmup)
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
    _zoo = signal_zoo()
    fam = {s.sid: s.family for s in _zoo}
    fam_list = [s.family for s in _zoo]

    # 风险分摊体检：信号族集中度 + 组合分散度 + **相关性调整后的有效独立数**
    w_final = res["weights_hist"][-1]
    try:
        _T, _N, _S = p3d.shape
        _ret = p3d.RET[horizon]
        _pay = np.full((_T, _S), np.nan)
        for _t in range(_T):
            _d = p3d.DIR[_t].astype(np.float32)
            _s = p3d.STR[_t].astype(np.float32) * (1.0 / 255.0)
            _r = _ret[_t]; _sg = p3d.SIGMA[_t]
            _ok = np.isfinite(_r) & np.isfinite(_sg) & (_sg > 1e-9)
            if _ok.sum() < 3:
                continue
            _y = _d[_ok] * (_r[_ok][:, None] / (_sg[_ok][:, None] * np.sqrt(horizon)))
            _act = _d[_ok] != 0
            _cnt = _act.sum(axis=0)
            _m = _cnt > 0
            _pay[_t, _m] = np.where(_act, _y, 0.0).sum(axis=0)[_m] / _cnt[_m]
        _corr = pd.DataFrame(_pay).corr().fillna(0.0).to_numpy().copy()
        np.fill_diagonal(_corr, 1.0)
        eff_corr = RK.effective_n_corr(w_final, _corr)
        _abs = np.abs(_corr - np.eye(_S))
        mirror_frac = float((_abs > 0.8).sum() / max(_S * (_S - 1), 1))
        corr_info = {"effective_n_corr": eff_corr,
                     "effective_n_weight": RK.effective_n(w_final),
                     "mirror_pair_frac": round(mirror_frac, 4),
                     "note": "按相关性算的有效独立下注数，远小于按权重算的数字"}
    except Exception as e:  # noqa: BLE001
        corr_info = {"error": f"{type(e).__name__}: {e}"}
    div_sig = RK.diversification_report(w_final, family_map=fam_list, min_effective=10.0)
    pw = RK.portfolio_weights(res["scores"][:, -1], p3d.SIGMA[-1],
                              sectors=None, top_k=10, max_weight=0.15)
    div_port = RK.diversification_report(pw["weights"], min_effective=5.0) if pw.get("ok") else {}
    div_sig["corr_adjusted"] = corr_info
    if corr_info.get("effective_n_corr") and corr_info["effective_n_corr"] < 45:
        div_sig["warnings"].append(
            f"相关性调整后有效独立信号仅 {corr_info['effective_n_corr']:.0f}/73，"
            "表面分散实则高度同质")
    risk_rep = {"signal_level": div_sig, "portfolio_level": div_port,
                "n_positions": pw.get("n_positions"),
                "note": "信号层只削峰（alpha 仍由学习器决定）；组合层逆波动 + 单票/行业上限"}
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
        "risk": risk_rep,
        "costs": {"market": market, "round_trip_bps": cost_bps,
                  "breakdown": rt["breakdown"], "verified_on": rt["verified_on"],
                  "table": CO.summary_table(),
                  "scenarios": CO.cost_scenarios(),
                  "drag": CO.annual_cost_drag(market, 0.7, horizon)},
        "edge": edge,
        "learned_vs_equal": eq_cmp,
        "meta": {"final_sign": meta["final_sign"], "verdict": meta["verdict"],
                 "final_mu": meta["final_mu"], "final_t": meta["final_t"],
                 "frac_inverted": meta["frac_inverted"], "frac_silent": meta["frac_silent"],
                 "compare": meta_cmp,
                 "meta_ic": {"mean": meta_ic["mean"], "t": meta_ic["t"], "p": meta_ic["p"]},
                 "monitor": {"frac_time_active": mon["frac_active"],
                             "final_state": mon["final_state"], "final_t": mon["final_t"]}},
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
           "confidence": calrep, "paper": cmp_, "paths": paths,
           "edge": edge, "meta": {"verdict": meta["verdict"], "sign": meta["final_sign"],
                                  "meta_ic_t": meta_ic["t"]},
           "learned_vs_equal": eq_cmp}

    if report_path:
        try:
            md = render(res, mets, bt, report_path)
            from .report_stats import append_stats
            append_stats(report_path, edge, eq_cmp, meta, meta_cmp, meta_ic, mon)
            from .report_stats import append_costs
            append_costs(report_path, market, cost_bps, rt, CO)
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
