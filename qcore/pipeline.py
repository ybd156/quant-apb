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
    market = str(market).upper()
    if market not in ("CN", "US", "HK"):
        market = "CN"

    # 按市场选池与加载器。不同市场的复权口径、涨跌停、停牌规则都不同，
    # 不能拿 A 股代码去喂美股加载器。
    loader, pad = None, True
    if market == "CN":
        pool = get_pool(top_n=pool_n)
    elif market == "US":
        from .us_data import get_pool_us, load_us_kline
        pool = get_pool_us(top_n=pool_n)
        loader, pad = load_us_kline, False
    else:  # HK
        from .us_data import get_pool_hk, load_hk_kline
        pool = get_pool_hk(top_n=pool_n)
        loader, pad = load_hk_kline, False

    codes = pool["code"].tolist()
    print(f"[1/7] data [{market}]: {len(codes)} stocks", flush=True)

    p3d = build_dataset(codes, start=start, horizon=horizon, extra_horizons=[1, 20],
                        verbose=False, loader=loader, pad=pad, market=market)
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
    # 学习器无效时给出具体原因（不是只说「无显著差异」）
    try:
        eq_cmp["diagnosis"] = AN.diagnose_learner(res, res_eq, p3d, h=horizon,
                                                   cmp=eq_cmp, sensitivity=False)
    except Exception as _e:  # noqa: BLE001
        eq_cmp["diagnosis"] = {"ok": False, "reason": str(_e)[:80]}

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
    from . import account_config as AC
    # 按市场取配置：A 股用 CN（美股/港股各有一套，不混用成本口径）
    _market = str(p3d.meta.get("market") or "CN").upper()
    if _market not in ("CN", "US", "HK"):
        _market = "CN"
    _acct = AC.load(_market)
    # 成本口径统一：回测与模拟盘必须站在同一个基准上。
    # 实测事故：account.json 写 cost_bps=10.0（低于 A 股法定下限 11.28bps），
    # 于是同一次输出里回测用 19.28bps、模拟盘用 10.0bps，差 9.28bps/期。
    _acct_cost = float(_acct["cost_bps"])
    # 容差 0.05bps：账户里存的是四舍五入后的 19.28，costs 算出 19.2815，
    # 用 1e-6 比会把「同一个口径」误报成不一致（自造的噪音告警）。
    if abs(_acct_cost - float(cost_bps)) > 0.05:
        print(f"      WARN cost basis mismatch: account={_acct_cost:.2f}bps "
              f"vs costs.round_trip={float(cost_bps):.2f}bps -> 用后者（唯一真源）",
              flush=True)
    print(f"      account[{_market}]: 初始资金 {_acct['initial_capital']:,.0f} "
          f"{_acct.get('currency','')} / 持仓 {_acct['top_k']} / "
          f"单票上限 {_acct['max_weight']:.0%} / 成本 {float(cost_bps):.2f}bps", flush=True)
    sim = simulate(res["scores_TN"], p3d.OPEN, p3d.CLOSE, p3d.VALID, p3d.dates,
                   PaperConfig(top_k=int(_acct["top_k"]), hold=horizon,
                               cost_bps=float(cost_bps),
                               initial=float(_acct["initial_capital"])),
                   start_t=cfg.warmup)
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
    # 代码->名称映射（UI 显示用）
    try:
        from . import names as _names_mod
        print("      names:", _names_mod.export_from_pool(pool), flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"      names export skipped: {_e}", flush=True)
    # 数据源健康度
    try:
        import shutil as _sh, os as _os
        _src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "data", "state", "source_health.json")
        if not _os.path.exists(_src):
            print("      source_health: 数据源未记录健康度", flush=True)
    except Exception:
        pass
    try:
        from .policy_calendar import to_state_json as _events_to_state
        _imp = []
        try:
            from . import impact as _IMP
            print("      impact: 事件研究计算中 ...", flush=True)
            _imp = _IMP.compute(pool, start="2016-01-01", max_stocks=min(len(pool), 300))
            print(f"      impact: {len(_imp)} 组（事件×板块×周期）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"      impact skipped: {_e}", flush=True)
        _events_to_state(impact=_imp)
    except Exception as _e:  # noqa: BLE001
        print(f"      events export skipped: {_e}", flush=True)
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
    paths = store.export_ui_bundle(res, p3d, family_of=fam, paper=paper_ui,
                                   market=_market)

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
    # 行为因子体检（真实计算，不是占位）
    try:
        from . import behavior_report as BR
        _codes = [str(c) for c in p3d.codes]
        _ind = {}
        _cap = {}
        try:
            _pool = get_pool(top_n=max(len(_codes), 50))
            _ind = dict(zip(_pool["code"].astype(str), _pool["industry"].astype(str)))
            _cap = dict(zip(_pool["code"].astype(str), _pool["mktcap"].astype(float)))
        except Exception:
            pass
        print("      behavior: 计算中 ...", flush=True)
        perf["behavior"] = BR.compute(_codes, start="2016-01-01", horizon=horizon,
                                      max_stocks=150, industry_map=_ind, cap_map=_cap)
        print(f"      behavior: {len(perf['behavior'])} 个因子", flush=True)
    except Exception as _e:  # noqa: BLE001
        perf["behavior"] = []
        perf["behavior_error"] = f"{type(_e).__name__}: {_e}"
        print(f"      behavior skipped: {_e}", flush=True)

    store.write_json("performance", _san(perf))

    # 成本口径落盘：/api/horizons 过去在**请求期**现算 annual_cost_drag，
    # 那是 UI 层在做计算（违反 INV-8），且与 performance.costs 两条来源。
    # 改成 qcore 一次算好、写进 costs.json，server 只读。
    try:
        from . import horizons as HZ
        store.write_json("costs", {
            "market": market,
            "round_trip_bps": round(float(cost_bps), 4),
            "buy_bps": round(float(rt["buy_bps"]), 4),
            "sell_bps": round(float(rt["sell_bps"]), 4),
            "annual_drag_pct": {str(h): round(HZ.annual_cost_drag(int(h), market) * 100, 2)
                                for h in (1, 3, 5, 10, 15, 20)},
            "turnover_assumption": 0.8,
            "verified_on": (perf.get("costs") or {}).get("verified_on"),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": "annual_drag_pct = 250/h × 换手0.8 × 往返bps；由 qcore 计算，UI 只读",
        })
    except Exception as _e:  # noqa: BLE001
        print(f"      costs.json skipped: {_e}", flush=True)

    led = res.get("ledger")
    if led is not None and len(led):
        store.append_ledger(led[led["as_of"] >= p3d.dates[-1] - np.timedelta64(90, "D")].to_dict("records"),
                        horizon=horizon, market=market)

    # 记录主台账属于哪个周期 —— 否则下游无法判断
    # 「已对账」该归到 h=1 还是 h=5，只能靠猜。
    store.write_json("ledger_meta", {
        "horizon": int(horizon),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n": int(len(led)) if led is not None else 0,
        "note": "主台账（pipeline 逐日重放）的周期归属",
    })

    st = store.read_json("status", default={}) or {}
    st["last_processed"] = str(pd.Timestamp(p3d.dates[-1]).date())
    st["rebuilt_at"] = datetime.now().isoformat(timespec="seconds")
    store.write_json("status", st)
    store.save_weighter(res["weighter"], market=market, horizon=horizon)

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
    ap.add_argument("--market", default="CN", choices=["CN", "US", "HK"],
                    help="市场：CN=A股，US=美股，HK=港股。各自独立成本与账户。")
    a = ap.parse_args(argv)
    out = run_full(a.pool, a.start, a.horizon, a.report, market=a.market)
    print(json.dumps({k: v for k, v in out.items() if k not in ("metrics", "backtest")},
                     ensure_ascii=False, default=_json_default)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
