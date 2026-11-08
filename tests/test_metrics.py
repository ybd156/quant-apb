"""
合成数据单测：已知答案验证评估三件套。

运行：python -m tests.test_metrics

原则：每个断言都有解析可算的真值（完全正相关 -> IC=1、命中率=1），
而不是"跑通就行"。所有 print 只用英文（控制台 GBK，中文会炸）。
"""
from __future__ import annotations

import math
import os
import shutil
import traceback

import numpy as np
import pandas as pd

from qcore.metrics import (assign_groups, decile_returns, evaluate_all,
                           hit_rate, rank_ic, spearman, weight_concentration)
from qcore.backtest import run_baselines, run_long_short
from qcore.report import render

# --------------------------------------------------------------------------- #
# 合成数据构造
# --------------------------------------------------------------------------- #
def make_res(n=51, t_days=420, drift_lo=-0.002, drift_hi=0.002,
             score_noise=0.0, seed=7, horizons=(1, 5, 20), const_score=True):
    """
    构造一个自洽的 res：
      CLOSE[i,d] = P0_i * (1+drift_i)^d
      OPEN[i,d]  = CLOSE[i,d-1]        (开盘=前收，于是 RET[h] 有解析式)
      RET[h][i,t] = OPEN[i,t+1+h]/OPEN[i,t+1] - 1 = (1+drift_i)^h - 1

    score 与 drift 单调同向（const_score=True 时 score 恒等于按 drift 的排名），
    因此 IC / 命中率 / 分组收益都有真值可对。
    """
    rng = np.random.default_rng(seed)
    drifts = np.linspace(drift_lo, drift_hi, n)
    base = np.full(n, 10.0)

    d_idx = np.arange(t_days, dtype=np.float64)
    CLOSE = base[:, None] * (1.0 + drifts)[:, None] ** d_idx[None, :]
    # 开盘价取 d-0.5 的指数位置：既保证 OPEN != 前收（避免期末收益恒为 0），
    # 又让 RET[h] 仍有解析式 (1+drift)^h - 1。
    OPEN = base[:, None] * (1.0 + drifts)[:, None] ** (d_idx[None, :] - 0.5)

    # score：按 drift 排序的 [-1,1] 均匀刻度（与 drift 完全单调同向）
    if n == 1:
        s0 = np.array([1.0])
    else:
        rank01 = np.argsort(np.argsort(drifts)) / (n - 1)  # 0..1
        s0 = rank01 * 2.0 - 1.0
    if const_score:
        scores = np.repeat(s0[:, None], t_days, axis=1)
    else:
        scores = np.repeat(s0[:, None], t_days, axis=1)
        scores = scores + rng.normal(0.0, score_noise, size=(n, t_days))

    RET = {}
    for h in horizons:
        r = np.full((n, t_days), np.nan)
        for t in range(t_days - 1 - h):
            r[:, t] = OPEN[:, t + 1 + h] / OPEN[:, t + 1] - 1.0
        RET[h] = r

    dates = pd.bdate_range("2020-01-01", periods=t_days).to_numpy()
    regime = np.tile(np.array([0, 1, 2, 3], dtype=np.int8), t_days // 4 + 1)[:t_days]

    # weights: 前 3 个信号占大头，其余均分
    S = 6
    w = np.zeros((t_days, S))
    w[:, 0], w[:, 1], w[:, 2] = 0.4, 0.25, 0.15
    w[:, 3:] = 0.20 / 3
    return dict(dates=dates, codes=[f"C{i:03d}" for i in range(n)], scores=scores,
                OPEN=OPEN, CLOSE=CLOSE, RET=RET, REGIME=regime,
                weights_hist=w, sids=[f"sig{j}" for j in range(S)],
                meta={"n_stocks": n, "n_days": t_days},
                config={"horizon": 5, "cost_bps": 10})


def _negate_scores(res):
    r = dict(res)
    r["scores"] = -np.asarray(res["scores"])
    return r


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("[PASS] " if cond else "[FAIL] ") + name + ("  " + detail if detail else ""))


def test_positive_monotone():
    res = make_res()
    sc, RET = res["scores"], res["RET"]
    for h in (1, 5, 20):
        ic = rank_ic(sc, RET, h, dates=res["dates"])
        hr = hit_rate(sc, RET, h, dates=res["dates"], regime=res["REGIME"])
        check(f"IC(h={h}) == 1.0", abs(ic["mean"] - 1.0) < 1e-9,
              f"mean={ic['mean']:.6f} n_days={ic['n_days']}")
        check(f"hit(h={h}) == 1.0", abs(hr["hit_rate"] - 1.0) < 1e-12,
              f"hit={hr['hit_rate']:.6f} n={hr['n_samples']}")
        check(f"IC>0 ratio(h={h}) == 1.0", abs(ic["positive_ratio"] - 1.0) < 1e-12,
              f"ratio={ic['positive_ratio']:.4f}")
        check(f"IC std(h={h}) == 0", (ic["std"] == 0) or abs(ic["std"]) < 1e-12,
              f"std={ic['std']:.3e}")


def test_negative_monotone():
    res = make_res()
    neg = _negate_scores(res)
    for h in (1, 5):
        ic = rank_ic(neg["scores"], res["RET"], h, dates=res["dates"])
        hr = hit_rate(neg["scores"], res["RET"], h, dates=res["dates"])
        check(f"neg IC(h={h}) == -1.0", abs(ic["mean"] + 1.0) < 1e-9,
              f"mean={ic['mean']:.6f}")
        check(f"neg hit(h={h}) == 0.0", abs(hr["hit_rate"]) < 1e-12,
              f"hit={hr['hit_rate']:.6f}")


def test_random_ic_near_zero():
    rng = np.random.default_rng(42)
    n, T = 50, 400
    drifts = rng.normal(0.0, 0.002, n)
    CLOSE = 10.0 * (1.0 + drifts)[:, None] ** np.arange(T)[None, :]
    OPEN = np.empty_like(CLOSE)
    OPEN[:, 0] = CLOSE[:, 0]
    OPEN[:, 1:] = CLOSE[:, :-1]
    RET = {1: np.full((n, T), np.nan), 5: np.full((n, T), np.nan)}
    for h in (1, 5):
        for t in range(T - 1 - h):
            RET[h][:, t] = OPEN[:, t + 1 + h] / OPEN[:, t + 1] - 1.0
    scores = rng.uniform(-1.0, 1.0, size=(n, T))
    for h in (1, 5):
        ic = rank_ic(scores, RET, h)
        check(f"random |IC(h={h})| < 0.05", abs(ic["mean"]) < 0.05,
              f"mean={ic['mean']:+.4f} std={ic['std']:.4f} n_days={ic['n_days']}")
        hr = hit_rate(scores, RET, h)
        check(f"random hit(h={h}) ~ 0.5", abs(hr["hit_rate"] - 0.5) < 0.05,
              f"hit={hr['hit_rate']:.4f}")


def test_decile_monotone():
    res = make_res()
    df = decile_returns(res["scores"], res["RET"], 5, n=5)
    rets = df["mean_ret"].to_numpy()
    check("decile mean_ret strictly increasing",
          bool(np.all(np.diff(rets) > 0)),
          "rets=" + ",".join(f"{v:.5f}" for v in rets))
    check("decile monotonicity == 1", abs(df.attrs["monotonicity"] - 1.0) < 1e-9,
          f"mono={df.attrs['monotonicity']:.6f}")
    check("decile top_minus_bottom > 0", df.attrs["top_minus_bottom"] > 0,
          f"spread={df.attrs['top_minus_bottom']:.6f}")
    check("decile hit_rate monotone", bool(np.all(np.diff(df["hit_rate"].to_numpy()) >= 0)),
          "hit=" + ",".join(f"{v:.3f}" for v in df["hit_rate"].to_numpy()))
    # 分组规模均衡
    sizes = df["n_obs"].to_numpy()
    check("decile group sizes balanced", sizes.max() - sizes.min() <= res["scores"].shape[1],
          f"n_obs={sizes.tolist()}")


def test_backtest_no_cost():
    res = make_res(n=51, t_days=420)
    bt = run_long_short(res, h=5, n_groups=5, cost_bps=0.0)
    check("bt ok", bt["ok"], str(bt.get("reason", "")))
    eq = bt["equity"]
    check("bt equity strictly increasing (no cost)",
          bool(np.all(np.diff(eq) > 0)), f"eq[0]={eq[0]:.4f} eq[-1]={eq[-1]:.4f}")
    check("bt LS ann > 0", bt["ann_return"] > 0,
          f"ann={bt['ann_return']*100:.2f}% sharpe={bt['sharpe']:.2f}")
    check("bt long ann > 0", bt["ann_long"] > 0, f"ann_long={bt['ann_long']*100:.2f}%")
    check("bt short ann < 0", bt["ann_short"] < 0, f"ann_short={bt['ann_short']*100:.2f}%")
    check("bt max_drawdown == 0", abs(bt["max_drawdown"]) < 1e-12,
          f"mdd={bt['max_drawdown']:.3e}")
    check("bt n_periods ~= (T-h)/h", abs(bt["n_periods"] - (420 - 5 - 1) / 5) <= 2,
          f"n_periods={bt['n_periods']}")
    check("bt group size == ceil(N/5)-ish", 9 <= bt["group_sizes"]["long"]["mean"] <= 11,
          f"long_mean={bt['group_sizes']['long']['mean']:.1f}")
    return bt


def test_backtest_cost_monotone():
    res = make_res(n=51, t_days=420)
    b0 = run_long_short(res, h=5, n_groups=5, cost_bps=0.0)
    b1 = run_long_short(res, h=5, n_groups=5, cost_bps=50.0)
    b2 = run_long_short(res, h=5, n_groups=5, cost_bps=500.0)
    check("cost 0 > cost 50 > cost 500 (ann)",
          b0["ann_return"] > b1["ann_return"] > b2["ann_return"],
          f"{b0['ann_return']*100:.2f}% > {b1['ann_return']*100:.2f}% > {b2['ann_return']*100:.2f}%")
    check("cost_total_bps grows with cost",
          b0["cost_total_bps"] < b1["cost_total_bps"] < b2["cost_total_bps"],
          f"{b0['cost_total_bps']:.1f} < {b1['cost_total_bps']:.1f} < {b2['cost_total_bps']:.1f}")


def test_backtest_turnover():
    res = make_res(n=51, t_days=420, score_noise=0.45, const_score=False, seed=11)
    bt = run_long_short(res, h=5, n_groups=5, cost_bps=10.0)
    check("noisy scores -> turnover > 0", bt["turnover_avg"] > 0.1,
          f"turnover_avg={bt['turnover_avg']:.3f} ann={bt['ann_return']*100:.2f}%")
    b0 = run_long_short(res, h=5, n_groups=5, cost_bps=0.0)
    check("turnover makes cost matter", b0["ann_return"] > bt["ann_return"],
          f"{b0['ann_return']*100:.2f}% -> {bt['ann_return']*100:.2f}%")


def test_baselines():
    res = make_res(n=51, t_days=420)
    base = make_res(n=51, t_days=420, score_noise=0.9, const_score=False, seed=3)
    df = run_baselines(res, base, h=5, cost_bps=10.0)
    check("baselines 4 rows", len(df) == 4, f"rows={len(df)}")
    names = df["策略"].tolist()
    check("baselines names", names == ["学习器(多空)", "等权集成(多空)", "随机打分(多空)", "买入持有(等权多头)"],
          str(names))
    learn = float(df.loc[df["策略"] == "学习器(多空)", "年化收益"].iloc[0])
    check("learner beats noise baseline", learn > 0, f"learner_ann={learn:.2f}%")
    hold = float(df.loc[df["策略"] == "买入持有(等权多头)", "年化收益"].iloc[0])
    check("buy&hold ann finite", np.isfinite(hold), f"hold_ann={hold:.2f}%")


def test_concentration():
    T, S = 120, 8
    rng = np.random.default_rng(5)
    w = rng.uniform(0.0, 1.0, size=(T, S))
    w = w / w.sum(axis=1, keepdims=True)
    c = weight_concentration(w, sids=[f"s{i}" for i in range(S)])
    check("hhi in (1/S, 1]", 1.0 / S - 1e-12 <= c["hhi_mean"] <= 1.0 + 1e-12,
          f"hhi_mean={c['hhi_mean']:.4f}")
    check("effective_n in [1, S]", 1.0 <= c["effective_n_mean"] <= S + 1e-9,
          f"eff_n={c['effective_n_mean']:.2f}")
    check("top5 share in (0.5, 1]", 0.5 < c["topk_mean"] <= 1.0,
          f"top5={c['topk_mean']:.4f}")

    onehot = np.zeros((T, S))
    onehot[:, 0] = 1.0
    c2 = weight_concentration(onehot)
    check("onehot hhi == 1", abs(c2["hhi_mean"] - 1.0) < 1e-12, f"{c2['hhi_mean']:.6f}")
    check("onehot effective_n == 1", abs(c2["effective_n_mean"] - 1.0) < 1e-12,
          f"{c2['effective_n_mean']:.6f}")
    check("onehot top5 == 1", abs(c2["topk_mean"] - 1.0) < 1e-12, f"{c2['topk_mean']:.6f}")

    eqw = np.full((T, S), 1.0 / S)
    c3 = weight_concentration(eqw)
    check("equal weight hhi == 1/S", abs(c3["hhi_mean"] - 1.0 / S) < 1e-12,
          f"{c3['hhi_mean']:.6f} vs {1.0/S:.6f}")
    check("equal weight eff_n == S", abs(c3["effective_n_mean"] - S) < 1e-9,
          f"{c3['effective_n_mean']:.6f}")


def test_spearman_helper():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    check("spearman identity", abs(spearman(a, a) - 1.0) < 1e-12, f"{spearman(a, a):.6f}")
    check("spearman reverse", abs(spearman(a, -a) + 1.0) < 1e-12, f"{spearman(a, -a):.6f}")
    # ties: [1,1,2,3] vs [1,2,3,4]
    t1 = np.array([1.0, 1.0, 2.0, 3.0])
    t2 = np.array([1.0, 2.0, 3.0, 4.0])
    v = spearman(t1, t2)
    check("spearman handles ties", np.isfinite(v) and -1 <= v <= 1, f"ties_rho={v:.4f}")
    check("spearman constant -> nan", np.isnan(spearman(np.ones(5), a)), "")


def test_assign_groups():
    v = np.array([5.0, 1.0, 3.0, 2.0, 4.0, np.nan, 6.0])
    g = assign_groups(v, 3)
    check("assign_groups returns 3 groups", g is not None and len(g) == 3,
          f"n={0 if g is None else len(g)}")
    if g:
        check("assign_groups low group is min", set(g[0].tolist()) == {1, 3}, str(g[0].tolist()))
        check("assign_groups high group is max", set(g[-1].tolist()) == {0, 6}, str(g[-1].tolist()))
        check("assign_groups no NaN member", all(not np.isnan(v[i]) for gg in g for i in gg), "")
        check("assign_groups covers all finite", sum(len(x) for x in g) == 6,
              f"covered={sum(len(x) for x in g)}")
        check("assign_groups equal size", max(len(x) for x in g) - min(len(x) for x in g) <= 1,
              str([len(x) for x in g]))
    check("assign_groups too few -> None", assign_groups(np.array([1.0, np.nan, np.nan]), 3) is None, "")


def test_layout_agnostic():
    """engine 目前输出 (T,N) 时间主序；必须与 (N,T) 得到完全一致的结果。"""
    res = make_res(n=41, t_days=200)
    sc_nt = res["scores"]
    sc_tn = np.ascontiguousarray(sc_nt.T)          # (T,N)
    for h in (1, 5):
        a = rank_ic(sc_nt, res["RET"], h)
        b = rank_ic(sc_tn, res["RET"], h)
        check(f"layout-agnostic IC(h={h})", abs(a["mean"] - b["mean"]) < 1e-12,
              f"{a['mean']:.6f} vs {b['mean']:.6f}")
        ha = hit_rate(sc_nt, res["RET"], h)
        hb = hit_rate(sc_tn, res["RET"], h)
        check(f"layout-agnostic hit(h={h})", abs(ha["hit_rate"] - hb["hit_rate"]) < 1e-12, "")

    r_tn = dict(res)
    r_tn["scores"] = sc_tn
    b1 = run_long_short(res, 5, 5, 10.0)
    b2 = run_long_short(r_tn, 5, 5, 10.0)
    check("layout-agnostic backtest ann",
          abs(b1["ann_return"] - b2["ann_return"]) < 1e-9,
          f"{b1['ann_return']:.6f} vs {b2['ann_return']:.6f}")

    df1 = run_baselines(res, r_tn, 5, 10.0)
    check("layout-agnostic baselines rows", len(df1) == 4, f"rows={len(df1)}")


def test_edge_cases():
    n, T = 20, 60
    res = make_res(n=n, t_days=T)

    # 1) 全 NaN 分数
    r_all_nan = dict(res)
    r_all_nan["scores"] = np.full((n, T), np.nan)
    hr = hit_rate(r_all_nan["scores"], res["RET"], 5)
    check("all-nan hit_rate -> nan, n=0",
          np.isnan(hr["hit_rate"]) and hr["n_samples"] == 0 and hr["insufficient"], "")
    ic = rank_ic(r_all_nan["scores"], res["RET"], 5)
    check("all-nan IC -> nan, n_days=0",
          np.isnan(ic["mean"]) and ic["n_days"] == 0 and ic["insufficient"], "")
    dec = decile_returns(r_all_nan["scores"], res["RET"], 5, n=5)
    check("all-nan decile -> 5 rows all nan",
          len(dec) == 5 and bool(np.all(np.isnan(dec["mean_ret"].to_numpy()))), "")
    bt = run_long_short(r_all_nan, h=5, n_groups=5, cost_bps=10)
    check("all-nan backtest -> ok False", not bt["ok"], str(bt.get("reason")))
    c = weight_concentration(np.full((T, 4), np.nan))
    check("all-nan weights -> insufficient", c["insufficient"] and np.isnan(c["hhi_mean"]), "")

    # 2) 单只股票
    r1 = make_res(n=1, t_days=T)
    ic1 = rank_ic(r1["scores"], r1["RET"], 5)
    check("single stock IC -> n_days=0", ic1["n_days"] == 0 and ic1["insufficient"], "")
    bt1 = run_long_short(r1, h=5, n_groups=5, cost_bps=10)
    check("single stock backtest -> ok False", not bt1["ok"], str(bt1.get("reason")))
    hr1 = hit_rate(r1["scores"], r1["RET"], 5)
    check("single stock hit_rate works", np.isfinite(hr1["hit_rate"]),
          f"hit={hr1['hit_rate']:.3f}")

    # 3) T 很小
    r_small = make_res(n=10, t_days=3)
    for fn, args in ((hit_rate, (r_small["scores"], r_small["RET"], 1)),
                     (rank_ic, (r_small["scores"], r_small["RET"], 1))):
        out = fn(*args)
        check(f"tiny T {fn.__name__} no crash", isinstance(out, dict), "")
    check("tiny T decile no crash",
          isinstance(decile_returns(r_small["scores"], r_small["RET"], 1, n=5), pd.DataFrame), "")
    bt_small = run_long_short(r_small, h=1, n_groups=5, cost_bps=10)
    check("tiny T backtest -> ok False", not bt_small["ok"], str(bt_small.get("reason")))

    # 4) h 不在 RET 里
    try:
        hit_rate(res["scores"], res["RET"], 999)
        check("missing horizon raises", False, "no exception")
    except KeyError:
        check("missing horizon raises KeyError", True, "")

    # 5) 空 weights_hist
    c0 = weight_concentration(np.zeros((0, 4)))
    check("empty weights no crash", isinstance(c0, dict), "")


def test_evaluate_all_and_report():
    res = make_res(n=51, t_days=420)
    m = evaluate_all(res, horizons=(1, 5, 20), n_groups=5)
    check("evaluate_all keys", set(["hit_rate", "rank_ic", "decile", "concentration"]) <= set(m.keys()),
          str(sorted(m.keys())))
    check("evaluate_all all h", m["horizons"] == [1, 5, 20], str(m["horizons"]))

    bt = {"long_short": run_long_short(res, 5, 5, 10.0),
          "baselines": run_baselines(res, make_res(n=51, t_days=420, score_noise=0.9,
                                                   const_score=False, seed=3), 5, 10.0)}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "data", "_test_report")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "report.md")
    out = render(res, m, bt, path)
    txt = open(out, encoding="utf-8").read()
    check("report file written", os.path.exists(out) and len(txt) > 1500,
          f"chars={len(txt)}")
    for kw in ("配置摘要", "样本区间", "方向命中率", "Rank IC", "分组收益",
               "多空组合绩效", "权重", "分年度稳定性", "结论", "局限"):
        check(f"report contains '{kw}'", kw in txt, "")
    check("report has LEAD_FILL placeholders", txt.count("LEAD_FILL") == 2,
          f"count={txt.count('LEAD_FILL')}")
    # 空 metrics 也不崩
    out2 = render(res, {}, {}, os.path.join(tmp, "report_empty.md"))
    check("report survives empty metrics", os.path.getsize(out2) > 200, "")
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_positive_monotone,
        test_negative_monotone,
        test_random_ic_near_zero,
        test_decile_monotone,
        test_backtest_no_cost,
        test_backtest_cost_monotone,
        test_backtest_turnover,
        test_baselines,
        test_concentration,
        test_spearman_helper,
        test_assign_groups,
        test_layout_agnostic,
        test_edge_cases,
        test_evaluate_all_and_report,
    ]
    failed = 0
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__} raised")
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    n_all = len(RESULTS)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_ok}/{n_all} assertions passed, {failed} test functions crashed")
    # 关键数值速览
    res = make_res()
    bt = run_long_short(res, 5, 5, 10.0)
    print(f"KEY  rank_ic(h=5)  = {rank_ic(res['scores'], res['RET'], 5)['mean']:+.6f}")
    print(f"KEY  hit_rate(h=5) = {hit_rate(res['scores'], res['RET'], 5)['hit_rate']:.6f}")
    print(f"KEY  ls_ann(h=5,10bps) = {bt['ann_return']*100:+.4f}%  sharpe={bt['sharpe']:.3f}")
    print("=" * 70)
    if n_ok != n_all or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
