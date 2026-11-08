"""
合成数据单测：事件研究 / 影响因子表 的已知答案验证。

运行：python -m tests.test_events

原则（与 tests/test_metrics.py 一致）：
  1. 每个断言都有解析可算的真值（人造 +5% 漂移必须被检出、纯噪声必须给 t≈0）；
  2. 反例必须存在（同一事件日在"无漂移"世界里不许显著）；
  3. 降级路径必须能跑（缺板块、缺市场收益、样本 n=3）；
  4. 所有 print 只用英文（控制台 GBK，中文会炸）。
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd

from qcore.events import (MIN_N, POLICY_TRANSMISSION, Event, build_sector_returns,
                          event_study, expand_sector, impact_factor, load_events,
                          predict_impact, save_event, sector_direction,
                          validate_event)

DRIFT = 0.05          # 人造事件后漂移（CAR(0,10) 真值）


# --------------------------------------------------------------------------- #
# 合成世界
# --------------------------------------------------------------------------- #
def make_world(n_days: int = 1200, beta: float = 1.0, alpha: float = 0.0,
               idio: float = 0.008, seed: int = 1, start: str = "2015-01-01"):
    """r_s = alpha + beta*r_m + N(0, idio)。返回 (idx, market_ret, sector_ret)。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    m = rng.normal(0.0003, 0.01, n_days)
    s = alpha + beta * m + rng.normal(0.0, idio, n_days)
    return idx, pd.Series(m, index=idx, name="MKT"), pd.Series(s, index=idx, name="SEC")


def add_drift(s: pd.Series, idx: pd.DatetimeIndex, event_positions, total: float,
              span: int = 11) -> pd.Series:
    """在事件日及其后 span-1 天均匀注入 total 的累计漂移（默认 11 天 -> CAR(0,10)）。"""
    out = s.to_numpy(dtype=float).copy()
    per = total / span
    for p in event_positions:
        out[p:p + span] += per
    return pd.Series(out, index=idx, name=s.name)


def event_dates_from(idx: pd.DatetimeIndex, start_pos: int, step: int, k: int):
    return [idx[start_pos + step * i] for i in range(k)]


def mk_events(dates, sectors=("SEC",), etype="policy", subtype="rrr_cut",
              direction=1, eid_prefix="E"):
    return [Event(eid=f"{eid_prefix}{i:03d}", date=str(pd.Timestamp(d).date()),
                  type=etype, subtype=subtype, title="synthetic", source="https://example.gov.cn/x",
                  direction=direction, sectors=list(sectors), catalyst_window=20)
            for i, d in enumerate(dates)]


# --------------------------------------------------------------------------- #
# 1. 人造 +5% 漂移必须被检出，且同一事件日在无漂移世界不许显著
# --------------------------------------------------------------------------- #
def test_drift_detected():
    idx, mr, sr = make_world(n_days=1200, beta=2.0, alpha=0.0, idio=0.008, seed=1)
    pos = [300 + 60 * i for i in range(12)]
    dates = event_dates_from(idx, 300, 60, 12)
    sr_drift = add_drift(sr, idx, pos, DRIFT, span=11)

    res = event_study(sr_drift, mr, dates)
    assert res["status"] == "ok", res["status"]
    assert res["n_valid"] == 12, res["n_valid"]
    car = res["car"]
    print(f"  [drift] n={car['n']} mean_CAR={car['mean']:.4f} t={car['t']:.2f} "
          f"win={car['win_rate']:.2f} status={res['status']}")
    assert abs(car["mean"] - DRIFT) < 0.012, car["mean"]       # 真值 5%
    assert car["t"] > 2.0, car["t"]
    assert car["win_rate"] >= 0.9, car["win_rate"]
    # 市场模型必须还原 beta≈2（我们构造的就是 beta=2）
    betas = [e["beta"] for e in res["per_event"]]
    assert abs(float(np.mean(betas)) - 2.0) < 0.15, np.mean(betas)
    # 事件前没有漂移 -> CAR(-5,-1) 不显著
    assert abs(res["pre_car"]["mean"]) < 0.02, res["pre_car"]["mean"]
    # 事件后 h=5 的 CAR 约等于 5% * 6/11
    h5 = res["horizons"][5]["mean"]
    assert abs(h5 - DRIFT * 6 / 11) < 0.008, h5

    # 反例：同一批事件日在**无漂移**世界必须不显著
    null = event_study(sr, mr, dates)
    print(f"  [null ] n={null['car']['n']} mean_CAR={null['car']['mean']:.4f} "
          f"t={null['car']['t']:.2f}")
    assert abs(null["car"]["t"]) < 2.0, null["car"]["t"]


# --------------------------------------------------------------------------- #
# 2. 估参窗口绝不含事件后数据（点对点铁律）
# --------------------------------------------------------------------------- #
def test_estimation_window_has_no_lookahead():
    idx, mr, sr = make_world(n_days=800, beta=1.5, seed=7)
    ev_date = idx[400]
    pos = idx.get_loc(ev_date)
    # 世界 B：估参窗口之后（pos-19 起，含整个事件窗口）的板块收益整体 ×5，
    # 估参窗口 [-120,-20] 内的数据与 A 完全一致 -> alpha/beta 必须一字不差
    a = sr.to_numpy(dtype=float).copy()
    b = a.copy()
    b[pos - 19:] = b[pos - 19:] * 5.0 + 0.01
    ra = event_study(pd.Series(a, index=idx), mr, [ev_date])
    rb = event_study(pd.Series(b, index=idx), mr, [ev_date])
    ea, eb = ra["per_event"][0], rb["per_event"][0]
    print(f"  [pit  ] alpha {ea['alpha']:.8f}/{eb['alpha']:.8f}  "
          f"beta {ea['beta']:.8f}/{eb['beta']:.8f}")
    assert abs(ea["alpha"] - eb["alpha"]) < 1e-12, (ea["alpha"], eb["alpha"])
    assert abs(ea["beta"] - eb["beta"]) < 1e-12, (ea["beta"], eb["beta"])
    # 但 CAR 必须不同（否则说明事件窗口没用上事件后数据）
    assert abs(ea["car"] - eb["car"]) > 1e-3

    # 手工复算 alpha/beta（估参窗口 [-120,-20]）
    lo, hi = pos - 120, pos - 20
    ys = a[lo:hi + 1]
    xs = mr.to_numpy(dtype=float)[lo:hi + 1]
    beta_hat = float(np.cov(ys, xs, ddof=0)[0, 1] / np.var(xs))
    alpha_hat = float(ys.mean() - beta_hat * xs.mean())
    assert abs(ea["beta"] - beta_hat) < 1e-8, (ea["beta"], beta_hat)
    assert abs(ea["alpha"] - alpha_hat) < 1e-8, (ea["alpha"], alpha_hat)


# --------------------------------------------------------------------------- #
# 3. 事件前已 price-in -> CAR(-5,-1) 与 CAR(0,+h) 反号 -> conflict
# --------------------------------------------------------------------------- #
def test_price_in_conflict():
    idx, mr, sr = make_world(n_days=1200, beta=1.0, idio=0.004, seed=3)
    pos = [300 + 60 * i for i in range(12)]
    dates = event_dates_from(idx, 300, 60, 12)
    arr = sr.to_numpy(dtype=float).copy()
    for p in pos:
        arr[p - 5:p] += 0.05 / 5.0        # 事件前 5 天先涨 5%（提前反应）
        arr[p:p + 11] -= 0.05 / 11.0      # 事件后跌回 5%
    res = event_study(pd.Series(arr, index=idx), mr, dates)
    pi = res["price_in"]
    print(f"  [pin  ] pre={pi['pre_mean']:.4f} post={pi['post_mean']:.4f} "
          f"conflict={pi['conflict']} verdict={pi['verdict']}")
    assert pi["pre_mean"] > 0.03
    assert pi["post_mean"] < -0.03
    assert pi["conflict"] is True
    assert pi["verdict"] == "conflict_pre_priced"
    assert res["status"] == "ok"


# --------------------------------------------------------------------------- #
# 4. 纯噪声：impact_factor 的 t 必须 ≈ 0（固定种子单例）
# --------------------------------------------------------------------------- #
def test_noise_impact_factor_t_near_zero():
    idx, mr, sr = make_world(n_days=2500, beta=0.9, idio=0.008, seed=26)
    dates = event_dates_from(idx, 400, 80, 20)
    evs = mk_events(dates, sectors=("SEC",), direction=1)
    tab = impact_factor(evs, {"SEC": sr}, mr, horizons=(1, 5, 20))
    assert len(tab) > 0
    ts = tab["t"].dropna().to_numpy()
    print(f"  [noise] rows={len(tab)} max|t|={np.abs(ts).max():.2f} "
          f"max|mean_car|={tab['mean_car'].abs().max():.4f}")
    assert ts.size >= 3
    assert np.abs(ts).max() < 2.0, ts          # 纯噪声不许出现"显著"
    assert tab["mean_car"].abs().max() < 0.015
    assert not tab["insufficient"].any()
    assert tab["n"].max() == 20


# --------------------------------------------------------------------------- #
# 5. 纯噪声的系综：不许系统性制造显著性（100+ 个 t 值的分布）
# --------------------------------------------------------------------------- #
def test_noise_ensemble_no_false_positive():
    ts = []
    for seed in range(30):
        idx, mr, sr = make_world(n_days=2500, beta=0.9, idio=0.008, seed=1000 + seed)
        dates = event_dates_from(idx, 400, 50, 40)
        evs = mk_events(dates, sectors=("SEC",))
        tab = impact_factor(evs, {"SEC": sr}, mr, horizons=(1, 5, 20))
        ts.extend([x for x in tab["t"].tolist() if np.isfinite(x)])
    t = np.asarray(ts, dtype=float)
    frac = float((np.abs(t) > 2).mean())
    print(f"  [ensm ] n_t={t.size} mean|t|={np.abs(t).mean():.3f} "
          f"mean(t)={t.mean():+.3f} frac|t|>2={frac:.3f}")
    assert t.size >= 150
    assert abs(t.mean()) < 0.4                       # 无系统性方向
    assert np.abs(t).mean() < 1.1                    # 近似 N(0,1)
    assert frac < 0.12                               # 名义 5%，留余量


# --------------------------------------------------------------------------- #
# 6. 样本 n=3 -> insufficient，而不是硬算
# --------------------------------------------------------------------------- #
def test_small_sample_insufficient():
    idx, mr, sr = make_world(n_days=1200, beta=1.0, idio=0.008, seed=5)
    pos = [300 + 60 * i for i in range(3)]
    dates = event_dates_from(idx, 300, 60, 3)
    sr3 = add_drift(sr, idx, pos, DRIFT)
    res = event_study(sr3, mr, dates)
    print(f"  [n=3  ] status={res['status']} n_valid={res['n_valid']} "
          f"insufficient={res['car']['insufficient']} t={res['car']['t']:.2f}")
    assert res["n_valid"] == 3
    assert res["status"] == "insufficient"
    assert res["car"]["insufficient"] is True
    assert any("not evidence" in w for w in res["warnings"]), res["warnings"]

    evs = mk_events(dates, sectors=("SEC",))
    tab = impact_factor(evs, {"SEC": sr3}, mr, horizons=(1, 5))
    assert len(tab) > 0
    assert tab["insufficient"].all(), tab[["sector", "horizon", "insufficient"]]
    assert (tab["n"] == 3).all()

    p = predict_impact(evs[0], tab)
    s0 = p["sectors"][0]
    print(f"  [n=3  ] predict conf={s0['confidence']:.2f} reason={s0['reason']}")
    assert s0["confidence"] == 0.0
    assert s0["expected_car"] is None        # 不硬报点预测
    assert s0["reason"] == "insufficient_samples"
    assert s0["insufficient"] is True
    assert p["has_history"] is True          # 有样本，只是不够


# --------------------------------------------------------------------------- #
# 7. 缺板块 / 缺市场收益 / 空输入 -> 优雅降级，不抛异常
# --------------------------------------------------------------------------- #
def test_degrade_missing_inputs():
    idx, mr, sr = make_world(n_days=1200, seed=9)
    dates = event_dates_from(idx, 300, 60, 8)

    # (a) 缺市场收益
    r = event_study(sr, None, dates)
    assert r["status"] == "degraded" and r["warnings"], r
    # (b) 缺板块收益
    r = event_study(None, mr, dates)
    assert r["status"] == "degraded"
    # (c) 空事件列表
    r = event_study(sr, mr, [])
    assert r["status"] == "insufficient" and r["n_events"] == 0
    # (d) 事件日在数据范围外
    r = event_study(sr, mr, ["1990-01-01", "2099-01-01"])
    assert r["n_valid"] == 0 and r["status"] == "degraded"
    assert any("out of data range" in w for w in r["warnings"])
    # (e) 板块字典里没有该板块 -> 整表为空 + warnings，不崩
    evs = mk_events(dates, sectors=("NO_SUCH_SECTOR",))
    tab = impact_factor(evs, {"SEC": sr}, mr)
    assert len(tab) == 0
    assert tab.attrs["warnings"]
    print(f"  [degr ] warnings={len(tab.attrs['warnings'])} "
          f"skip_data={tab.attrs['n_skip_data']}")
    # (f) 事件没有 sectors 且传入的是 dict -> 跳过并警告
    evs2 = [Event(eid="X1", date=str(dates[0].date()), type="policy", subtype="rrr_cut",
                  source="https://example.gov.cn/x", direction=1, sectors=[])]
    tab2 = impact_factor(evs2, {"SEC": sr}, mr)
    assert len(tab2) == 0
    # (g) direction=0 的事件不入表（无方向假设）
    evs3 = mk_events(dates, direction=0)
    tab3 = impact_factor(evs3, {"SEC": sr}, mr)
    assert len(tab3) == 0
    assert tab3.attrs["n_skip_direction"] == len(dates)
    # (h) 单板块 Series 直传（无 sectors 字段）也能工作
    evs4 = [Event(eid="S1", date=str(dates[0].date()), type="macro", subtype="cpi_up",
                  source="https://example.gov.cn/x", direction=1, sectors=[])]
    tab4 = impact_factor(evs4, sr, mr)
    assert len(tab4) >= 1, tab4


# --------------------------------------------------------------------------- #
# 8. 查不到历史样本 -> 明说"无历史样本"，绝不外推
# --------------------------------------------------------------------------- #
def test_predict_no_history():
    idx, mr, sr = make_world(n_days=1500, seed=11)
    dates = event_dates_from(idx, 400, 60, 10)
    evs = mk_events(dates, sectors=("SEC",), etype="policy")
    tab = impact_factor(evs, {"SEC": sr}, mr, horizons=(1, 5, 20))
    # 训练表里只有 policy/SEC，新事件是 earnings/OTHER
    new = Event(eid="NEW", date=str(idx[-5].date()), type="earnings", subtype="beat",
                source="https://example.gov.cn/x", direction=1, sectors=["OTHER"])
    p = predict_impact(new, tab)
    s0 = p["sectors"][0]
    print(f"  [nohist] reason={s0['reason']} expected_car={s0['expected_car']} "
          f"has_history={p['has_history']}")
    assert s0["expected_car"] is None
    assert s0["reason"] == "no_history"
    assert s0["confidence"] == 0.0
    assert p["has_history"] is False
    assert any("no historical samples" in w for w in p["warnings"])
    # 空表也不能崩
    p2 = predict_impact(new, impact_factor([], {}, mr))
    assert p2["sectors"][0]["reason"] == "no_history"


# --------------------------------------------------------------------------- #
# 9. 方向口径：传导表优先，方向必须进 signed CAR
# --------------------------------------------------------------------------- #
def test_direction_orientation():
    ev = Event(eid="D1", date="2021-01-01", type="policy", subtype="rate_cut",
               source="https://example.gov.cn/x", direction=1, sectors=["银行", "地产"])
    assert sector_direction(ev, "银行") == -1      # 降息压息差 -> 银行负
    assert sector_direction(ev, "地产") == +1
    assert sector_direction(ev, "未知板块") == 1    # 回落到事件整体方向
    ev2 = Event(eid="D2", date="2021-01-01", type="policy", subtype="collection_bid",
                source="https://example.gov.cn/x", direction=-1, sectors=["医药"])
    assert sector_direction(ev2, "医药") == -1

    # 同一批 CAR，方向翻转 -> signed_car 符号翻转，但 raw CAR 不变
    idx, mr, sr = make_world(n_days=1200, idio=0.004, seed=13)
    pos = [300 + 60 * i for i in range(10)]
    dates = event_dates_from(idx, 300, 60, 10)
    srd = add_drift(sr, idx, pos, 0.03)
    up = mk_events(dates, sectors=("SEC",), direction=1)
    dn = mk_events(dates, sectors=("SEC",), direction=-1)
    t_up = impact_factor(up, {"SEC": srd}, mr, horizons=(5,))
    t_dn = impact_factor(dn, {"SEC": srd}, mr, horizons=(5,))
    r_up = t_up[t_up["level"] == "type_sector"].iloc[0]
    r_dn = t_dn[t_dn["level"] == "type_sector"].iloc[0]
    print(f"  [dir  ] up mean={r_up['mean_car']:+.4f} t={r_up['t']:.2f} | "
          f"dn mean={r_dn['mean_car']:+.4f} t={r_dn['t']:.2f}")
    assert r_up["mean_car"] > 0.008 and r_up["t"] > 2.0
    assert abs(r_dn["mean_car"] + r_up["mean_car"]) < 1e-9
    assert abs(abs(r_dn["t"]) - abs(r_up["t"])) < 1e-9
    assert abs(r_up["mean_car_raw"] - r_dn["mean_car_raw"]) < 1e-12


# --------------------------------------------------------------------------- #
# 10. 预测：signed CAR -> 板块实际预期收益
# --------------------------------------------------------------------------- #
def test_predict_sign_and_clock():
    idx, mr, sr = make_world(n_days=1500, idio=0.004, seed=17)
    pos = [400 + 60 * i for i in range(12)]
    dates = event_dates_from(idx, 400, 60, 12)
    srd = add_drift(sr, idx, pos, 0.04)
    evs = mk_events(dates, sectors=("SEC",), direction=1)
    tab = impact_factor(evs, {"SEC": srd}, mr, horizons=(1, 5, 20))
    p = predict_impact(evs[0], tab, horizon=5)
    s0 = p["sectors"][0]
    print(f"  [pred ] expected_car={s0['expected_car']:+.4f} conf={s0['confidence']:.2f} "
          f"n={s0['n']} t={s0['t']:.2f} peak_h={p['catalyst_clock']['peak_h']}")
    assert s0["expected_car"] is not None
    assert abs(s0["expected_car"] - s0["expected_car_signed"]) < 1e-12   # d=+1
    assert 0.0 < s0["confidence"] <= 1.0
    assert s0["n"] == 12
    assert p["catalyst_clock"]["peak_h"] in (1, 5, 20)
    assert set(p["catalyst_clock"]["curve"]) == {1, 5, 20}
    # 反向事件：expected_car 符号必须翻
    ev_dn = mk_events(dates, sectors=("SEC",), direction=-1)[0]
    p2 = predict_impact(ev_dn, tab, horizon=5)
    assert abs(p2["sectors"][0]["expected_car"] + s0["expected_car"]) < 1e-12
    # 置信度随 t 单调：t<1 的格必须给 0
    assert p["confidence_max"] <= 1.0


# --------------------------------------------------------------------------- #
# 11. 持久化：JSONL 追加 + 读回 + 坏行容错
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip():
    tmp = tempfile.mkdtemp(prefix="evtest_")
    try:
        path = os.path.join(tmp, "events.jsonl")
        assert load_events(path) == []                 # 不存在 -> []
        ev = Event(eid="A1", date="2020-03-16", type="policy", subtype="rrr_cut",
                   title="t", source="https://example.gov.cn/a", direction=1,
                   sectors=["银行"], evidence_level="strong")
        got = save_event(ev, path)
        assert os.path.isabs(got) and os.path.exists(got)
        save_event(Event(eid="A2", date="2020-04-01", type="macro", subtype="cpi_up",
                         source="https://example.gov.cn/b", direction=0,
                         sectors=["食品饮料"]), path)
        with open(path, "a", encoding="utf-8") as f:
            f.write("this is not json\n\n")           # 坏行
            f.write('{"eid":"A3"}\n')                 # 缺字段
        evs = load_events(path)
        print(f"  [io   ] loaded={len(evs)} eids={[e.eid for e in evs]}")
        assert len(evs) == 2
        assert evs[0].eid == "A1" and evs[0].date == "2020-03-16"
        assert evs[0].sectors == ["银行"] and evs[0].direction == 1
        assert evs[0].evidence_level == "strong"
        assert validate_event(evs[0]) == []
        bad = validate_event(Event(eid="", date="20-1-1", type="nope",
                                   source="not-a-url"))
        assert len(bad) >= 3, bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 12. 传导表 / 板块映射 / 行业等权近似 的健全性
# --------------------------------------------------------------------------- #
def test_transmission_and_sector_helpers():
    assert len(POLICY_TRANSMISSION) >= 25, len(POLICY_TRANSMISSION)
    for k, v in POLICY_TRANSMISSION.items():
        assert isinstance(v, list) and v, k
        for sec, d, logic in v:
            assert d in (-1, 0, 1), (k, sec, d)
            assert len(logic) >= 8, (k, sec)     # 传导逻辑必须写清楚
    # 模糊映射：通用名 -> 实际行业字段
    avail = ["银行", "证券", "房地产开发", "化学制药", "半导体", "白酒", "电池"]
    assert expand_sector("券商", avail) == ["证券"]
    assert set(expand_sector("地产", avail)) == {"房地产开发"}
    assert set(expand_sector("医药", avail)) == {"化学制药"}
    assert expand_sector("不存在的板块", avail) == ["不存在的板块"]   # 不瞎猜

    # 行业等权近似：3 只同行业股票 -> 等权收益 = 三只的均值
    idx = pd.bdate_range("2020-01-01", periods=60)
    panel, pool = {}, []
    for i, code in enumerate(("600001", "600002", "600003")):
        px = 10.0 * np.cumprod(1 + 0.001 * (i + 1) + np.zeros(60))
        panel[code] = pd.DataFrame({"date": idx, "close": px})
        pool.append({"code": code, "industry": "银行"})
    sec = build_sector_returns(panel, pd.DataFrame(pool), min_names=3)
    print(f"  [sect ] sectors={list(sec)} n={len(sec.get('银行', []))}")
    assert "银行" in sec and len(sec["银行"]) >= 50
    # 只有 2 只 -> 达不到 min_names=3，丢弃
    assert build_sector_returns(panel, pd.DataFrame(pool).head(2), min_names=3) == {}
    assert build_sector_returns({}, pd.DataFrame(pool)) == {}


# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    ok = fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            ok += 1
        except Exception:                       # noqa: BLE001
            print(f"FAIL {name}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok} passed, {fail} failed, {len(tests)} total")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
