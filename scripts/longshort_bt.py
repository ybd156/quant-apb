#!/usr/bin/env python3
"""
多空 vs 纯多头 —— 真实样本外对照。

## 为什么要单独跑这个

`qcore/longshort.py` 有 `backtest()`，但那是通用函数。
要回答"该不该做空"，必须用**真正的样本外数据**跑一次，
而且要逐项拆开成本：交易成本、借券成本、空头腿单独贡献。

## 设计（避免自欺）

- 训练集：2008-01-01 ~ 2023-12-31
- 测试集：2024-01-01 ~ 最新
- 模型只在训练集上 fit，测试集**只做预测**
- 同一批分数同时喂给三种组合：
    ① 纯多头（top_k 等权）
    ② 多空等额（top_k 多 / bottom_k 空，净敞口 0）
    ③ 多空 2:1（多头 2 倍空头，净敞口 +33%）
- 每个市场单独跑（CN / US / HK 成本与做空可行性都不同）

## 输出

每个市场一张表：年化、夏普、HAC t、胜率、各项成本。
最后给一句实话：值不值得做。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name, fn):
    import importlib.util
    here = os.path.join(ROOT, "scripts")
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_market(market: str = "CN", universe: str = "csi300", stocks: int = 300,
               horizon: int = 5, top_k: int = 20, bottom_k: int = 20,
               train_end: str = "2023-12-31", verbose: bool = True) -> dict:
    from qcore.gbdt import GBDT
    from qcore import longshort as LS
    from qcore import costs as CO

    bench = _load("bench", "benchmark_alpha158.py")
    from qcore import qlib_data as QD

    t0 = time.time()
    codes = QD.universe_codes(universe, start=train_end)
    if verbose:
        print(f"[{market}] {len(codes)} 只，训练截至 {train_end}", flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=horizon,
                             max_stocks=stocks, verbose=verbose)
    if not d.get("ok"):
        return {"ok": False, "reason": "build_features"}
    X, y, dates = d["X"], d["y"], d["dates"]
    Xn = bench._cs_rank_norm(X)

    ud = np.unique(dates)
    # _flatten 要的是 **时间轴索引**，不是日期值。传日期会得到空集。
    tr_idx = np.where(dates <= np.datetime64(train_end))[0]
    if tr_idx.size < 100:
        return {"ok": False, "reason": f"训练天数不足 {tr_idx.size}"}
    Xtr, ytr = bench._flatten(Xn, y, tr_idx)
    if Xtr.shape[0] < 10000:
        return {"ok": False, "reason": f"训练样本不足 {Xtr.shape[0]}"}

    if verbose:
        print(f"[{market}] 训练 {Xtr.shape[0]} 行，{time.time()-t0:.0f}s", flush=True)
    model = GBDT(n_estimators=250, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    model.fit(Xtr, ytr)

    # 测试集：逐日预测
    te_idx = np.where(dates > np.datetime64(train_end))[0]
    if te_idx.size < 30:
        return {"ok": False, "reason": f"测试集太短 {te_idx.size}"}
    te_dates = ud[ud > np.datetime64(train_end)]

    scores_TN = np.full((len(dates), X.shape[1]), np.nan)
    for t in te_idx:
        m = np.isfinite(Xn[t]).all(axis=1)
        if m.sum() < 20:
            continue
        scores_TN[t, m] = model.predict(Xn[t][m])

    # 前向收益（与 horizon 对齐）
    RET = y  # y 已是 horizon 日收益
    VALID = np.isfinite(Xn).all(axis=2) & np.isfinite(RET)
    start_t = int(te_idx[0])

    # round_trip_bps 返回的是 dict（含明细），要取 round_trip_bps 字段
    _c = CO.round_trip_bps(market, notional=1e6)
    cost_bps = float(_c.get('round_trip_bps') or _c.get('buy_bps', 0) + _c.get('sell_bps', 0)) if isinstance(_c, dict) else float(_c)
    res = {"market": market, "horizon": horizon, "top_k": top_k,
           "bottom_k": bottom_k, "cost_bps": round(float(cost_bps), 2),
           "n_test_days": int(te_dates.size), "train_rows": int(Xtr.shape[0])}

    # ① 纯多头
    r_long = LS.backtest(scores_TN, RET, VALID, market=market, top_k=top_k,
                         bottom_k=bottom_k, hold=horizon,
                         gross_long=1.0, gross_short=0.0,
                         cost_bps=cost_bps, start_t=start_t)
    res["long_only"] = r_long

    # ② 多空等额（净敞口 0）
    r_neutral = LS.backtest(scores_TN, RET, VALID, market=market, top_k=top_k,
                            bottom_k=bottom_k, hold=horizon,
                            gross_long=0.5, gross_short=0.5,
                            cost_bps=cost_bps, start_t=start_t)
    res["long_short_neutral"] = r_neutral

    # ③ 多空 2:1
    r_2to1 = LS.backtest(scores_TN, RET, VALID, market=market, top_k=top_k,
                         bottom_k=bottom_k, hold=horizon,
                         gross_long=0.67, gross_short=0.33,
                         cost_bps=cost_bps, start_t=start_t)
    res["long_short_2to1"] = r_2to1

    # 结论
    lo = (r_long or {}).get("ann_long_only")
    ls = (r_neutral or {}).get("ann_long_short")
    t_ls = (r_neutral or {}).get("t_hac")
    borrow = (r_neutral or {}).get("borrow_bps_per_period")
    if lo is None or ls is None:
        res["verdict"] = "回测样本不足，无法判断"
    else:
        better = ls > lo
        sig = (t_ls is not None and abs(t_ls) >= 2)
        res["verdict"] = (
            f"多空 {ls:+.2%} vs 纯多 {lo:+.2%} → "
            + ("多空更优" if better else "多空不如纯多")
            + ("" if sig else f"；但 HAC t={t_ls} 不显著")
            + f"；空头每期借券 {borrow}bps"
        )
    if not LS.spec(market).allowed:
        res["verdict"] += "；该市场借不到券，实际不可做空"
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["ok"] = True
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="CN")
    ap.add_argument("--universe", default="csi300")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "state", "longshort_backtest.json"))
    a = ap.parse_args(argv)

    allres = []
    for m in [x.strip() for x in a.markets.split(",") if x.strip()]:
        print("=" * 60)
        print(f"  市场 {m}")
        print("=" * 60)
        r = run_market(m, universe=a.universe, horizon=a.horizon)
        allres.append(r)
        if r.get("ok"):
            print(f"\n  纯多头年化     {r['long_only'].get('ann_long_only')}")
            print(f"  多空(中性)年化 {r['long_short_neutral'].get('ann_long_short')}")
            print(f"  多空(2:1)年化  {r['long_short_2to1'].get('ann_long_short')}")
            print(f"  {r['verdict']}\n")
        else:
            print(f"  失败：{r.get('reason')}\n")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "results": allres}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"已写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
