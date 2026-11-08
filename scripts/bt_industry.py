"""
行业中性化到底能不能变成钱？—— 把 IC 提升落到组合超额上。

## 为什么要做这一步

子任务报告：行业中性化让 IC 从 +0.0260 升到 +0.0377（+45%，配对 t=+4.66）。
但 **IC 提升 ≠ 组合收益提升**——本项目已经栽过好几次「指标好看、实盘归零」。

所以必须跑完整链路：
    标签改造 → 训练 → 逐日打分 → 低换手回测（扣 19.28bps）→ 相对池等权的超额

## 三组对照

| 组 | 标签 |
|---|---|
| A  | 原始前向收益（基线） |
| B  | 行业内去均值（行业残差） |
| B2 | 行业内 rank(pct)（只学行业内排序） |

B2 在子任务里 RankIC 是基线的 3.2 倍，但 Pearson IC 只 +16.8%。
本脚本用**组合超额**来裁决，而不是看 IC。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bench = _load("bench", "benchmark_alpha158.py")

from qcore import qlib_data as QD
from qcore import bt as BT
from qcore.neutral import industry_neutralize, industry_rank
from qcore.gbdt import GBDT


def _industry_map(codes) -> np.ndarray:
    """代码 -> 行业名。剥 sh/sz/bj 前缀后去 get_pool 的 industry 列查。"""
    try:
        from qcore.data import get_pool
        pool = get_pool(top_n=600)
        m = {}
        for r in pool.itertuples(index=False):
            c = str(getattr(r, "code", "")).zfill(6)
            ind = getattr(r, "industry", None)
            if c and isinstance(ind, str) and ind:
                m[c] = ind
        out = []
        for c in codes:
            bare = str(c).lower()
            bare = bare[2:] if bare[:2] in ("sh", "sz", "bj") else bare
            out.append(m.get(bare.zfill(6), "NA"))
        return np.array(out, dtype=object)
    except Exception:
        return np.array(["NA"] * len(codes), dtype=object)


def prepare(variant: str, universe: str = "csi300", stocks: int = 300,
            trees: int = 250, train_end: str = "2019-12-31",
            test_start: str = "2021-01-01", test_end: str = "2026-11-06",
            verbose: bool = True) -> Optional[Dict]:
    codes = QD.universe_codes(universe, start=test_start)[:stocks]
    if verbose:
        print(f"  {universe}: {len(codes)} 只", flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=1,
                             max_stocks=stocks, verbose=verbose)
    if not d.get("ok"):
        return None
    X, y, dates, cl = d["X"], d["y"], d["dates"], d["codes"]
    if verbose:
        print(f"  面板 T={X.shape[0]} N={X.shape[1]} K={X.shape[2]}", flush=True)

    t0 = time.time()
    Xn = bench._cs_rank_norm(X)
    if verbose:
        print(f"  标准化 {time.time()-t0:.0f}s", flush=True)

    # ---- 标签改造 ----
    ind = _industry_map(cl)
    y2 = y.copy()
    if variant in ("B", "B2"):
        for t in range(y.shape[0]):
            row = y[t]
            m = np.isfinite(row)
            if m.sum() < 20:
                continue
            if variant == "B":
                y2[t][m] = industry_neutralize(row[m], ind[m])
            else:
                y2[t][m] = industry_rank(row[m], ind[m])
        if verbose:
            print(f"  标签已改造为 {variant}（行业数 {len(set(ind))}）", flush=True)

    sp = bench._split(dates, test_start, test_end, train_end, "2020-01-01", "2020-12-31")
    Xtr, ytr = bench._flatten(Xn, y2, sp["train"])
    if Xtr.shape[0] < 5000:
        return None
    model = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    t0 = time.time()
    model.fit(Xtr, ytr)
    if verbose:
        print(f"  训练 {time.time()-t0:.0f}s / {Xtr.shape[0]:,} 样本", flush=True)

    idx = sp["test"]
    scores = np.full((len(idx), X.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(idx):
        row = Xn[t]
        m = np.isfinite(row).all(axis=1)
        if m.sum() >= 20:
            scores[i, m] = model.predict(row[m])

    close = np.full((len(idx), X.shape[1]), np.nan, dtype=float)
    for j, c in enumerate(cl):
        df = QD.load_stock(c, start="2008-01-01")
        if df is None:
            continue
        ser = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        close[:, j] = ser.reindex(dates[idx]).to_numpy(dtype=float)

    return {"scores": scores, "close": close, "dates": dates[idx],
            "codes": cl, "y": y[idx], "model": model}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--variants", default="A,B,B2")
    ap.add_argument("--cost-bps", type=float, default=19.28)
    a = ap.parse_args(argv)

    print("=" * 74)
    print("行业中性化 -> 组合超额验证（CSI300 / 2021-2026 / 扣 %.2fbps）" % a.cost_bps)
    print("=" * 74)

    rows = []
    for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
        print(f"\n=== 变体 {v} ===", flush=True)
        d = prepare(v, stocks=a.stocks, trees=a.trees)
        if d is None:
            print("  准备失败")
            continue
        ic = bench._ic(d["scores"], d["y"])
        print(f"  信号 IC={ic['ic']:+.4f} (t={ic['t']:+.2f}) RankIC={ic['rank_ic']:+.4f}",
              flush=True)

        g = BT.grid(d["scores"], d["close"], d["dates"],
                    holds=(10, 20), bands=(0.5, 1.0),
                    top_ks=(30, 50), cost_bps=a.cost_bps)
        best = g.sort_values("ex_sharpe", ascending=False).iloc[0]
        print(f"  最佳 k={int(best.top_k)} 持有{int(best.hold)}日 缓冲{best.band:.1f}: "
              f"净年化 {best.annual*100:+.2f}% 池等权 {best.bench*100:+.2f}% "
              f"**超额 {best.excess*100:+.2f}%** 超额夏普 {best.ex_sharpe:.2f}", flush=True)
        rows.append({"variant": v, "ic": ic["ic"], "t_ic": ic["t"],
                     "rank_ic": ic["rank_ic"],
                     "annual": best.annual, "bench": best.bench,
                     "excess": best.excess, "ex_sharpe": best.ex_sharpe,
                     "ex_mdd": best.ex_mdd,
                     "k": int(best.top_k), "hold": int(best.hold), "band": best.band})

    print("\n" + "=" * 74)
    print(f"{'变体':<6}{'IC':>9}{'RankIC':>9}{'净年化':>9}{'池等权':>9}"
          f"{'超额':>9}{'超额夏普':>9}")
    for r in rows:
        print(f"{r['variant']:<6}{r['ic']:>+9.4f}{r['rank_ic']:>+9.4f}"
              f"{r['annual']*100:>8.2f}%{r['bench']*100:>8.2f}%"
              f"{r['excess']*100:>8.2f}%{r['ex_sharpe']:>9.2f}")
    if len(rows) >= 2:
        base = rows[0]
        print()
        for r in rows[1:]:
            print(f"  {r['variant']} vs A: 超额变化 {(r['excess']-base['excess'])*100:+.2f}pp，"
                  f"夏普变化 {r['ex_sharpe']-base['ex_sharpe']:+.2f}")
        best = max(rows, key=lambda r: r["ex_sharpe"])
        print(f"\n→ 按超额夏普最优：{best['variant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
