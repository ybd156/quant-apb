"""
低换手回测 —— 决定 Alpha158+GBDT 到底能不能扣成本后赚钱。

用法：
    python scripts/lowturnover_bt.py --universe csi300
    python scripts/lowturnover_bt.py --universe csi500 --stocks 500

流程：
  1. 载入点对点成分股 + Alpha158 因子
  2. 2008-2019 训练 GBDT（250 棵，不早停）
  3. 2021-2026 逐日打分
  4. 跑 (top_k × 持有期 × 缓冲带) 网格，扣 19.28bps 往返成本
  5. 输出净年化 / 夏普 / 回撤 / 换手
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import qlib_data as QD
from qcore import bt as BT
from qcore.gbdt import GBDT
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_alpha158.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def pit_codes(universe: str = "csi300", start: str = "2008-01-01",
              end: str = "2021-01-01") -> list:
    """点对点成分：凡在 [start, end] 内**曾经**属于该指数的票都算进来。

    为什么（评审 R4）：`universe_codes(start=test_start)` 只取"测试起点那一天"的成分，
    于是训练期 2008–2019 的样本是"按 2021 年是否还在指数里"筛过的（未来条件选择）。
    `pit_membership()` 本来就提供了逐段区间，这里只是把它接上。
    """
    from datetime import datetime as _dt
    d0 = _dt.strptime(start[:10], "%Y-%m-%d").date()
    d1 = _dt.strptime(end[:10], "%Y-%m-%d").date()
    mem = QD.pit_membership(universe)
    out = []
    for code, spans in mem.items():
        for (s0, s1) in spans:
            a = s0.date() if hasattr(s0, "date") else s0
            b = s1.date() if hasattr(s1, "date") else s1
            if b >= d0 and a <= d1:
                out.append(code)
                break
    return list(dict.fromkeys(out))


def prepare(universe: str = "csi300", stocks: int = 300,
            train_end: str = "2019-12-31", test_start: str = "2021-01-01",
            test_end: str = "2026-11-06", horizon: int = 1,
            trees: int = 250, verbose: bool = True, pit: bool = False):
    """训练模型并返回测试期的分数、价格、日期。

    pit=True 时用**点对点成分**（含期间调出/退市的票），避免用 2021 快照选 2008–2019 样本。
    """
    if pit:
        codes = pit_codes(universe, start="2008-01-01", end=test_start)
        if verbose:
            print(f"  [pit] 点对点成分 {len(codes)} 只（2008–{test_start} 曾入选）", flush=True)
    else:
        codes = QD.universe_codes(universe, start=test_start)
    if verbose:
        print(f"  {universe} 成分股 {len(codes)} 只（{test_start} 时点）", flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=horizon,
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

    sp = bench._split(dates, test_start, test_end, train_end, "2020-01-01", "2020-12-31")
    Xtr, ytr = bench._flatten(Xn, y, sp["train"])
    if Xtr.shape[0] < 5000:
        return None
    model = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    t0 = time.time()
    model.fit(Xtr, ytr)
    if verbose:
        print(f"  训练 {time.time()-t0:.0f}s  {trees} 棵 / {Xtr.shape[0]:,} 样本", flush=True)

    test_idx = sp["test"]
    scores = np.full((len(test_idx), X.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(test_idx):
        row = Xn[t]
        m = np.isfinite(row).all(axis=1)
        if m.sum() >= 20:
            scores[i, m] = model.predict(row[m])

    close = np.full((len(test_idx), X.shape[1]), np.nan, dtype=float)
    for j, c in enumerate(cl):
        df = QD.load_stock(c, start="2008-01-01")
        if df is None:
            continue
        ser = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        close[:, j] = ser.reindex(dates[test_idx]).to_numpy(dtype=float)

    return {"scores": scores, "close": close, "dates": dates[test_idx],
            "codes": cl, "model": model, "X_test": Xn[test_idx], "y": y[test_idx]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="csi300")
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--cost-bps", type=float, default=19.28)
    a = ap.parse_args(argv)

    print("=" * 70)
    print(f"低换手回测 — {a.universe.upper()} / 2021-2026 / 成本 {a.cost_bps}bps")
    print("=" * 70)
    d = prepare(a.universe, a.stocks, trees=a.trees)
    if d is None:
        print("准备失败")
        return 1

    # 先看信号本身（毛，未扣成本、无换手约束）
    ic = bench._ic(d["scores"], d["y"])
    print(f"\n信号质量（测试期）：IC={ic['ic']:+.4f} (t={ic['t']:+.2f}) "
          f"RankIC={ic['rank_ic']:+.4f} n={ic['n']}")

    print(f"\n参数网格（扣 {a.cost_bps}bps 往返成本）：")
    g = BT.grid(d["scores"], d["close"], d["dates"],
                holds=(5, 10, 20), bands=(0.0, 0.5, 1.0),
                top_ks=(30, 50), cost_bps=a.cost_bps)
    g = g.sort_values("ex_sharpe", ascending=False)
    hdr = f"{'k':>3}{'持有':>5}{'带':>5}{'净年化':>9}{'池等权':>9}{'超额':>9}{'超额夏普':>9}{'超额回撤':>9}{'换手':>7}"
    print()
    print(hdr)
    for r in g.itertuples(index=False):
        print(f"{r.top_k:>3}{r.hold:>5}{r.band:>5.1f}"
              f"{r.annual*100:>8.2f}%{r.bench*100:>8.2f}%{r.excess*100:>8.2f}%"
              f"{r.ex_sharpe:>9.2f}{r.ex_mdd*100:>8.1f}%{r.turnover*100:>6.0f}%")

    best = g.iloc[0]
    print()
    print(f"最佳（按超额夏普）：k={int(best.top_k)} 持有{int(best.hold)}日 缓冲{best.band:.1f}")
    print(f"  净年化 {best.annual*100:+.2f}%  池等权 {best.bench*100:+.2f}%  超额 {best.excess*100:+.2f}%")
    print(f"  超额夏普 {best.ex_sharpe:.2f}  超额回撤 {best.ex_mdd*100:.1f}%  换手 {best.turnover*100:.0f}%")
    if best.excess > 0.03 and best.ex_sharpe > 0.5:
        print("  → 扣成本后有真实超额（相对池等权），值得进一步验证")
    elif best.excess > 0:
        print("  → 超额为正但偏薄，需要更低换手或更强信号")
    else:
        print("  → 超额为负：净收益全是 beta，没有选股能力")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
