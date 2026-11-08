"""
扩展技术指标到底有没有用？—— A/B/C 对照实验。

| 配置 | 特征 |
|---|---|
| A | Alpha158（158） |
| B | Alpha158 + 35 扩展指标（193） |
| C | Alpha158 + 35 扩展 + 22 经典（215） |

口径：CSI300 / 2008-2019 训练 / 2021-2026 测试 / h=1 / **不早停 200 棵**。

⚠️ 关键教训：早停会让树数掉到个位数，此时大量预测并列，RankIC 不可信。
本脚本强制不早停，确保三次对比的树数一致。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench", os.path.join(_here, "benchmark_alpha158.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

from qcore import qlib_data as QD
from qcore.alpha158 import alpha158
from qcore.tech_extra import extra_indicators
from qcore.indicators2 import compute_indicators2
from qcore.gbdt import GBDT


def build(use_extra: bool, use_i2: bool, stocks: int = 200, verbose: bool = True):
    codes = QD.universe_codes("csi300", start="2021-01-01")[:stocks]
    per, close = {}, {}
    t0 = time.time()
    for i, c in enumerate(codes, 1):
        df = QD.load_stock(c, start="2008-01-01")
        if df is None or len(df) < 300:
            continue
        try:
            parts = [alpha158(df)]
            if use_extra:
                parts.append(extra_indicators(df))
            if use_i2:
                i2 = compute_indicators2(df)
                i2.index = pd.DatetimeIndex(df["date"])
                parts.append(i2)
            f = pd.concat(parts, axis=1)
        except Exception:
            continue
        f.index = pd.DatetimeIndex(df["date"])
        per[c] = f
        close[c] = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        if verbose and i % 100 == 0:
            print(f"  feat {i}/{stocks}  {time.time()-t0:.0f}s", flush=True)
    dates = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in per.values()])))
    names = list(per[list(per)[0]].columns)
    T, N, K = len(dates), len(per), len(names)
    X = np.full((T, N, K), np.nan, dtype=np.float32)
    for j, c in enumerate(per):
        X[:, j, :] = per[c].reindex(dates)[names].to_numpy(dtype=np.float32)
    cm = pd.DataFrame({c: s.reindex(dates) for c, s in close.items()}, index=dates)
    cv = cm.to_numpy(float)
    y = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T - 2):
        with np.errstate(invalid="ignore", divide="ignore"):
            y[t] = cv[t + 2] / cv[t + 1] - 1.0
    y[~np.isfinite(y)] = np.nan
    return X, y, dates, names


def run(tag: str, use_extra: bool, use_i2: bool, trees: int = 200) -> dict:
    print(f"\n=== {tag} ===", flush=True)
    X, y, dates, names = build(use_extra, use_i2)
    print(f"  特征 {X.shape[2]} 个，T={X.shape[0]} N={X.shape[1]}", flush=True)
    Xn = bench._cs_rank_norm(X)
    sp = bench._split(dates, "2021-01-01", "2026-11-06",
                      "2019-12-31", "2020-01-01", "2020-12-31")
    Xtr, ytr = bench._flatten(Xn, y, sp["train"])
    m = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.1,
             min_samples_leaf=200, subsample=0.8, colsample=0.8,
             l2=1.0, max_bins=32, random_state=42)
    t0 = time.time()
    m.fit(Xtr, ytr)                      # 不早停
    dt = time.time() - t0
    idx = sp["test"]
    sc = np.full((len(idx), X.shape[1]), np.nan, dtype=np.float32)
    yt = np.full((len(idx), X.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(idx):
        r = Xn[t]
        ok = np.isfinite(r).all(axis=1)
        if ok.sum() >= 20:
            sc[i, ok] = m.predict(r[ok])
        yt[i] = y[t]
    ic = bench._ic(sc, yt)
    print(f"  IC={ic['ic']:+.4f} (t={ic['t']:+.2f}) RankIC={ic['rank_ic']:+.4f} "
          f"树={trees} 训练={dt:.0f}s", flush=True)
    return {"tag": tag, "K": X.shape[2], "ic": ic["ic"], "t": ic["t"],
            "rank_ic": ic["rank_ic"], "trees": trees}


def main() -> int:
    res = [
        run("A 仅 Alpha158", False, False),
        run("B +35 扩展指标", True, False),
        run("C +35 扩展 +22 经典", True, True),
    ]
    print("\n" + "=" * 62)
    print(f"{'配置':<22}{'特征数':>8}{'IC':>10}{'t':>8}{'RankIC':>10}")
    for r in res:
        print(f"{r['tag']:<22}{r['K']:>8}{r['ic']:>+10.4f}{r['t']:>8.2f}{r['rank_ic']:>+10.4f}")
    best = max(res, key=lambda r: r["ic"])
    base = res[0]
    print()
    if best is base:
        print("→ 加技术指标没有提升，Alpha158 已经够用")
    else:
        print(f"→ {best['tag']} 最好，IC 提升 {(best['ic']-base['ic'])*10000:.0f} bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
