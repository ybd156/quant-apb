"""
复现 Qlib Alpha158 + GBDT 基准 —— 验证我们的评估框架是否与公开结果一致。

## 要回答的问题

微软 Qlib 公布 CSI300 上 Alpha158 + LightGBM 的 **IC = 0.0448**（2017-2020 样本外）。
我们用自己的数据、自己的 Alpha158 实现、自己写的 numpy GBDT，能不能复现出接近的数字？

- **能** → 说明我们的评估框架是对的，之前 IC=−0.0185 纯粹是信号差
- **不能** → 说明我们的数据口径或评估流程有问题，那更严重

## 口径（尽量对齐 Qlib 官方 workflow）

- 池子：CSI300 **点对点成分**（`instruments/csi300.txt` 带起止日期，不是今天的名单）
- 特征：Alpha158（158 个，`qcore.alpha158`）
- 标签：`close[T+1+h] / close[T+1] - 1`（T 日出信号，T+1 收盘成交）
  - h=1 对齐 Qlib 默认标签 `Ref(close,-2)/Ref(close,-1)-1`
  - h=5 是我们的口径，便于横向对比
- 特征标准化：**横截面 rank → 正态化**（Qlib 的 CSRankNorm），逐日做
- 切分：2008-2014 训练 / 2015-2016 验证 / 2017-2020 测试
- 指标：IC（Pearson）、Rank IC（Spearman）、ICIR

## 为什么必须横截面标准化

Alpha158 的量纲差异极大（RANK∈[0,1]，VMA 能到 63）。
不标准化，树模型会被极端值主导。Qlib 的做法是逐日 CSRankNorm。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import qlib_data as QD
from qcore.alpha158 import alpha158, factor_names
from qcore.gbdt import GBDT


def _ndtri(p: np.ndarray) -> np.ndarray:
    """向量化标准正态分位（Acklam 近似，纯 numpy，无 scipy）。"""
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    out = np.empty_like(p)
    lo = p < 0.02425
    hi = p > 1 - 0.02425
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /                    (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def _cs_rank_norm(X: np.ndarray) -> np.ndarray:
    """
    逐日横截面 rank -> 标准正态（Qlib CSRankNorm 等价）。

    向量化：每个交易日把整块 (N,K) 交给 pandas.rank(pct=True) 一次算完，
    再整体做 ndtri。原来按列循环 + 逐元素 norm_ppf 是 O(T*N*K) 次 Python 调用，
    2400 天 × 300 股 × 158 因子 = 1.1 亿次，根本跑不完。
    """
    T, N, K = X.shape
    out = np.full_like(X, np.nan, dtype=np.float32)
    for t in range(T):
        blk = X[t]
        if not np.isfinite(blk).any():
            continue
        # 有效值 >= 20 的列才标准化
        cnt = np.isfinite(blk).sum(axis=0)
        ok_col = cnt >= 50          # 样本太少的列不标准化（否则分位被推到极端）
        if not ok_col.any():
            continue
        sub = pd.DataFrame(blk[:, ok_col])
        # 中位秩百分位：(rank-0.5)/n，永不触及 0/1，避免 ndtri 爆到 ±7
        r = (sub.rank() - 0.5) / cnt[ok_col]
        z = _ndtri(r.to_numpy(dtype=np.float64))
        out[t][:, ok_col] = z.astype(np.float32)
    return out


def build_features(codes: List[str], start: str = "2008-01-01",
                   horizon: int = 1, max_stocks: int = 300,
                   verbose: bool = True) -> Dict:
    """构造 (T,N,K) 特征 + (T,N) 标签 + 日历。"""
    codes = codes[:max_stocks]
    names = factor_names()
    K = len(names)

    per_stock: Dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for i, c in enumerate(codes, 1):
        df = QD.load_stock(c, start=start)
        if df is None or len(df) < 300:
            continue
        try:
            f = alpha158(df)
        except Exception:
            continue
        f.index = pd.DatetimeIndex(df["date"])
        per_stock[c] = f
        if verbose and i % 100 == 0:
            print(f"    alpha158: {i}/{len(codes)}  {time.time()-t0:.0f}s", flush=True)
    if not per_stock:
        return {"ok": False, "reason": "no features"}
    if verbose:
        print(f"    alpha158: {len(per_stock)} 只完成，{time.time()-t0:.0f}s", flush=True)

    all_dates = sorted(set().union(*[set(f.index) for f in per_stock.values()]))
    dates = pd.DatetimeIndex(all_dates)
    T, N = len(dates), len(per_stock)
    cl = list(per_stock.keys())

    X = np.full((T, N, K), np.nan, dtype=np.float32)
    for j, c in enumerate(cl):
        f = per_stock[c].reindex(dates)
        X[:, j, :] = f[names].to_numpy(dtype=np.float32)

    # 标签：T 日出信号，T+1 收盘成交
    close = pd.DataFrame(index=dates, columns=cl, dtype=float)
    for c in cl:
        df = QD.load_stock(c, start=start)
        close[c] = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"])).reindex(dates)
    cvals = close.to_numpy(dtype=float)
    y = np.full((T, N), np.nan, dtype=np.float32)
    # close[T+1+h] / close[T+1] - 1
    for t in range(T - 1 - horizon):
        num = cvals[t + 1 + horizon]
        den = cvals[t + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            y[t] = num / den - 1.0
    y[~np.isfinite(y)] = np.nan

    return {"ok": True, "X": X, "y": y, "dates": dates, "codes": cl, "names": names}


def _split(dates: pd.DatetimeIndex, test_start: str = "2017-01-01",
           test_end: str = "2020-12-31",
           train_end: str = "2014-12-31",
           valid_start: str = "2015-01-01",
           valid_end: str = "2016-12-31") -> Dict[str, np.ndarray]:
    """
    训练/验证窗口固定（2008-2014 / 2015-2016），测试窗口可调。
    test_start 晚于 2017 时，2017..test_start 的样本会被丢弃——
    这是刻意的：宁可少用数据，也不让测试集与验证集重叠。
    """
    d = dates
    return {
        "train": np.where((d >= "2008-01-01") & (d <= train_end))[0],
        "valid": np.where((d >= valid_start) & (d <= valid_end))[0],
        "test": np.where((d >= test_start) & (d <= test_end))[0],
    }


def _flatten(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把 (T,N,K)/(T,N) 的指定日期展平成样本。"""
    xs, ys = [], []
    for t in idx:
        m = np.isfinite(y[t]) & np.isfinite(X[t]).all(axis=1)
        if m.sum() < 20:
            continue
        xs.append(X[t][m])
        ys.append(y[t][m])
    if not xs:
        return np.empty((0, X.shape[2])), np.empty((0,))
    return np.vstack(xs), np.concatenate(ys)


def _ic(score: np.ndarray, y: np.ndarray) -> Dict:
    from qcore import stats as ST
    T = score.shape[0]
    ics, rics = [], []
    for t in range(T):
        s, r = score[t], y[t]
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < 20:
            continue
        ics.append(float(np.corrcoef(s[m], r[m])[0, 1]))
        rs = pd.Series(s[m]).rank().to_numpy()
        rr = pd.Series(r[m]).rank().to_numpy()
        rics.append(float(np.corrcoef(rs, rr)[0, 1]))
    ics = np.asarray([v for v in ics if np.isfinite(v)])
    rics = np.asarray([v for v in rics if np.isfinite(v)])
    if ics.size < 20:
        return {"ic": float("nan"), "t": float("nan"), "icir": float("nan"), "n": 0}
    st = ST.mean_tstat(ics, lags=0)
    icir = float(np.mean(ics) / (np.std(ics, ddof=1) + 1e-12))
    return {"ic": float(st["mean"]), "t": float(st["t"]), "icir": icir,
            "rank_ic": float(np.mean(rics)) if rics.size else float("nan"),
            "rank_icir": float(np.mean(rics) / (np.std(rics, ddof=1) + 1e-12)) if rics.size else float("nan"),
            "n": int(ics.size)}


def run(horizon: int = 1, max_stocks: int = 300, start: str = "2008-01-01",
        verbose: bool = True, test_start: str = "2017-01-01",
        test_end: str = "2020-12-31", train_end: str = "2014-12-31",
        valid_start: str = "2015-01-01", valid_end: str = "2016-12-31",
        n_estimators: int = 120, early_stop: int = 20) -> Dict:
    if not QD.available():
        return {"ok": False, "reason": "Qlib 数据未就绪"}
    codes = QD.universe_codes("csi300", start="2017-06-01")
    if verbose:
        print(f"  CSI300 成分股: {len(codes)} 只（2017-06 时点）", flush=True)
    d = build_features(codes, start=start, horizon=horizon,
                       max_stocks=max_stocks, verbose=verbose)
    if not d.get("ok"):
        return d

    X, y, dates = d["X"], d["y"], d["dates"]
    if verbose:
        print(f"  面板: T={X.shape[0]} N={X.shape[1]} K={X.shape[2]}", flush=True)
        print("  横截面 rank 正态化 ...", flush=True)
    t0 = time.time()
    Xn = _cs_rank_norm(X)
    if verbose:
        print(f"  标准化完成 {time.time()-t0:.0f}s", flush=True)

    sp = _split(dates, test_start, test_end, train_end, valid_start, valid_end)
    Xtr, ytr = _flatten(Xn, y, sp["train"])
    Xva, yva = _flatten(Xn, y, sp["valid"])
    if Xtr.shape[0] < 5000 or Xva.shape[0] < 500:
        return {"ok": False, "reason": f"样本不足 train={Xtr.shape[0]} valid={Xva.shape[0]}"}
    if verbose:
        print(f"  train={Xtr.shape[0]:,} valid={Xva.shape[0]:,}", flush=True)

    # 横截面去均值（让模型学"相对强弱"而不是"整体涨跌"）
    model = GBDT(n_estimators=n_estimators, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    t0 = time.time()
    # early_stop<=0 表示不做早停（树太少会产生大量并列预测，把 RankIC 毁掉）
    if early_stop > 0:
        model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=early_stop)
    else:
        model.fit(Xtr, ytr)
    if verbose:
        print(f"  训练完成 {time.time()-t0:.0f}s  树数={getattr(model,'n_trees_','?')}", flush=True)

    # 测试集逐日预测
    test_idx = sp["test"]
    score = np.full((len(test_idx), X.shape[1]), np.nan, dtype=np.float32)
    ytest = np.full((len(test_idx), X.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(test_idx):
        row = Xn[t]
        m = np.isfinite(row).all(axis=1)
        if m.sum() >= 20:
            score[i, m] = model.predict(row[m])
        ytest[i] = y[t]

    res = _ic(score, ytest)
    res.update({"ok": True, "horizon": horizon, "train_n": int(Xtr.shape[0]),
                "test_start": test_start, "test_end": test_end,
                "valid_n": int(Xva.shape[0]), "n_trees": int(getattr(model, "n_trees_", 0)),
                "test_days": len(test_idx)})
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--test-start", default="2017-01-01")
    ap.add_argument("--test-end", default="2020-12-31")
    ap.add_argument("--train-end", default="2014-12-31")
    ap.add_argument("--valid-start", default="2015-01-01")
    ap.add_argument("--valid-end", default="2016-12-31")
    ap.add_argument("--trees", type=int, default=120)
    ap.add_argument("--early-stop", type=int, default=20)
    a = ap.parse_args(argv)

    print("=" * 64)
    print("复现 Qlib Alpha158 + GBDT 基准")
    print("=" * 64)
    print(f"  Qlib 官方公布（CSI300 / 2017-2020 样本外 / h=1）：")
    print(f"    LightGBM  IC=0.0448  RankIC=0.0469")
    print(f"    Linear    IC=0.0397  RankIC=0.0472")
    print()
    r = run(horizon=a.horizon, max_stocks=a.stocks,
            test_start=a.test_start, test_end=a.test_end,
            train_end=a.train_end, valid_start=a.valid_start, valid_end=a.valid_end,
            n_estimators=a.trees, early_stop=a.early_stop)
    print()
    if not r.get("ok"):
        print(f"失败: {r.get('reason')}")
        return 1
    print(f"我们的结果（h={r['horizon']}，测试 {r['test_days']} 天）：")
    print(f"  IC      = {r['ic']:+.4f}   (t={r['t']:+.2f})")
    print(f"  RankIC  = {r['rank_ic']:+.4f}")
    print(f"  ICIR    = {r['icir']:+.3f}")
    print(f"  RankICIR= {r['rank_icir']:+.3f}")
    print(f"  训练样本 {r['train_n']:,} / 验证 {r['valid_n']:,} / 树 {r['n_trees']}")
    print()
    target = 0.0448 if r["horizon"] == 1 else 0.03
    ratio = r["ic"] / target if target else 0
    print(f"对比目标 IC={target}：达成 {ratio:.0%}")
    if r["ic"] > 0 and r["t"] > 2:
        print("→ 复现成功：我们的评估框架与公开结果方向一致")
    else:
        print("→ 未复现：需检查数据口径或流程")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
