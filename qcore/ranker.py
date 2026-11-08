"""
预测目标改造 —— 不预测收益，改预测"排序"或"概率"。

## 为什么回归目标可能不对

回归 `E[r|X]` 有个隐含问题：**它被极端值主导**。
A 股单日 ±10% 的样本会主导最小二乘的损失，而这些样本往往不可交易（涨跌停、停牌）。

而我们真正需要的不是"收益是多少"，而是"**谁比谁强**"——
这正是排序问题（Learning to Rank），而不是回归问题。

## 三种目标

| 目标 | 损失 | 适合什么 |
|---|---|---|
| **回归**（基线） | MSE | 想估绝对幅度 |
| **排序**（pairwise, RankNet） | `log(1+exp(-(s_i-s_j)))` | 只关心相对强弱 → 选股 |
| **分类**（top vs rest） | 逻辑损失 | 关心"是不是头部" |
| **分位数回归** | pinball loss | 关心"下/上分位"，抗极值 |

## 全部纯 numpy，因果训练

每种都是"扩张窗口训练 + 单步预测"：
- 第 t 天只用 ≤ t−1−h 的已到期样本训练
- 训练样本按天抽样（全量太慢），但**预测用全部特征**
- 学习率固定小步，不做早停（早停会引入未来信息）

## 诚实提醒

换目标不等于有 alpha。如果特征本身没有信息，三种目标都会给出接近 0 的 IC。
本模块的价值在于**排除"目标选错了"这个解释**。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TargetConfig:
    lr: float = 0.05
    epochs: int = 8
    n_pairs_per_day: int = 200
    top_quantile: float = 0.30
    quantile: float = 0.30          # 分位数回归的目标分位
    l2: float = 1e-3
    train_days_sample: int = 200    # 每次训练抽多少天
    min_train_days: int = 250
    warmup: int = 120
    seed: int = 42


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (X - mu) / sd


def _build_train(X: np.ndarray, y: np.ndarray, t_end: int, cfg: TargetConfig,
                 rng: np.random.Generator):
    """从 [0, t_end) 抽训练样本。X: (T,N,K)；y: (T,N)。"""
    days = np.arange(0, max(t_end, 0))
    if days.size == 0:
        return None, None
    if days.size > cfg.train_days_sample:
        days = rng.choice(days, cfg.train_days_sample, replace=False)
    xs, ys = [], []
    for d in days:
        m = np.isfinite(y[d]) & np.isfinite(X[d]).all(axis=1)
        if m.sum() >= 10:
            xs.append(X[d][m]); ys.append(y[d][m])
    if not xs:
        return None, None
    return np.vstack(xs), np.concatenate(ys)


def _fit_regression(X, y, cfg, rng):
    A = X.T @ X + cfg.l2 * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(A, b)


def _fit_pairwise(X, y, cfg, rng, k=1):
    """RankNet 风格：log(1+exp(-(s_i - s_j)))，y_i > y_j。"""
    n, K = X.shape
    w = np.zeros(K)
    for _ in range(cfg.epochs):
        i = rng.integers(0, n, cfg.n_pairs_per_day)
        j = rng.integers(0, n, cfg.n_pairs_per_day)
        better = y[i] > y[j]
        if better.sum() == 0:
            continue
        i, j = i[better], j[better]
        diff = X[i] - X[j]
        s = diff @ w
        g = -1.0 / (1.0 + np.exp(s))          # dL/ds
        w -= cfg.lr * (diff.T @ g) / len(i) + cfg.l2 * w
    return w


def _fit_logistic(X, y, cfg, rng):
    """分类：标签 = 当天收益是否进 top_quantile。"""
    thr = np.quantile(y, 1.0 - cfg.top_quantile)
    lab = (y >= thr).astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(cfg.epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (p - lab) / len(y)
        w -= cfg.lr * (g + cfg.l2 * w)
    return w


def _fit_quantile(X, y, cfg, rng):
    """分位数回归（pinball loss）。"""
    w = np.zeros(X.shape[1])
    q = cfg.quantile
    for _ in range(cfg.epochs):
        r = y - X @ w
        g = np.where(r > 0, -q, 1.0 - q)       # dL/dr
        w -= cfg.lr * (X.T @ g) / len(y) + cfg.l2 * w
    return w


FITTERS = {"regression": _fit_regression, "rank": _fit_pairwise,
           "classify": _fit_logistic, "quantile": _fit_quantile}


def fit_predict(X: np.ndarray, RET: np.ndarray, h: int = 5,
                target: str = "rank", cfg: TargetConfig = TargetConfig()) -> np.ndarray:
    """
    X: (T,N,K) 特征；RET: (T,N)。返回 (T,N) 分数。
    每 20 天重训一次（不必每天），仍严格因果。
    """
    T, N, K = X.shape
    rng = np.random.default_rng(cfg.seed)
    out = np.full((T, N), np.nan, dtype=np.float32)
    fit = FITTERS.get(target)
    if fit is None:
        raise ValueError(f"unknown target {target}")
    w = np.zeros(K)

    for t in range(cfg.warmup, T):
        if (t - cfg.warmup) % 20 == 0:
            t_end = t - 1 - h
            if t_end < cfg.min_train_days:
                continue
            Xt, yt = _build_train(X, RET, t_end, cfg, rng)
            if Xt is None or len(yt) < 500:
                continue
            Xs = _standardize(Xt)
            w = fit(Xs, yt, cfg, rng)
        row = X[t]
        ok = np.isfinite(row).all(axis=1)
        out[t] = np.where(ok, row @ w, np.nan)
    return out


def compare_targets(X: np.ndarray, RET: np.ndarray, h: int = 5,
                    targets=("regression", "rank", "classify", "quantile"),
                    cfg: TargetConfig = TargetConfig()) -> Dict:
    """同一批特征，比四种目标哪个方向更准。"""
    from . import analysis as A, stats as ST
    res = {}
    for tg in targets:
        sc = fit_predict(X, RET, h=h, target=tg, cfg=cfg)
        ic = A.daily_ic_series(sc.T, RET.T)
        ic = ic[np.isfinite(ic)]
        st = ST.mean_tstat(ic, lags=h - 1)
        T, N = sc.shape
        sp = []
        for t in range(T):
            x = sc[t]; y = RET[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 50:
                continue
            q1, q9 = np.percentile(x[m], 20), np.percentile(x[m], 80)
            sp.append(y[m][x[m] >= q9].mean() - y[m][x[m] <= q1].mean())
        sp = np.asarray(sp)
        s2 = ST.mean_tstat(sp, lags=h - 1) if sp.size > 30 else {"mean": np.nan, "t": np.nan}
        res[tg] = {"ic": st["mean"], "t_ic": st["t"],
                   "spread": s2["mean"], "t_spread": s2["t"]}
    best = max(res.items(), key=lambda kv: abs(kv[1]["t_ic"]) if np.isfinite(kv[1]["t_ic"]) else -1)
    return {"results": res, "best_target": best[0]}
