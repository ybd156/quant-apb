"""
均值-方差优化 + 换手惩罚 —— 把"选谁"和"买多少"联合优化。

## 现在的问题

权重（learner）和仓位（position）是**分开优化**的：
- learner 只看信号准不准，不管交易成本
- position 只看回撤，不管组合风险结构

结果就是：两个局部最优，合起来不是最优。

## 目标函数

    max_w   μᵀw − (γ/2)·wᵀΣw − κ·‖w − w_prev‖₁

    第一项：预期收益
    第二项：风险惩罚（γ = 风险厌恶系数）
    第三项：换手惩罚（κ 把交易成本显式写进目标）

第三项是关键：**换手会直接扣分**，所以优化器会自动避开"为了微小 alpha 大幅调仓"。

## 求解：投影梯度 + L1 软阈值

L1 换手惩罚的梯度是次梯度，用软阈值（soft-thresholding）处理：

    w ← soft(w − lr·∇, lr·κ)

再投影到约束集（非负、单票上限、行业上限、权重和=1）。

## 为什么这比"先排序再等权"好

等权忽略了两件事：① 信号强度差异 ② 股票间相关性。
两只高相关股票各配 10%，实际风险敞口是 20% 而不是 10%。
均值-方差用 Σ 把这个考虑进去。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class OptimizerConfig:
    risk_aversion: float = 5.0        # γ
    turnover_penalty: float = 0.002   # κ（每单位换手的惩罚）
    lr: float = 0.05
    iters: int = 300
    max_weight: float = 0.15
    max_sector: float = 0.35
    long_only: bool = True


def _project(w: np.ndarray, cfg: OptimizerConfig) -> np.ndarray:
    """投影到 {w ≥ 0, Σw = 1, w_i ≤ max_weight}。"""
    w = np.asarray(w, dtype=float)
    if cfg.long_only:
        w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 1e-12:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / s
    # 单票上限（注水法）
    if cfg.max_weight * len(w) >= 1.0:
        for _ in range(50):
            over = w > cfg.max_weight + 1e-12
            if not over.any():
                break
            excess = (w[over] - cfg.max_weight).sum()
            w[over] = cfg.max_weight
            free = ~over
            if not free.any():
                break
            w[free] += excess * (w[free] / w[free].sum())
    return w / max(w.sum(), 1e-12)


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def optimize(mu: np.ndarray, Sigma: np.ndarray, w_prev: Optional[np.ndarray] = None,
             sectors: Optional[Sequence[str]] = None,
             cfg: OptimizerConfig = OptimizerConfig()) -> Dict:
    """
    mu: (N,) 预期收益；Sigma: (N,N) 协方差。
    返回最优权重与分解（收益项/风险项/换手项）。
    """
    mu = np.nan_to_num(np.asarray(mu, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    S = np.nan_to_num(np.asarray(Sigma, dtype=float), nan=1e-6, posinf=1e-6, neginf=1e-6)
    n = mu.size
    if n == 0:
        return {"ok": False, "reason": "empty"}
    if w_prev is None:
        w_prev = np.full(n, 1.0 / n)
    w_prev = np.asarray(w_prev, dtype=float)

    w = w_prev.copy()
    for _ in range(cfg.iters):
        grad = mu - cfg.risk_aversion * (S @ w)
        step = w + cfg.lr * grad
        # L1 换手惩罚的软阈值（相对于 w_prev）
        step = _soft_threshold(step - w_prev, cfg.lr * cfg.turnover_penalty) + w_prev
        w_new = _project(step, cfg)
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new

    # 行业上限（迭代收紧）
    if sectors is not None:
        sec = np.array(sectors, dtype=object)
        for _ in range(30):
            changed = False
            for k in set(sec.tolist()):
                m = sec == k
                tot = w[m].sum()
                if tot > cfg.max_sector + 1e-9:
                    scale = cfg.max_sector / tot
                    freed = tot * (1 - scale)
                    w[m] *= scale
                    other = ~m
                    if other.any() and w[other].sum() > 0:
                        w[other] += freed * (w[other] / w[other].sum())
                    changed = True
            if not changed:
                break
        w = _project(w, cfg)

    ret_term = float(mu @ w)
    risk_term = float(w @ S @ w)
    turn_term = float(np.abs(w - w_prev).sum())
    obj = ret_term - 0.5 * cfg.risk_aversion * risk_term - cfg.turnover_penalty * turn_term
    return {"ok": True, "weights": w, "objective": obj,
            "expected_return": ret_term, "risk": risk_term,
            "turnover": turn_term,
            "effective_n": float(1.0 / max((w ** 2).sum(), 1e-12)),
            "sector_share": ({str(k): round(float(w[sec == k].sum()), 4)
                              for k in set(sec.tolist())} if sectors is not None else None)}


def turnover_aware_backtest(mu_series: np.ndarray, ret_series: np.ndarray,
                            Sigma_fn=None, cfg: OptimizerConfig = OptimizerConfig(),
                            rebalance_every: int = 5, warmup: int = 60) -> Dict:
    """
    对比"等权"与"均值-方差+换手惩罚"的净收益（扣换手成本）。
    mu_series: (T,N) 每期预期收益；ret_series: (T,N) 实际收益。
    """
    mu_series = np.asarray(mu_series, dtype=float)
    ret_series = np.asarray(ret_series, dtype=float)
    # 容错：(N,T) 传入时自动转置
    if mu_series.shape != ret_series.shape:
        if mu_series.T.shape == ret_series.shape:
            mu_series = mu_series.T
        elif ret_series.T.shape == mu_series.shape:
            ret_series = ret_series.T
    T, N = mu_series.shape
    w_eq = np.full(N, 1.0 / N)
    w_prev = np.full(N, 1.0 / N)
    nav_eq = nav_mv = 1.0
    to_eq, to_mv = [], []
    cost = 10.0 / 1e4

    for t in range(T):
        r = np.nan_to_num(ret_series[t], nan=0.0)
        if t < warmup:
            continue
        # 等权（每期再平衡）
        turn_eq = float(np.abs(w_eq - w_prev).sum())
        nav_eq *= (1 + float(w_eq @ r) - cost * turn_eq)
        to_eq.append(turn_eq)

        if (t - warmup) % rebalance_every != 0:
            nav_mv *= (1 + float(w_prev @ r))
            continue
        if Sigma_fn is not None:
            S = Sigma_fn(t)
        else:
            v = np.nanvar(ret_series[max(0, t - 250):t], axis=0, ddof=0)
            S = np.diag(np.nan_to_num(v, nan=1e-4, posinf=1e-4, neginf=1e-4) + 1e-6)
        mu_t = np.nan_to_num(mu_series[t], nan=0.0, posinf=0.0, neginf=0.0)
        res = optimize(mu_t, S, w_prev, cfg=cfg)
        if res.get("ok"):
            w = res["weights"]
            turn_mv = float(np.abs(w - w_prev).sum())
            nav_mv *= (1 + float(w @ r) - cost * turn_mv)
            w_prev = w
            to_mv.append(turn_mv)
        else:
            nav_mv *= (1 + float(w_prev @ r))

    return {"equal_weight": {"final_nav": float(nav_eq),
                             "total_return": float(nav_eq - 1),
                             "avg_turnover": float(np.mean(to_eq))},
            "mean_variance": {"final_nav": float(nav_mv),
                              "total_return": float(nav_mv - 1),
                              "avg_turnover": float(np.mean(to_mv)) if to_mv else float("nan")},
            "diff": float(nav_mv - nav_eq)}
