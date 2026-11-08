"""
生存分析 —— 预测"多久会涨"，而不是"会不会涨"。

## 为什么换个问法

现在所有模型都在回答「未来 h 天涨还是跌」。但交易真正关心的是：

    **多久能到目标价？中途会不会先触发止损？**

这是**时间到事件（time-to-event）**问题，不是分类问题。
标准工具是生存分析（Cox 1979），它天然处理**删失**（censoring）——
持有期结束时还没到目标价，不算"失败"，只算"未观测完"。

## Kaplan-Meier

    S(t) = Π_{t_i ≤ t} (1 − d_i / n_i)

    d_i = 在 t_i 时刻发生事件的个数
    n_i = 在 t_i 时刻仍处于风险中的个数

给出"持有 N 天后仍未达标"的概率曲线。

## Cox 比例风险（简化版）

    h(t|X) = h₀(t) · exp(βᵀX)

用偏似然（partial likelihood）估 β：

    L = Π_{事件} exp(βᵀX_i) / Σ_{j∈风险集} exp(βᵀX_j)

**风险比 exp(β) > 1** 表示该特征加速"达标"，< 1 表示延缓。

## 本项目的用法

对每只股票每天，定义事件 = "未来 20 日内涨幅首次达到 +5%"。
未被触发即删失。用 Cox 估出哪些信号能加速达标，输出风险分。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SurvivalConfig:
    target_return: float = 0.05     # 事件阈值：涨 5%
    max_horizon: int = 20           # 观测窗口
    l2: float = 1e-2
    iters: int = 200
    lr: float = 0.01


def build_events(RET_daily: np.ndarray, cfg: SurvivalConfig = SurvivalConfig()
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    从日收益构造 (time, event)：
      time  = 首次达到 target_return 所需天数（未达到则 = max_horizon）
      event = 1 表示在窗口内达到，0 表示删失
    RET_daily: (T,N) 日收益。
    """
    T, N = RET_daily.shape
    time = np.full((T, N), np.nan)
    event = np.zeros((T, N), dtype=np.int8)
    for t in range(T - cfg.max_horizon):
        cum = np.ones(N)
        for k in range(cfg.max_horizon):
            cum *= (1.0 + np.nan_to_num(RET_daily[t + k]))
            hit = (cum - 1.0) >= cfg.target_return
            newly = hit & (event[t] == 0) & np.isfinite(RET_daily[t + k])
            event[t][newly] = 1
            time[t][newly & ~np.isfinite(time[t])] = k + 1
        # 未命中 -> 删失在 max_horizon
        time[t][event[t] == 0] = cfg.max_horizon
    return time, event


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """KM 生存曲线。"""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    m = np.isfinite(t)
    t, e = t[m], e[m]
    if t.size == 0:
        return pd.DataFrame(columns=["t", "at_risk", "events", "survival"])
    rows = []
    surv = 1.0
    for tt in np.sort(np.unique(t)):
        at_risk = int((t >= tt).sum())
        d = int(((t == tt) & (e == 1)).sum())
        if at_risk > 0 and d > 0:
            surv *= (1.0 - d / at_risk)
        rows.append({"t": float(tt), "at_risk": at_risk, "events": d, "survival": surv})
    return pd.DataFrame(rows)


def cox_fit(X: np.ndarray, time: np.ndarray, event: np.ndarray,
            cfg: SurvivalConfig = SurvivalConfig()) -> Dict:
    """
    简化 Cox：按时间排序做偏似然梯度上升。
    X: (n,K)；time: (n,)；event: (n,)
    """
    m = np.isfinite(time) & np.isfinite(X).all(axis=1)
    X, time, event = X[m], time[m], event[m]
    if X.shape[0] < 50:
        return {"ok": False, "reason": f"样本 {X.shape[0]} 不足"}
    mu = X.mean(axis=0); sd = X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xs = (X - mu) / sd

    order = np.argsort(-time)          # 从大到小，风险集累积
    Xs, ev = Xs[order], event[order]
    n, K = Xs.shape
    beta = np.zeros(K)
    for _ in range(cfg.iters):
        eta = Xs @ beta
        eta = np.clip(eta, -20, 20)
        w = np.exp(eta)
        # 风险集累积和（从末尾往前）
        cum_w = np.cumsum(w[::-1])[::-1]
        cum_wx = np.cumsum((w[:, None] * Xs)[::-1], axis=0)[::-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            grad = np.zeros(K)
            for i in range(n):
                if ev[i] != 1:
                    continue
                grad += Xs[i] - cum_wx[i] / max(cum_w[i], 1e-12)
        grad = grad / max(ev.sum(), 1) - cfg.l2 * beta
        beta += cfg.lr * grad
    hr = np.exp(np.clip(beta, -5, 5))
    return {"ok": True, "beta": beta, "hazard_ratio": hr,
            "n": int(n), "n_events": int(ev.sum()),
            "verdict": ("风险比 >1 的特征会**加速**达标（越早涨到位）"
                        if (hr > 1).any() else "无加速特征")}


def survival_scores(X: np.ndarray, cox: Dict) -> np.ndarray:
    """用 Cox 的 β 给每行打"达标速度"分（越大越快）。"""
    if not cox.get("ok"):
        return np.full(X.shape[0], np.nan)
    b = cox["beta"]
    return X @ b


def survival_report(RET_daily: np.ndarray, cfg: SurvivalConfig = SurvivalConfig()) -> Dict:
    """快速体检：达标概率与中位时间。"""
    time, event = build_events(RET_daily, cfg)
    km = kaplan_meier(time, event)
    if len(km) == 0:
        return {"ok": False, "reason": "no_events"}
    hit_rate = float(np.nanmean(event[np.isfinite(time)]))
    # 中位达标时间：生存曲线首次跌破 0.5
    below = km[km["survival"] <= 0.5]
    median_t = float(below["t"].iloc[0]) if len(below) else float(cfg.max_horizon)
    return {"ok": True, "target": cfg.target_return, "horizon": cfg.max_horizon,
            "hit_rate": hit_rate, "median_days": median_t,
            "survival_at_h": float(km["survival"].iloc[-1]),
            "km": km}
