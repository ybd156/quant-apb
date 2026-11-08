"""
多源组合层 —— 让数据决定"技术/行为/基本面各占多少"，而不是拍脑袋。

## 为什么不能简单加权

实测（283 只 / 2013-2026 / h=5）：
    技术分（meta 翻正后）  IC = +0.0154  t = +1.84
    行为分                IC = +0.0352  t = +5.14
    两者横截面相关        −0.43

负相关说明它们**确实提供不同信息**，但等权相加反而稀释 —— 因为两者信噪比差 2 倍。
正确做法：按**各自的信息含量**加权，而信息含量必须从数据里估，不能拍。

## 方法：因果多元回归 + 收缩 + 显著性闸门

对每个到期日 s，把各分量的横截面值去均值（与 IC 同口径），
累积回归  r_s = Σ_k β_k · x_{k,s} + ε

    β_k = 第 k 个来源的边际信息含量
    β 用 n/(n+n0) 收缩，样本少时压向 0
    某个分量 |t| < t_min 时把它的 β 置 0（不显著就不给权重）

全部扩张窗口、只用已到期样本 —— 与 meta 层同一套纪律。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ComboConfig:
    n0: float = 2000.0        # 收缩强度（等效样本数）
    min_n: int = 500          # 最少到期样本
    t_min: float = 1.5        # 单个分量的显著性门槛，低于此置 0
    ridge: float = 1e-6       # 数值稳定


def _xs_demean(x: np.ndarray) -> np.ndarray:
    """横截面去均值（按天）。"""
    out = np.array(x, dtype=np.float64, copy=True)
    mu = np.nanmean(out, axis=1, keepdims=True)
    out = out - mu
    return out


def learn_combo(scores: Dict[str, np.ndarray], RET: np.ndarray, h: int,
                warmup: int = 120, cfg: ComboConfig = ComboConfig()) -> Dict:
    """
    scores: {name: (T,N)}；RET: (T,N)。
    返回因果的组合权重历史、组合分、各分量 t 值。
    """
    names = list(scores)
    K = len(names)
    T, N = RET.shape
    X = np.stack([_xs_demean(np.asarray(scores[k], dtype=np.float64)) for k in names], axis=0)  # (K,T,N)

    beta_hist = np.zeros((T, K))
    t_hist = np.zeros((T, K))
    n_hist = np.zeros(T, dtype=np.int64)
    combo = np.full((T, N), np.nan, dtype=np.float32)

    # 累积矩：Σxx (K,K), Σxy (K), n
    Sxx = np.zeros((K, K))
    Sxy = np.zeros(K)
    n = 0.0
    beta = np.zeros(K)
    tvals = np.zeros(K)

    for t in range(T):
        s = t - 1 - h
        if s >= 0:
            x = X[:, s, :]                       # (K,N)
            y = RET[s]
            m = np.isfinite(y) & np.isfinite(x).all(axis=0)
            if m.sum() >= 5:
                xm = x[:, m]; ym = y[m]
                Sxx += xm @ xm.T
                Sxy += xm @ ym
                n += float(m.sum())
                if n >= cfg.min_n:
                    A = Sxx + np.eye(K) * (cfg.ridge * max(np.trace(Sxx) / K, 1.0))
                    try:
                        b = np.linalg.solve(A, Sxy)
                        # 近似标准误：残差方差 × (X'X)^{-1} 对角
                        resid = ym - xm.T @ b
                        dof = max(n - K, 1)
                        s2 = float(resid @ resid) / dof
                        cov = s2 * np.linalg.inv(A)
                        se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
                        shrink = n / (n + cfg.n0)
                        b_sh = b * shrink
                        tvals = b / se
                        beta = np.where(np.abs(tvals) >= cfg.t_min, b_sh, 0.0)
                    except np.linalg.LinAlgError:
                        beta = np.zeros(K)

        beta_hist[t] = beta
        t_hist[t] = tvals
        n_hist[t] = int(n)
        if t >= warmup:
            row = np.nansum(beta[:, None] * X[:, t, :], axis=0)
            valid = np.isfinite(X[:, t, :]).all(axis=0)
            combo[t] = np.where(valid, row, np.nan).astype(np.float32)

    return {"combo": combo, "beta": beta_hist, "t": t_hist, "n": n_hist,
            "names": names,
            "final_beta": dict(zip(names, beta_hist[-1].tolist())),
            "final_t": dict(zip(names, t_hist[-1].tolist())),
            "active": [k for k, v in zip(names, beta_hist[-1]) if abs(v) > 1e-12]}


def explain(res: Dict) -> str:
    """一句话说明当前谁在起作用。"""
    act = res.get("active") or []
    if not act:
        return "所有分量都不显著，组合层选择不表态"
    parts = []
    for k in act:
        parts.append(f"{k}(β={res['final_beta'][k]:+.4f}, t={res['final_t'][k]:+.2f})")
    return "当前有效分量：" + "、".join(parts)
