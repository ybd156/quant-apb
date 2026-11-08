"""
Meta 层 —— 二级学习：让系统自己发现"该不该反着用"。

为什么需要这一层：
    一级集成把 73 个信号加权平均，得到 score ∈ [-1,1]。但"平均"有个结构性缺陷：
    如果所有信号在当前市场都**系统性偏反**（技术信号在短周期上的均值回复效应），
    加权平均仍然是偏反的，权重怎么调都救不回来 —— 因为权重只能调整"听谁的多"，
    不能调整"要不要反着听"。

    实测就撞上了这件事：方向命中率 0.483~0.496（低于抛硬币），分层收益却单调。

Meta 层做的事（严格因果）：
    用**只有到期**的样本，在线回归  r_h = α + β·score
        β > 0  → 顺势用
        β < 0  → 反着用（系统自动翻转）
        β ≈ 0  → 没有信息，输出接近 0（等于"不表态"）
    再用样本量做收缩 β ← β·n/(n+n0)，避免小样本乱翻。

输出 `adj = β·score`，量纲是"预期 h 日收益"，比无量纲 score 更好解释。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MetaConfig:
    n0: float = 3000.0     # 收缩强度（等效样本数）：样本少时 β 被压向 0
    min_n: int = 500       # 低于此样本量直接输出 0（不表态）
    clip_beta: float = 0.05  # β 截断，防止极端值


def learn_meta(scores: np.ndarray, RET: np.ndarray, h: int, warmup: int = 120,
               cfg: MetaConfig = MetaConfig()) -> Dict:
    """
    scores / RET 均为 (T,N)。返回 beta (T,)、alpha (T,)、adj (T,N)、n_used (T,)。
    第 t 天的 β 只用到期的（as_of ≤ t-1-h）样本。
    """
    T, N = scores.shape
    beta_hist = np.zeros(T, dtype=np.float64)
    alpha_hist = np.zeros(T, dtype=np.float64)
    n_hist = np.zeros(T, dtype=np.int64)
    adj = np.zeros((T, N), dtype=np.float32)

    # 累积和（因果）
    n = 0.0; sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
    beta = 0.0; alpha = 0.0

    for t in range(T):
        s = t - 1 - h
        if s >= 0:
            sc, r = scores[s], RET[s]
            m = np.isfinite(sc) & np.isfinite(r)
            if m.any():
                x = sc[m].astype(np.float64); y = r[m].astype(np.float64)
                n += x.size; sx += x.sum(); sy += y.sum()
                sxx += (x * x).sum(); sxy += (x * y).sum()
                if n >= cfg.min_n:
                    vx = sxx - sx * sx / n
                    if vx > 1e-12:
                        b = (sxy - sx * sy / n) / vx
                        a = (sy - b * sx) / n
                        shrink = n / (n + cfg.n0)
                        beta = float(np.clip(b * shrink, -cfg.clip_beta, cfg.clip_beta))
                        alpha = float(a * shrink)

        beta_hist[t] = beta
        alpha_hist[t] = alpha
        n_hist[t] = int(n)
        if t >= warmup:
            row = scores[t]
            adj[t] = np.where(np.isfinite(row), alpha + beta * row, np.nan).astype(np.float32)

    # 方向统计
    sign_flips = int(np.sum(np.diff(np.sign(beta_hist)) != 0))
    return {
        "beta": beta_hist, "alpha": alpha_hist, "adj": adj, "n_used": n_hist,
        "beta_final": float(beta_hist[-1]), "alpha_final": float(alpha_hist[-1]),
        "sign_flips": sign_flips,
        "verdict": ("反着用" if beta_hist[-1] < -1e-6 else
                    ("顺势用" if beta_hist[-1] > 1e-6 else "无信息")),
        "n_final": int(n_hist[-1]),
    }


def compare_raw_vs_meta(scores: np.ndarray, RET: np.ndarray, meta: Dict,
                        warmup: int = 120) -> Dict:
    """同一份预测，比"原始 score"和"meta 调整后"哪个方向更准。"""
    adj = meta["adj"]
    out = {}
    for name, sc in (("raw", scores), ("meta", adj)):
        m = np.isfinite(sc) & np.isfinite(RET)
        if m.sum() == 0:
            out[name] = {"hit_rate": None, "n": 0}
            continue
        hit = (np.sign(sc[m]) == np.sign(RET[m]))
        out[name] = {
            "hit_rate": float(hit.mean()),
            "n": int(m.sum()),
            "mean_signed_ret": float((np.sign(sc[m]) * RET[m]).mean()),
        }
    return out
