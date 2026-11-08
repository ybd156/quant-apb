"""
多周期融合 —— 短期反转 + 长期动量，学术上最稳的组合之一。

## 为什么单周期注定平庸

同一批信号在 h=1 和 h=20 上的方向经常相反：
- **h=1（日频）**：A 股散户追涨杀跌 → 短期反转占优
- **h=20（月频）**：动量效应占优（Jegadeesh & Titman 1993）

只用 h=5 一个周期，等于把两种相反的力量混在一起，互相抵消。
本项目实测就是这个结果：h=5 的 IC = −0.0185（弱负），而 h=1 是 −0.0068、h=20 是 −0.0254。

## 融合方法

对每个周期 h 分别得到分数 s_h，然后：

    s_fused = Σ_h w_h · z(s_h)

w_h 由**因果回归**估计：用已到期的样本回归 r_t → [z(s_1), z(s_5), z(s_20)]，
取各周期的边际信息含量。**允许 w 为负**——这正是"短期反着用、长期顺着用"的数学表达。

## 关键：融合必须在"标准化后"做

三个周期的分数尺度不同（h 越长，收益方差越大）。不标准化直接加权，
等于让 h=20 主导一切。所以每个周期先做横截面 rank 标准化到 [-1,1]。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class MultiHorizonConfig:
    horizons: Tuple[int, ...] = (1, 5, 20)
    ridge: float = 1.0
    n0: float = 2000.0
    min_n: int = 500
    t_gate: float = 1.5
    warmup: int = 120


def xs_rank(x: np.ndarray) -> np.ndarray:
    """逐日横截面 rank → [-1,1]。"""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        m = np.isfinite(row)
        if m.sum() < 20:
            continue
        out[t, m] = pd.Series(row[m]).rank(pct=True).to_numpy() * 2.0 - 1.0
    return out


def fuse_horizons(scores_by_h: Dict[int, np.ndarray], RET: Dict[int, np.ndarray],
                  cfg: MultiHorizonConfig = MultiHorizonConfig()) -> Dict:
    """
    scores_by_h: {h: (T,N)}；RET: {h: (T,N)}。
    因果估计各周期权重并融合。
    """
    hs = [h for h in cfg.horizons if h in scores_by_h]
    if not hs:
        return {"ok": False, "reason": "no horizons"}
    K = len(hs)
    Z = {h: xs_rank(scores_by_h[h]) for h in hs}
    T, N = Z[hs[0]].shape

    # 目标：用最长周期的收益作为"总目标"（也可用各周期分别回归后平均）
    Y = RET[hs[-1]]

    beta_hist = np.zeros((T, K))
    t_hist = np.zeros((T, K))
    fused = np.full((T, N), np.nan, dtype=np.float32)
    Szz = np.zeros((K, K)); Szy = np.zeros(K)
    n = 0.0; beta = np.zeros(K); tvals = np.zeros(K)

    for t in range(T):
        s = t - 1 - hs[-1]
        if s >= 0:
            z = np.stack([Z[h][s] for h in hs], axis=0)     # (K,N)
            y = Y[s]
            m = np.isfinite(y) & np.isfinite(z).all(axis=0)
            if m.sum() >= 10:
                zm = z[:, m]; ym = y[m]
                Szz += zm @ zm.T; Szy += zm @ ym; n += float(m.sum())
                if n >= cfg.min_n:
                    A = Szz + np.eye(K) * (cfg.ridge * max(np.trace(Szz) / K, 1.0))
                    try:
                        Ainv = np.linalg.inv(A); b = Ainv @ Szy
                        resid = ym - zm.T @ b
                        s2 = float(resid @ resid) / max(n - K, 1)
                        se = np.sqrt(np.maximum(np.diag(Ainv) * s2, 1e-18))
                        tvals = b / se
                        beta = np.where(np.abs(tvals) >= cfg.t_gate, b * n / (n + cfg.n0), 0.0)
                    except np.linalg.LinAlgError:
                        beta = np.zeros(K)
        beta_hist[t] = beta; t_hist[t] = tvals
        if t >= cfg.warmup:
            zt = np.stack([Z[h][t] for h in hs], axis=0)
            ok = np.isfinite(zt).all(axis=0)
            fused[t] = np.where(ok, np.nansum(beta[:, None] * zt, axis=0), np.nan)

    return {"ok": True, "fused": fused, "horizons": hs,
            "beta": beta_hist, "t": t_hist,
            "final_beta": dict(zip([f"h{h}" for h in hs], beta_hist[-1].tolist())),
            "final_t": dict(zip([f"h{h}" for h in hs], t_hist[-1].tolist())),
            "n": int(n)}


def regime_interaction(scores: np.ndarray, RET: np.ndarray, regime: np.ndarray,
                       cfg: MultiHorizonConfig = MultiHorizonConfig()) -> Dict:
    """
    regime × 信号交互：**每个 regime 单独估一套系数**，而不是一套权重全局用。

    现在 learner 的 regime 表只调整"听谁的多"，但信号本身的**方向**在不同 regime 可能相反
    （例如动量在趋势市有效、在震荡市失效甚至反向）。本函数直接估 regime 特定的 β。
    """
    Z = xs_rank(scores)
    T, N = Z.shape
    regs = sorted(set(int(r) for r in np.unique(regime) if np.isfinite(r)))
    out = np.full((T, N), np.nan, dtype=np.float32)
    stats = {}
    for r in regs:
        idx = np.where(regime == r)[0]
        if idx.size < 60:
            stats[r] = {"ok": False, "reason": f"样本 {idx.size}"}
            continue
        # 因果：只用该 regime 下已到期的样本
        num = den = 0.0; n = 0
        b_hist = np.zeros(T)
        for t in range(T):
            s = t - 1 - 5
            if s >= 0 and regime[s] == r:
                z = Z[s]; y = RET[s]
                m = np.isfinite(z) & np.isfinite(y)
                if m.sum() >= 10:
                    xm = z[m] - z[m].mean(); ym = y[m] - y[m].mean()
                    num += float(xm @ ym); den += float(xm @ xm); n += int(m.sum())
            b_hist[t] = num / den if den > 1e-12 else 0.0
            if t >= cfg.warmup and regime[t] == r and den > 1e-12:
                out[t] = (b_hist[t] * Z[t]).astype(np.float32)
        stats[r] = {"ok": True, "beta": float(b_hist[-1]), "n": n}
    return {"ok": True, "scores": out, "per_regime": stats}


def evaluate(comp: np.ndarray, RET: np.ndarray, h: int = 5) -> Dict:
    from . import analysis as A, stats as ST
    ic = A.daily_ic_series(comp.T, RET.T); ic = ic[np.isfinite(ic)]
    st = ST.mean_tstat(ic, lags=h - 1)
    T, N = comp.shape
    sp = []
    for t in range(T):
        x = comp[t]; y = RET[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50:
            continue
        q1, q9 = np.percentile(x[m], 20), np.percentile(x[m], 80)
        sp.append(y[m][x[m] >= q9].mean() - y[m][x[m] <= q1].mean())
    sp = np.asarray(sp)
    s2 = ST.mean_tstat(sp, lags=h - 1) if sp.size > 30 else {"mean": np.nan, "t": np.nan}
    return {"ic": st["mean"], "t_ic": st["t"], "spread": s2["mean"], "t_spread": s2["t"]}
