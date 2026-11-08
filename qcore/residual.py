"""
残差学习 —— 直接针对"选股效应为负"这个实测问题。

## 问题诊断

Brinson 归因实测（本项目，283 只 / 2013-2026 / h=5）：

    总超额      +0.006% / 期
    行业配置    +0.055%      ← 收益来源
    行业内选股  −0.227%      ← **在减分**
    交互        +0.179%

也就是说：模型在同行业内**选错股**，正超额全靠行业配置。

## 为什么会这样

因为我们一直在预测**原始收益** r_i,t+h。
但原始收益里，绝大部分方差来自三个与"选股能力"无关的东西：

    r_i = α_market  +  β_i·r_market  +  γ_industry  +  δ_size  +  ε_i
          ↑市场beta    ↑个股beta        ↑行业         ↑市值       ↑残差

模型在预测 r_i 时，会不自觉地去拟合 β、γ、δ —— 那些**不需要选股能力就能拿到**的部分。
结果就是"行业配置对了、选股错了"，而总超额被配置项撑着。

## 解决办法：只预测残差

    ① 每天横截面回归 r_i = a + b·log_size_i + Σ_k c_k·1[行业=k] + ε_i
    ② 用 ε_i（残差）作为**学习目标**
    ③ 预测 ε̂_i 后，再决定买谁

这样模型被迫只学"同行业、同市值下谁更强"——**这才是选股**。

## 理论支撑

- Fama & MacBeth (1973)：横截面回归 + 残差，是因子研究的标准做法
- Blitz, Huij & Martens (2011)：残差动量（residual momentum）比原始动量更稳健
- 本项目自己的教训：amihud 中性化后价差从 +0.566% 掉到 +0.031%
  —— 说明**不中性化的因子测到的都是 beta 伪装**

## 双层结构

残差学习只解决"选股"。总收益 = 择时 × 选股，所以还要一层：

    择时层：决定总仓位（市场 beta 暴露）→ 用市场级信号（宽度/波动/趋势）
    选股层：决定相对配置（alpha）      → 用残差学习

两层分离的好处：**能分别评估**。现在混在一起，看不出是"看错大盘"还是"选错股"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ResidualConfig:
    size_col: str = "log_amount"     # 市值代理（PIT 安全：用成交额而非今日市值）
    min_names: int = 30
    clip_resid: float = 0.10         # 残差截断（防极端值主导）


def residualize(RET: np.ndarray, SZ: np.ndarray, IND: np.ndarray,
                cfg: ResidualConfig = ResidualConfig()) -> np.ndarray:
    """
    逐日横截面回归，去掉行业与市值效应，返回残差收益。
    RET (T,N)；SZ (T,N) log 市值代理；IND (N,) 行业编码。
    """
    T, N = RET.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    uniq = np.unique(IND)
    for t in range(T):
        y = RET[t]
        s = SZ[t]
        m = np.isfinite(y) & np.isfinite(s)
        if m.sum() < cfg.min_names:
            continue
        yv, sv, iv = y[m], s[m], IND[m]
        # 行业去均值
        yd = yv.copy()
        for k in uniq:
            sel = iv == k
            if sel.sum() >= 3:
                yd[sel] -= yv[sel].mean()
        # 对 size 回归取残差
        sv2 = sv - sv.mean()
        beta = float((yd * sv2).sum() / max((sv2 * sv2).sum(), 1e-12))
        r = yd - beta * sv2
        r = np.clip(r, -cfg.clip_resid, cfg.clip_resid)
        out[t, m] = r
    return out


def residual_momentum(RET: np.ndarray, SZ: np.ndarray, IND: np.ndarray,
                      lookback: int = 60, skip: int = 5) -> np.ndarray:
    """
    残差动量（Blitz et al. 2011）：对残差收益累加，跳过最近 skip 天（防短期反转）。
    返回 (T,N)，值越大 = 残差动量越强。
    """
    res = residualize(RET, SZ, IND)
    df = pd.DataFrame(res)
    cum = df.rolling(lookback, min_periods=lookback // 2).sum().shift(skip)
    return cum.to_numpy()


# ------------------------------------------------------------------ 双层
def two_layer(scores_resid: np.ndarray, market_ret: np.ndarray,
              timing_signal: Optional[np.ndarray] = None) -> Dict:
    """
    双层分离：
      选股层 = scores_resid（残差预测）
      择时层 = timing_signal（市场级，可为空 → 满仓）
    返回两层各自的评估口径。
    """
    T, N = scores_resid.shape
    # 选股层：纯 alpha（残差收益的横截面排序）
    return {"selection_scores": scores_resid,
            "timing": timing_signal if timing_signal is not None else np.ones(T),
            "note": "选股层用残差分数，择时层用市场级信号；两层可独立评估"}


def selection_alpha(scores: np.ndarray, RESID: np.ndarray, h: int = 5,
                    top_k: int = 10) -> Dict:
    """
    直接测"选股能力"：按分数选股，看它们相对**同行业同市值**的超额（即残差收益）。
    这才是干净的 alpha 度量。
    """
    T, N = scores.shape
    sc = np.asarray(scores, dtype=float)
    if sc.shape[0] != T:      # 容忍 (N,T) 传入
        sc = sc.T
    sp = []
    for t in range(T - 1 - h):
        s = sc[:, t]
        y = RESID[t]
        m = np.isfinite(s) & np.isfinite(y)
        if m.sum() < top_k * 2:
            continue
        ii = np.where(m)[0]
        pick = ii[np.argsort(-s[ii])][:top_k]
        sp.append(float(y[pick].mean()))          # 选中组的残差收益
    if not sp:
        return {"ok": False, "reason": "样本不足"}
    from . import stats as ST
    st = ST.mean_tstat(np.asarray(sp), lags=h - 1)
    return {"ok": True, "alpha": st["mean"], "t": st["t"],
            "n_periods": st["n"],
            "verdict": ("选股能力显著为正" if st["t"] >= 2 else
                        "选股能力显著为负" if st["t"] <= -2 else "选股能力不显著")}
