"""
统一模型 —— 把技术、人性、情绪、政策、资金放进同一个数学框架。

## 为什么需要一个统一模型

现在有 6 个信息层，各自产出"分数"，但分数之间**不可比**：
技术分是 vote 加权平均、行为分是 rank 均值、事件分是 CAR 估计、资金分是净流入 z 值。
直接相加等于把苹果和橘子混在一起。

统一模型做的事：**把所有层映射到同一个尺度——预期收益**。

## 数学结构

对每个交易日 t，横截面去均值后：

    E[r_i,t+h] = Σ_k β_k,t · z_k,i,t

  z_k,i,t  = 第 k 个信息层在第 t 日对股票 i 的标准化分数（横截面 rank → [-1,1]）
  β_k,t    = 该层的"边际信息含量"，由**只用已到期样本**的岭回归估计

β 的估计：

    β̂ = (ZᵀZ + λI)⁻¹ Zᵀy            # 岭回归，λ 对抗多重共线
    β_shrunk = β̂ · n/(n+n₀)          # 小样本收缩
    β_used = β_shrunk · 1[|t_k| ≥ t_gate]   # 不显著就置 0（不表态）

## 六个层与它们的可回测性（关键约束）

| 层 | 可回测 | 说明 |
|---|---|---|
| 技术 technical | ✅ | 73 个信号，全历史 |
| 人性 behavioral | ✅ | 10 个因子，全历史 |
| 经典指标 indicators | ✅ | 22 个指标，全历史 |
| 资金 capital | ⚠️ 部分 | 个股资金流仅 120 日；ETF 份额无历史 |
| 情绪 sentiment | ❌ | 股吧无法按历史日期翻页 |
| 政策 policy | ⚠️ 稀疏 | 56 个事件，年度级 |

**只有前三层能进回测。** 后三层只能前向积累 —— 这是数据决定的，不是设计选择。
在回测里硬塞情绪/资金，等于用今天的信息解释昨天的涨跌。

## 输出

- `composite`: 统一模型分（预期 h 日收益，单位与收益一致）
- `beta_hist`: 各层 β 的时间序列（可看"市场在奖励哪种信息"）
- `attribution`: 当前各层贡献占比
- `gates`: 各层是否通过显著性门槛
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class UnifiedConfig:
    ridge: float = 1.0          # 岭回归强度（标准化输入后 1.0 合适）
    n0: float = 2000.0          # 收缩强度（等效样本数）
    min_n: int = 500            # 最少到期样本
    t_gate: float = 1.5         # 单层显著性门槛
    warmup: int = 120


def _xs_rank(x: np.ndarray) -> np.ndarray:
    """逐日横截面 rank → [-1,1]，NaN 保持 NaN。"""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        m = np.isfinite(row)
        if m.sum() < 20:
            continue
        r = pd.Series(row[m]).rank(pct=True).to_numpy() * 2.0 - 1.0
        out[t, m] = r
    return out


def standardize_layers(layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把每一层压成 [-1,1] 的横截面 rank 分数（同尺度，可比）。"""
    return {k: _xs_rank(np.asarray(v, dtype=np.float64)) for k, v in layers.items()}


def fit_unified(layers: Dict[str, np.ndarray], RET: np.ndarray, h: int,
                cfg: UnifiedConfig = UnifiedConfig()) -> Dict:
    """
    layers: {name: (T,N)}；RET: (T,N)。全部因果，只用到期样本估 β。
    """
    names = list(layers)
    K = len(names)
    Z = standardize_layers(layers)
    T, N = RET.shape

    Zstack = np.stack([Z[k] for k in names], axis=0)      # (K,T,N)
    beta_hist = np.zeros((T, K))
    t_hist = np.zeros((T, K))
    n_hist = np.zeros(T, dtype=np.int64)
    comp = np.full((T, N), np.nan, dtype=np.float32)

    Szz = np.zeros((K, K))
    Szy = np.zeros(K)
    n = 0.0
    beta = np.zeros(K)
    tvals = np.zeros(K)

    for t in range(T):
        s = t - 1 - h
        if s >= 0:
            z = Zstack[:, s, :]                  # (K,N)
            y = RET[s]
            m = np.isfinite(y) & np.isfinite(z).all(axis=0)
            if m.sum() >= 5:
                zm = z[:, m]
                ym = y[m]
                Szz += zm @ zm.T
                Szy += zm @ ym
                n += float(m.sum())
                if n >= cfg.min_n:
                    A = Szz + np.eye(K) * (cfg.ridge * max(np.trace(Szz) / K, 1.0))
                    try:
                        Ainv = np.linalg.inv(A)
                        b = Ainv @ Szy
                        resid = ym - zm.T @ b
                        dof = max(n - K, 1)
                        s2 = float(resid @ resid) / dof
                        se = np.sqrt(np.maximum(np.diag(Ainv) * s2, 1e-18))
                        tvals = b / se
                        shrink = n / (n + cfg.n0)
                        beta = np.where(np.abs(tvals) >= cfg.t_gate, b * shrink, 0.0)
                    except np.linalg.LinAlgError:
                        beta = np.zeros(K)
        beta_hist[t] = beta
        t_hist[t] = tvals
        n_hist[t] = int(n)
        if t >= cfg.warmup:
            row = np.nansum(beta[:, None] * Zstack[:, t, :], axis=0)
            valid = np.isfinite(Zstack[:, t, :]).all(axis=0)
            comp[t] = np.where(valid, row, np.nan).astype(np.float32)

    # 归因：当前各层对组合分的贡献占比
    contrib = {}
    if np.isfinite(Zstack[:, -1, :]).all(axis=0).any():
        parts = beta[:, None] * np.nan_to_num(Zstack[:, -1, :])
        total = np.abs(parts).sum()
        contrib = {names[k]: float(np.abs(parts[k]).sum() / total) if total > 0 else 0.0
                   for k in range(K)}

    return {
        "composite": comp, "beta": beta_hist, "t": t_hist, "n": n_hist,
        "names": names,
        "final_beta": dict(zip(names, beta_hist[-1].tolist())),
        "final_t": dict(zip(names, t_hist[-1].tolist())),
        "active": [k for k, v in zip(names, beta_hist[-1]) if abs(v) > 1e-12],
        "attribution": contrib,
        "n_used": int(n),
    }


def summarize(res: Dict) -> str:
    """一句话说明当前谁在起作用。"""
    act = res.get("active") or []
    if not act:
        return "所有层都不显著 → 统一模型选择不表态（预期收益记 0）"
    parts = [f"{k}(β={res['final_beta'][k]:+.4f}, t={res['final_t'][k]:+.2f})"
             for k in act]
    return "有效层：" + "、".join(parts)


def evaluate(comp: np.ndarray, RET: np.ndarray, dates: np.ndarray, h: int = 5) -> Dict:
    """统一模型分本身的检验（IC / 五分位价差，HAC 校正）。"""
    from . import analysis as A, stats as ST
    ic = A.daily_ic_series(comp.T, RET.T)
    ic = ic[np.isfinite(ic)]
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
    return {"ic": st["mean"], "t_ic": st["t"], "spread": s2["mean"], "t_spread": s2["t"],
            "n_days": int(ic.size),
            "verdict": ("有效" if (abs(st["t"]) >= 2 and s2["mean"] > 0) else
                        "统计显著但不可交易" if abs(st["t"]) >= 2 else "无优势")}
