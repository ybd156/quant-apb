"""
分布漂移监控（PSI / KL）—— 输入变了，输出失效只是时间问题。

## 为什么必须监控输入而不是只看输出

现在所有监控都在看**输出**（IC、回撤、命中率）。但输出是滞后的：
等 IC 掉下来，往往已经亏了几个月。

输入分布漂移是**更早的预警**：
- 换手率的分布整体上移（市场变热）
- 信号发声频率骤降（数据源出问题 / 市场结构变了）
- 特征相关性结构变化（原本互补的信号开始同向）

## 两个指标

**PSI（Population Stability Index）** —— 分箱后比分布，业界最常用

    PSI = Σ (实际占比 − 期望占比) · ln(实际占比 / 期望占比)

    < 0.10  稳定
    0.10~0.25  有变化，关注
    > 0.25  显著漂移，模型大概率失效

**KL 散度** —— 信息论视角，对尾部更敏感

    KL(P‖Q) = Σ P·ln(P/Q)

## 用法

    psi = psi_series(feature_history, baseline_window=250, step=20)

返回每个时间点的 PSI，画出曲线就能看到"什么时候开始不对"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DriftConfig:
    n_bins: int = 10
    baseline_window: int = 250
    step: int = 20
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    eps: float = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray,
        n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI：以 expected 的分箱边界为准，比较两者占比。"""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if e.size < 30 or a.size < 30:
        return float("nan")
    qs = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if qs.size < 3:
        return 0.0
    e_hist, _ = np.histogram(e, bins=qs)
    a_hist, _ = np.histogram(a, bins=qs)
    e_pct = np.maximum(e_hist / max(e_hist.sum(), 1), eps)
    a_pct = np.maximum(a_hist / max(a_hist.sum(), 1), eps)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def kl_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 20,
                  eps: float = 1e-9) -> float:
    """KL(P‖Q)：以 p 的分布为基准。"""
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    q = np.asarray(q, dtype=float); q = q[np.isfinite(q)]
    if p.size < 30 or q.size < 30:
        return float("nan")
    lo = min(p.min(), q.min()); hi = max(p.max(), q.max())
    if not (lo < hi):
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ph, _ = np.histogram(p, bins=bins)
    qh, _ = np.histogram(q, bins=bins)
    pp = np.maximum(ph / max(ph.sum(), 1), eps)
    qq = np.maximum(qh / max(qh.sum(), 1), eps)
    return float(np.sum(pp * np.log(pp / qq)))


def psi_series(values: np.ndarray, cfg: DriftConfig = DriftConfig()) -> pd.DataFrame:
    """
    values: (T,) 或 (T,K) 特征序列。返回每 step 个点的 PSI。
    基准 = 前 baseline_window 个点（滚动更新，因果）。
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    T, K = v.shape
    rows = []
    for t in range(cfg.baseline_window, T, cfg.step):
        base = v[t - cfg.baseline_window:t]
        cur = v[max(0, t - 60):t]
        rec = {"t": t}
        for k in range(K):
            rec[f"psi_{k}"] = psi(base[:, k], cur[:, k], cfg.n_bins, cfg.eps)
        rec["psi_mean"] = float(np.nanmean([rec[f"psi_{k}"] for k in range(K)]))
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df["level"] = np.where(df["psi_mean"] >= cfg.psi_alert, "alert",
                               np.where(df["psi_mean"] >= cfg.psi_warn, "warn", "ok"))
    return df


def drift_report(psi_df: pd.DataFrame, cfg: DriftConfig = DriftConfig()) -> Dict:
    if psi_df is None or len(psi_df) == 0:
        return {"ok": False, "reason": "no_data"}
    last = psi_df.iloc[-1]
    level = last["level"]
    return {
        "ok": True, "current_psi": float(last["psi_mean"]),
        "level": level,
        "n_alert": int((psi_df["level"] == "alert").sum()),
        "n_warn": int((psi_df["level"] == "warn").sum()),
        "verdict": ("输入分布显著漂移 → 模型大概率已失效，应降仓或重训"
                    if level == "alert" else
                    "输入分布有变化 → 关注，暂不调整"
                    if level == "warn" else "输入分布稳定"),
    }


def feature_drift_from_signals(DIR: np.ndarray, cfg: DriftConfig = DriftConfig()) -> pd.DataFrame:
    """
    从信号矩阵算漂移：每天"有多少信号发声"的序列。
    发声率骤降通常意味着数据源异常或市场结构变化。
    """
    T, N, S = DIR.shape
    fire_rate = (DIR != 0).mean(axis=(1, 2))
    return psi_series(fire_rate, cfg)
