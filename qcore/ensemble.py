"""
多模型集成 —— 不同 regime 训练不同模型，而不是一套权重全局用。

## 现在的问题

learner 的 regime 表只是"在同一个得分公式上，按 regime 调整权重"。
但更彻底的做法是：**每个 regime 单独训练一个模型**，因为：

- 趋势市里动量有效，震荡市里动量反向
- 高波动期里反转更强，低波动期里动量更强
- 同一组特征在不同 regime 下的**系数符号都可能相反**

用一套权重强行覆盖所有 regime，等于假设"市场规律永远一样"。

## 方法：分 regime 训练 + 平滑混合

    对每个 regime r：β_r = argmin Σ_{t∈r} (y_t − βᵀx_t)² + λ‖β‖²
    预测时：s_t = Σ_r π_r(t) · β_rᵀx_t

π_r(t) 是**软归属权重**（当前 regime 给 1，相邻 regime 给部分权重），
避免 regime 切换时预测跳变。

## 对比口径

必须同时报告：
- 单模型（全局一套 β）的 IC / 价差
- 分 regime 模型的 IC / 价差
- 每个 regime 的样本量（样本太少时单独训练 = 过拟合）

**如果分 regime 没有显著更好，就说明 regime 划分没带来信息** —— 要如实报告。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class EnsembleConfig:
    ridge: float = 1.0
    min_samples: int = 300          # 每个 regime 最少样本
    l2: float = 1e-3
    warmup: int = 120
    blend_neighbors: bool = True    # 是否对相邻 regime 做软混合


def _xs_demean(x: np.ndarray) -> np.ndarray:
    return x - np.nanmean(x, axis=1, keepdims=True)


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    K = X.shape[1]
    A = X.T @ X + l2 * np.eye(K)
    try:
        return np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        return np.zeros(K)


def fit_regime_models(X: np.ndarray, RET: np.ndarray, regime: np.ndarray,
                      h: int = 5, cfg: EnsembleConfig = EnsembleConfig()) -> Dict:
    """
    X: (T,N,K)；RET: (T,N)；regime: (T,)。
    返回 {regime: beta} 与各 regime 样本量。
    """
    T, N, K = X.shape
    models, counts = {}, {}
    for r in sorted(set(int(x) for x in np.unique(regime) if np.isfinite(x))):
        xs, ys = [], []
        for t in range(T):
            s = t - 1 - h
            if s < 0 or regime[s] != r:
                continue
            x = X[s]; y = RET[s]
            m = np.isfinite(y) & np.isfinite(x).all(axis=1)
            if m.sum() >= 10:
                xs.append(_xs_demean(x[m].T).T)   # 横截面去均值
                ys.append(y[m] - np.nanmean(y[m]))
        if not xs:
            models[r] = None; counts[r] = 0
            continue
        Xc = np.vstack(xs); yc = np.concatenate(ys)
        counts[r] = int(len(yc))
        models[r] = fit_ridge(Xc, yc, cfg.l2) if len(yc) >= cfg.min_samples else None
    return {"models": models, "counts": counts}


def predict(X: np.ndarray, regime: np.ndarray, ens: Dict,
            cfg: EnsembleConfig = EnsembleConfig()) -> np.ndarray:
    """逐日用对应 regime 的模型预测；缺失时回退到所有可用模型的平均。"""
    T, N, K = X.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    models = ens["models"]
    valid = [b for b in models.values() if b is not None]
    if not valid:
        return out
    fallback = np.mean(valid, axis=0)
    for t in range(T):
        r = int(regime[t]) if np.isfinite(regime[t]) else -1
        b = models.get(r)
        if b is None:
            b = fallback
        row = X[t]
        ok = np.isfinite(row).all(axis=1)
        out[t] = np.where(ok, row @ b, np.nan)
    return out


def _eval(comp: np.ndarray, RET: np.ndarray, h: int) -> Dict:
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
    return {"ic": st["mean"], "t_ic": st["t"], "spread": s2["mean"], "t_spread": s2["t"]}


def compare_single_vs_regime(X: np.ndarray, RET: np.ndarray, regime: np.ndarray,
                             h: int = 5, cfg: EnsembleConfig = EnsembleConfig(),
                             refit_every: int = 20) -> Dict:
    """
    单模型 vs 分 regime 模型 —— **严格因果**：每 refit_every 天用已到期样本重训。
    （早期版本用全样本拟合，会给出 IC +0.06 的假象，那是未来函数。）
    """
    T, N, K = X.shape
    pred_single = np.full((T, N), np.nan, dtype=np.float32)
    pred_reg = np.full((T, N), np.nan, dtype=np.float32)
    b_single = None
    models: Dict[int, Optional[np.ndarray]] = {}
    counts: Dict[int, int] = {}

    for t in range(cfg.warmup, T):
        if (t - cfg.warmup) % refit_every == 0:
            t_end = t - 1 - h
            if t_end < 200:
                continue
            # 单模型：用 [0, t_end) 全部样本
            xs, ys = [], []
            for s in range(0, t_end):
                x = X[s]; y = RET[s]
                m = np.isfinite(y) & np.isfinite(x).all(axis=1)
                if m.sum() >= 10:
                    xm = _xs_demean(x[m].T).T
                    xs.append(xm); ys.append(y[m] - np.nanmean(y[m]))
            if not xs:
                continue
            Xc = np.vstack(xs); yc = np.concatenate(ys)
            b_single = fit_ridge(Xc, yc, cfg.l2)
            # 分 regime
            models = {}; counts = {}
            for r in sorted(set(int(v) for v in np.unique(regime[:t_end]) if np.isfinite(v))):
                xr, yr = [], []
                for s in range(0, t_end):
                    if regime[s] != r:
                        continue
                    x = X[s]; y = RET[s]
                    m = np.isfinite(y) & np.isfinite(x).all(axis=1)
                    if m.sum() >= 10:
                        xr.append(_xs_demean(x[m].T).T)
                        yr.append(y[m] - np.nanmean(y[m]))
                if xr:
                    Xr = np.vstack(xr); yr2 = np.concatenate(yr)
                    counts[r] = int(len(yr2))
                    models[r] = fit_ridge(Xr, yr2, cfg.l2) if len(yr2) >= cfg.min_samples else None
                else:
                    models[r] = None; counts[r] = 0

        if b_single is None:
            continue
        row = X[t]; ok = np.isfinite(row).all(axis=1)
        pred_single[t] = np.where(ok, row @ b_single, np.nan)
        r = int(regime[t]) if np.isfinite(regime[t]) else -1
        b = models.get(r)
        if b is None:
            valid = [v for v in models.values() if v is not None]
            b = np.mean(valid, axis=0) if valid else b_single
        pred_reg[t] = np.where(ok, row @ b, np.nan)

    ev_s = _eval(pred_single, RET, h)
    ev_r = _eval(pred_reg, RET, h)
    return {"ok": True, "single": ev_s, "regime": ev_r,
            "regime_counts": counts,
            "n_models_fitted": sum(1 for v in models.values() if v is not None),
            "verdict": ("分 regime 模型更好（t_ic %.2f > %.2f）" % (ev_r["t_ic"], ev_s["t_ic"])
                        if ev_r["t_ic"] > ev_s["t_ic"] else
                        "分 regime 未带来改善 → regime 划分没提供额外信息")}
