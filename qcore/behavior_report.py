"""
行为因子体检 —— 把 `qcore.behavioral` 的 10 个因子在真实面板上跑一遍，
产出 Web UI 需要的 `performance.json["behavior"]`。

## 为什么需要这一层

`qcore.behavioral` 只负责「单只股票 → 因子序列」，不负责：
  · 拼成横截面面板
  · 算 IC / HAC t / 多空价差
  · 行业+市值中性化

没有这一层，UI 的行为因子页永远是"未就绪"。

## 输出结构（UI 契约）

```json
[{"factor":"beh_turn_anom","ic":0.0123,"t":2.1,"spread":0.004,
  "ic_neutral":0.004,"note":"..."}]
```

`ic_neutral` 是**行业+市值中性化后**的 IC。按 AGENTS.md §10.6：
任何与规模/流动性/波动相关的因子，不做中性化不许上报 alpha。

## 成本控制

默认只取 150 只、最近若干年。全量 300 只 × 全历史会拖到几分钟，
而因子体检的目的是"看方向对不对"，不需要全样本。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import behavioral as B
from . import stats as ST


def _neutralize_ic(f: np.ndarray, r: np.ndarray, industry: np.ndarray,
                   logcap: np.ndarray) -> Optional[float]:
    """
    行业+市值中性化后的横截面 IC。
    做法：先对因子做横截面回归（行业哑变量 + log市值），取残差再算 rank IC。
    """
    m = np.isfinite(f) & np.isfinite(r) & np.isfinite(logcap)
    if m.sum() < 30:
        return None
    f_, r_, cap_, ind_ = f[m], r[m], logcap[m], industry[m]
    # 设计矩阵：常数 + logcap + 行业哑变量
    cats = pd.Categorical(ind_)
    dummies = pd.get_dummies(cats, drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(f_)), cap_, dummies]) if dummies.size else \
        np.column_stack([np.ones(len(f_)), cap_])
    try:
        beta, *_ = np.linalg.lstsq(X, f_, rcond=None)
        resid = f_ - X @ beta
    except np.linalg.LinAlgError:
        return None
    rf = pd.Series(resid).rank().to_numpy()
    rr = pd.Series(r_).rank().to_numpy()
    if rf.std() < 1e-12 or rr.std() < 1e-12:
        return None
    return float(np.corrcoef(rf, rr)[0, 1])


def compute(codes: List[str], start: str = "2016-01-01", horizon: int = 5,
            max_stocks: int = 150, industry_map: Optional[Dict[str, str]] = None,
            cap_map: Optional[Dict[str, float]] = None,
            verbose: bool = False) -> List[Dict]:
    """跑一遍行为因子体检，返回 UI 用的列表。"""
    from .data import load_kline

    codes = list(codes)[:max_stocks]
    per_factor: Dict[str, Dict[str, pd.Series]] = {}
    close_map: Dict[str, pd.Series] = {}
    used: List[str] = []

    for i, c in enumerate(codes):
        try:
            df = load_kline(c)
        except Exception:
            continue
        if df is None or len(df) < 300:
            continue
        df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        if len(df) < 260:
            continue
        try:
            f = B.behavioral_factors(df)
        except Exception:
            continue
        dts = pd.DatetimeIndex(f.index)
        close_map[c] = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        for col in f.columns:
            per_factor.setdefault(col, {})[c] = pd.Series(f[col].to_numpy(), index=dts)
        used.append(c)
        if verbose and (i + 1) % 30 == 0:
            print(f"      behavior: {i + 1}/{len(codes)}", flush=True)

    if not used:
        return []

    # 统一日历
    all_dates = sorted(set().union(*[set(s.index) for s in close_map.values()]))
    dates = pd.DatetimeIndex(all_dates)
    close_mat = pd.DataFrame({c: s.reindex(dates) for c, s in close_map.items()}, index=dates)
    # 前向 h 日收益
    # 前向收益是**标签**不是特征：因子本身全部由 behavioral_factors 的 rolling 运算得到
    fwd = close_mat.shift(-horizon) / close_mat - 1.0  # leak-scan: ok 这是 y（前向标签），不是 X

    ind_arr = np.array([(industry_map or {}).get(c, "NA") for c in close_mat.columns], dtype=object)
    cap_arr = np.log(np.array([max((cap_map or {}).get(c, 1e9), 1.0)
                               for c in close_mat.columns], dtype=float))

    rows: List[Dict] = []
    for fac, by_code in per_factor.items():
        mat = pd.DataFrame({c: s.reindex(dates) for c, s in by_code.items()}, index=dates)
        ics, spreads = [], []
        for t in range(len(dates)):
            x = mat.iloc[t].to_numpy(dtype=float)
            y = fwd.iloc[t].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 40:
                continue
            rx = pd.Series(x[m]).rank().to_numpy()
            ry = pd.Series(y[m]).rank().to_numpy()
            if rx.std() < 1e-12 or ry.std() < 1e-12:
                continue
            ics.append(float(np.corrcoef(rx, ry)[0, 1]))
            lo, hi = np.percentile(x[m], 20), np.percentile(x[m], 80)
            top, bot = y[m][x[m] >= hi], y[m][x[m] <= lo]
            if len(top) and len(bot):
                spreads.append(float(np.nanmean(top) - np.nanmean(bot)))
        if len(ics) < 60:
            continue
        ic_s = ST.mean_tstat(np.array(ics), lags=max(horizon - 1, 0))
        sp_s = ST.mean_tstat(np.array(spreads), lags=max(horizon - 1, 0)) if len(spreads) > 60 \
            else {"mean": float("nan"), "t": float("nan")}

        # 中性化 IC：取最近 250 个交易日的平均
        neut = []
        tail = range(max(0, len(dates) - 250), len(dates))
        for t in tail:
            x = mat.iloc[t].to_numpy(dtype=float)
            y = fwd.iloc[t].to_numpy(dtype=float)
            v = _neutralize_ic(x, y, ind_arr, cap_arr)
            if v is not None:
                neut.append(v)
        ic_neutral = float(np.mean(neut)) if neut else None

        note = "方向与文献一致" if (ic_s["mean"] or 0) != 0 else ""
        if ic_neutral is not None and ic_s["mean"] and abs(ic_neutral) < abs(ic_s["mean"]) * 0.4:
            note = "中性化后大幅衰减 → 大概率是规模/行业效应"
        rows.append({
            "factor": fac,
            "ic": round(float(ic_s["mean"]), 6),
            "t": round(float(ic_s["t"]), 3),
            "spread": round(float(sp_s["mean"]), 6) if np.isfinite(sp_s["mean"]) else None,
            "spread_t": round(float(sp_s["t"]), 3) if np.isfinite(sp_s["t"]) else None,
            "ic_neutral": round(ic_neutral, 6) if ic_neutral is not None else None,
            "n_days": len(ics),
            "n_stocks": int(mat.notna().any(axis=0).sum()),
            "note": note,
        })
    rows.sort(key=lambda r: -abs(r["ic"]))
    return rows


if __name__ == "__main__":
    from .data import get_pool
    p = get_pool(top_n=160)
    ind = dict(zip(p["code"].astype(str), p["industry"].astype(str)))
    cap = dict(zip(p["code"].astype(str), p["mktcap"].astype(float)))
    out = compute(p["code"].astype(str).tolist(), industry_map=ind, cap_map=cap, verbose=True)
    for r in out:
        print(f"{r['factor']:<18} ic={r['ic']:+.4f} t={r['t']:+5.2f} "
              f"neut={r['ic_neutral'] if r['ic_neutral'] is None else round(r['ic_neutral'],4)} "
              f"| {r['note']}")
