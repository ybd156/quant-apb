"""
形势研判 —— 回答两个问题：
  ① 当时形势：站在 as_of 这一天，只看到当天为止，市场/个股处于什么状态？
  ② 历史形势：当前这个状态，在历史上出现过多少次？出现之后通常怎么走？

第 ② 条用的是"历史类比"（analog forecasting）：把每天的状态压成一个特征向量，
找历史上最像的 K 天，看它们之后 h 天的真实分布。这比"模型说会涨"更容易让人信服，
因为它给出的是**样本**而不是**断言**，而且能顺带给出可信度（样本够不够、分布散不散）。

全部因果：特征用滚动统计标准化，类比样本只取"结果已知"的历史日（t' ≤ as_of-1-h）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ 特征
def _features(df: pd.DataFrame) -> pd.DataFrame:
    """把日线压成状态特征（全部因果，rolling/expanding）。"""
    c = df["close"].astype(float)
    r = c.pct_change()
    out = pd.DataFrame(index=df.index)
    out["ret20"] = c / c.shift(20) - 1.0
    out["ret60"] = c / c.shift(60) - 1.0
    out["vol20"] = r.rolling(20, min_periods=20).std()
    out["vol_ratio"] = out["vol20"] / r.rolling(120, min_periods=40).std()
    ma60 = c.rolling(60, min_periods=60).mean()
    out["dist_ma60"] = c / ma60 - 1.0
    out["dd"] = c / c.rolling(250, min_periods=60).max() - 1.0
    # 成交量能：量比
    v = df["volume"].astype(float)
    out["volu"] = v / v.rolling(20, min_periods=20).mean()
    # 波动分位（自身历史）
    out["vol_pct"] = out["vol20"].rolling(250, min_periods=60).rank(pct=True)
    return out


def _zscore_causal(feat: pd.DataFrame, win: int = 250) -> pd.DataFrame:
    mu = feat.rolling(win, min_periods=60).mean()
    sd = feat.rolling(win, min_periods=60).std()
    return ((feat - mu) / sd.replace(0, np.nan)).clip(-4, 4)


# ------------------------------------------------------------------ 当时形势
def now_context(df: pd.DataFrame, as_of=None, name: str = "") -> Dict:
    """站在 as_of（默认最后一根）看当下状态。"""
    d = df[df["date"] <= pd.Timestamp(as_of)] if as_of is not None else df
    if len(d) < 80:
        return {"ok": False, "reason": "insufficient_history", "n": len(d)}
    feat = _features(d)
    z = _zscore_causal(feat)
    i = len(d) - 1
    row = {k: float(feat[k].iloc[i]) for k in feat.columns}
    zrow = {k: float(z[k].iloc[i]) for k in z.columns}

    trend = "up" if row["dist_ma60"] > 0 and row["ret60"] > 0 else (
        "down" if row["dist_ma60"] < 0 and row["ret60"] < 0 else "sideways")
    vol_state = "high" if (row.get("vol_pct") or 0.5) > 0.6 else (
        "low" if (row.get("vol_pct") or 0.5) < 0.4 else "mid")

    return {
        "ok": True, "as_of": str(pd.Timestamp(d["date"].iloc[-1]).date()), "name": name,
        "close": float(d["close"].iloc[-1]),
        "trend": trend, "vol_state": vol_state,
        "ret20": row["ret20"], "ret60": row["ret60"],
        "vol20_ann": float(row["vol20"] * np.sqrt(252)) if np.isfinite(row["vol20"]) else None,
        "drawdown_from_250d_high": row["dd"],
        "dist_ma60": row["dist_ma60"],
        "vol_percentile": row.get("vol_pct"),
        "volume_ratio": row["volu"],
        "z": zrow,
        "summary": (f"趋势{ {'up':'向上','down':'向下','sideways':'震荡'}[trend] }、"
                    f"波动{ {'high':'偏高','low':'偏低','mid':'中性'}[vol_state] }；"
                    f"距 250 日高点 {row['dd']:.1%}，20 日 {row['ret20']:+.1%}。"),
    }


# ------------------------------------------------------------------ 历史形势
def analog_forecast(df: pd.DataFrame, as_of=None, h: int = 5, k: int = 30,
                    min_gap: int = 20, feature_cols: Optional[List[str]] = None) -> Dict:
    """
    历史类比：找历史上状态最像的 k 天，统计它们之后 h 天的收益分布。

    min_gap：类比样本与当前时点至少隔开 min_gap 天，避免"几乎是同一天"的重复样本。
    """
    d = df[df["date"] <= pd.Timestamp(as_of)] if as_of is not None else df
    d = d.reset_index(drop=True)
    if len(d) < 300:
        return {"ok": False, "reason": "insufficient_history", "n": len(d)}

    feat = _features(d)
    cols = feature_cols or ["ret20", "vol20", "dist_ma60", "dd", "volu", "vol_ratio"]
    cols = [c for c in cols if c in feat.columns]
    z = _zscore_causal(feat)[cols].to_numpy(dtype=np.float64)

    T = len(d)
    fwd = np.full(T, np.nan)
    o = d["close"].to_numpy(dtype=float)
    fwd[:T - h] = o[h:] / o[:T - h] - 1.0

    cur = z[T - 1]
    cand = np.arange(0, T - 1 - h - min_gap)          # 结果已知且不与当下重叠
    if len(cand) < 10:
        return {"ok": False, "reason": "insufficient_analogs", "n": int(len(cand))}
    sub = z[cand]
    ok = np.isfinite(sub).all(axis=1) & np.isfinite(cur).all()
    cand, sub = cand[ok], sub[ok]
    if len(cand) < 10:
        return {"ok": False, "reason": "insufficient_analogs", "n": int(len(cand))}

    dist = np.sqrt(((sub - cur) ** 2).sum(axis=1))
    k = int(min(k, len(cand)))
    pick = np.argsort(dist)[:k]
    idx = cand[pick]
    f = fwd[idx]
    f = f[np.isfinite(f)]
    if len(f) < 5:
        return {"ok": False, "reason": "no_forward_data", "n": int(len(f))}

    q = np.quantile(f, [0.1, 0.25, 0.5, 0.75, 0.9])
    hit = float((f > 0).mean())
    se = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 2 else float("nan")

    # 可信度：样本量 + 分布离散度 + 类比距离（越像越可信）
    n_score = min(1.0, len(f) / 30.0)
    disp_score = float(np.clip(1.0 - f.std() / 0.15, 0.0, 1.0))
    dist_score = float(np.clip(1.0 - dist[pick].mean() / 3.0, 0.0, 1.0))
    cred = 0.4 * n_score + 0.35 * disp_score + 0.25 * dist_score

    return {
        "ok": True, "as_of": str(pd.Timestamp(d["date"].iloc[-1]).date()),
        "h": h, "n_analogs": int(len(f)),
        "mean": float(f.mean()), "median": float(np.median(f)), "std": float(f.std()),
        "se": se, "hit_rate": hit,
        "q10": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
        "q75": float(q[3]), "q90": float(q[4]),
        "avg_distance": float(dist[pick].mean()),
        "credibility": float(cred),
        "analog_dates": [str(pd.Timestamp(d["date"].iloc[i]).date()) for i in idx[:10]],
        "summary": (f"历史上最像的 {len(f)} 个时点，之后 {h} 日：中位数 {np.median(f):+.1%}，"
                    f"上涨概率 {hit:.0%}，区间 [{q[0]:+.1%}, {q[4]:+.1%}]，类比可信度 {cred:.0%}。"),
    }


# ------------------------------------------------------------------ 组合视角
def stock_context(df: pd.DataFrame, as_of=None, h: int = 5, k: int = 30,
                  name: str = "") -> Dict:
    """个股完整形势：当时 + 历史 + 类比，一次给全。"""
    return {
        "now": now_context(df, as_of, name),
        "analog": analog_forecast(df, as_of, h=h, k=k),
        "as_of": str(pd.Timestamp(df["date"].iloc[-1]).date()),
    }
