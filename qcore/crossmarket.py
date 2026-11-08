"""
跨市场传导 —— A 股不是孤立的，隔夜美股、同日港股、美元都在推它。

## 为什么这可能是真信号

A 股 9:30 开盘时，美股刚收盘（北京时间凌晨 4:00/5:00），
**这个信息在 A 股开盘前就完全公开了**，但市场对它的定价常常滞后。

这不是"预测未来"，而是"用已知的、但没被充分定价的信息"。

## 关键设计：必须是横截面

纯市场级因子（如"隔夜标普涨 1%"）对所有股票一样，**横截面 IC 恒为 0**。
必须和个股特征交互，才有区分度：

    score_i = overnight_us_ret × beta_i × vol_scale_i

含义：隔夜美股大涨时，**高 beta 股票**应该涨更多。如果市场没充分定价，
高 beta 组的表现会优于低 beta 组 → 产生横截面 IC。

同理：
    港股同日（上午）→ 用 HK 上午收益 × 个股港股关联度
    美元指数 → 出口行业 vs 内需行业

## 因果性（严格）

- 美股隔夜：取 A 股 t 日开盘前**最近一个已收盘**的美股交易日收益
- 港股：港股 9:30-12:00 与 A 股 9:30-11:30 重叠，但港股 16:00 才收盘，
  **用港股全天收益就是未来函数**。这里只用 t−1 日港股收益（完全已知）
- 美元指数：同理用 t−1 日

## 数据来源

SPY / HSI / UUP 等 ETF 或指数，走 qcore.us_data（已验证可取到 1990 年至今）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_series(ticker: str, market: str = "US") -> Optional[pd.Series]:
    """取一条日线收盘序列（index=date）。失败返回 None。"""
    try:
        from . import us_data as U
        df = U.load_us_kline(ticker) if market == "US" else U.load_hk_kline(ticker)
        if df is None or len(df) < 200:
            return None
        s = pd.Series(df["close"].to_numpy(dtype=float),
                      index=pd.DatetimeIndex(df["date"]))
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception:
        return None


def overnight_us(dates: pd.DatetimeIndex,
                 ticker: str = "SPY") -> Optional[pd.Series]:
    """
    对每个 A 股交易日 t，取**开盘前最近一个已收盘**的美股收益。
    用 reindex(method='ffill') 保证只取 ≤ t−1 的美股数据，绝不用 t 日美股。
    """
    s = _load_series(ticker)
    if s is None:
        return None
    us_ret = s.pct_change()
    # A 股 t 日可用的是「t 之前最后一个美股收盘」——先把美股日期整体前移一天再 ffill
    shifted = us_ret.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    out = shifted.reindex(pd.DatetimeIndex(dates), method="ffill")
    out.name = f"overnight_{ticker}"
    return out


def prev_day_return(dates: pd.DatetimeIndex, ticker: str,
                    market: str = "HK") -> Optional[pd.Series]:
    """t−1 日的港股/其他市场收益（完全已知，无未来函数）。"""
    s = _load_series(ticker, market)
    if s is None:
        return None
    r = s.pct_change()
    r.index = r.index + pd.Timedelta(days=1)
    out = r.reindex(pd.DatetimeIndex(dates), method="ffill")
    out.name = f"prev_{ticker}"
    return out


def beta_to(market_ret: pd.Series, stock_ret: pd.DataFrame,
            window: int = 120) -> pd.DataFrame:
    """
    滚动 beta：每只股票对市场收益的敏感度。严格用 ≤t 的数据（rolling，不 center）。
    """
    cov = stock_ret.rolling(window, min_periods=window // 2).cov(market_ret)
    var = market_ret.rolling(window, min_periods=window // 2).var()
    b = cov.div(var, axis=0)
    return b


def spillover_panel(p3d, us_ticker: str = "SPY",
                    vol_scale: bool = True) -> Dict:
    """
    构造跨市场传导因子面板 (T,N)。

    返回 {"overnight": (T,N), "score": (T,N), "diag": {...}}
      overnight : 每个 A 股交易日的隔夜美股收益（市场级，T 维）
      score     : 横截面得分 = overnight × beta × vol_scale
    """
    dates = pd.DatetimeIndex(p3d.dates)
    T, N = p3d.CLOSE.shape

    ov = overnight_us(dates, us_ticker)
    if ov is None:
        return {"ok": False, "reason": f"取不到 {us_ticker}"}
    ov_arr = ov.to_numpy(dtype=float)

    # 个股日收益（用 CLOSE）
    close = pd.DataFrame(p3d.CLOSE, index=dates)
    sret = close.pct_change()

    # 市场收益 = 池内等权（避免依赖单一指数）
    mkt = sret.mean(axis=1)
    beta = beta_to(mkt, sret, window=120)
    beta_arr = beta.to_numpy(dtype=float)
    # 缺失 beta 用 1（中性），并记录比例
    miss = float(np.isnan(beta_arr).mean())
    beta_arr = np.where(np.isfinite(beta_arr), beta_arr, 1.0)
    beta_arr = np.clip(beta_arr, -0.5, 3.0)

    score = ov_arr[:, None] * beta_arr
    if vol_scale:
        sig = np.asarray(p3d.SIGMA, dtype=float)
        med = np.nanmedian(sig, axis=1, keepdims=True)
        scale = np.where((med > 0) & np.isfinite(med), sig / med, 1.0)
        scale = np.clip(np.nan_to_num(scale, nan=1.0), 0.3, 3.0)
        score = score * scale

    # 只保留可交易
    score = np.where(p3d.VALID, score, np.nan)
    return {"ok": True, "overnight": ov_arr, "score": score,
            "diag": {"ticker": us_ticker, "beta_missing_frac": round(miss, 4),
                     "mean_abs_overnight": float(np.nanmean(np.abs(ov_arr)))}}


def evaluate(p3d, horizon: int = 5, us_ticker: str = "SPY") -> Dict:
    """算这个跨市场因子的 IC / HAC t / 多空价差。"""
    from . import analysis as A, stats as ST

    p = spillover_panel(p3d, us_ticker)
    if not p.get("ok"):
        return p
    score = p["score"]
    ret = p3d.RET[horizon]

    # 注意：daily_ic_series 约定 (N,T)，这里必须转置
    ic = A.daily_ic_series(score.T, ret.T)
    ic = ic[np.isfinite(ic)]
    if ic.size < 60:
        return {"ok": False, "reason": f"IC 样本不足 {ic.size}"}
    st = ST.mean_tstat(ic, lags=max(horizon - 1, 0))

    T = score.shape[0]
    sp = []
    for t in range(T):
        x, y = score[t], ret[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < max(10, int(0.2 * score.shape[1])):
            continue
        lo, hi = np.percentile(x[m], 20), np.percentile(x[m], 80)
        sp.append(np.nanmean(y[m][x[m] >= hi]) - np.nanmean(y[m][x[m] <= lo]))
    sp = np.asarray(sp)
    sps = ST.mean_tstat(sp, lags=max(horizon - 1, 0)) if sp.size > 60 else \
        {"mean": np.nan, "t": np.nan}

    return {"ok": True, "ic": float(st["mean"]), "t_ic": float(st["t"]),
            "spread": float(sps["mean"]), "t_spread": float(sps["t"]),
            "n_days": int(ic.size), "diag": p["diag"]}


if __name__ == "__main__":
    from qcore.data import get_pool
    from qcore.dataset import build_dataset

    pool = get_pool(top_n=120)
    p3d = build_dataset(pool["code"].tolist(), start="2016-01-01",
                        horizon=5, verbose=False)
    for tk in ("SPY", "QQQ"):
        r = evaluate(p3d, 5, tk)
        if r.get("ok"):
            print(f"{tk}: IC={r['ic']:+.4f} t={r['t_ic']:+.2f} "
                  f"价差={r['spread']*100:+.4f}% t={r['t_spread']:+.2f} "
                  f"(n={r['n_days']})")
        else:
            print(f"{tk}: {r.get('reason')}")
