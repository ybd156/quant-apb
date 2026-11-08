"""
扩展技术指标 —— 在 Alpha158 之外再加一批经典指标，供 GBDT 自己决定要不要用。

## 为什么值得加

之前测过「22 个经典指标单独当信号」，结果 0/22 过三道门。
但那是**单独使用**——每个指标自己发买卖信号。

放进 GBDT 里是另一回事：树模型可以把「MACD 金叉 × 低波动 × 行业」这类
**交互**挖出来，而单独用 MACD 完全做不到。所以值得再试一次，
但结论必须实测，不能假设加了就有用。

## 与 Alpha158 的区分

Alpha158 已经覆盖：ROC/MA/STD/BETA/RSQR/RESI/MAX/MIN/QTL/RANK/RSV/
IMAX/IMIN/IMXD/CORR/CORD/CNTP/CNTN/CNTD/SUMP/SUMN/SUMD/VMA/VSTD/WVMA/VSUMP…

本模块**只加 Alpha158 没有的**，避免重复：
MACD 族、KDJ 族、BOLL 族、ATR/NATR、DMI/ADX、TRIX、VR、AR/BR、EMV、
CHO、DMA、MTM 加速度、RSI 背离、量价背离、波动率锥位置。

## 因果性

全部用 rolling / ewm，第 t 行只用 ≤t 数据。
`ewm` 在 pandas 里是因果的（只往回看），可以用。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """
    真正的「往后挪 k 行」，前面填 NaN。
    不能用 np.roll——它会把末尾的值环绕到开头，
    既制造未来函数式的伪值，又破坏截断不变性。
    """
    if k <= 0:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder 平滑（RSI/ATR/DMI 用）。"""
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _roll(x: np.ndarray, w: int, how: str = "mean") -> np.ndarray:
    s = pd.Series(x).rolling(w, min_periods=max(2, w // 2))
    return getattr(s, how)().to_numpy()


def extra_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入单只股票日线（date/open/high/low/close/volume[/amount]），
    输出 index=date、columns=扩展指标（全部相对量，无量纲）。
    """
    d = df.reset_index(drop=True)
    o = d["open"].to_numpy(dtype=float)
    h = d["high"].to_numpy(dtype=float)
    lo = d["low"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    v = d["volume"].to_numpy(dtype=float)
    n = len(c)
    out = pd.DataFrame(index=pd.DatetimeIndex(d["date"]))
    eps = 1e-12

    ret = np.empty(n); ret[0] = np.nan
    ret[1:] = c[1:] / c[:-1] - 1.0

    # ---------------- MACD 族（归一化到价格，消除量纲）----------------
    dif = _ema(c, 12) - _ema(c, 26)
    dea = _ema(dif, 9)
    out["x_macd_dif"] = dif / np.maximum(c, eps)
    out["x_macd_dea"] = dea / np.maximum(c, eps)
    out["x_macd_hist"] = (dif - dea) * 2.0 / np.maximum(c, eps)
    out["x_macd_hist_slope"] = pd.Series(out["x_macd_hist"]).diff(3).to_numpy()

    # ---------------- KDJ ----------------
    ll = _roll(lo, 9, "min"); hh = _roll(h, 9, "max")
    rsv = (c - ll) / np.maximum(hh - ll, eps)
    k = _rma(np.nan_to_num(rsv, nan=0.5), 3)
    dd = _rma(k, 3)
    out["x_kdj_k"] = k
    out["x_kdj_d"] = dd
    out["x_kdj_j"] = 3.0 * k - 2.0 * dd
    out["x_kdj_kd"] = k - dd

    # ---------------- BOLL ----------------
    ma20 = _roll(c, 20, "mean")
    sd20 = _roll(c, 20, "std")
    out["x_boll_pos"] = (c - ma20) / np.maximum(2.0 * sd20, eps)   # 在带内的位置
    out["x_boll_width"] = 4.0 * sd20 / np.maximum(ma20, eps)       # 带宽（波动率锥）
    out["x_boll_squeeze"] = out["x_boll_width"] / \
        np.maximum(_roll(out["x_boll_width"].to_numpy(), 120, "mean"), eps)

    # ---------------- ATR / NATR ----------------
    tr = np.maximum.reduce([h - lo, np.abs(h - _shift(c, 1)), np.abs(lo - _shift(c, 1))])
    tr[0] = h[0] - lo[0]
    atr14 = _rma(tr, 14)
    out["x_atr14"] = atr14 / np.maximum(c, eps)
    out["x_atr_ratio"] = atr14 / np.maximum(_roll(tr, 60, "mean"), eps)

    # ---------------- DMI / ADX ----------------
    up = np.diff(h, prepend=h[0]); dn = -np.diff(lo, prepend=lo[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100.0 * _rma(plus_dm, 14) / np.maximum(atr14, eps)
    mdi = 100.0 * _rma(minus_dm, 14) / np.maximum(atr14, eps)
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, eps)
    out["x_dmi_pdi"] = pdi / 100.0
    out["x_dmi_mdi"] = mdi / 100.0
    out["x_dmi_diff"] = (pdi - mdi) / 100.0
    out["x_adx14"] = _rma(dx, 14) / 100.0

    # ---------------- TRIX ----------------
    e1 = _ema(c, 12); e2 = _ema(e1, 12); e3 = _ema(e2, 12)
    out["x_trix12"] = (e3 - _shift(e3, 1)) / np.maximum(np.abs(_shift(e3, 1)), eps)

    # ---------------- VR / AR / BR ----------------
    up_v = np.where(ret > 0, v, 0.0); dn_v = np.where(ret < 0, v, 0.0)
    out["x_vr26"] = _roll(up_v, 26, "sum") / np.maximum(_roll(dn_v, 26, "sum"), eps)
    out["x_ar26"] = _roll(h - o, 26, "sum") / np.maximum(_roll(o - lo, 26, "sum"), eps)
    out["x_br26"] = _roll(np.maximum(h - _shift(c, 1), 0.0), 26, "sum") / \
        np.maximum(_roll(np.maximum(_shift(c, 1) - lo, 0.0), 26, "sum"), eps)

    # ---------------- EMV（简易波动）----------------
    mid = (h + lo) / 2.0
    emv = (mid - _shift(mid, 1)) * (h - lo) / np.maximum(v, eps)
    out["x_emv14"] = _roll(emv, 14, "mean") * 1e6

    # ---------------- CHO（佳庆指标）----------------
    # 用滚动 250 日累加代替从上市起 cumsum：cumsum 虽然因果，
    # 但截断后起点不同会导致数值不一致（破坏截断不变性）
    _adflow = ((2 * c - h - lo) / np.maximum(h - lo, eps)) * v
    ad = pd.Series(_adflow).rolling(250, min_periods=50).sum().to_numpy()
    out["x_cho"] = (_ema(ad, 3) - _ema(ad, 10)) / np.maximum(np.abs(_ema(ad, 10)), eps)

    # ---------------- DMA（差离）----------------
    out["x_dma"] = (_roll(c, 10, "mean") - _roll(c, 50, "mean")) / np.maximum(c, eps)

    # ---------------- MTM 加速度 ----------------
    mom = c / np.maximum(_shift(c, 12), eps) - 1.0
    out["x_mtm12"] = mom
    out["x_mtm_accel"] = pd.Series(mom).diff(5).to_numpy()

    # ---------------- RSI 多周期 + 背离 ----------------
    gain = np.where(ret > 0, ret, 0.0); loss = np.where(ret < 0, -ret, 0.0)
    rsi6 = 100 - 100 / (1 + _rma(gain, 6) / np.maximum(_rma(loss, 6), eps))
    rsi24 = 100 - 100 / (1 + _rma(gain, 24) / np.maximum(_rma(loss, 24), eps))
    out["x_rsi6"] = rsi6 / 100.0
    out["x_rsi24"] = rsi24 / 100.0
    out["x_rsi_spread"] = (rsi6 - rsi24) / 100.0
    # 价格新高但 RSI 未新高 = 顶背离
    p_hi = c >= _roll(c, 20, "max") - 1e-9
    r_hi = rsi6 >= _roll(rsi6, 20, "max") - 1e-9
    out["x_rsi_divergence"] = (p_hi & ~r_hi).astype(float)

    # ---------------- 量价背离 ----------------
    p_up = ret > 0
    v_hi = v >= _roll(v, 20, "max") - 1e-9
    out["x_pv_bearish"] = (p_up & ~v_hi).astype(float)      # 价涨量不涨
    out["x_pv_bullish"] = ((~p_up) & v_hi).astype(float)    # 价跌放量

    # ---------------- 波动率锥位置 ----------------
    vol20 = _roll(ret, 20, "std")
    out["x_vol_cone"] = vol20 / np.maximum(_roll(vol20, 250, "mean"), eps)

    # ---------------- 收盘位置 / 跳空 ----------------
    out["x_close_loc"] = (c - lo) / np.maximum(h - lo, eps)
    out["x_gap"] = (o - _shift(c, 1)) / np.maximum(_shift(c, 1), eps)

    # 预热期清零为 NaN（避免伪造信号）
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def extra_names() -> List[str]:
    return list(extra_indicators(_dummy()).columns)


def _dummy() -> pd.DataFrame:
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"date": idx, "open": c * 1.001, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": rng.integers(1e5, 1e6, n)})


if __name__ == "__main__":
    from qcore.data import load_kline
    import time

    names = extra_names()
    print(f"扩展指标 {len(names)} 个")
    df = load_kline("600519")
    t0 = time.time()
    f = extra_indicators(df)
    dt = time.time() - t0
    print(f"600519 {len(df)} 行，耗时 {dt*1000:.0f} ms")
    print(f"输出 {f.shape[1]} 列")

    # 截断不变性
    f_half = extra_indicators(df.iloc[: len(df) // 2])
    a = f.iloc[: len(df) // 2].to_numpy()
    b = f_half.to_numpy()
    same = np.array_equal(np.isnan(a), np.isnan(b)) and \
        np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-12)
    print(f"截断不变性: {'PASS' if same else 'FAIL'}"
          f"  max|diff|={np.nanmax(np.abs(a-b)):.3e}")

    # 统计
    st = pd.DataFrame({"非NaN": f.notna().sum(), "均值": f.mean().round(4),
                       "标准差": f.std().round(4)})
    print(st.head(12).to_string())
    _arr = np.array(f.corr().abs().to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(_arr, 0.0)
    corr = pd.DataFrame(_arr, index=f.columns, columns=f.columns)
    print(f"因子间最大|相关| = {corr.max().max():.3f}")
