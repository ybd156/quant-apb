"""qcore/signals.py —— 参数化技术信号库（严格因果 / 纯 numpy+pandas / 全向量化）

对应 docs/02_SPEC.md §0（铁律）与 §2（信号契约）。

设计约束
--------
1. 因果性：第 t 行只依赖 <= t 行。所有标准化只用 rolling(w) 口径，
   绝不出现 shift(-n)、rolling(center=True)、全样本 mean/std/quantile/min/max。
   滚动量一律用"前缀累加"实现（cumsum / 分块前后缀极值），因此
   截断序列（df.iloc[:t+1]）与全序列在第 t 行的浮点结果**逐位相同**。
2. 预热期：指标不足处输出 DIR=0 / STR=0，不传播 NaN（收口处统一把非有限值归 0）。
3. 不抛异常：缺列 / 空表 / 全 NaN / 除零一律降级为 0。
4. 价格口径：数据层已统一为**后复权**（fqt=2，乘法复权，价格恒正、口径可比），
   故价格基准 px 直接取 close；仅当某行 close 缺失/非正时才回退到
   成交均价 vwap = amount/(volume*100)，再不行就前向填充（绝不回填未来）。
   日内 OHLC 按 px/close 同比例缩放到同一价格框架，并在 (high-low)/close 过大
   等畸形行上退化为 px（形态/缺口类信号在这些行置 0）。
   后复权价与今日真实价不可比，因此**全库禁止任何绝对价格阈值**
   （close>10 之类），一切阈值都是比值 / z-score / 滚动分位 / ATR 倍数。
   收益率一律由价格序列自算（等价 close.pct_change），不读 pct_chg 字段。
5. 输出 dtype 统一 float32（N100 小主机内存预算）。
6. 增量模式：compute_signals(df, tail=k) 只返回最后 k 行，指标预热仍用全量历史；
   信号求值在带余量的切片上做（余量 > 任何派生滚动窗口），结果与全量逐元素相同。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["SignalSpec", "signal_zoo", "compute_signals", "FAMILIES"]

_EPS = 1e-12
_TAIL_MARGIN = 300          # tail 模式下的求值余量（> 任何派生滚动窗口 120 + shift）
_F32 = np.float32


# ======================================================================
# 0. 数据结构
# ======================================================================
@dataclass(frozen=True)
class SignalSpec:
    """一个"专家"：信号族 + 参数组合。sid 全局唯一且稳定可读。"""
    sid: str
    family: str
    kind: str
    params: dict
    note: str


# ======================================================================
# 1. 基础数值工具（全部 O(n) 向量化，前缀累加型 => 天然因果）
# ======================================================================
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """向后平移 k 行（只看过去），前 k 行 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k <= 0:
        return x.copy()
    if k < x.size:
        out[k:] = x[:-k]
    return out


def _sdiv(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """安全除法：分母过小或结果非有限时返回 fill，永不 inf/NaN 传播。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.true_divide(a, b)
    bad = ~np.isfinite(out)
    bad |= np.abs(b) <= _EPS
    if bad.any():
        out[bad] = fill
    return out


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """滚动求和（前缀累加差分），前 w-1 行 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 0 or n < w:
        return out
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    return _roll_sum(x, w) / float(w)


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    """滚动总体标准差（ddof=0）。"""
    x = np.asarray(x, dtype=np.float64)
    m = _roll_mean(x, w)
    m2 = _roll_mean(x * x, w)
    var = m2 - m * m
    np.maximum(var, 0.0, out=var)          # 防浮点负零
    out = np.sqrt(var)
    out[~np.isfinite(m)] = np.nan
    return out


def _roll_extreme(x: np.ndarray, w: int, is_max: bool = True) -> np.ndarray:
    """滚动极值（分块前缀/后缀极值法，O(n)）。

    窗口 [i-w+1, i] 跨至多两个 w 大小的块：
      同一块内 <=> 窗口正好铺满整块，取块内前缀极值即可；
      跨两块   <=> max(起点所在块的后缀极值, 终点所在块的前缀极值)。
    两种情形都只用到 <= i 的数据。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 0 or n < w:
        return out
    nb = (n + w - 1) // w
    pad = nb * w - n
    fill = -np.inf if is_max else np.inf
    xp = np.concatenate([x, np.full(pad, fill, dtype=np.float64)])
    blocks = xp.reshape(nb, w)
    if is_max:
        pre = np.maximum.accumulate(blocks, axis=1).ravel()[:n]
        suf = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()[:n]
        res = np.maximum(suf[: n - w + 1], pre[w - 1:])
    else:
        pre = np.minimum.accumulate(blocks, axis=1).ravel()[:n]
        suf = np.minimum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()[:n]
        res = np.minimum(suf[: n - w + 1], pre[w - 1:])
    out[w - 1:] = res
    return out


def _roll_quantile(x: np.ndarray, w: int, q: float) -> np.ndarray:
    """滚动分位（滑动窗口 + partition，向量化，无 Python 循环）。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 0 or n < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, w)   # (n-w+1, w) 视图
    k = int(np.clip(round(q * (w - 1)), 0, w - 1))
    part = np.partition(win, k, axis=1)[:, k]
    out[w - 1:] = part
    return out


def _ema(x: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """指数平滑（递归定义，天然因果）；用 pandas 的 C 实现保证速度。"""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    return s.ewm(alpha=alpha, adjust=False, min_periods=max(1, min_periods)).mean().to_numpy()


def _sign(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    out[np.isfinite(x) & (x > 0)] = 1.0
    out[np.isfinite(x) & (x < 0)] = -1.0
    return out


def _emit(d: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """统一收口：DIR 归入 {-1,0,1}，STR 归入 [0,1]，DIR=0 处 STR=0。

    就地改写（impl 返回的都是新数组），避免大数组反复分配 —— N100 上这是热点。
    """
    d = np.asarray(d, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if d.size == 0:
        return d, s
    bad = ~np.isfinite(d)
    if bad.any():
        d[bad] = 0.0
    bad = ~np.isfinite(s)
    if bad.any():
        s[bad] = 0.0
    np.clip(d, -1.0, 1.0, out=d)
    np.rint(d, out=d)                      # 只有 {-1,0,1} 三种取值
    np.clip(s, 0.0, 1.0, out=s)
    # 安全网：方向已成立但强度算出 0（NaN 降级）时给最小可信强度
    s[(d != 0.0) & (s <= 0.0)] = 0.5
    s[d == 0.0] = 0.0
    return d, s


# ======================================================================
# 2. 指标上下文（一次构建，全信号复用）
# ======================================================================
def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """取数值列（缺失列返回全 NaN，绝不抛异常）。"""
    if name not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float64)
    arr = df[name].to_numpy()
    if arr.dtype.kind in "fiub":                 # 已是数值：零拷贝/一次转换
        return arr.astype(np.float64, copy=False)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)


class _Ctx:
    """单只股票的全部基础序列 + 指标缓存。价格基准为成交均价 px。"""

    __slots__ = ("n", "index", "c", "o", "h", "l", "v", "turn", "amt",
                 "ohlc_ok", "bar_ok", "idx_c", "_k")

    def __init__(self, df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None):
        n = len(df)
        self.n = n
        self._k: Dict[Any, np.ndarray] = {}

        if "date" in df.columns:
            dt = np.asarray(df["date"].to_numpy(), dtype="datetime64[ns]")
            idx = pd.DatetimeIndex(dt)
        else:
            idx = pd.DatetimeIndex(df.index)
        self.index = idx

        amt = _col(df, "amount")
        vol = _col(df, "volume")
        close = _col(df, "close")
        o_raw = _col(df, "open")
        h_raw = _col(df, "high")
        l_raw = _col(df, "low")
        self.turn = _col(df, "turnover")
        self.amt = amt
        self.v = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)

        # --- 价格基准：后复权 close 优先（恒正、口径可比）
        #     缺失/非正时才退到成交均价 vwap（原始价，仅作兜底），最后前向填充。
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = amt / (vol * 100.0)
        vwap = np.where(np.isfinite(vwap) & (vwap > 0), vwap, np.nan)
        good_close = np.isfinite(close) & (close > 0)
        px = np.where(good_close, close, vwap)
        px = pd.Series(px).ffill().to_numpy(dtype=np.float64)   # 只向前填充 => 因果
        self.c = px
        self.bar_ok = np.isfinite(px)

        # --- 日内 OHLC：按 px/close 同比例缩放到同一价格框架
        finite_ohlc = (np.isfinite(o_raw) & np.isfinite(h_raw) & np.isfinite(l_raw)
                       & np.isfinite(close) & (close > 0) & (o_raw > 0))
        sane = finite_ohlc & (h_raw >= l_raw) & ((h_raw - l_raw) <= 0.5 * np.abs(close))
        s = np.where(sane, _sdiv(px, close, 1.0), 1.0)
        self.o = np.where(sane, o_raw * s, px)
        self.h = np.where(sane, h_raw * s, px)
        self.l = np.where(sane, l_raw * s, px)
        self.ohlc_ok = sane

        # --- 基准指数（相对强弱族用）：按日期对齐，只取 <=t 的值
        self.idx_c: Optional[np.ndarray] = None
        try:
            if index_df is not None:
                d_int, c_val = _index_arrays(index_df)
            else:
                d_int, c_val = _load_index_arrays()
            self.idx_c = _align_ffill(d_int, c_val, idx.asi8)
        except Exception:      # noqa: BLE001  数据异常 -> 该族降级为 0
            self.idx_c = None

    # ---------------- 价格类指标（带缓存） ----------------
    def sma(self, w: int) -> np.ndarray:
        key = ("sma", w)
        v = self._k.get(key)
        if v is None:
            v = _roll_mean(self.c, w)
            self._k[key] = v
        return v

    def rstd(self, w: int) -> np.ndarray:
        key = ("std", w)
        v = self._k.get(key)
        if v is None:
            v = _roll_std(self.c, w)
            self._k[key] = v
        return v

    def ema(self, w: int) -> np.ndarray:
        key = ("ema", w)
        v = self._k.get(key)
        if v is None:
            v = _ema(self.c, 2.0 / (w + 1.0), w)
            self._k[key] = v
        return v

    def rsi(self, w: int) -> np.ndarray:
        key = ("rsi", w)
        v = self._k.get(key)
        if v is None:
            d = np.diff(self.c, prepend=np.nan)
            up = np.where(np.isfinite(d) & (d > 0), d, 0.0)
            dn = np.where(np.isfinite(d) & (d < 0), -d, 0.0)
            au = _ema(up, 1.0 / w, w)
            ad = _ema(dn, 1.0 / w, w)
            rs = _sdiv(au, ad, np.inf)
            v = 100.0 - 100.0 / (1.0 + rs)
            v = np.where(~np.isfinite(ad) | ~np.isfinite(au), np.nan, v)
            v = np.where(np.isfinite(ad) & (ad <= _EPS), 100.0, v)
            self._k[key] = v
        return v

    def atr(self, w: int) -> np.ndarray:
        key = ("atr", w)
        v = self._k.get(key)
        if v is None:
            pc = _shift(self.c, 1)
            tr = np.fmax(self.h - self.l, np.fmax(np.abs(self.h - pc), np.abs(self.l - pc)))
            tr = np.where(np.isfinite(tr), tr, np.nan)
            v = _ema(tr, 1.0 / w, w)
            self._k[key] = v
        return v

    def kdj(self, n: int, k: int = 3, d: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("kdj", n, k, d)
        v = self._k.get(key)
        if v is None:
            ll = _roll_extreme(self.l, n, is_max=False)
            hh = _roll_extreme(self.h, n, is_max=True)
            rsv = 100.0 * _sdiv(self.c - ll, hh - ll, 0.5)
            rsv = np.where(np.isfinite(hh) & np.isfinite(ll), rsv, np.nan)
            K = _ema(rsv, 1.0 / k, n)
            D = _ema(K, 1.0 / d, n)
            v = (K, D, 3.0 * K - 2.0 * D)
            self._k[key] = v
        return v

    def macd(self, f: int, s: int, sig: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("macd", f, s, sig)
        v = self._k.get(key)
        if v is None:
            dif = self.ema(f) - self.ema(s)
            dea = _ema(dif, 2.0 / (sig + 1.0), sig)
            v = (dif, dea, dif - dea)
            self._k[key] = v
        return v

    def boll(self, n: int, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("boll", n, k)
        v = self._k.get(key)
        if v is None:
            mid = self.sma(n)
            sd = self.rstd(n)
            v = (mid, mid + k * sd, mid - k * sd)
            self._k[key] = v
        return v

    def vma(self, w: int) -> np.ndarray:
        key = ("vma", w)
        v = self._k.get(key)
        if v is None:
            v = _roll_mean(np.nan_to_num(self.v, nan=0.0), w)
            self._k[key] = v
        return v

    def vratio(self, w: int) -> np.ndarray:
        """量比：当日量 / 前 w 日均量（不含当日）。"""
        key = ("vr", w)
        v = self._k.get(key)
        if v is None:
            prev = _shift(self.vma(w), 1)
            v = _sdiv(self.v, prev, 0.0)
            self._k[key] = v
        return v


class _View:
    """tail 模式下的切片视图：基础序列切片，指标先全量算再切片。"""

    __slots__ = ("_ctx", "_s", "n", "index", "c", "o", "h", "l", "v", "turn",
                 "amt", "ohlc_ok", "bar_ok", "idx_c")

    def __init__(self, ctx: _Ctx, start: int):
        self._ctx = ctx
        self._s = int(start)
        self.n = ctx.n - self._s
        self.index = ctx.index[self._s:]
        for name in ("c", "o", "h", "l", "v", "turn", "amt", "ohlc_ok", "bar_ok"):
            setattr(self, name, getattr(ctx, name)[self._s:])
        self.idx_c = None if ctx.idx_c is None else ctx.idx_c[self._s:]

    def sma(self, w: int) -> np.ndarray:
        return self._ctx.sma(w)[self._s:]

    def rstd(self, w: int) -> np.ndarray:
        return self._ctx.rstd(w)[self._s:]

    def ema(self, w: int) -> np.ndarray:
        return self._ctx.ema(w)[self._s:]

    def rsi(self, w: int) -> np.ndarray:
        return self._ctx.rsi(w)[self._s:]

    def atr(self, w: int) -> np.ndarray:
        return self._ctx.atr(w)[self._s:]

    def kdj(self, n: int, k: int = 3, d: int = 3):
        a, b, c = self._ctx.kdj(n, k, d)
        return a[self._s:], b[self._s:], c[self._s:]

    def macd(self, f: int, s: int, sig: int):
        a, b, c = self._ctx.macd(f, s, sig)
        return a[self._s:], b[self._s:], c[self._s:]

    def boll(self, n: int, k: float):
        a, b, c = self._ctx.boll(n, k)
        return a[self._s:], b[self._s:], c[self._s:]

    def vma(self, w: int) -> np.ndarray:
        return self._ctx.vma(w)[self._s:]

    def vratio(self, w: int) -> np.ndarray:
        return self._ctx.vratio(w)[self._s:]


_INDEX_CACHE: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}


def _index_arrays(ix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(日期 int64 升序, close) 二元组；同一基准只解析一次。"""
    col = "date" if "date" in ix.columns else ix.columns[0]
    d = np.asarray(ix[col].to_numpy(), dtype="datetime64[ns]").astype(np.int64)
    c = _col(ix, "close")
    order = np.argsort(d, kind="stable")
    d, c = d[order], c[order]
    keep = np.ones(d.size, dtype=bool)
    keep[1:] = d[1:] != d[:-1]          # 去重日期，保留首条
    return d[keep], c[keep]


def _align_ffill(d_src: np.ndarray, v_src: np.ndarray, d_dst: np.ndarray) -> np.ndarray:
    """把源序列按日期前向填充对齐到目标日期（只取 <=t 的值，纯因果）。"""
    pos = np.searchsorted(d_src, d_dst, side="right") - 1
    out = np.full(d_dst.size, np.nan, dtype=np.float64)
    ok = pos >= 0
    out[ok] = v_src[pos[ok]]
    return out


def _load_index_arrays(code: str = "000300") -> Tuple[np.ndarray, np.ndarray]:
    """读基准指数缓存（只读本地 CSV，绝不触发联网）；失败返回空数组。"""
    if code in _INDEX_CACHE:
        got = _INDEX_CACHE[code]
        if got is None:
            return np.empty(0, np.int64), np.empty(0)
        return got
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None
    try:
        import os
        from .data import INDEX_DIR
        for name in (f"{code}_h2.csv", f"{code}.csv"):
            p = os.path.join(INDEX_DIR, name)
            if os.path.exists(p):
                out = _index_arrays(pd.read_csv(p, parse_dates=["date"]))
                break
    except Exception:      # noqa: BLE001
        out = None
    _INDEX_CACHE[code] = out
    if out is None:
        return np.empty(0, np.int64), np.empty(0)
    return out


# ======================================================================
# 3. 各族实现：f(ctx, params) -> (DIR, STR)  numpy float64
# ======================================================================
def _f_ma_cross(ctx, p: dict):
    """双均线金叉/死叉（事件型）。"""
    fast, slow = int(p["fast"]), int(p["slow"])
    mf, ms = ctx.sma(fast), ctx.sma(slow)
    sep = _sdiv(mf - ms, ms, 0.0)
    prev = _shift(sep, 1)
    ok = np.isfinite(mf) & np.isfinite(ms)
    up = ok & (sep > 0) & (prev <= 0)
    dn = ok & (sep < 0) & (prev >= 0)
    d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    s = 0.5 + 0.5 * np.clip(np.abs(sep) / 0.05, 0.0, 1.0)
    return d, s


def _f_ma_slope(ctx, p: dict):
    """均线斜率（状态型）/ 多头排列。"""
    if p.get("mode") == "align":
        ns = [int(v) for v in p["ns"]]
        ms = [ctx.sma(v) for v in ns]
        ok = np.ones(ctx.n, dtype=bool)
        for m in ms:
            ok &= np.isfinite(m)
        bull = ok.copy()
        bear = ok.copy()
        for i in range(len(ms) - 1):
            bull &= ms[i] > ms[i + 1]
            bear &= ms[i] < ms[i + 1]
        d = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
        s = np.clip(np.abs(_sdiv(ms[0] - ms[-1], ms[-1], 0.0)) / 0.05, 0.0, 1.0)
        return d, s
    n, k = int(p["n"]), int(p["k"])
    m = ctx.sma(n)
    base = _shift(m, k)
    sl = _sdiv(m - base, base, 0.0)
    sl = np.where(np.isfinite(m) & np.isfinite(base), sl, np.nan)
    d = _sign(sl)
    s = np.clip(np.abs(sl) / 0.05, 0.0, 1.0)
    return d, s


def _f_macd(ctx, p: dict):
    """MACD 金叉死叉 / 零轴穿越 / 柱状放大。"""
    dif, dea, hist = ctx.macd(int(p["fast"]), int(p["slow"]), int(p["sig"]))
    px = np.abs(ctx.c)
    mode = p.get("mode", "cross")
    if mode == "zero":
        d1, d0 = dif, _shift(dif, 1)
        up = (d1 > 0) & (d0 <= 0)
        dn = (d1 < 0) & (d0 >= 0)
        d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
        s = 0.5 + 0.5 * np.clip(_sdiv(np.abs(dif), 0.01 * px, 0.0), 0.0, 1.0)
        return d, s
    if mode == "hist":
        h0, h1, h2 = hist, _shift(hist, 1), _shift(hist, 2)
        grow = np.isfinite(h2) & (np.abs(h0) > np.abs(h1)) & (np.abs(h1) > np.abs(h2))
        d = np.where(grow, _sign(hist), 0.0)
        s = np.clip(_sdiv(np.abs(hist), 0.02 * px, 0.0), 0.0, 1.0)
        return d, s
    x, y = dif, dea
    x0, y0 = _shift(dif, 1), _shift(dea, 1)
    up = (x > y) & (x0 <= y0)
    dn = (x < y) & (x0 >= y0)
    d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    s = 0.5 + 0.5 * np.clip(_sdiv(np.abs(hist), 0.01 * px, 0.0), 0.0, 1.0)
    return d, s


def _f_rsi(ctx, p: dict):
    """RSI 超卖反转 / 超买回落 / 中轴突破。"""
    r = ctx.rsi(int(p["n"]))
    th = float(p["th"])
    r1 = _shift(r, 1)
    mode = p.get("mode", "os")
    if mode == "ob":
        hit = (r < th) & (r1 >= th)
        d = np.where(hit, -1.0, 0.0)
        s = 0.5 + 0.5 * np.clip(_sdiv(r1 - th, 100.0 - th, 0.0), 0.0, 1.0)
        return d, s
    if mode == "mid":
        up = (r > th) & (r1 <= th)
        dn = (r < th) & (r1 >= th)
        d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
        s = 0.5 + 0.5 * np.clip(np.abs(r - _shift(r, 3)) / 15.0, 0.0, 1.0)
        return d, s
    hit = (r > th) & (r1 <= th)
    d = np.where(hit, 1.0, 0.0)
    s = 0.5 + 0.5 * np.clip(_sdiv(th - r1, th, 0.0), 0.0, 1.0)
    return d, s


def _f_kdj(ctx, p: dict):
    """KDJ 金叉 / J 值极值 / 超买超卖区。"""
    n, k, dd = int(p["n"]), int(p.get("k", 3)), int(p.get("d", 3))
    K, D, J = ctx.kdj(n, k, dd)
    mode = p.get("mode", "cross")
    if mode == "j":
        lo, hi = float(p["lo"]), float(p["hi"])
        d = np.where(J < lo, 1.0, np.where(J > hi, -1.0, 0.0))
        s = np.clip(np.abs(J - 50.0) / 50.0, 0.0, 1.0)
        return d, s
    if mode == "obos":
        lo, hi = float(p["lo"]), float(p["hi"])
        d = np.where(K < lo, 1.0, np.where(K > hi, -1.0, 0.0))
        s = np.clip(np.abs(K - 50.0) / 50.0, 0.0, 1.0)
        return d, s
    K0, D0 = _shift(K, 1), _shift(D, 1)
    up = (K > D) & (K0 <= D0)
    dn = (K < D) & (K0 >= D0)
    d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    s = 0.5 + 0.5 * np.clip(np.abs(K - D) / 20.0, 0.0, 1.0)
    return d, s


def _f_boll(ctx, p: dict):
    """布林：下轨反弹 / 上轨突破 / 带宽收口。"""
    n, k = int(p["n"]), float(p["k"])
    mid, up, lo = ctx.boll(n, k)
    mode = p.get("mode", "lb")
    if mode == "ub":
        hit = np.isfinite(up) & (ctx.c > up) & (_shift(ctx.c, 1) <= _shift(up, 1))
        d = np.where(hit, 1.0, 0.0)
        s = 0.5 + 0.5 * np.clip(_sdiv(ctx.c - up, np.maximum(up - mid, _EPS), 0.0), 0.0, 1.0)
        return d, s
    if mode == "sq":
        bw = _sdiv(up - lo, mid, 0.0)
        bw = np.where(np.isfinite(up), bw, np.nan)
        bwm = _roll_mean(np.nan_to_num(bw, nan=0.0), 120)
        sq = np.isfinite(bwm) & (bwm > 0) & (bw < float(p["th"]) * bwm)
        d = np.where(sq, _sign(ctx.c - mid), 0.0)
        s = np.clip(1.0 - _sdiv(bw, bwm, 1.0), 0.0, 1.0)
        return d, s
    hit = np.isfinite(lo) & (ctx.l <= lo) & (ctx.c > lo)
    d = np.where(hit, 1.0, 0.0)
    s = 0.5 + 0.5 * np.clip(_sdiv(lo - ctx.l, np.maximum(up - mid, _EPS), 0.0), 0.0, 1.0)
    return d, s


def _f_vol_break(ctx, p: dict):
    """放量突破 N 日高点（量比阈值）。"""
    n, vr = int(p["n"]), float(p["vr"])
    hh = _shift(_roll_extreme(ctx.h, n, is_max=True), 1)     # 前 n 日最高（不含当日）
    ratio = ctx.vratio(n)
    hit = np.isfinite(hh) & (ctx.c > hh) & (ratio > vr)
    d = np.where(hit, 1.0, 0.0)
    s = 0.5 + 0.5 * np.clip(_sdiv(ratio, vr, 0.0) - 1.0, 0.0, 1.0)
    return d, s


def _f_vol_shrink(ctx, p: dict):
    """缩量回调到支撑（趋势内）。"""
    nm, vr, tn = int(p["n"]), float(p["vr"]), int(p["trend"])
    ma = ctx.sma(nm)
    mat = ctx.sma(tn)
    in_trend = np.isfinite(mat) & (ctx.c > mat) & (mat > _shift(mat, 10))
    near = np.abs(_sdiv(ctx.c - ma, ma, 9.0)) < float(p.get("near", 0.03))
    shrink = ctx.vratio(nm) < vr
    pull = ctx.c < _shift(ctx.c, 1)
    hit = in_trend & near & shrink & pull
    d = np.where(hit, 1.0, 0.0)
    s = 0.5 + 0.5 * np.clip(1.0 - _sdiv(ctx.vratio(nm), vr, 1.0), 0.0, 1.0)
    return d, s


def _f_volume_anom(ctx, p: dict):
    """量能异动（量比 / 换手率分位），方向跟随当日涨跌。"""
    ret1 = _sdiv(ctx.c - _shift(ctx.c, 1), _shift(ctx.c, 1), 0.0)
    if p.get("mode") == "turn":
        n, th = int(p["n"]), float(p["th"])
        q = _roll_quantile(np.nan_to_num(ctx.turn, nan=0.0), n, float(p["q"]))
        hit = np.isfinite(q) & (ctx.turn > q) & np.isfinite(ctx.turn)
        d = np.where(hit, _sign(ret1), 0.0)
        s = 0.5 + 0.5 * np.clip(_sdiv(ctx.turn, q, 0.0) - 1.0, 0.0, 1.0)
        return d, s
    n, th = int(p["n"]), float(p["th"])
    ratio = ctx.vratio(n)
    hit = np.isfinite(ratio) & (ratio > th)
    d = np.where(hit, _sign(ret1), 0.0)
    s = 0.5 + 0.5 * np.clip(_sdiv(ratio, th, 0.0) - 1.0, 0.0, 1.0)
    return d, s


def _f_atr(ctx, p: dict):
    """ATR 通道突破 / 波动压缩。"""
    n = int(p["n"])
    a = ctx.atr(n)
    if p.get("mode") == "squeeze":
        ap = _sdiv(a, ctx.c, 0.0)
        ap = np.where(np.isfinite(a), ap, np.nan)
        apm = _roll_mean(np.nan_to_num(ap, nan=0.0), 120)
        sq = np.isfinite(apm) & (apm > 0) & (ap < float(p["th"]) * apm)
        d = np.where(sq, _sign(ctx.c - ctx.sma(20)), 0.0)
        s = np.clip(1.0 - _sdiv(ap, apm, 1.0), 0.0, 1.0)
        return d, s
    k = float(p["k"])
    mid = ctx.sma(n)
    upper = mid + k * a
    lower = mid - k * a
    up = np.isfinite(upper) & (ctx.c > upper) & (_shift(ctx.c, 1) <= _shift(upper, 1))
    dn = np.isfinite(lower) & (ctx.c < lower) & (_shift(ctx.c, 1) >= _shift(lower, 1))
    d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    s = 0.5 + 0.5 * np.clip(_sdiv(np.abs(ctx.c - mid) - k * a, np.maximum(a, _EPS), 0.0), 0.0, 1.0)
    return d, s


def _f_mom(ctx, p: dict):
    """N 日动量 / 反转（标准化 z 口径）。"""
    n = int(p["n"])
    m = _sdiv(ctx.c - _shift(ctx.c, n), _shift(ctx.c, n), 0.0)
    m = np.where(np.isfinite(_shift(ctx.c, n)), m, np.nan)
    if p.get("mode") == "pair":
        m2 = _sdiv(ctx.c - _shift(ctx.c, int(p["n2"])), _shift(ctx.c, int(p["n2"])), 0.0)
        diff = m - m2
        d = _sign(diff)
        sd = _roll_std(np.nan_to_num(diff, nan=0.0), 120)
        z = _sdiv(diff, sd, 0.0)
        s = np.clip(np.abs(z) / 2.0, 0.0, 1.0)
        return d, s
    sd = _roll_std(np.nan_to_num(m, nan=0.0), 120)
    z = _sdiv(m, sd, 0.0)
    z = np.where(np.isfinite(m) & np.isfinite(sd), z, np.nan)
    rev = bool(p.get("rev", False))
    d = _sign(z) * (-1.0 if rev else 1.0)
    d = np.where(np.abs(z) >= 0.3, d, 0.0)
    s = np.clip(np.abs(z) / 2.0, 0.0, 1.0)
    return d, s


def _f_gap(ctx, p: dict):
    """跳空缺口方向。"""
    th = float(p["th"])
    ph, pl = _shift(ctx.h, 1), _shift(ctx.l, 1)
    ok = ctx.ohlc_ok & np.isfinite(ph) & np.isfinite(pl)
    up = ok & (ctx.o > ph * (1.0 + th))
    dn = ok & (ctx.o < pl * (1.0 - th))
    d = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    s = 0.5 + 0.5 * np.clip(np.abs(_sdiv(ctx.o - ph, ph, 0.0)) / max(th * 2.0, _EPS), 0.0, 1.0)
    return d, s


def _f_pattern(ctx, p: dict):
    """K 线形态：锤子线 / 吞没 / 十字星反转 / 三连阴后大阳。"""
    o, h, l, c = ctx.o, ctx.h, ctx.l, ctx.c
    rng = h - l
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    ok = ctx.ohlc_ok & np.isfinite(rng) & (rng > 0)
    mode = p.get("mode", "hammer")
    d = np.zeros(ctx.n, dtype=np.float64)
    s = np.zeros(ctx.n, dtype=np.float64)
    if mode == "hammer":
        sh = float(p.get("shadow", 2.0))
        hit = ok & (body < 0.35 * rng) & (lower > sh * np.maximum(body, 0.05 * rng)) & (upper < 0.25 * rng)
        d[hit] = 1.0
        s[hit] = 0.5 + 0.5 * np.clip(_sdiv(lower, rng, 0.0)[hit], 0.0, 1.0)
        return d, s
    o1, c1 = _shift(o, 1), _shift(c, 1)
    if mode in ("engulf_bull", "engulf_bear"):
        if mode == "engulf_bull":
            hit = ok & (c1 < o1) & (c > o) & (c >= o1) & (o <= c1)
            d[hit] = 1.0
        else:
            hit = ok & (c1 > o1) & (c < o) & (c <= o1) & (o >= c1)
            d[hit] = -1.0
        s[hit] = 0.5 + 0.5 * np.clip(_sdiv(body, np.maximum(np.abs(c1 - o1), _EPS), 0.0)[hit], 0.0, 1.0)
        return d, s
    if mode == "doji_rev":
        c2, c3, c4 = _shift(c, 2), _shift(c, 3), _shift(c, 4)
        doji = ok & (body < 0.12 * rng)
        down3 = (c1 < c2) & (c2 < c3) & (c3 < c4)
        up3 = (c1 > c2) & (c2 > c3) & (c3 > c4)
        d[doji & down3] = 1.0
        d[doji & up3] = -1.0
        s[doji] = 0.5
        return d, s
    # 三连阴后大阳
    c2, c3, c4 = _shift(c, 2), _shift(c, 3), _shift(c, 4)
    down3 = (c1 < c2) & (c2 < c3) & (c3 < c4)
    big = ok & down3 & (c > o) & (_sdiv(c - c1, c1, 0.0) > float(p.get("th", 0.02)))
    d[big] = 1.0
    s[big] = 0.5 + 0.5 * np.clip(_sdiv(c - c1, c1, 0.0)[big] / 0.06, 0.0, 1.0)
    return d, s


def _f_rs(ctx, p: dict):
    """相对指数强弱（超额动量）。"""
    if ctx.idx_c is None:
        z = np.zeros(ctx.n)
        return z, z.copy()
    n = int(p["n"])
    rs_ = _sdiv(ctx.c - _shift(ctx.c, n), _shift(ctx.c, n), 0.0)
    ri = _sdiv(ctx.idx_c - _shift(ctx.idx_c, n), _shift(ctx.idx_c, n), 0.0)
    ex = rs_ - ri
    ex = np.where(np.isfinite(_shift(ctx.c, n)) & np.isfinite(_shift(ctx.idx_c, n)), ex, np.nan)
    if p.get("mode") == "z":
        sd = _roll_std(np.nan_to_num(ex, nan=0.0), 120)
        z = _sdiv(ex, sd, 0.0)
        z = np.where(np.isfinite(ex) & np.isfinite(sd), z, np.nan)
        d = np.where(z > float(p.get("th", 2.0)), 1.0, np.where(z < -float(p.get("th", 2.0)), -1.0, 0.0))
        s = np.clip(np.abs(z) / 4.0, 0.0, 1.0)
        return d, s
    d = _sign(ex)
    s = np.clip(np.abs(ex) / float(p.get("scale", 0.15)), 0.0, 1.0)
    return d, s


def _f_dist_ma(ctx, p: dict):
    """乖离率极值（均值回复语义）。"""
    n = int(p["n"])
    ma = ctx.sma(n)
    bias = _sdiv(ctx.c - ma, ma, 0.0)
    bias = np.where(np.isfinite(ma), bias, np.nan)
    mode = p.get("mode", "neg")
    if mode == "z":
        sd = _roll_std(np.nan_to_num(bias, nan=0.0), 120)
        z = _sdiv(bias, sd, 0.0)
        z = np.where(np.isfinite(bias) & np.isfinite(sd), z, np.nan)
        th = float(p.get("th", 2.0))
        d = np.where(z < -th, 1.0, np.where(z > th, -1.0, 0.0))
        s = np.clip(np.abs(z) / 4.0, 0.0, 1.0)
        return d, s
    th = float(p["th"])
    if mode == "pos":
        hit = bias > th
        d = np.where(hit, -1.0, 0.0)
    else:
        hit = bias < -th
        d = np.where(hit, 1.0, 0.0)
    s = 0.5 + 0.5 * np.clip(_sdiv(np.abs(bias), th, 0.0) - 1.0, 0.0, 1.0)
    return d, s


_IMPL: Dict[str, Callable[[Any, dict], Tuple[np.ndarray, np.ndarray]]] = {
    "ma_cross": _f_ma_cross,
    "ma_slope": _f_ma_slope,
    "macd": _f_macd,
    "rsi": _f_rsi,
    "kdj": _f_kdj,
    "boll": _f_boll,
    "vol_break": _f_vol_break,
    "vol_shrink": _f_vol_shrink,
    "volume_anom": _f_volume_anom,
    "atr_stop": _f_atr,
    "mom": _f_mom,
    "gap": _f_gap,
    "pattern": _f_pattern,
    "rs": _f_rs,
    "dist_ma": _f_dist_ma,
}

_KIND = {
    "ma_cross": "cross", "ma_slope": "trend", "macd": "trend", "rsi": "reversal",
    "kdj": "reversal", "boll": "reversal", "vol_break": "volume", "vol_shrink": "volume",
    "volume_anom": "volume", "atr_stop": "volatility", "mom": "momentum",
    "gap": "pattern", "pattern": "pattern", "rs": "momentum", "dist_ma": "reversal",
}

# 需要"可信 OHLC"的族（加性除息污染区置 0）
_NEEDS_OHLC = {"gap", "pattern"}


# ======================================================================
# 4. 信号动物园（参数网格 = 可学习的参数空间）
# ======================================================================
def _table() -> List[tuple]:
    """(family, params, sid, note)。sid 稳定可读：<family>_<paramhash>。"""
    t: List[tuple] = []
    # ---- ma_cross (4) 金叉/死叉
    for f, s in ((5, 20), (10, 30), (20, 60), (5, 60)):
        t.append(("ma_cross", {"fast": f, "slow": s}, f"ma_cross_{f}_{s}",
                  f"MA{f} 上穿 MA{s} 视为金叉(+1)，下穿为死叉(-1)"))
    # ---- ma_slope (5) 斜率 / 多头排列
    for n, k in ((20, 5), (60, 10), (120, 20), (250, 20)):
        t.append(("ma_slope", {"mode": "slope", "n": n, "k": k}, f"ma_slope_{n}_{k}",
                  f"MA{n} 相对 {k} 日前抬升为正(+1)，走平/下行(-1)"))
    t.append(("ma_slope", {"mode": "align", "ns": [5, 10, 20, 60]}, "ma_align_5_10_20_60",
              "MA5>MA10>MA20>MA60 多头排列(+1)，反向为空头(-1)"))
    # ---- macd (5)
    for f, s, g in ((12, 26, 9), (8, 17, 9)):
        t.append(("macd", {"mode": "cross", "fast": f, "slow": s, "sig": g}, f"macd_cross_{f}_{s}_{g}",
                  f"MACD({f},{s},{g}) DIF 上穿 DEA 金叉(+1)，下穿死叉(-1)"))
        t.append(("macd", {"mode": "zero", "fast": f, "slow": s, "sig": g}, f"macd_zero_{f}_{s}_{g}",
                  f"MACD({f},{s},{g}) DIF 上穿零轴(+1)，下穿(-1)"))
    t.append(("macd", {"mode": "hist", "fast": 12, "slow": 26, "sig": 9}, "macd_hist_12_26_9",
              "MACD 柱体连续放大时顺柱体方向(+1/-1)"))
    # ---- rsi (7)
    for n, th in ((14, 30), (6, 20), (21, 35)):
        t.append(("rsi", {"mode": "os", "n": n, "th": th}, f"rsi_{n}_th{th}",
                  f"RSI{n} 自下方上穿 {th} 视为超卖反转(+1)"))
    for n, th in ((14, 70), (6, 80), (21, 65)):
        t.append(("rsi", {"mode": "ob", "n": n, "th": th}, f"rsi_{n}_ob{th}",
                  f"RSI{n} 自上方下穿 {th} 视为超买回落(-1)"))
    t.append(("rsi", {"mode": "mid", "n": 14, "th": 50}, "rsi_14_mid50",
              "RSI14 上穿 50 中轴(+1)，下穿(-1)"))
    # ---- kdj (4)
    t.append(("kdj", {"mode": "cross", "n": 9, "k": 3, "d": 3}, "kdj_cross_9_3_3",
              "KDJ(9,3,3) K 上穿 D 金叉(+1)，下穿死叉(-1)"))
    t.append(("kdj", {"mode": "cross", "n": 14, "k": 3, "d": 3}, "kdj_cross_14_3_3",
              "KDJ(14,3,3) K 上穿 D 金叉(+1)，下穿死叉(-1)"))
    t.append(("kdj", {"mode": "j", "n": 9, "k": 3, "d": 3, "lo": 0.0, "hi": 100.0}, "kdj_j_9_3_3",
              "J 值 <0 超卖(+1)，>100 超买(-1)"))
    t.append(("kdj", {"mode": "obos", "n": 9, "k": 3, "d": 3, "lo": 20.0, "hi": 80.0}, "kdj_obos_9_3_3",
              "K<20 超卖区(+1)，K>80 超买区(-1)"))
    # ---- boll (5)
    for n, k in ((20, 2.0), (60, 2.0)):
        t.append(("boll", {"mode": "lb", "n": n, "k": k}, f"boll_lb_{n}_{int(k)}",
                  f"BOLL({n},{k:g}) 下影触及下轨后收回(+1)"))
        t.append(("boll", {"mode": "ub", "n": n, "k": k}, f"boll_ub_{n}_{int(k)}",
                  f"BOLL({n},{k:g}) 收盘上穿上轨(+1)"))
    t.append(("boll", {"mode": "sq", "n": 20, "k": 2.0, "th": 0.7}, "boll_sq_20_2",
              "布林带宽低于 120 日均值 70% 视为收口，方向跟随中轨上下"))
    # ---- vol_break (5)
    for n, vr in ((20, 1.5), (60, 1.8), (120, 2.0), (20, 2.5), (60, 1.2)):
        sid = f"vol_break_{n}_{str(vr).replace('.', 'p')}"
        t.append(("vol_break", {"n": n, "vr": vr}, sid,
                  f"收盘突破前 {n} 日最高且量比 >{vr:g} 视为放量突破(+1)"))
    # ---- vol_shrink (4)
    for n, vr, tn in ((20, 0.7, 60), (20, 0.6, 120), (60, 0.7, 120), (10, 0.8, 60)):
        sid = f"vol_shrink_{n}_{str(vr).replace('.', 'p')}_{tn}"
        t.append(("vol_shrink", {"n": n, "vr": vr, "trend": tn}, sid,
                  f"趋势内回踩 MA{n} 且量能 <{vr:g} 倍前 {n} 日均量(+1)"))
    # ---- volume_anom (5)
    for n, th in ((20, 2.0), (20, 3.0), (5, 1.5)):
        sid = f"vol_anom_{n}_{str(th).replace('.', 'p')}"
        t.append(("volume_anom", {"mode": "ratio", "n": n, "th": th}, sid,
                  f"量比(前 {n} 日均量)> {th:g} 视为异动，方向跟随当日涨跌"))
    for n, q in ((20, 0.90), (60, 0.95)):
        sid = f"vol_anom_turn{int(q * 100)}_{n}"
        t.append(("volume_anom", {"mode": "turn", "n": n, "q": q, "th": q}, sid,
                  f"换手率高于前 {n} 日 {q:.0%} 分位视为异动，方向跟随当日涨跌"))
    # ---- atr_stop (4)
    for n, k in ((14, 2.0), (20, 1.5)):
        sid = f"atr_chan_{n}_{str(k).replace('.', 'p')}"
        t.append(("atr_stop", {"mode": "chan", "n": n, "k": k}, sid,
                  f"收盘突破 MA{n}±{k:g}×ATR{n} 通道(+1 上破 / -1 下破)"))
    for n, th in ((14, 0.6), (20, 0.7)):
        sid = f"atr_sq_{n}_{str(th).replace('.', 'p')}"
        t.append(("atr_stop", {"mode": "squeeze", "n": n, "th": th}, sid,
                  f"ATR{n}/价格 低于 120 日均值 {th:g} 倍视为波动压缩，方向跟随 MA20"))
    # ---- mom (6)
    t.append(("mom", {"n": 1, "rev": True}, "mom_1_rev", "1 日反转：昨日跌则今日看多(-1×动量)"))
    t.append(("mom", {"n": 5, "rev": True}, "mom_5_rev", "5 日反转：短期超跌看多(-1×动量)"))
    for n in (20, 60, 120):
        t.append(("mom", {"n": n, "rev": False}, f"mom_{n}", f"{n} 日动量标准化后顺动量(+1 强动量)"))
    t.append(("mom", {"n": 5, "n2": 20, "mode": "pair"}, "mom_5_20",
              "5 日动量减 20 日动量 >0 视为加速(+1)"))
    # ---- gap (3)
    for th in (0.005, 0.01, 0.02):
        sid = f"gap_{str(th * 100).replace('.', 'p')}"
        t.append(("gap", {"th": th}, sid, f"开盘较昨日最高跳空 >{th:.1%}(+1)，向下跳空(-1)"))
    # ---- pattern (6)
    t.append(("pattern", {"mode": "hammer", "shadow": 2.0}, "pat_hammer_2",
              "锤子线：下影 >2 倍实体且实体小(+1)"))
    t.append(("pattern", {"mode": "hammer", "shadow": 3.0}, "pat_hammer_3",
              "长下影锤子线：下影 >3 倍实体(+1)"))
    t.append(("pattern", {"mode": "engulf_bull"}, "pat_engulf_bull", "看涨吞没：阳线实体吞没前一根阴线(+1)"))
    t.append(("pattern", {"mode": "engulf_bear"}, "pat_engulf_bear", "看跌吞没：阴线实体吞没前一根阳线(-1)"))
    t.append(("pattern", {"mode": "doji_rev"}, "pat_doji_rev",
              "十字星出现在三连阴之后(+1)/三连阳之后(-1)"))
    t.append(("pattern", {"mode": "3yin_yang", "th": 0.02}, "pat_3yin_yang",
              "三连阴后当日大阳(>2%)视为反包(+1)"))
    # ---- rs (5)
    for n in (5, 20, 60, 120):
        t.append(("rs", {"mode": "raw", "n": n, "scale": 0.15}, f"rs_{n}",
                  f"{n} 日超额收益（个股-沪深300）为正(+1)"))
    t.append(("rs", {"mode": "z", "n": 20, "th": 2.0}, "rs_20_z",
              "20 日超额收益 z 值 >2 视为显著强于指数(+1)，<-2 反之(-1)"))
    # ---- dist_ma (5)
    for n, th, mode in ((20, 0.08, "neg"), (20, 0.08, "pos"), (60, 0.15, "neg"), (60, 0.15, "pos")):
        sid = f"dist_ma_{n}_{mode}{int(th * 100)}"
        if mode == "neg":
            t.append(("dist_ma", {"mode": "neg", "n": n, "th": th}, sid,
                      f"乖离率(收盘/MA{n}-1) < -{th:.0%} 视为超跌回归(+1)"))
        else:
            t.append(("dist_ma", {"mode": "pos", "n": n, "th": th}, sid,
                      f"乖离率(收盘/MA{n}-1) > {th:.0%} 视为超买回归(-1)"))
    t.append(("dist_ma", {"mode": "z", "n": 20, "th": 2.0}, "dist_ma_20_z2",
              "20 日乖离率 z 值 <-2 超跌(+1)，>2 超买(-1)"))
    return t


def _build_zoo() -> List[SignalSpec]:
    out: List[SignalSpec] = []
    seen = set()
    for family, params, sid, note in _table():
        if sid in seen:
            raise ValueError(f"duplicate sid: {sid}")
        seen.add(sid)
        out.append(SignalSpec(sid=sid, family=family, kind=_KIND[family],
                              params=dict(params), note=note))
    return out


_ZOO: List[SignalSpec] = _build_zoo()
FAMILIES: Tuple[str, ...] = tuple(dict.fromkeys(s.family for s in _ZOO))


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p").replace("-", "m")
    if isinstance(v, (list, tuple)):
        return "_".join(_fmt_num(x) for x in v)
    return str(v)


def _auto_sid(family: str, params: dict, idx: int) -> str:
    parts = [family]
    for k in sorted(params):
        parts.append(f"{k}{_fmt_num(params[k])}")
    return "_".join(parts) + (f"_{idx}" if idx else "")


def signal_zoo(cfg: Optional[dict] = None) -> List[SignalSpec]:
    """返回全部 (family, params) 组合；默认 73 个变体，覆盖 SPEC §2.2 的 15 个族。

    cfg 可选键：
      families: list[str]        只保留这些族
      grid:     {family: [dict]} 用自定义参数网格替换该族默认网格
    """
    specs = list(_ZOO)
    if not cfg:
        return specs
    fams = cfg.get("families")
    if fams:
        keep = set(fams)
        specs = [s for s in specs if s.family in keep]
    grid = cfg.get("grid") or {}
    if grid:
        out: List[SignalSpec] = []
        for s in specs:
            if s.family not in grid:
                out.append(s)
                continue
            for i, params in enumerate(grid[s.family]):
                params = dict(params)
                out.append(SignalSpec(sid=_auto_sid(s.family, params, i), family=s.family,
                                      kind=s.kind, params=params,
                                      note=f"{s.family} 自定义参数变体"))
        specs = out
    return specs


# ======================================================================
# 5. 主入口
# ======================================================================
def compute_signals(df: pd.DataFrame,
                    tail: Optional[int] = None,
                    index_df: Optional[pd.DataFrame] = None,
                    specs: Optional[Sequence[SignalSpec]] = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """单只股票日线 -> (DIR, STR)。

    DIR: index=date, columns=sid, 取值 {-1,0,+1} float32
    STR: index=date, columns=sid, 取值 [0,1]    float32

    tail=None 全历史；tail=k 只返回最后 k 行（指标预热仍用全量历史，
    且第 n-1 行结果与全量模式逐元素相同）。任何数据异常都不抛异常，降级为 0。
    """
    specs = list(specs) if specs is not None else signal_zoo()
    sids = [s.sid for s in specs]

    n = len(df)
    if n == 0:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=sids, dtype=_F32)
        return empty.copy(), empty.copy()

    ctx = _Ctx(df, index_df=index_df)

    k = 0 if tail is None else int(tail)
    if k > 0 and k < n:
        start = max(0, n - k - _TAIL_MARGIN)
        view: Any = _View(ctx, start)
        cut = view.n - k
    else:
        view = ctx
        cut = 0

    m = view.n
    rows = m - cut
    nsig = len(specs)
    out_d = np.zeros((rows, nsig), dtype=_F32)
    out_s = np.zeros((rows, nsig), dtype=_F32)
    bar_ok = view.bar_ok
    ohlc_ok = view.ohlc_ok

    for j, spec in enumerate(specs):
        fn = _IMPL.get(spec.family)
        if fn is None:
            continue
        try:
            d, s = fn(view, spec.params)
            d = np.asarray(d, dtype=np.float64).reshape(-1)
            s = np.asarray(s, dtype=np.float64).reshape(-1)
            if d.size != m or s.size != m:
                continue
        except Exception:      # noqa: BLE001  单信号异常不得拖垮整体
            continue
        d, s = _emit(d, s)
        if spec.family in _NEEDS_OHLC:
            d *= ohlc_ok
            s *= ohlc_ok
        d *= bar_ok
        s *= bar_ok
        out_d[:, j] = d[cut:]
        out_s[:, j] = s[cut:]

    idx = view.index[-rows:] if rows else view.index[:0]
    DIR = pd.DataFrame(out_d, index=idx, columns=sids)
    STR = pd.DataFrame(out_s, index=idx, columns=sids)
    return DIR, STR
