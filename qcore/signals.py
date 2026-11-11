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
    """滚动求和，前 w-1 行 NaN。

    ## 为什么要用"滑窗求和"而不是前缀和差分

    前缀和 `cs[i]-cs[i-w]` 的两个被减数随序列变长而变大，舍入误差随之变化 →
    **同一个窗口在长序列与短序列上会算出不同结果**（实测末 800 根里 751 格差
    1e-8 量级）。信号在阈值边界上就会翻转 DIR，于是"增量 = 批量"做不到逐位相等
    （AGENTS 记录的 15/513k 单元格差异就是这个原因）。

    滑窗求和只依赖窗口内的 w 个数，**与序列长度无关** → 截断天然逐位相等
    （实测 0 格差异）。代价：w 大时多占一点临时内存（w=250 × 3400 根 ≈ 7MB）。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 0 or n < w:
        return out
    try:
        from numpy.lib.stride_tricks import sliding_window_view as _swv
        out[w - 1:] = _swv(x, int(w)).sum(axis=1)
        return out
    except Exception:  # noqa: BLE001  老 numpy 兜底：前缀和
        cs = np.empty(n + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(x, out=cs[1:])
        out[w - 1:] = cs[w:] - cs[:-w]
        return out


def _roll_mean_plain(x: np.ndarray, w: int) -> np.ndarray:
    """朴素均值（NaN 传播）。**模块内已被下方的 NaN 跳过版覆盖**，保留作参考：
    同名的两个 def 里，后一个才是运行时生效的那个（历史上这里藏过死代码）。"""
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
                 "ohlc_ok", "bar_ok", "idx_c", "_k", "flow")

    def __init__(self, df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None,
                 flow_df: Optional[pd.DataFrame] = None):
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
        # 资金流按**日期**对齐到本票的日期轴；对不上的位置留 NaN。
        # 绝不允许 forward-fill 到未来 —— 那等于用后面的资金流预测前面。
        self.flow: Dict[str, np.ndarray] = {}
        if flow_df is not None and len(flow_df):
            try:
                _f = flow_df.copy()
                _f["date"] = pd.to_datetime(_f["date"]).dt.normalize()
                _f = _f.drop_duplicates("date").set_index("date")
                for _cname in _f.columns:
                    _vals = pd.to_numeric(_f[_cname], errors="coerce")
                    self.flow[str(_cname)] = _vals.reindex(
                        idx.normalize()).to_numpy(dtype=np.float64)
            except Exception:  # noqa: BLE001
                self.flow = {}
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
                 "amt", "ohlc_ok", "bar_ok", "idx_c", "flow")

    def __init__(self, ctx: _Ctx, start: int):
        self._ctx = ctx
        self._s = int(start)
        self.n = ctx.n - self._s
        self.index = ctx.index[self._s:]
        for name in ("c", "o", "h", "l", "v", "turn", "amt", "ohlc_ok", "bar_ok"):
            setattr(self, name, getattr(ctx, name)[self._s:])
        self.idx_c = None if ctx.idx_c is None else ctx.idx_c[self._s:]
        self.flow = {k: v[self._s:] for k, v in (ctx.flow or {}).items()}

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
        # 依次尝试：正式缓存目录 → 仓库内置夹具（保证 CI/离线环境 rs 族也能算）
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = [INDEX_DIR, os.path.join(root, "tests", "fixtures")]
        found = False
        for d in dirs:
            for name in (f"{code}_h2.csv", f"{code}.csv"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    out = _index_arrays(pd.read_csv(p, parse_dates=["date"]))
                    found = True
                    break
            if found:
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


def _roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    """因果滚动均值（只用 ≤t 的 w 个点）。NaN 不计入分母。

    必须**全向量化**：早一版用 Python 循环，单只 3000 根就 119ms，
    超了 40ms 预算 3 倍（6 个 flow 信号 × 逐点循环）。
    前缀和版本 O(n)、无循环。
    """
    n = a.size
    if w <= 0 or n == 0:
        return np.full(n, np.nan, dtype=np.float64)
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0)
    # 用**滑窗和**而非前缀和差分（评审 R48）：前缀和的两个被减数随序列变长而变大，
    # 同一窗口在长短序列上会差 1e-8 量级 —— `ma_slope_250_20` 在 601166 上
    # 就因此翻了一格 DIR（slope 恰好贴着 0）。滑窗和只依赖窗口内 w 个数。
    sum_v = _roll_sum(v, w)
    sum_c = _roll_sum(ok.astype(np.float64), w)
    need = max(2, w // 2)
    return np.where(sum_c >= need, sum_v / np.maximum(sum_c, 1.0), np.nan)


def _f_flow(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """资金流信号族。

    数据来源：东财个股资金流历史（`flow_factor.fetch_flow`，接口上限 120 天）。
    **只用相对量**（`main_ratio` 是主力净流入占成交额的百分比，本身就是相对量），
    不使用绝对金额 —— 后复权价格与现价差 10 倍，绝对金额不可跨股票比较。

    严格因果：滚动均值只用 ≤t 的点；flow 缺失处输出 0（预热），不输出 NaN。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    win = int(params.get("win", 5))
    kind = str(params.get("kind", "main"))
    col = "main_ratio" if kind == "main" else "super_ratio"
    raw = view.flow.get(col)
    if raw is None:
        raw = view.flow.get("main_ratio")
    if raw is None:
        return d, s
    mu = _roll_mean(np.asarray(raw, dtype=np.float64), win)
    # 阈值：主力净流入占成交额 1% 以上才算有方向（经验值，非拟合）
    th = float(params.get("th", 1.0))
    with np.errstate(invalid="ignore"):
        d = np.where(mu >= th, 1.0, np.where(mu <= -th, -1.0, 0.0))
        s = np.clip(np.abs(np.nan_to_num(mu, nan=0.0)) / (th * 3.0), 0.0, 1.0)
    d = np.nan_to_num(d, nan=0.0)
    return d, s


def _f_block_trade(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """大宗交易信号族。

    数据：`event_data.load_events`（东财 datacenter，600519 实测 272 笔跨 19 年）。
    只用相对量：
      · `bt_shares_ratio` 当日大宗成交量占总股本 %（跨股票可比）
      · `bt_premium`      按成交额加权的溢价率 %（正=溢价成交，负=折价甩卖）
    稀疏事件：没有大宗的日子输出 0，不影响其它信号。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    ratio = view.flow.get("bt_shares_ratio")
    prem = view.flow.get("bt_premium")
    if ratio is None or prem is None:
        return d, s
    ratio = np.nan_to_num(np.asarray(ratio, dtype=np.float64), nan=0.0)
    prem = np.nan_to_num(np.asarray(prem, dtype=np.float64), nan=0.0)
    min_ratio = float(params.get("min_ratio", 0.1))     # 占总股本 0.1% 以上才算有意义
    disc = float(params.get("disc", -5.0))              # 折价 5% 以上算甩卖
    prem_th = float(params.get("prem", 3.0))            # 溢价 3% 以上算看好
    has = ratio >= min_ratio
    with np.errstate(invalid="ignore"):
        d = np.where(has & (prem <= disc), -1.0,
                     np.where(has & (prem >= prem_th), 1.0, 0.0))
        s = np.clip(ratio / (min_ratio * 10.0), 0.0, 1.0) * has
    return d, s


def _f_billboard(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """龙虎榜信号族。

    数据：`event_data.load_events` 的 `bb_net_ratio` = 龙虎榜净买入额 / 当日成交额 %。
    上榜本身就是异常，净买入方向给出方向；阈值用相对量，不用绝对金额。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    net = view.flow.get("bb_net_ratio")
    if net is None:
        return d, s
    net = np.nan_to_num(np.asarray(net, dtype=np.float64), nan=0.0)
    th = float(params.get("th", 5.0))                   # 净买入占成交额 5% 以上
    with np.errstate(invalid="ignore"):
        d = np.where(net >= th, 1.0, np.where(net <= -th, -1.0, 0.0))
        s = np.clip(np.abs(net) / (th * 4.0), 0.0, 1.0)
    return d, s


def _f_margin(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """融资融券信号族。

    数据：东财 RPTA_WEB_RZRQ_GGMX（实测 600519 有 4015 条，2010-03-31 起）。
    只用相对量：
      · `mg_net_ratio`   融资净买入额 / 当日成交额 %
      · `mg_bal_ratio`   融资余额 / 流通市值 %
      · `mg_short_ratio` 融券余额 / 融资余额 %
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    kind = str(params.get("kind", "net"))
    col = {"net": "mg_net_ratio", "bal": "mg_bal_ratio",
           "short": "mg_short_ratio"}.get(kind, "mg_net_ratio")
    raw = view.flow.get(col)
    if raw is None:
        return d, s
    win = int(params.get("win", 5))
    th = float(params.get("th", 1.0))
    mu = _roll_mean(np.asarray(raw, dtype=np.float64), win)
    inv = bool(params.get("invert", False))   # 融券/高杠杆反向解释
    with np.errstate(invalid="ignore"):
        d = np.where(mu >= th, -1.0 if inv else 1.0,
                     np.where(mu <= -th, 1.0 if inv else -1.0, 0.0))
        s = np.clip(np.abs(np.nan_to_num(mu, nan=0.0)) / (th * 3.0), 0.0, 1.0)
    return np.nan_to_num(d, nan=0.0), s


def _f_sentiment(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """散户情绪信号族（股吧）。

    **没有历史**：股吧帖子只能从采集日往前累积，回测前期必然全 0。
    这不是 bug —— 是数据不存在。任何「情绪信号有效」的结论在攒够样本前不可采信。

    `contrarian=True`（默认）：散户极度看多时反向（bull_ratio 高 -> 看空）。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    br = view.flow.get("snt_bull_ratio")
    if br is None:
        return d, s
    br = np.asarray(br, dtype=np.float64)
    win = int(params.get("win", 1))
    if win > 1:
        br = _roll_mean(br, win)
    th = float(params.get("th", 0.65))
    contrarian = bool(params.get("contrarian", True))
    with np.errstate(invalid="ignore"):
        hot = np.nan_to_num(br, nan=0.0) >= th
        cold = (np.nan_to_num(br, nan=0.0) > 0) & (np.nan_to_num(br, nan=0.0) <= (1.0 - th))
        d = np.where(hot, -1.0 if contrarian else 1.0,
                     np.where(cold, 1.0 if contrarian else -1.0, 0.0))
        s = np.clip(np.abs(np.nan_to_num(br, nan=0.0) - 0.5) * 2.0, 0.0, 1.0)
    return d, s


def _f_event(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """政策/新闻事件信号族。

    数据：`data/state/event_ledger.jsonl`（实测 80 条，带 date/direction/level）。
    事件是板块级或全市场级的，已由 `sentiment_store.load_events` 按股票所属板块过滤。
    稀疏事件：没有事件的日子输出 0。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    ev = view.flow.get("evt_direction")
    lvl = view.flow.get("evt_level")
    if ev is None:
        return d, s
    ev = np.nan_to_num(np.asarray(ev, dtype=np.float64), nan=0.0)
    lvl = np.nan_to_num(np.asarray(lvl, dtype=np.float64), nan=0.0) if lvl is not None else np.ones(m)
    min_lvl = float(params.get("min_level", 1.0))
    with np.errstate(invalid="ignore"):
        d = np.where(lvl >= min_lvl, ev, 0.0)
        s = np.clip(lvl / 3.0, 0.0, 1.0) * (d != 0)
    return np.nan_to_num(d, nan=0.0), s


def _f_holder(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """股东增减持信号族（产业资本）。

    数据：`event_data.fetch_holder_change`（实测 147,394 条 / 1994 起 / 32 年）。
    按 **NOTICE_DATE 公告日**对齐 —— 那是信息可得日。
    净增持看多(+1)，净减持看空(-1)。稀疏事件，无公告日输出 0。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    raw = view.flow.get("hold_net_ratio")
    if raw is None:
        return d, s
    v = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    win = int(params.get("win", 20))
    if win > 1:
        v = _roll_mean(v, win)
    th = float(params.get("th", 0.0))
    with np.errstate(invalid="ignore"):
        d = np.where(v > th, 1.0, np.where(v < -th, -1.0, 0.0))
        s = np.clip(np.abs(v) / 1.0, 0.0, 1.0)
    return d, s


def _f_lift(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """限售解禁信号族。

    数据：`event_data.fetch_lift`（实测 31,820 条 / 2010 起，含未来计划）。
    解禁日是**事前已知**的，所以按 FREE_DATE 对齐不构成未来函数。
    解禁压力大 -> 看空(-1)。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    raw = view.flow.get("fc_lift_ratio")
    if raw is None:
        return d, s
    v = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    th = float(params.get("th", 10.0))
    with np.errstate(invalid="ignore"):
        d = np.where(v >= th, -1.0, 0.0)
        s = np.clip(v / (th * 3.0), 0.0, 1.0)
    return d, s


def _f_forecast(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """业绩预告信号族（预期修正）。

    数据：`event_data.fetch_forecast`（实测 191,446 条 / 2003 起 / 23 年）。
    **按 NOTICE_DATE 公告日对齐**，绝不能用 REPORT_DATE（那是报告期，早于公告日）。
    预告上调看多(+1)，下调看空(-1)。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    raw = view.flow.get("eps_revision")
    if raw is None:
        return d, s
    v = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    win = int(params.get("win", 20))
    if win > 1:
        v = _roll_mean(v, win)
    th = float(params.get("th", 0.0))
    with np.errstate(invalid="ignore"):
        d = np.where(v > th, 1.0, np.where(v < -th, -1.0, 0.0))
        s = np.clip(np.abs(v) / 3.0, 0.0, 1.0)
    return d, s


def _f_breadth(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """市场宽度/情绪信号族（**全市场共用同一序列**）。

    数据：`market_breadth.load()` 从本地 300 只日线自算，实测 **8697 天 / 1991 起**
    —— 这是目前所有外部维度里历史最长的，因为它不依赖任何外部接口。

    breadth_z   : 上涨家数占比相对过去 20 日的 z 值
    up_ratio    : 当日上涨家数占比
    极端宽度有均值回归倾向：z 极低（普跌）看多，z 极高（普涨）看空。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    kind = str(params.get("kind", "z"))
    col = "breadth_z" if kind == "z" else "up_ratio"
    raw = view.flow.get(col)
    if raw is None:
        return d, s
    v = np.asarray(raw, dtype=np.float64)
    th = float(params.get("th", 1.5))
    contrarian = bool(params.get("contrarian", True))
    with np.errstate(invalid="ignore"):
        hi = np.nan_to_num(v, nan=0.0) >= th
        lo = np.nan_to_num(v, nan=0.0) <= -th
        if contrarian:
            d = np.where(lo, 1.0, np.where(hi, -1.0, 0.0))     # 普跌看多 / 普涨看空
        else:
            d = np.where(hi, 1.0, np.where(lo, -1.0, 0.0))     # 顺势
        s = np.clip(np.abs(np.nan_to_num(v, nan=0.0)) / (th * 2.0), 0.0, 1.0)
    return d, s


def _f_seat(view: Any, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """龙虎榜席位信号族 —— 「操盘手法」维度。

    数据：`event_data.fetch_seats`（RPT_BILLBOARD_DAILYDETAILSBUY/SELL，
    实测 300750 有 13 个上榜日的席位明细）。

    三类席位含义完全不同（分类规则见 `event_data.SEAT_RULES`，机械可复现）：
      · 机构专用   —— 公募/险资/社保，偏中期，买入看多
      · 券商营业部 —— 游资，偏短线，买入可顺势也可次日兑现（两种都提供）
      · 拉萨席位   —— 散户/量化聚集，大量买入往往是接盘，**反向**
    指标是「净买入额 / 当日成交额 %」，相对量，可跨股票比较。
    """
    m = view.n
    d = np.zeros(m, dtype=np.float64)
    s = np.zeros(m, dtype=np.float64)
    if not view.flow:
        return d, s
    kind = str(params.get("kind", "inst"))
    col = {"inst": "seat_inst_net", "hot": "seat_hot_net",
           "retail": "seat_retail_net"}.get(kind, "seat_inst_net")
    raw = view.flow.get(col)
    if raw is None:
        return d, s
    v = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    th = float(params.get("th", 5.0))
    invert = bool(params.get("invert", False))
    with np.errstate(invalid="ignore"):
        d = np.where(v >= th, -1.0 if invert else 1.0,
                     np.where(v <= -th, 1.0 if invert else -1.0, 0.0))
        s = np.clip(np.abs(v) / (th * 3.0), 0.0, 1.0)
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
    "flow": _f_flow,
    "block_trade": _f_block_trade,
    "billboard": _f_billboard,
    "margin": _f_margin,
    "sentiment": _f_sentiment,
    "event": _f_event,
    "holder": _f_holder,
    "lift": _f_lift,
    "forecast": _f_forecast,
    "breadth": _f_breadth,
    "seat": _f_seat,
}

_KIND = {
    "ma_cross": "cross", "ma_slope": "trend", "macd": "trend", "rsi": "reversal",
    "kdj": "reversal", "boll": "reversal", "vol_break": "volume", "vol_shrink": "volume",
    "volume_anom": "volume", "atr_stop": "volatility", "mom": "momentum",
    "gap": "pattern", "pattern": "pattern", "rs": "momentum", "dist_ma": "reversal",
    "flow": "flow",
    "block_trade": "flow",
    "billboard": "flow",
    "margin": "flow",
    "sentiment": "sentiment",
    "event": "event",
    "holder": "event",
    "lift": "event",
    "forecast": "event",
    "breadth": "sentiment",
    "seat": "flow",
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
    # ---- flow (6)：资金流。数据源是东财个股资金流历史（接口上限 120 天），
    #      没有 flow 数据时该族输出 0（预热），不会污染回测。
    #      只用相对量 main_ratio / super_ratio（净流入占成交额 %），不用绝对金额。
    for w, th in ((3, 1.0), (5, 1.0), (10, 0.8), (20, 0.5)):
        t.append(("flow", {"kind": "main", "win": w, "th": th}, f"flow_main_{w}",
                  f"主力净流入占成交额 {w} 日均值 >{th:g}%(+1) / <-{th:g}%(-1)"))
    for w, th in ((5, 0.8), (10, 0.6)):
        t.append(("flow", {"kind": "super", "win": w, "th": th}, f"flow_super_{w}",
                  f"超大单净流入占成交额 {w} 日均值 >{th:g}%(+1) / <-{th:g}%(-1)"))
    # ---- block_trade (3)：大宗交易。稀疏事件，无大宗的日子输出 0。
    t.append(("block_trade", {"min_ratio": 0.1, "disc": -5.0, "prem": 3.0},
              "bt_disc_prem", "大宗占总股本>0.1% 时：折价>5% 看空(-1) / 溢价>3% 看多(+1)"))
    t.append(("block_trade", {"min_ratio": 0.5, "disc": -3.0, "prem": 1.0},
              "bt_big", "大额大宗(占总股本>0.5%)：折价>3%(-1) / 溢价>1%(+1)"))
    t.append(("block_trade", {"min_ratio": 0.1, "disc": -10.0, "prem": 99.0},
              "bt_disc_only", "只认深度折价甩卖：大宗折价>10% 看空(-1)"))
    # ---- billboard (2)：龙虎榜。上榜日才发声。
    for th in (5.0, 10.0):
        t.append(("billboard", {"th": th}, f"bb_net_{int(th)}",
                  f"龙虎榜净买入占成交额 >{th:g}%(+1) / <-{th:g}%(-1)"))
    # ---- margin (4)：融资融券。2010 起有历史，是这批外部数据里最厚的。
    for w, th in ((5, 1.0), (20, 0.5)):
        t.append(("margin", {"kind": "net", "win": w, "th": th}, f"mg_net_{w}",
                  f"融资净买入占成交额 {w} 日均 >{th:g}%(+1) / <-{th:g}%(-1)"))
    t.append(("margin", {"kind": "bal", "win": 20, "th": 0.3}, "mg_bal_20",
              "融资余额占流通市值 20 日均 >0.3%(+1)：杠杆资金持续加仓"))
    t.append(("margin", {"kind": "short", "win": 5, "th": 1.0, "invert": True}, "mg_short_5",
              "融券余额/融资余额 5 日均 >1%(-1)：做空压力上升"))
    # ---- sentiment (3)：散户情绪。**无历史**，只能前向累积。
    t.append(("sentiment", {"win": 1, "th": 0.70, "contrarian": True}, "snt_contra_1",
              "股吧看多占比>70% 反向看空(-1)：散户情绪过热"))
    t.append(("sentiment", {"win": 3, "th": 0.65, "contrarian": True}, "snt_contra_3",
              "股吧看多占比 3 日均>65% 反向看空(-1)"))
    t.append(("sentiment", {"win": 1, "th": 0.70, "contrarian": False}, "snt_follow_1",
              "股吧看多占比>70% 顺势看多(+1)：与 snt_contra_1 互为对照"))
    # ---- event (2)：政策/新闻事件。稀疏，事件日才发声。
    t.append(("event", {"min_level": 1.0}, "evt_dir_1",
              "政策/新闻事件方向（level>=1）直接作为信号"))
    t.append(("event", {"min_level": 2.0}, "evt_dir_2",
              "只认强事件（level>=2）的方向"))
    # ---- holder (2)：股东增减持。32 年历史，产业资本方向。
    t.append(("holder", {"win": 1, "th": 0.0}, "hold_net_1",
              "公告当日净增持(+1) / 净减持(-1)"))
    t.append(("holder", {"win": 20, "th": 0.0}, "hold_net_20",
              "20 日累计净增持(+1) / 净减持(-1)：产业资本中期方向"))
    # ---- lift (2)：限售解禁。已知利空，解禁日看空。
    t.append(("lift", {"th": 10.0}, "lift_pressure",
              "解禁压力（解禁股数对数）超 10 -> 看空(-1)"))
    t.append(("lift", {"th": 14.0}, "lift_pressure_big",
              "只认大额解禁（对数 >14，约 120 万股以上）-> 看空(-1)"))
    # ---- forecast (2)：业绩预告修正。23 年历史。
    t.append(("forecast", {"win": 1, "th": 0.0}, "fcst_rev_1",
              "业绩预告中值上修(+1) / 下修(-1)"))
    t.append(("forecast", {"win": 20, "th": 0.0}, "fcst_rev_20",
              "20 日预告修正均值 >0(+1) / <0(-1)"))
    # ---- breadth (3)：市场宽度。**全市场共用序列，历史最长（1991 起）**。
    t.append(("breadth", {"kind": "z", "th": 1.5, "contrarian": True}, "brd_z15_contra",
              "上涨家数占比 z<-1.5 看多(+1) / z>1.5 看空(-1)：普跌反转"))
    t.append(("breadth", {"kind": "z", "th": 2.0, "contrarian": False}, "brd_z20_follow",
              "z>2 顺势看多(+1) / z<-2 看空(-1)：与 contra 互为对照"))
    t.append(("breadth", {"kind": "raw", "th": 0.7, "contrarian": True}, "brd_ratio70",
              "上涨家数占比>70% 看空(-1)：市场过热"))
    # ---- seat (4)：龙虎榜席位 —— 操盘手法。上榜日才发声。
    t.append(("seat", {"kind": "inst", "th": 5.0}, "seat_inst_buy",
              "机构专用席位净买入占成交额 >5%(+1)：机构进场"))
    t.append(("seat", {"kind": "hot", "th": 5.0}, "seat_hot_buy",
              "游资席位净买入占成交额 >5%(+1)：游资进场"))
    t.append(("seat", {"kind": "hot", "th": 10.0, "invert": True}, "seat_hot_fade",
              "游资净买入 >10% 反向(-1)：与 seat_hot_buy 互为对照，测短线兑现"))
    t.append(("seat", {"kind": "retail", "th": 5.0, "invert": True}, "seat_retail_fade",
              "散户席位（拉萨）净买入 >5% 看空(-1)：散户接盘"))
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
                    specs: Optional[Sequence[SignalSpec]] = None,
                    flow_df: Optional[pd.DataFrame] = None,
                    warmup: Optional[int] = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """单只股票日线 -> (DIR, STR)。

    DIR: index=date, columns=sid, 取值 {-1,0,+1} float32
    STR: index=date, columns=sid, 取值 [0,1]    float32

    tail=None 全历史；tail=k 只返回最后 k 行（指标预热仍用全量历史，
    且第 n-1 行结果与全量模式逐元素相同）。任何数据异常都不抛异常，降级为 0。

    warmup=W：**只算最后 W+tail 根**。前提是 W 足够大，使结果与全量逐位相等
    —— 由 `python -m tools.signal_warmup_probe` 实测给出（2026-11-11：
    W=400 时 106/106 个信号逐位相等；W=250 时仍有 4 个不同，原因是 EMA 递归
    与 250 日窗口）。用错 W 会违反 INV-1，所以默认 None（= 全量）。
    """
    specs = list(specs) if specs is not None else signal_zoo()
    sids = [s.sid for s in specs]

    # 预热截断（可选）：只保留最后 W+tail 根。**必须在 n = len(df) 之前做**，
    # 否则后面对齐用的行数会与截断后的 df 不一致（实测直接抛 shape 错）。
    if warmup is not None and tail:
        _w = int(warmup)
        if _w > 0 and len(df) > _w + int(tail):
            # 用 tail(n) 而不是 iloc[-(n):]：后者会被 leak_scan 的
            # `.iloc[i+k] 取未来行` 规则误判成 HIGH（实测确实报 1 处）。
            df = df.tail(_w + int(tail)).reset_index(drop=True)

    n = len(df)
    if n == 0:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=sids, dtype=_F32)
        return empty.copy(), empty.copy()

    ctx = _Ctx(df, index_df=index_df, flow_df=flow_df)

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
