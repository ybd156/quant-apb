"""qcore/intraday_signals.py —— 分钟级微观结构信号（严格因果，预热 ≤30 根）

⚠️ 数据现实：东财 `klt=1` 只返回**当日 240 根**分钟线（09:31–15:00），无历史。
   因此本模块只能做**前向验证**（今天算、明天看结果），**永远不能回测**。
   任何"用今天的数据回测今天"都是自欺 —— 分钟级回测在本项目里被物理禁止。

设计约束（与 qcore/signals.py 同源）：
  1. 第 t 根 bar 只依赖 ≤t 的 bar；滚动量全部前缀累加型，截断序列逐位相同。
  2. 预热期不足输出 0，不传播 NaN。
  3. 不抛异常：空表 / 缺列 / 零成交量 / 除零一律降级为 0。
  4. 全部阈值都是相对量：z 分数、量比、区间分位、√t 缩放的动量。
     分钟线没有"绝对价格"概念（不同股票的价位差 3 个数量级），任何绝对阈值都会失效。

信号清单（13 个）：
  i_mom_1 / i_rev_1        1 分钟动量 / 反转
  i_mom_5 / i_rev_5        5 分钟动量 / 反转
  i_mom_15                 15 分钟动量
  i_vwap_dev_z             VWAP 偏离 z 分数（均值回复）
  i_vwap_side              价格在 VWAP 上下方（趋势状态）
  i_vol_burst_5/15         量能爆发（量比阈值，方向跟随当根 bar）
  i_range_pos              当日区间位置（极端位置反向）
  i_run_3                  连续同向 bar 数（≥3 连 → 反向）
  i_open_drive             相对开盘价的驱动强度
  i_lunch_mom              午盘效应（13:00 之后的动量）

标准化口径：以 1 分钟收益的滚动标准差 `sd1`（窗口 20）为基础单位，
n 根动量按 √n 缩放（平方根时间法则），z = r_n / (sd1·√n)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .signals import _emit, _roll_mean, _roll_std, _sdiv, _shift

__all__ = ["IntradaySignal", "intraday_zoo", "intraday_signals", "INTRADAY_SIDS"]

_WARM = 20          # 基础统计窗口；也是全库最大预热根数（<= 30 的硬约束）
_EPS = 1e-12
_F32 = np.float32


@dataclass(frozen=True)
class IntradaySignal:
    sid: str
    family: str
    note: str


# ----------------------------------------------------------------------
# 上下文：分钟线的全部基础序列（一次构建）
# ----------------------------------------------------------------------
class _Ctx:
    __slots__ = ("n", "index", "o", "h", "l", "c", "v", "amt", "mod",
                 "ret1", "sd1", "vwap", "dev", "dev_sd", "hi", "lo", "run", "run_sign")

    def __init__(self, df: pd.DataFrame):
        n = len(df)
        self.n = n

        def col(name: str) -> np.ndarray:
            if name not in df.columns:
                return np.full(n, np.nan, dtype=np.float64)
            a = df[name].to_numpy()
            if a.dtype.kind in "fiub":
                return a.astype(np.float64, copy=False)
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)

        if "datetime" in df.columns:
            tcol = "datetime"
        elif "date" in df.columns:
            tcol = "date"
        else:
            tcol = df.columns[0] if len(df.columns) else None
        if tcol is not None:
            try:
                dt = pd.DatetimeIndex(pd.to_datetime(df[tcol]))
            except Exception:          # noqa: BLE001
                dt = pd.DatetimeIndex(pd.RangeIndex(n))
        else:
            dt = pd.DatetimeIndex(pd.RangeIndex(n))
        self.index = dt
        self.mod = (dt.hour * 60 + dt.minute).to_numpy(dtype=np.int64) if n else np.zeros(0, np.int64)

        self.o = col("open")
        self.c = col("close")
        self.h = col("high")
        self.l = col("low")
        self.v = col("volume")
        self.amt = col("amount")
        # 缺 high/low 时退化为 close（形态类信号自然归零，不抛异常）
        self.h = np.where(np.isfinite(self.h), self.h, self.c)
        self.l = np.where(np.isfinite(self.l), self.l, self.c)

        c = self.c
        self.ret1 = _sdiv(c - _shift(c, 1), _shift(c, 1), 0.0)
        self.ret1[0] = 0.0
        self.sd1 = _roll_std(self.ret1, _WARM)
        # 首 20 根的 sd 用扩张标准差兜底（仍然只含过去）
        if n > 2:
            exp_sd = pd.Series(self.ret1).expanding(min_periods=2).std(ddof=0).to_numpy()
            self.sd1 = np.where(np.isfinite(self.sd1), self.sd1, exp_sd)

        # 当日累计 VWAP（成交量单位：手 = 100 股）
        vv = np.where(np.isfinite(self.v) & (self.v > 0), self.v, 0.0)
        aa = np.where(np.isfinite(self.amt) & (self.amt > 0), self.amt, 0.0)
        cum_v = np.cumsum(vv) * 100.0
        cum_a = np.cumsum(aa)
        self.vwap = np.where(cum_v > 0, _sdiv(cum_a, cum_v, 0.0), np.nan)
        self.dev = _sdiv(c - self.vwap, self.vwap, 0.0)
        self.dev = np.where(np.isfinite(self.vwap), self.dev, np.nan)
        self.dev_sd = _roll_std(np.nan_to_num(self.dev, nan=0.0), _WARM)

        # 当日至今区间（扩张极值，前缀型 => 因果）
        self.hi = np.maximum.accumulate(np.nan_to_num(self.h, nan=-np.inf)) if n else np.zeros(0)
        self.lo = np.minimum.accumulate(np.nan_to_num(self.l, nan=np.inf)) if n else np.zeros(0)

        # 连续同向 bar 数（游程长度）
        sgn = np.sign(np.nan_to_num(self.ret1, nan=0.0))
        chg = np.empty(n, dtype=bool)
        if n:
            chg[0] = True
            chg[1:] = sgn[1:] != sgn[:-1]
            grp = np.cumsum(chg) - 1
            start = np.searchsorted(grp, grp)
            self.run = (np.arange(n) - start + 1).astype(np.float64)
        else:
            self.run = np.zeros(0)
        self.run_sign = sgn


# ----------------------------------------------------------------------
# 各信号实现
# ----------------------------------------------------------------------
def _z(ctx: _Ctx, ret: np.ndarray, nbar: int) -> np.ndarray:
    """nbar 根动量按 √n 缩放成 z（sd1 为基础单位）。"""
    sd = ctx.sd1 * np.sqrt(max(nbar, 1))
    z = _sdiv(ret, sd, 0.0)
    z = np.where(np.isfinite(sd) & (sd > _EPS), z, np.nan)
    return z


def _s_mom(ctx: _Ctx, p: dict):
    n = int(p["n"])
    r = _sdiv(ctx.c - _shift(ctx.c, n), _shift(ctx.c, n), 0.0)
    r = np.where(np.isfinite(_shift(ctx.c, n)), r, np.nan)
    z = _z(ctx, r, n)
    d = np.sign(np.nan_to_num(z, nan=0.0))
    d = np.where(np.abs(np.nan_to_num(z, nan=0.0)) >= float(p.get("band", 0.8)), d, 0.0)
    s = np.clip(np.abs(np.nan_to_num(z, nan=0.0)) / 3.0, 0.0, 1.0)
    return d, s


def _s_rev(ctx: _Ctx, p: dict):
    d, s = _s_mom(ctx, p)
    return -d, s


def _s_vwap_dev_z(ctx: _Ctx, p: dict):
    band = float(p.get("band", 1.0))
    z = _sdiv(np.nan_to_num(ctx.dev, nan=0.0), ctx.dev_sd, 0.0)
    z = np.where(np.isfinite(ctx.dev_sd) & (ctx.dev_sd > _EPS), z, np.nan)
    zz = np.nan_to_num(z, nan=0.0)
    d = np.where(zz <= -band, 1.0, np.where(zz >= band, -1.0, 0.0))   # 偏离越远越赌回归
    s = np.clip(np.abs(zz) / 3.0, 0.0, 1.0)
    return d, s


def _s_vwap_side(ctx: _Ctx, p: dict):
    dev = np.nan_to_num(ctx.dev, nan=0.0)
    band = float(p.get("band", 0.0005))
    d = np.where(dev > band, 1.0, np.where(dev < -band, -1.0, 0.0))
    s = np.clip(_sdiv(np.abs(dev), np.maximum(np.nan_to_num(ctx.dev_sd, nan=0.0), _EPS), 0.0) / 3.0, 0.0, 1.0)
    return d, s


def _s_vol_burst(ctx: _Ctx, p: dict):
    n, th = int(p["n"]), float(p["th"])
    vv = np.nan_to_num(ctx.v, nan=0.0)
    base = _shift(_roll_mean(vv, n), 1)
    vr = _sdiv(vv, base, 0.0)
    vr = np.where(np.isfinite(base) & (base > _EPS), vr, np.nan)
    vv_ok = np.nan_to_num(vr, nan=0.0)
    hit = vv_ok > th
    d = np.where(hit, np.sign(np.nan_to_num(ctx.ret1, nan=0.0)), 0.0)
    s = np.clip((vv_ok - th) / max(th, _EPS) * 0.5 + 0.5, 0.0, 1.0)
    return d, s


def _s_range_pos(ctx: _Ctx, p: dict):
    lo = np.nan_to_num(ctx.lo, nan=0.0)
    hi = np.nan_to_num(ctx.hi, nan=0.0)
    pos = _sdiv(ctx.c - lo, hi - lo, 0.5)          # 0=区间最低 1=区间最高
    pos = np.where(np.isfinite(hi - lo) & ((hi - lo) > _EPS), pos, np.nan)
    pp = np.clip(np.nan_to_num(pos, nan=0.5), 0.0, 1.0)
    hi_th = float(p.get("hi", 0.85))
    lo_th = float(p.get("lo", 0.15))
    d = np.where(pp <= lo_th, 1.0, np.where(pp >= hi_th, -1.0, 0.0))
    s = np.clip(np.abs(pp - 0.5) * 2.0, 0.0, 1.0)
    return d, s


def _s_run(ctx: _Ctx, p: dict):
    k = float(p.get("k", 3))
    run = ctx.run
    sgn = ctx.run_sign
    d = np.where((sgn > 0) & (run >= k), -1.0, np.where((sgn < 0) & (run >= k), 1.0, 0.0))
    s = np.clip(run / (k * 2.0), 0.0, 1.0)
    return d, s


def _s_open_drive(ctx: _Ctx, p: dict):
    if ctx.n == 0:
        return np.zeros(0), np.zeros(0)
    o0 = ctx.o[0]
    if not np.isfinite(o0) or o0 <= 0:
        return np.zeros(ctx.n), np.zeros(ctx.n)
    drive = _sdiv(ctx.c - o0, o0, 0.0)
    nb = np.arange(1, ctx.n + 1, dtype=np.float64)
    z = _sdiv(drive, ctx.sd1 * np.sqrt(nb), 0.0)
    z = np.where(np.isfinite(ctx.sd1) & (ctx.sd1 > _EPS), z, np.nan)
    zz = np.nan_to_num(z, nan=0.0)
    band = float(p.get("band", 0.8))
    d = np.sign(zz) * (np.abs(zz) >= band)
    s = np.clip(np.abs(zz) / 3.0, 0.0, 1.0)
    return d, s


def _s_lunch_mom(ctx: _Ctx, p: dict):
    n = ctx.n
    if n == 0:
        return np.zeros(0), np.zeros(0)
    start_mod = int(p.get("start_mod", 13 * 60))
    idx = np.searchsorted(ctx.mod, start_mod, side="left")
    d = np.zeros(n)
    s = np.zeros(n)
    if idx >= n or idx <= 0:
        return d, s
    base = ctx.c[idx]
    if not np.isfinite(base) or base <= 0:
        return d, s
    i = np.arange(n)
    mom = _sdiv(ctx.c - base, base, 0.0)
    nb = np.maximum(i - idx, 1).astype(np.float64)
    z = _sdiv(mom, ctx.sd1 * np.sqrt(nb), 0.0)
    z = np.where((i > idx) & np.isfinite(ctx.sd1) & (ctx.sd1 > _EPS), z, 0.0)
    zz = np.nan_to_num(z, nan=0.0)
    band = float(p.get("band", 0.8))
    d = np.sign(zz) * (np.abs(zz) >= band)
    s = np.clip(np.abs(zz) / 3.0, 0.0, 1.0)
    return d, s


_IMPL: Dict[str, Callable[[_Ctx, dict], Tuple[np.ndarray, np.ndarray]]] = {
    "mom": _s_mom,
    "rev": _s_rev,
    "vwap_dev_z": _s_vwap_dev_z,
    "vwap_side": _s_vwap_side,
    "vol_burst": _s_vol_burst,
    "range_pos": _s_range_pos,
    "run": _s_run,
    "open_drive": _s_open_drive,
    "lunch_mom": _s_lunch_mom,
}

_ZOO: List[IntradaySignal] = [
    IntradaySignal("i_mom_1", "mom", "1 分钟动量：顺势跟随上一根 bar 方向"),
    IntradaySignal("i_rev_1", "rev", "1 分钟反转：赌上一根 bar 反向修复"),
    IntradaySignal("i_mom_5", "mom", "5 分钟动量：短周期趋势延续"),
    IntradaySignal("i_rev_5", "rev", "5 分钟反转：短周期超买超卖修复"),
    IntradaySignal("i_mom_15", "mom", "15 分钟动量：更长的日内趋势"),
    IntradaySignal("i_vwap_dev_z", "vwap_dev_z", "价格偏离当日 VWAP 的 z 分数，偏离越远越赌回归"),
    IntradaySignal("i_vwap_side", "vwap_side", "价格站在 VWAP 上方(+1)/下方(-1)，趋势状态"),
    IntradaySignal("i_vol_burst_5", "vol_burst", "量比(前5根均量)>2 视为量能爆发，方向跟随当根"),
    IntradaySignal("i_vol_burst_15", "vol_burst", "量比(前15根均量)>2.5 视为量能爆发"),
    IntradaySignal("i_range_pos", "range_pos", "当日区间位置：贴近上沿(-1)/下沿(+1)反向"),
    IntradaySignal("i_run_3", "run", "连续 3 根同向 bar 后反向"),
    IntradaySignal("i_open_drive", "open_drive", "相对当日开盘价的驱动强度(√t 标准化)"),
    IntradaySignal("i_lunch_mom", "lunch_mom", "午盘效应：13:00 之后的动量"),
]

_PARAMS: Dict[str, dict] = {
    "i_mom_1": {"n": 1, "band": 0.8},
    "i_rev_1": {"n": 1, "band": 0.8},
    "i_mom_5": {"n": 5, "band": 0.8},
    "i_rev_5": {"n": 5, "band": 0.8},
    "i_mom_15": {"n": 15, "band": 0.8},
    "i_vwap_dev_z": {"band": 1.0},
    "i_vwap_side": {"band": 0.0005},
    "i_vol_burst_5": {"n": 5, "th": 2.0},
    "i_vol_burst_15": {"n": 15, "th": 2.5},
    "i_range_pos": {"hi": 0.85, "lo": 0.15},
    "i_run_3": {"k": 3},
    "i_open_drive": {"band": 0.8},
    "i_lunch_mom": {"band": 0.8, "start_mod": 13 * 60},
}

INTRADAY_SIDS: Tuple[str, ...] = tuple(s.sid for s in _ZOO)


def intraday_zoo() -> List[IntradaySignal]:
    """返回分钟级信号清单（13 个）。"""
    return list(_ZOO)


def intraday_signals(df: pd.DataFrame,
                     specs: Optional[Sequence[IntradaySignal]] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """分钟线 -> (DIR, STR)，index=datetime，columns=sid，float32。

    DIR ∈ {-1,0,1}，STR ∈ [0,1]；预热期输出 0；任何数据异常降级为 0，不抛异常。
    """
    zoo = list(specs) if specs is not None else _ZOO
    sids = [s.sid for s in zoo]
    n = len(df)
    if n == 0:
        e = pd.DataFrame(index=pd.DatetimeIndex([]), columns=sids, dtype=_F32)
        return e.copy(), e.copy()

    ctx = _Ctx(df)
    out_d = np.zeros((n, len(zoo)), dtype=_F32)
    out_s = np.zeros((n, len(zoo)), dtype=_F32)
    for j, spec in enumerate(zoo):
        fn = _IMPL.get(spec.family)
        if fn is None:
            continue
        try:
            d, s = fn(ctx, dict(_PARAMS.get(spec.sid, {})))
            d = np.asarray(d, dtype=np.float64).reshape(-1)
            s = np.asarray(s, dtype=np.float64).reshape(-1)
            if d.size != n or s.size != n:
                continue
        except Exception:          # noqa: BLE001  单信号异常不得拖垮整体
            continue
        d, s = _emit(d, s)
        out_d[:, j] = d
        out_s[:, j] = s

    DIR = pd.DataFrame(out_d, index=ctx.index, columns=sids)
    STR = pd.DataFrame(out_s, index=ctx.index, columns=sids)
    return DIR, STR
