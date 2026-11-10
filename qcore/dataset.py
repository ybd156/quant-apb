"""
数据集层（性能优化版）—— 为 N100 这类小主机的"持续运行"而设计。

三条性能决策：
  1. **时间主序布局 (T,N,S)**：每天取 `DIR[t]` 是连续内存切片，而不是跨步访问。
     这是逐日重放引擎的关键 —— 5000 天 × 300 股 × 90 信号，跨步访问会慢 3-5 倍。
  2. **位宽压缩**：DIR 存 int8（-1/0/1），STR 存 uint8（0-255 量化）。
     300 股 × 5000 天 × 90 信号 = 1.35 亿格 → int8+uint8 共 270MB，float32 要 1.1GB。
  3. **向量化收益**：收益矩阵一次性算完（T,N），不再逐股循环。

内存预算（300 股 / 12 年 / 90 信号）：DIR 135MB + STR 135MB + 价格类 ~20MB ≈ 300MB。
若内存更紧，传 memmap=True 落盘映射，RSS 可压到 100MB 以内。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import build_panel, load_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Panel3D:
    dates: np.ndarray            # (T,) datetime64[ns]
    codes: List[str]             # (N,)
    sids: List[str]              # (S,)
    DIR: np.ndarray              # (T,N,S) int8   in {-1,0,1}
    STR: np.ndarray              # (T,N,S) uint8  in [0,255]（或 float32，见 str_dtype）
    OPEN: np.ndarray             # (T,N) float32
    CLOSE: np.ndarray            # (T,N) float32
    RET: Dict[int, np.ndarray]   # h -> (T,N) float32
    SIGMA: np.ndarray            # (T,N) float32
    REGIME: np.ndarray           # (T,) int8
    VALID: np.ndarray            # (T,N) bool  可交易（有价、非停牌）
    UNIV: np.ndarray = field(default=None)   # (T,N) bool 当日是否在"点对点股票池"内
    index_close: np.ndarray = field(default=None)
    str_dtype: str = "uint8"
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return len(self.dates), len(self.codes), len(self.sids)

    def str_float(self, t: int) -> np.ndarray:
        """取第 t 天的强度矩阵 (N,S) float32（仅这一天的，内存 O(N*S)）。"""
        a = self.STR[t]
        if a.dtype == np.uint8:
            return a.astype(np.float32) * (1.0 / 255.0)
        return a

    def dir_float(self, t: int) -> np.ndarray:
        return self.DIR[t].astype(np.float32)

    def nbytes(self) -> int:
        tot = self.DIR.nbytes + self.STR.nbytes + self.OPEN.nbytes + \
            self.CLOSE.nbytes + self.SIGMA.nbytes + self.REGIME.nbytes + self.VALID.nbytes
        tot += sum(v.nbytes for v in self.RET.values())
        return tot


# ------------------------------------------------------------------ regime
def build_regime(index_df: pd.DataFrame, dates: np.ndarray,
                 trend_win: int = 60, vol_win: int = 20, vol_pct_win: int = 250) -> np.ndarray:
    """4 档市场状态（全部因果）：0=跌低波 1=跌高波 2=涨低波 3=涨高波"""
    ix = index_df.set_index("date").sort_index()
    close = ix["close"].astype(float)
    ma = close.rolling(trend_win, min_periods=trend_win).mean()
    up = (close > ma) & (ma > ma.shift(5))
    vol = close.pct_change().rolling(vol_win, min_periods=vol_win).std()
    vol_pct = vol.rolling(vol_pct_win, min_periods=vol_win).rank(pct=True)
    reg = (up.astype(int) * 2 + (vol_pct > 0.5).astype(int)).reindex(pd.DatetimeIndex(dates))
    return reg.fillna(0).astype(np.int8).to_numpy()


# ------------------------------------------------------------------ build
def build_dataset(codes: List[str], start: str = "2013-01-01", horizon: int = 5,
                  extra_horizons: Optional[List[int]] = None,
                  exec_lag: int = 1, str_dtype: str = "uint8", n_univ: int = 100,
                  chunk: int = 25, cutoff: Optional[str] = None,
                  verbose: bool = True, loader=None, pad: bool = True,
                  market: str = "CN") -> Panel3D:
    from .signals import compute_signals, signal_zoo

    zoo = signal_zoo()
    sids = [s.sid for s in zoo]
    S = len(sids)

    panel = build_panel(codes, start=start, verbose=verbose, loader=loader, pad=pad)
    if not panel:
        raise RuntimeError("empty panel")
    if cutoff is not None:
        # 泄漏检验的正确底座：在**算任何指标之前**就砍掉未来行情，
        # 这样 RET/rolling/信号全部基于 ≤cutoff 的数据重算。
        cut = pd.Timestamp(cutoff)
        panel = {c: v[v["date"] <= cut].reset_index(drop=True) for c, v in panel.items()}
        panel = {c: v for c, v in panel.items() if len(v) >= 200}
    code_list = sorted(panel.keys())
    N = len(code_list)

    ix = load_index("000300")
    # regime 必须用「完整指数历史」算，再 reindex 到本面板的日期。
    # 旧实现拿 start 之后的切片算 vol_pct(250)，窗口前 250 天全 NaN，
    # build_regime 又 fillna(0) → 增量窗口(900 天)与全量历史在同一历史日
    # 给出不同 regime（实测 57/610 天 = 9.3%），而 regime 决定用哪套权重，
    # 于是「今天的分」≠「回放的分」。
    # 注意：build_regime 内部全部是向后滚动/位移，只用 ≤t 的信息，
    # 因此保留 start 之前的行不会引入未来函数。
    ix_full = ix.copy()
    ix = ix[(ix["date"] >= pd.Timestamp(start))].reset_index(drop=True)
    if cutoff is not None:
        ix = ix[ix["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
    dates = ix["date"].to_numpy()
    T = len(dates)
    didx = pd.DatetimeIndex(dates)

    # ---- 价格矩阵（一次性向量化）
    def _mat(col):
        s = {c: pd.Series(panel[c][col].to_numpy(dtype=np.float64),
                          index=pd.DatetimeIndex(panel[c]["date"])) for c in code_list}
        return pd.DataFrame(s).reindex(didx).to_numpy(dtype=np.float32)

    OPEN = _mat("open")
    CLOSE = _mat("close")
    VOLUME = _mat("volume")
    AMOUNT = _mat("amount")
    traded = np.isfinite(OPEN) & np.isfinite(CLOSE) & (VOLUME > 0)

    # ---- 点对点股票池（防生存者偏差）：过去 60 日平均成交额排名前 n_univ
    #      只用 ≤t 的信息；横截面排名只用到当日，收盘后即可观测。
    amt60 = pd.DataFrame(AMOUNT).rolling(60, min_periods=20).mean().to_numpy()
    rk = pd.DataFrame(amt60).rank(axis=1, ascending=False, method="first").to_numpy()
    UNIV = np.isfinite(rk) & (rk <= n_univ)
    VALID = traded & UNIV
    del AMOUNT, amt60, rk

    # ---- 条件波动率（因果 rolling，整矩阵一次算）
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.vstack([np.full((1, N), np.nan, np.float32),
                         CLOSE[1:] / CLOSE[:-1] - 1.0]).astype(np.float32)
    SIGMA = pd.DataFrame(pct).rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)

    # ---- 收益矩阵：r_h(t) = OPEN[t+lag+h]/OPEN[t+lag] - 1（越界 NaN）
    horizons = sorted({horizon, *(extra_horizons or [])})
    RET: Dict[int, np.ndarray] = {}
    # 停牌缺口掩码：若 [t+exec_lag, t+exec_lag+h] 区间内该股有非交易日（主日历上是 NaN），
    # 说明这段时间停牌过，复牌跳变（实测最大 +622%）不是可交易收益，必须剔除。
    gap = (~traded).astype(np.int32)
    cgap = np.cumsum(gap, axis=0)
    for h in horizons:
        arr = np.full((T, N), np.nan, dtype=np.float32)
        if T > exec_lag + h:
            with np.errstate(invalid="ignore", divide="ignore"):
                arr[:T - exec_lag - h] = OPEN[exec_lag + h:] / OPEN[exec_lag:T - h] - 1.0
            # 区间 (t+exec_lag, t+exec_lag+h] 内的缺口数
            span = cgap[exec_lag + h:] - cgap[exec_lag:T - h]
            n = span.shape[0]
            arr[:n] = np.where(span == 0, arr[:n], np.nan)
        RET[h] = arr

    # ---- 信号矩阵（唯一需要逐股的地方，分块填充以控制峰值内存）
    DIR = np.zeros((T, N, S), dtype=np.int8)
    STR = np.zeros((T, N, S), dtype=np.uint8 if str_dtype == "uint8" else np.float32)

    for n, code in enumerate(code_list):
        df = panel[code]
        try:
            dir_df, str_df = compute_signals(df)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [signal-fail] {code}: {e}")
            continue
        pos = np.searchsorted(dates, df["date"].to_numpy(), side="left")
        ok = (pos < T) & (dates[np.clip(pos, 0, T - 1)] == df["date"].to_numpy())
        pos = pos[ok]
        if len(pos) == 0:
            continue
        d_arr = np.nan_to_num(dir_df.to_numpy(dtype=np.float32), nan=0.0)[ok]
        s_arr = np.clip(np.nan_to_num(str_df.to_numpy(dtype=np.float32), nan=0.0), 0.0, 1.0)[ok]
        DIR[pos, n, :] = np.clip(np.rint(d_arr), -1, 1).astype(np.int8)
        if str_dtype == "uint8":
            STR[pos, n, :] = np.rint(s_arr * 255.0).astype(np.uint8)
        else:
            STR[pos, n, :] = s_arr.astype(np.float32)
        if verbose and (n + 1) % chunk == 0:
            print(f"  ... {n + 1}/{N}", flush=True)

    regime = build_regime(ix_full, dates)
    p3d = Panel3D(
        dates=dates, codes=code_list, sids=sids,
        DIR=DIR, STR=STR, OPEN=OPEN, CLOSE=CLOSE, RET=RET, SIGMA=SIGMA,
        REGIME=regime, VALID=VALID, UNIV=UNIV,
        index_close=ix["close"].to_numpy(dtype=np.float32),
        str_dtype=str_dtype,
        meta={"start": start, "horizon": horizon, "exec_lag": exec_lag,
              "market": str(market).upper(),
              "n_signals": S, "n_stocks": N, "n_days": T, "n_univ": n_univ,
              "date_from": str(pd.Timestamp(dates[0]).date()),
              "date_to": str(pd.Timestamp(dates[-1]).date())},
    )
    return p3d


def truncate(p3d: Panel3D, as_of) -> Panel3D:
    """截断到 as_of（含）。用于截断不变性检验；只切第一维，不重算任何指标。"""
    cut = np.datetime64(pd.Timestamp(as_of))
    k = int(np.searchsorted(p3d.dates, cut, side="right"))
    return Panel3D(
        dates=p3d.dates[:k], codes=p3d.codes, sids=p3d.sids,
        DIR=p3d.DIR[:k], STR=p3d.STR[:k],
        OPEN=p3d.OPEN[:k], CLOSE=p3d.CLOSE[:k],
        RET={h: v[:k] for h, v in p3d.RET.items()},
        SIGMA=p3d.SIGMA[:k], REGIME=p3d.REGIME[:k], VALID=p3d.VALID[:k],
        UNIV=p3d.UNIV[:k] if p3d.UNIV is not None else None,
        index_close=p3d.index_close[:k] if p3d.index_close is not None else None,
        str_dtype=p3d.str_dtype,
        meta={**p3d.meta, "truncated_at": str(pd.Timestamp(cut).date())},
    )


def save_panel(p3d: Panel3D, path: str) -> None:
    """整块落盘（npz，不压缩以换取读取速度；~300MB）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             dates=p3d.dates, codes=np.array(p3d.codes), sids=np.array(p3d.sids),
             DIR=p3d.DIR, STR=p3d.STR, OPEN=p3d.OPEN, CLOSE=p3d.CLOSE,
             SIGMA=p3d.SIGMA, REGIME=p3d.REGIME, VALID=p3d.VALID, UNIV=p3d.UNIV,
             index_close=p3d.index_close,
             ret_keys=np.array(list(p3d.RET.keys())),
             **{f"RET_{h}": v for h, v in p3d.RET.items()})


def load_panel(path: str) -> Panel3D:
    z = np.load(path, allow_pickle=False)
    ret_keys = [int(x) for x in z["ret_keys"]]
    return Panel3D(
        dates=z["dates"], codes=[str(c) for c in z["codes"]],
        sids=[str(s) for s in z["sids"]],
        DIR=z["DIR"], STR=z["STR"], OPEN=z["OPEN"], CLOSE=z["CLOSE"],
        RET={h: z[f"RET_{h}"] for h in ret_keys},
        SIGMA=z["SIGMA"], REGIME=z["REGIME"], VALID=z["VALID"], UNIV=z["UNIV"],
        index_close=z["index_close"], meta={},
    )
