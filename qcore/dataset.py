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
def _health_pick(need: str) -> Optional[str]:
    """从 `data/state/datasource_health.json` 读上次探测选中的源（不联网）。

    为什么读缓存而不是每次探测：探测要打 3~5 个接口，300 只票就是上千次请求。
    健康表由 `python -m qcore.datasource` 定期刷新，回测只读它。
    没有健康表时返回 None，调用方走内置默认源（向后兼容）。
    """
    try:
        import json as _json
        p = os.path.join(ROOT, "data", "state", "datasource_health.json")
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as f:
            d = _json.load(f)
        rec = ((d.get("picked") or {}).get(need) or {})
        return str(rec.get("source") or "") or None
    except Exception:  # noqa: BLE001
        return None


#: 与 code 无关的全局表（市场宽度、全市场事件）进程内缓存。
#: 重要：`external_loader` 是**逐票**调用的（300 只），而这两张表是全市场共用。
#: 不做缓存的话，300 次 `pd.read_csv(market_breadth.csv 650KB)` + 300 次
#: 40KB jsonl 解析，会把一次 pipeline 从 ~100s 拖到十几分钟（实测）。
_GLOBAL_CACHE = {"ts": 0.0, "breadth": None, "events": None}
#: 「数据日 → 可用日」的 T+1 平移缓存。
#: 每只票的事件表都有 4000+ 个日期，逐个调 `next_trading_day()` 是 4054 次
#: 日历查询 × 300 只 = 120 万次（实测单票 1.6s，R54 的真凶）。
#: 日期在各票之间是重复的，缓存后单票只剩 dict 查找。
_SHIFT_CACHE: Dict[str, object] = {}
_GLOBAL_TTL = 60.0


def _global_frames():
    import time as _t
    now = _t.time()
    if _GLOBAL_CACHE["ts"] and (now - _GLOBAL_CACHE["ts"]) < _GLOBAL_TTL:
        return _GLOBAL_CACHE["breadth"], _GLOBAL_CACHE["events"]
    breadth = events = None
    try:
        from . import market_breadth as MB
        bw = MB.load()
        if bw is not None and len(bw):
            bw = bw.copy()
            bw["date"] = pd.to_datetime(bw["date"]).dt.normalize()
            breadth = bw[["date", "breadth_z", "up_ratio"]]
    except Exception:  # noqa: BLE001
        breadth = None
    try:
        from . import sentiment_store as SS
        ev = SS.load_events("")
        if ev is not None and len(ev):
            ev = ev.copy()
            ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
            events = ev
    except Exception:  # noqa: BLE001
        events = None
    _GLOBAL_CACHE.update({"ts": now, "breadth": breadth, "events": events})
    return breadth, events



def _merge_on_date(frames):
    """按 date 做外连接（numpy 版）。

    为什么不用 `pd.merge(..., how="outer")`：`external_loader` 是**逐票**调用，
    每票都要把 4–5 张表合并一遍，pandas 的 merge 会排序+哈希+重排索引，
    实测 120 只票 6.2s → >120s（评审 R54）。
    这里改成"日期取并集 + searchsorted 定位"，语义不变（每日期一行、缺值 NaN），
    但只做一次排序与一次并集。
    """
    import numpy as np

    parts = {}
    for f in frames:
        if f is None or not len(f) or "date" not in f.columns:
            continue
        days = pd.to_datetime(f["date"]).dt.normalize().to_numpy()
        days = days.astype("datetime64[D]").astype(np.int64)
        _u, idx = np.unique(days, return_index=True)       # 同一天多行时保留第一行
        for c in f.columns:
            if c == "date":
                continue
            try:
                vals = pd.to_numeric(f[c], errors="coerce").to_numpy(dtype=np.float64)[idx]
            except Exception:  # noqa: BLE001
                continue
            parts[str(c)] = (_u, vals)
    if not parts:
        return None
    all_days = np.unique(np.concatenate([d for d, _ in parts.values()]))
    out = {"date": pd.to_datetime(all_days.astype("datetime64[D]"))}
    for c, (d, v) in parts.items():
        arr = np.full(all_days.size, np.nan, dtype=np.float64)
        arr[np.searchsorted(all_days, d)] = v
        out[c] = arr
    return pd.DataFrame(out)


def external_loader(code: str, cache_only: bool = True):
    """把「资金流 + 事件」合并成一张按日索引的外部数据表，供 compute_signals 用。

    三者都是**外部数据**，取不到就返回 None（对应信号族输出 0，绝不报错）。
    资金流只有 120 天（东财接口上限），事件数据有多年（实测 600519 272 笔跨 19 年）。
    """
    frames = []
    # 选源：有健康表就按它（`python -m qcore.datasource` 定期刷新），
    # 没有就用内置默认 —— 保证任何部署都能跑，不会因为缺健康表就崩。
    src = {"flow": _health_pick("flow") or "eastmoney-fflow",
           "event": _health_pick("block_trade") or "eastmoney-blocktrade"}
    try:
        from . import flow_factor as FF
        # 资金流：cache_only 时只用磁盘缓存（读不到就跳过，绝不联网）。
        raw = None
        if cache_only:
            try:
                raw = FF._cache_read(code)
            except Exception:  # noqa: BLE001
                raw = None
            if raw is not None and len(raw):
                raw = raw[FF.PUBLIC_COLUMNS].copy()
        else:
            raw = FF.fetch_flow(code)
        if raw is not None and len(raw):
            f = raw.copy()
            f["date"] = pd.to_datetime(f["date"]).dt.normalize()
            # fetch_flow 已带 main_ratio（主力净流入占成交额 %）；
            # super_ratio 不一定存在，有就带上，没有 _f_flow 会回退到 main_ratio。
            keep = ["date", "main_ratio"] + (["super_ratio"] if "super_ratio" in f.columns else [])
            frames.append(f[keep])
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import event_data as ED
        # cache_only=True：**回测不联网**。没缓存就返回空表（信号族输出 0）。
        # 需要更新缓存时用 `python -m qcore.event_data --refresh` 或定时任务。
        e = ED.load_events(code, cache_only=cache_only)
        if e is not None and len(e):
            e = e.copy()
            e["date"] = pd.to_datetime(e["date"]).dt.normalize()
            keep = ["date"] + [c for c in ("bt_shares_ratio", "bt_premium",
                                           "bb_net_ratio", "mg_net_ratio",
                                           "mg_bal_ratio", "mg_short_ratio",
                                           "hold_net_ratio", "fc_lift_ratio",
                                           "eps_revision", "seat_inst_net",
                                           "seat_hot_net", "seat_retail_net")
                               if c in e.columns]
            e = e[keep]
            # 两融/大宗/龙虎榜都是 **T+1 披露**（T 日盘后公布 T 日数据，
            # 但你在 T 日盘中拿不到）。信号在 t 日只能用 t-1 及更早的值，
            # 否则就是用明天的数据做今天的决策（INV-1）。
            # 做法：把这些列的日期整体后移 1 个交易日，让「数据日」变成「可用日」。
            _lag = [c for c in keep if c != "date"]
            if _lag:
                _d = pd.to_datetime(e["date"]).dt.normalize()
                try:
                    from . import calendar_cn as _CAL
                    _shift = {}
                    for d in _d.dropna().unique():
                        _ts = pd.Timestamp(d)
                        _k = str(_ts.date())
                        _v = _SHIFT_CACHE.get(_k)
                        if _v is None:
                            _v = pd.Timestamp(_CAL.next_trading_day(_ts.date()))
                            _SHIFT_CACHE[_k] = _v
                        _shift[_ts] = _v
                    e = e.copy()
                    e["date"] = _d.map(_shift)
                except Exception:  # noqa: BLE001
                    e = e.copy()
                    e["date"] = _d + pd.Timedelta(days=1)
            frames.append(e)
    except Exception:  # noqa: BLE001
        pass
    # 散户情绪：只读已落盘快照（绝不联网）。没有快照时为空，信号族输出 0。
    try:
        from . import sentiment_store as SS
        sn = SS.load_sentiment(code)
        if sn is not None and len(sn):
            sn = sn.copy()
            sn["date"] = pd.to_datetime(sn["date"]).dt.normalize()
            frames.append(sn)
    except Exception:  # noqa: BLE001
        pass
    # 市场宽度 + 全市场事件：与 code 无关 → 走进程内缓存，绝不逐票重读
    _bw, _ev = _global_frames()
    if _bw is not None:
        frames.append(_bw)
    # 政策/新闻事件：同一张全局表（缓存），仍按 code 与板块过滤
    if _ev is not None:
        frames.append(_ev)
    if not frames:
        return None
    out = _merge_on_date(frames)
    if out is None:
        return None
    # 把实际选用的数据源挂到 attrs，便于事后审计「这批数字是哪来的」
    out.attrs["sources"] = src
    return out


def build_dataset(codes: List[str], start: str = "2013-01-01", horizon: int = 5,
                  signal_warmup: Optional[int] = None,
                  extra_horizons: Optional[List[int]] = None,
                  exec_lag: int = 1, str_dtype: str = "uint8", n_univ: int = 100,
                  chunk: int = 25, cutoff: Optional[str] = None,
                  verbose: bool = True, loader=None, pad: bool = True,
                  market: str = "CN", extra_loader=None,
                  live_quotes: Optional[dict] = None) -> Panel3D:
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
        # 实时行情补丁：盘中把**今天这根未收盘 bar** 的收盘/高低/量换成实时快照，
        # 让信号反映当前状态，而不是昨天收盘。
        #
        # 三个约束：
        #   1. 只改「日期 == 今天」的那一根，历史 bar 一个字节都不动
        #   2. 只改价格类字段，不改日期轴（不能凭空多一根 bar）
        #   3. 只用于 live/paper 路径 —— 回测永远不传 live_quotes（INV-1 不受影响）
        if live_quotes:
            q = live_quotes.get(str(code).zfill(6))
            if q and q.get("price"):
                try:
                    _d = pd.to_datetime(df["date"])
                    if len(df):
                        # 打的是**最后一根** bar，不是「日期==今天」的那根。
                        # 原因：`_strip_forming_bar` 盘中会把「今天未收盘」那根丢掉
                        # （为了不让部分 bar 污染信号），于是盘中根本没有今天的行 ——
                        # 按日期匹配会永远匹配不到，补丁静默失效（实测 DIR 差异 0 格）。
                        # 改成打最后一根，并校验它离今天不超过 1 个交易日，避免给
                        # 陈旧数据凭空造一个假 bar。
                        _last = _d.iloc[-1]
                        _ok = True
                        try:
                            from . import calendar_cn as _CAL
                            _ref = _CAL.prev_trading_day(pd.Timestamp.now().date())
                            _ok = _last.normalize() >= pd.Timestamp(_ref)
                        except Exception:  # noqa: BLE001
                            _ok = (_last.normalize() >= pd.Timestamp.now().normalize()
                                   - pd.Timedelta(days=4))
                        if _ok:
                            _i = df.index[-1]
                            _px = float(q["price"])
                            df = df.copy()
                            _old_close = float(df.loc[_i, "close"])
                            df.loc[_i, "close"] = _px
                            for _c in ("high", "low"):
                                if _c in df.columns:
                                    _old = float(df.loc[_i, _c])
                                    if _old > 0:
                                        df.loc[_i, _c] = max(_old, _px) if _c == "high" else min(_old, _px)
                            df.attrs["live_patched"] = {
                                "date": str(_last.date()), "old_close": _old_close,
                                "live": _px}
                except Exception:  # noqa: BLE001
                    pass
        # 外部数据（资金流 / 大宗交易 / 龙虎榜）按票加载；取不到就 None，
        # 对应信号族输出 0（预热），不影响其余信号。
        _extra = None
        if extra_loader is not None:
            try:
                _extra = extra_loader(code)
            except Exception:  # noqa: BLE001
                _extra = None
        # 预热裁剪（评审 R48）：`build_panel(clip=False)` 保留上市以来全部历史，
        # 于是每只票都按 6000+ 根算信号 —— 而引擎只读 start 之后的行。
        # 传 signal_warmup=W 时只保留「start 前 W 根 + start 之后全部」：
        # 由 tools/signal_warmup_probe 实测，W=400 时结果与全量**逐位相等**。
        # 每票独立守卫：start 之前的行数不足 W 就不裁（否则预热不够，信号不可信）。
        _dfs = df
        if signal_warmup:
            try:
                _n_from = int((df["date"] < pd.Timestamp(start)).sum())
                _w = int(signal_warmup)
                if _n_from >= _w:
                    _dfs = df.iloc[_n_from - _w:].reset_index(drop=True)
            except Exception:  # noqa: BLE001
                _dfs = df
        try:
            dir_df, str_df = compute_signals(_dfs, flow_df=_extra)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [signal-fail] {code}: {e}")
            continue
        # 映射必须用**裁剪后**的 _dfs：dir_df 只有 _dfs 那些行，
        # 拿全量 df 的日期做掩码会 shape 不匹配（实测 IndexError 8515 vs 3761）。
        pos = np.searchsorted(dates, _dfs["date"].to_numpy(), side="left")
        ok = (pos < T) & (dates[np.clip(pos, 0, T - 1)] == _dfs["date"].to_numpy())
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
