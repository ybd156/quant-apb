#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qcore/intraday_factor.py —— 分钟级（5min）因子，聚合成日频（字段口径已真实请求核实）

## 数据接口（实测 2026-11-08）

    https://push2his.eastmoney.com/api/qt/stock/kline/get
      ?secid=1.600519 &fields1=f1,f2,f3
      &fields2=f51..f56 &klt=5&fqt=2&beg=20240101&end=20500101&lmt=5000

`data.klines` 每行是逗号分隔字符串。**实测 `fields2=f51..f56` 只有 6 个字段**，
没有成交额和振幅（任务描述里的"8 字段"与实测不符，这里按实测写）：

| 位置 | 字段 | 含义 | 单位 |
|---|---|---|---|
| 0 | datetime | K 线**结束**时刻（09:35 … 15:00） | - |
| 1 | open   | 开 | 元（后复权） |
| 2 | close  | 收 | 元（后复权） |
| 3 | high   | 高 | 元（后复权） |
| 4 | low    | 低 | 元（后复权） |
| 5 | volume | 成交量 | 手 |

要拿成交额必须把 `f57` 加进 fields2，加完后是 7 字段：`..., volume, amount`。
字段顺序验证：600519 2026-11-06 当日 48 根 5min 的 volume 合计 42864 手，
与同日日线 volume 42848 手基本一致（差 16 手为集合竞价撮合口径），
且 `open/close/high/low` 满足 `low ≤ open,close ≤ high` 逐行成立。

## 三个必须知道的口径陷阱

1. **历史只有 31 个交易日**（实测 1488 根，`beg`/`lmt` 怎么调都一样，是接口硬限制）。
   所以本模块的日频因子最多只能产出 31 行 —— 它**不能**用来做长历史回测，
   只能做最近一个月的横截面/事件观察。这一点不能含糊。
2. **价格是后复权，成交额是不复权的原始元**（`fqt=2` 只作用于价格）。
   实测 600519 2026-11-06：5min 收盘 8906.67（后复权），
   而 `amount/volume` 反推的均价是 1446.87（真实价）。
   若直接算 `close / (amount/volume/100)` 会得到 **+515%** 的假偏离。
   所以本模块的 VWAP **用「成交量加权的 5min 收盘价」** 计算，全程留在后复权空间：
   同一天内复权因子是常数，加权平均后自动抵消，比值仍等于真实 VWAP 偏离。
3. K 线时间戳是**区间结束时刻**：09:35 那根覆盖 09:30–09:35。
   因此"收盘前 30 分钟"= 最后 6 根（14:35…15:00），"开盘前 30 分钟"= 前 6 根（09:35…10:00）。

## 因子的经济含义

日频因子都由**当日盘中结构**决定，不含跨日窗口，所以没有预热期；
但**未走完的交易日会被剔除**（最后一根不是 15:00 的那天），
否则盘中跑一次就会把半天数据当成一整天，且会破坏截断不变性。

## 因果性

第 t 行只用第 t 日及之前的 bar：全部按 `day` 分组，组内不跨日；
无 `shift(-n)`、无 `center=True`、无全样本 mean/std/quantile。
`audit/leak_scan.py` HIGH=0。

用法：
    from qcore.intraday_factor import fetch_intraday, intraday_factors
    bars = fetch_intraday("600519")        # DataFrame[datetime, open, close, ...]
    f    = intraday_factors("600519")      # index=date, 6 个因子列
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["fetch_intraday", "intraday_factors", "INTRADAY_FACTORS", "BAR_COLUMNS"]

# ------------------------------------------------------------------ 常量

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://quote.eastmoney.com/",
}

_KLINE_URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
              "?secid={secid}&fields1=f1,f2,f3"
              "&fields2=f51,f52,f53,f54,f55,f56,f57"
              "&klt={klt}&fqt=2&beg={beg}&end=20500101&lmt={lmt}")

#: 接口字段（实测 7 个，f57=amount）
BAR_COLUMNS = ["datetime", "open", "close", "high", "low", "volume", "amount"]

#: 本模块产出的 6 个因子
INTRADAY_FACTORS = [
    "intraday_vol_share",    # 收盘前 30 分钟成交量占比（尾盘异动）
    "intraday_open_share",   # 开盘前 30 分钟成交量占比
    "intraday_ret_skew",     # 当日 5min 收益偏度
    "intraday_ret_ac1",      # 当日 5min 收益一阶自相关
    "intraday_high_time",    # 当日最高价出现时刻 / 48
    "intraday_vwap_dev",     # 收盘价 / 当日 VWAP − 1
]

_CLOSE_MINUTE = 15 * 60       # 15:00，用于判定"这一天走完了"
_SESSION_MINUTES = 240        # A 股连续竞价 4 小时
_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "intraday"
_CACHE_TTL = 86400.0          # 1 天
_RETRIES = 3
_TIMEOUT = 30
_MIN_INTERVAL = 0.25
_last_call = [0.0]


# ------------------------------------------------------------------ 基础设施

def _norm(code: str) -> str:
    """'600519' / '600519.SH' / 'sh600519' -> '600519'。"""
    s = str(code).strip().upper()
    for suf in (".SH", ".SZ", ".BJ"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = s.replace("SH", "").replace("SZ", "").replace("BJ", "").strip()
    if not s.isdigit():
        raise ValueError(f"无法识别的股票代码: {code!r}")
    return s.zfill(6)


def _secid(code: str) -> str:
    """东财 secid：沪市（6/9 开头）用 1.，深市（0/2/3）与北交所（4/8）用 0.。"""
    c = _norm(code)
    return f"1.{c}" if c[0] in "69" else f"0.{c}"


def _get(url: str) -> str:
    """HTTP GET，失败重试 3 次，间隔递增（1s / 2s / 3s），最终失败抛异常。"""
    last: Optional[Exception] = None
    for i in range(_RETRIES):
        wait = _MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            _last_call[0] = time.time()
            return body
        except Exception as exc:                      # noqa: BLE001 - 网络异常类型很杂
            _last_call[0] = time.time()
            last = exc
            if i < _RETRIES - 1:
                time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"请求失败（已重试 {_RETRIES} 次）: {url} :: {last!r}")


def _cache_paths(code: str, klt: int):
    stem = _CACHE_DIR / f"{_norm(code)}_{int(klt)}m"
    return Path(str(stem) + ".parquet"), Path(str(stem) + ".pkl")


def _cache_read(code: str, klt: int) -> Optional[pd.DataFrame]:
    pq, pk = _cache_paths(code, klt)
    for path, reader in ((pq, pd.read_parquet), (pk, pd.read_pickle)):
        if not path.exists():
            continue
        if time.time() - path.stat().st_mtime > _CACHE_TTL:
            return None
        try:
            return reader(path)
        except Exception:                             # noqa: BLE001 - 缓存损坏当作未命中
            return None
    return None


def _cache_write(code: str, klt: int, df: pd.DataFrame) -> Path:
    """优先 parquet；没有 pyarrow 时自动降级 pickle（本机实测无 pyarrow）。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pq, pk = _cache_paths(code, klt)
    try:
        df.to_parquet(pq, index=False)
        if pk.exists():
            pk.unlink()
        return pq
    except Exception:                                 # noqa: BLE001 - 无 pyarrow
        df.to_pickle(pk)
        if pq.exists():
            pq.unlink()
        return pk


# ------------------------------------------------------------------ 抓取

def fetch_intraday(code: str, klt: int = 5, lmt: int = 5000,
                   beg: str = "20240101", use_cache: bool = True) -> pd.DataFrame:
    """
    返回 DataFrame[datetime, open, close, high, low, volume, amount]，按时间升序。

    单位：价格后复权（元）；volume 手；**amount 为不复权的原始元**
    （`fqt=2` 只作用于价格，这一点见模块 docstring 陷阱 2）。

    实测 klt=5 最多只有 31 个交易日（1488 根），lmt/beg 调大都无效。
    失败抛异常由调用方处理（已内置 3 次重试）。
    """
    klt = int(klt)
    if klt not in (1, 5, 15, 30, 60, 120):
        raise ValueError(f"klt={klt} 不支持，A 股分钟线常用 1/5/15/30/60/120")

    if use_cache:
        cached = _cache_read(code, klt)
        if cached is not None and len(cached) > 0:
            return cached

    url = _KLINE_URL.format(secid=_secid(code), klt=klt, beg=beg, lmt=int(lmt))
    payload = json.loads(_get(url))
    data = payload.get("data") or {}
    rows = data.get("klines") or []
    if not rows:
        raise RuntimeError(
            f"{code}: 分钟线接口返回空（klt={klt}）。该代码可能已退市/停牌，"
            f"或 secid 判定错误。"
        )

    split = [r.split(",") for r in rows]
    width = len(split[0])
    if width < len(BAR_COLUMNS):
        raise RuntimeError(
            f"{code}: 分钟线字段数 {width} < 期望 {len(BAR_COLUMNS)}，"
            f"接口口径可能变了，请重新核实字段顺序。首行={rows[0]!r}"
        )

    df = pd.DataFrame([r[: len(BAR_COLUMNS)] for r in split], columns=BAR_COLUMNS)
    for col in BAR_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = (df.dropna(subset=["datetime"])
            .drop_duplicates(subset=["datetime"], keep="last")
            .sort_values("datetime")
            .reset_index(drop=True))

    _cache_write(code, klt, df)
    return df


# ------------------------------------------------------------------ 因子

def _expected_bars(klt: int) -> int:
    """一个完整交易日的 K 线根数（连续竞价 240 分钟）。这是日历常数，不是样本统计量。"""
    return max(1, _SESSION_MINUTES // int(klt))


def _intraday_factors_from_frame(df: pd.DataFrame, klt: int = 5) -> pd.DataFrame:
    """
    把分钟线聚合成日频 6 因子。全部按 day 分组，组内不跨日 → 天然因果。

    未走完的交易日（最后一根不是 15:00）直接剔除，不给 NaN 行：
    这样任何前缀截断都不会改变已有行的取值（截断不变性成立）。
    """
    cols = INTRADAY_FACTORS
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols, dtype=float)

    d = df.sort_values("datetime").reset_index(drop=True)
    d["day"] = d["datetime"].dt.normalize()
    day = d["day"]
    expected = _expected_bars(klt)
    w30 = max(1, 30 // int(klt))          # "30 分钟"对应的 bar 数（klt=5 → 6 根）

    g = d.groupby("day", sort=True)
    n_bars = g.size()                     # 每日 bar 数
    pos = g.cumcount()                    # 每日组内 0-based 序号
    n_map = day.map(n_bars)               # 每行所属日的 bar 数

    # ---- 完整性判定：最后一根必须收在 15:00，且 bar 数不少于应有根数−2 ----
    # 用「时间戳」而不是「根数」做主判据：截断数据永远不会以 15:00 结尾，
    # 因此盘中数据/前缀截断都会被正确剔除。
    last_dt = g["datetime"].max()          # leak-scan: ok 组内(按 day)取末根时间戳，不跨日
    last_min = last_dt.dt.hour * 60 + last_dt.dt.minute
    complete = (n_bars >= expected - 2) & (last_min == _CLOSE_MINUTE)

    vol = d["volume"].astype(float)
    close = d["close"].astype(float)
    # 全天成交量为 0（停牌/异常）时置 NaN，避免占比与 VWAP 出现 inf
    tot_vol = vol.groupby(day).sum().where(lambda s: s > 0)

    out = pd.DataFrame(index=tot_vol.index)

    # ---- 1. intraday_vol_share：收盘前 30 分钟成交量 / 全天成交量 ----------
    # 经济含义：尾盘异动。放量拉升的尾盘常是主力表态（或指数基金收盘调仓）；
    # 尾盘放量下跌则常是出货。占比高说明"当天的信息主要在最后半小时定价"。
    is_tail = pos.to_numpy() >= (n_map.to_numpy() - w30)
    tail_vol = vol.where(is_tail, 0.0).groupby(day).sum()
    out["intraday_vol_share"] = tail_vol / tot_vol

    # ---- 2. intraday_open_share：开盘前 30 分钟成交量 / 全天成交量 ---------
    # 经济含义：隔夜信息释放强度。开盘半小时占比高 = 隔夜有重大消息/情绪极端，
    # 之后趋于平静；占比低 = 全天慢慢消化，通常配合趋势更稳。
    is_head = pos.to_numpy() < w30
    head_vol = vol.where(is_head, 0.0).groupby(day).sum()
    out["intraday_open_share"] = head_vol / tot_vol

    # ---- 当日 5min 收益（组内 shift，不跨日）----
    # 每根 bar 的收益 = 本根收盘 / 上一根收盘 − 1；每天第一根为 NaN。
    # 48 根 bar 只能产出 47 个收益（首根缺前值），这一点如实标注。
    prev_close = close.groupby(day).shift(1)
    ret = close / prev_close - 1.0
    ret = ret.where(np.isfinite(ret))

    # ---- 3. intraday_ret_skew：当日 5min 收益偏度 ------------------------
    # 经济含义：盘中收益分布的对称性。右偏（正）说明少数几根急拉贡献了大部分涨幅
    # （脉冲式、易回吐）；左偏（负）说明是急跌砸出来的坑（恐慌，常伴超跌反弹）。
    # 用 pandas 的样本无偏偏度（含小样本修正），组内计算。
    out["intraday_ret_skew"] = ret.groupby(day).skew()

    # ---- 4. intraday_ret_ac1：5min 收益一阶自相关 ------------------------
    # 经济含义：微观结构。负自相关 = 买卖价差跳动/做市商撮合造成的"噪声反转"，
    # 日内流动性越好、价差越小，负得越明显；正自相关 = 订单流被切碎后同向跟进
    # （动量/追单），通常意味着冲击成本高。
    # 用矩的闭式解向量化（GroupBy 没有 autocorr，避免 apply 拖慢）：
    #   ac1 = cov(r_t, r_{t-1}) / sqrt(var(r_t)·var(r_{t-1}))
    out["intraday_ret_ac1"] = _lag1_autocorr(ret.to_numpy(float), day, len(out))

    # ---- 5. intraday_high_time：当日最高价出现时刻 / 48 -------------------
    # 经济含义：越晚创新高 = 买盘越持续、越强（全天在往上找价格）；
    # 越早创新高后回落 = 高开低走/冲高回落，多头衰竭。
    # 输出 ∈ (0, 1]：1/48 ≈ 开盘第一根，48/48 = 收盘前最后一根。
    # 同日最高价并列时取**首次**出现（保守口径，不夸大盘中强度）。
    day_max = d["high"].astype(float).groupby(day).max()
    is_max = d["high"].to_numpy(float) == day.map(day_max).to_numpy(float)
    first_max_pos = pd.Series(np.where(is_max, pos.to_numpy(float), np.nan)).groupby(day).min()
    out["intraday_high_time"] = (first_max_pos + 1.0) / expected

    # ---- 6. intraday_vwap_dev：收盘价 / 当日 VWAP − 1 --------------------
    # 经济含义：收盘价相对当日成交均价的偏离 = 当天最后一段的"溢价/折价"。
    # 收在 VWAP 之上（正）= 尾盘买方愿意付高于均价的价格，当日买方占优；
    # 收在 VWAP 之下（负）= 尾盘抛压把价格压到均价下方。
    # VWAP 用「成交量加权的 5min 收盘价」，全程后复权空间（见 docstring 陷阱 2）。
    pv = (close * vol)
    vwap = pv.groupby(day).sum() / tot_vol
    close_last = close.groupby(day).last()
    out["intraday_vwap_dev"] = close_last / vwap - 1.0

    # ---- 剔除未走完的交易日 ----
    mask = complete.reindex(out.index).fillna(False).to_numpy(bool)
    out = out.loc[mask]
    out.index.name = "date"
    return out[cols].astype(float)


def _lag1_autocorr(ret: np.ndarray, day: pd.Series, k: int) -> np.ndarray:
    """
    按日分组的一阶自相关（总体矩闭式解，向量化）。

    ac1 = cov(r_t, r_{t−1}) / sqrt(var(r_t) · var(r_{t−1}))
    用 np.bincount 做分组求和，避免 GroupBy.apply（慢 10 倍以上）。
    返回长度 k 的数组，按 day 的 groupby 顺序（与 tot_vol.index 对齐）。
    """
    codes, _uniq = pd.factorize(day.to_numpy(), sort=True)
    r = np.asarray(ret, dtype=float)
    lag = np.full_like(r, np.nan)
    if len(r) > 1:
        # 只在同一天内连 r[i-1]；跨日（codes 变化）留 NaN，不把隔夜跳空当日内收益
        same_day = codes[1:] == codes[:-1]
        lag[1:] = np.where(same_day, r[:-1], np.nan)

    m = np.isfinite(r) & np.isfinite(lag)
    out = np.full(k, np.nan, dtype=float)
    if not m.any():
        return out

    c = codes[m]
    x = r[m]
    y = lag[m]
    cnt = np.bincount(c, minlength=k).astype(float)
    ok = cnt >= 3                                   # 至少 3 个配对才谈得上自相关
    if not ok.any():
        return out

    s_x = np.bincount(c, weights=x, minlength=k)
    s_y = np.bincount(c, weights=y, minlength=k)
    s_xx = np.bincount(c, weights=x * x, minlength=k)
    s_yy = np.bincount(c, weights=y * y, minlength=k)
    s_xy = np.bincount(c, weights=x * y, minlength=k)

    with np.errstate(invalid="ignore", divide="ignore"):
        mx = s_x / cnt
        my = s_y / cnt
        cov = s_xy / cnt - mx * my
        var_x = s_xx / cnt - mx * mx
        var_y = s_yy / cnt - my * my
        denom = np.sqrt(var_x * var_y)
        ac = cov / denom
    ac[~ok] = np.nan
    ac[~np.isfinite(ac)] = np.nan
    out[:] = ac[:k]
    return out


def intraday_factors(code: str, klt: int = 5, use_cache: bool = True) -> pd.DataFrame:
    """
    返回 index=date（交易日）、columns=INTRADAY_FACTORS 的日频因子表（6 列）。

    注意：接口只给最近约 31 个交易日（klt=5），所以最多 ~31 行。
    未走完的交易日被剔除；失败抛异常（已重试 3 次）——不编造数据。
    """
    bars = fetch_intraday(code, klt=klt, use_cache=use_cache)
    return _intraday_factors_from_frame(bars, klt=klt)


# ------------------------------------------------------------------ 自检

def _synthetic_bars(n_days: int = 104, klt: int = 5, seed: int = 11) -> pd.DataFrame:
    """合成约 5000 根 5min bar（104 个完整交易日 + 1 个残缺日），仅用于性能/截断测试。"""
    rng = np.random.default_rng(seed)
    times = []
    for i in range(0, 24):                       # 09:35 … 11:30
        times.append((9 * 60 + 35) + 5 * i)
    for i in range(0, 24):                       # 13:05 … 15:00
        times.append((13 * 60 + 5) + 5 * i)

    dates = pd.bdate_range("2026-06-01", periods=n_days)
    stamps = []
    for dt in dates:
        for tm in times:
            stamps.append(pd.Timestamp(dt) + pd.Timedelta(minutes=int(tm)))
    # 再补 8 根残缺日（模拟盘中数据/截断），凑到 ~5000 根
    extra_day = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
    while extra_day.weekday() >= 5:
        extra_day = extra_day + pd.Timedelta(days=1)
    for i in range(8):
        stamps.append(extra_day + pd.Timedelta(minutes=int(times[i])))

    n = len(stamps)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.0015, n))
    high = price * (1.0 + np.abs(rng.normal(0.0, 0.0012, n)))
    low = price * (1.0 - np.abs(rng.normal(0.0, 0.0012, n)))
    vol = rng.lognormal(6.0, 0.6, n)
    return pd.DataFrame({
        "datetime": pd.DatetimeIndex(stamps),
        "open": price, "close": price, "high": high, "low": low,
        "volume": vol, "amount": vol * 100.0 * price,
    })


def _print_stats(f: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} 因子统计 (n_rows={len(f)}"
          f"{', 区间 ' + str(f.index.min()) + ' .. ' + str(f.index.max()) if len(f) else ''}) ---")
    if len(f) == 0:
        print("  （空表，无法统计）")
        return
    desc = f.describe().T
    head = f"{'因子':<22}{'非NaN':>7}{'均值':>13}{'标准差':>13}{'最大值':>13}{'最小值':>13}"
    print(head)
    print("-" * len(head))
    for name in INTRADAY_FACTORS:
        if name not in desc.index:
            print(f"{name:<22}{'不可用':>7}")
            continue
        r = desc.loc[name]
        print(f"{name:<22}{int(r['count']):>7}{r['mean']:>13.6f}"
              f"{r['std']:>13.6f}{r['max']:>13.6f}{r['min']:>13.6f}")


def _truncation_test(bars: pd.DataFrame, n_trunc: int, label: str, klt: int = 5) -> bool:
    """
    截断不变性：只用前 n_trunc 根 bar 算 vs 用全部 bar 算，已产出的日必须逐位相同。

    前缀截断会让最后一天残缺，本模块会剔除未走完的交易日，
    因此比较的是「前缀能产出的那些日」与「全量对应日」。
    """
    n_total = len(bars)
    n_trunc = int(min(n_trunc, n_total))
    full = _intraday_factors_from_frame(bars, klt=klt)
    part = _intraday_factors_from_frame(bars.iloc[:n_trunc].copy(), klt=klt)
    a = full.iloc[: len(part)].to_numpy(float)
    b = part.to_numpy(float)
    if a.shape != b.shape:
        print(f"  [FAIL] {label}: shape 不一致 full[:{len(part)}]={a.shape} part={b.shape}")
        return False
    idx_same = bool(np.array_equal(full.index[: len(part)].to_numpy(),
                                   part.index.to_numpy()))
    ok = bool(np.array_equal(a, b, equal_nan=True)) and idx_same
    nan_a = int(np.isnan(a).sum())
    nan_b = int(np.isnan(b).sum())
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: 前 {n_trunc}/{n_total} 根 bar "
          f"-> {len(part)} 个交易日，逐位相同={ok}  (NaN 个数 {nan_a} vs {nan_b})")
    return ok


def _corr_report(f: pd.DataFrame, label: str) -> None:
    sub = f.dropna()
    if len(sub) < 3:
        print(f"  {label} 相关系数：有效行仅 {len(sub)}，不足以计算（不可用）")
        return
    corr = sub.corr().to_numpy(float)
    iu = np.triu_indices(corr.shape[0], 1)
    vals = np.abs(corr[iu])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"  {label} 相关系数：无有效配对")
        return
    mx = float(vals.max())            # leak-scan: ok 仅自检诊断，不参与因子计算
    k = int(np.argmax(vals))
    i, j = iu[0][k], iu[1][k]
    flag = "  ⚠ >0.9 因子可能重复" if mx > 0.9 else ""
    print(f"  {label} 上三角 |corr| 最大 = {mx:.4f}"
          f"（{INTRADAY_FACTORS[i]} vs {INTRADAY_FACTORS[j]}）{flag}")


def _main() -> int:
    # Windows 控制台默认 GBK，打印中文/特殊符号会 UnicodeEncodeError，强制 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                 # noqa: BLE001 - 老环境无 reconfigure
        pass

    codes = ["600519", "000001"]
    all_ok = True

    for code in codes:
        print("=" * 92)
        print(f"[分钟级因子] {code}")
        print("=" * 92)

        t0 = time.perf_counter()
        try:
            bars = fetch_intraday(code, klt=5)
        except Exception as exc:                      # noqa: BLE001
            print(f"  不可用：分钟线接口取数失败 -> {exc}")
            all_ok = False
            continue
        t_fetch = time.perf_counter() - t0

        n_days = bars["datetime"].dt.normalize().nunique()   # leak-scan: ok 仅自检诊断，统计交易日个数
        per_day = bars.groupby(bars["datetime"].dt.normalize()).size()
        print(f"  原始 5min bar: {len(bars)} 根 / {n_days} 个交易日 "
              f"({bars['datetime'].min()} .. {bars['datetime'].max()})  取数 {t_fetch:.2f}s")
        print(f"  每日 bar 数: 唯一值={sorted(per_day.unique().tolist())} "
              f"(完整日应为 {_expected_bars(5)})")
        if n_days <= 40:
            print(f"  ⚠ 注意：该接口只给最近 {n_days} 个交易日，无法用于长历史回测")

        t1 = time.perf_counter()
        try:
            f = intraday_factors(code)
        except Exception as exc:                      # noqa: BLE001
            print(f"  不可用：因子聚合失败 -> {exc}")
            all_ok = False
            continue
        t_api = time.perf_counter() - t1

        # 纯计算耗时（不含任何网络 IO），这才是"计算性能"该看的数字
        bars2 = fetch_intraday(code, klt=5)
        t2 = time.perf_counter()
        f2 = _intraday_factors_from_frame(bars2, klt=5)
        t_calc = time.perf_counter() - t2
        if not bool(np.array_equal(f.to_numpy(float), f2.to_numpy(float), equal_nan=True)):
            print("  [FAIL] intraday_factors() 与 _intraday_factors_from_frame() 结果不一致")
            all_ok = False

        _print_stats(f, f"{code} 真实数据")
        print(f"  耗时: 取数 {t_fetch:.2f}s; intraday_factors() 全流程 {t_api:.2f}s; "
              f"纯计算 {t_calc * 1000:.2f} ms / {len(bars2)} 根 bar")

        print("\n  截断不变性（真实数据，按 bar 截断到 500 根）:")
        all_ok &= _truncation_test(bars, 500, f"{code} 真实 {len(bars)} 根")

        print("\n  截断不变性（合成 ~5000 根 bar）:")
        syn = _synthetic_bars(104, 5)
        print(f"    合成 bar 数 = {len(syn)}")
        all_ok &= _truncation_test(syn, 500, "合成 5000 根")

        print("\n  性能（合成 5000 根 bar 纯计算，要求 < 2s）:")
        t2 = time.perf_counter()
        _intraday_factors_from_frame(syn, klt=5)
        dt = time.perf_counter() - t2
        print(f"    intraday_factors(5000 bar) = {dt * 1000:.1f} ms  "
              f"[{'PASS' if dt < 2.0 else 'FAIL'}]")

        print("\n  因子间相关性:")
        _corr_report(f, code)

    print("\n" + "=" * 92)
    print(f"总判定: {'全部通过' if all_ok else '存在失败项（见上）'}")
    print("=" * 92)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
