"""美股 / 港股数据层：东方财富直连（纯标准库 urllib + pandas），本地 CSV 缓存。

与 `qcore/data.py`（A股）**列名完全一致**，下游 `qcore/dataset.py` 可直接吃：

    date, open, close, high, low, volume, amount, amplitude, pct_chg, change, turnover, code

⚠️ 复权口径（沿用 data.py 的铁律）
    东财 `fqt=1`（前复权）是**减法复权**。A股上会把高分红老股压成负价；
    美股/港股上同样崩坏 —— 实测 `105.AAPL` fqt=1：1990-01-02 收盘 **-10.26**，
    全序列 5239 根非正价。绝对禁用。
    统一 `fqt=2`（后复权）：`105.AAPL` 1990-01-02 = 76.40 → 2026-11-06 = 79066.08，
    与"1990 年 AAPL 约 $37.4、累计 2:1/7:1/4:1 拆股 + 分红再投"的量级吻合；
    港股 `116.00700` 2004-06-16 = 4.15 → 2026-11-06 = 2379.89（含 2014 年 1:5 拆股）。
    后复权价格绝对值远大于现价，本项目只用收益率/相对量，无影响。

⚠️ 美股/港股**无涨跌停**，接口给的 pct_chg/change 在复权口径下不可靠，一律用本地 close 重算。

数据质量：US 后复权序列偶见"水平位移"毛刺（复权因子在某日整体错位，形成单日
|pct_chg| 巨大、与真实拆股/分红日无关）。`_repair_level_shifts` 按 k=close[t]/close[t-1]
把该 bar **之前的整段**缩放（等价一次拆股调整）：只抹掉该 bar 自身那一天，
其余 bar 的收益分毫不动、不留日期缺口。修复清单进 `df.attrs["repaired"]`。
⚠️ 该规则也会抹掉"真实的单日 ±50% 极端行情"（实测 AAPL 2000-09-29 −50.1% 是真跌）——
收益率/波动率类信号受不了这种 bar，故按规则剔除并登记；要保留原始口径就传 `repair=False`。

其他设计
  1. 单只失败抛清晰异常；批量函数（get_us_pool 的逐只补数、download）try/except 继续；
  2. 限速 0.22s + 5 次指数退避**复用 data.py 的 `_get_json`**（同一个 push2his 主机，
     必须共享同一个限速器，否则 A股+美股并发会直接把连接打断）；
  3. 缓存文件名带复权标记 `_h{adjust}`，避免新旧口径混用。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # 包内导入（`python -m tests.test_us_data`）
    from qcore.data import ADJUST, KLINE_COLS, KLINE_FIELDS, RAW_DIR, _get_json
except ImportError:  # 直接 `python qcore/us_data.py` 时的兜底
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qcore.data import ADJUST, KLINE_COLS, KLINE_FIELDS, RAW_DIR, _get_json

# ---------------------------------------------------------------- paths
US_KLINE_DIR = os.path.join(RAW_DIR, "kline", "us")
HK_KLINE_DIR = os.path.join(RAW_DIR, "kline", "hk")
META_DIR = os.path.join(RAW_DIR, "meta")
for _d in (US_KLINE_DIR, HK_KLINE_DIR, META_DIR):
    os.makedirs(_d, exist_ok=True)

US_MAP_PATH = os.path.join(META_DIR, "us_map.json")
US_POOL_PATH = os.path.join(META_DIR, "us_pool.csv")
US_POOL_META_PATH = os.path.join(META_DIR, "us_pool_meta.json")

# ---------------------------------------------------------------- endpoints
_SUGGEST_URL = ("https://searchapi.eastmoney.com/api/suggest/get"
                "?input={q}&type=14&token=D43BF722C8E33BDC906FB84D85E326E8&count=20")
_KLINE_URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
              "?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2={fields}"
              "&klt=101&fqt={fqt}&beg={beg}&end={end}&lmt=100000")
_CLIST_URL = ("https://push2.eastmoney.com/api/qt/clist/get"
              "?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f20&fs={fs}"
              "&fields=f12,f13,f14,f20,f6,f3,f100")
_QUOTE_URL = ("https://push2.eastmoney.com/api/qt/stock/get?secid={secid}"
              "&fields=f43,f57,f58,f116,f117,f47,f48,f170")

# 东财市场号：105=NASDAQ 106=NYSE 107=AMEX 116=港股 100=港股指数
US_MKTS = ("105", "106", "107")
US_MKT_NAMES = {"105": "NASDAQ", "106": "NYSE", "107": "AMEX"}
HK_MKT = "116"
US_CLIST_FS = "m:105,m:106,m:107"

BENCH_US = "SPY"
BENCH_US_SECID = "107.SPY"       # AMEX 上市的 SPDR S&P 500 ETF
BENCH_HK = "HSI"
BENCH_HK_SECID = "100.HSI"       # 恒生指数

MAX_JUMP = 50.0                  # repair=True（无脑修复）的检出阈值（%）
AUTO_JUMP = 30.0                 # repair="auto" 的检出阈值（%）：先检出，再用 fqt=0 交叉验证
VERIFY_TOL = 2.5                 # 交叉验证容差（百分点）：两口径同日收益差 > 此值即判为因子缺陷
MEGA_CAP = 1.0e12                # 异常标的筛查起点：市值 > $1T
POOL_MIN_BARS = 500              # 进池最少历史 bar 数
DEFAULT_BEG = "19900101"
DEFAULT_END = "20500101"

# 后备池：人工维护的流动大盘股（**非官方成分股快照**，仅在 clist 接口不可用时兜底）。
# 来源：S&P100 / Nasdaq-100 并集中的长期大市值成员，人工核对；更新日期 2026-11-08。
# 明确不做"成分股"宣称 —— 官方名单每季度变动，硬编码必然过期。
_FALLBACK_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK.B", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "WMT",
    "NFLX", "ABBV", "BAC", "ORCL", "CRM", "CVX", "MRK", "KO", "AMD", "PEP",
    "TMO", "ADBE", "CSCO", "ACN", "MCD", "LIN", "ABT", "WFC", "QCOM", "INTU",
    "TXN", "DHR", "AMGN", "IBM", "PM", "CAT", "GE", "NOW", "UBER", "SPGI",
    "RTX", "ISRG", "BKNG", "PFE", "GS", "HON", "T", "VRTX", "BLK", "MU",
    "LRCX", "AMAT", "PANW", "ADP", "SBUX", "GILD", "MDT", "REGN", "LMT", "SYK",
    "MMC", "PLTR", "KLAC", "ADI", "INTC", "BA", "MDLZ", "CI", "ELV", "SO",
]


# ---------------------------------------------------------------- helpers
def _norm_us_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    for suf in (".US", ".O", ".N", ".A"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    if not t:
        raise ValueError("empty ticker")
    return t


def _is_hk(ticker: str) -> bool:
    t = str(ticker).strip().upper()
    if t.endswith(".HK"):
        return True
    return t.isdigit() and len(t) in (4, 5)


def _norm_hk_code(code: str) -> str:
    """'00700' / '0700.HK' / '700' -> '00700'（东财港股固定 5 位）。"""
    t = str(code).strip().upper()
    if t.endswith(".HK"):
        t = t[:-3]
    t = t.strip()
    if not t.isdigit():
        raise ValueError(f"bad HK code: {code!r}")
    if len(t) > 5:
        raise ValueError(f"bad HK code (too long): {code!r}")
    return t.zfill(5)


def _safe_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def _kline_url(secid: str, adjust: int, beg: str, end: str, lmt: int = 100000) -> str:
    return _KLINE_URL.format(secid=secid, fields=KLINE_FIELDS, fqt=int(adjust),
                             beg=beg, end=end, lmt=int(lmt))


def _raw_frame(secid: str, adjust: int, beg: str, end: str,
               lmt: int = 100000) -> Tuple[str, pd.DataFrame]:
    """拉原始 kline -> DataFrame（未清洗）。lmt 只取最后 N 根（轻量探测用）。"""
    d = _get_json(_kline_url(secid, adjust, beg, end, lmt=lmt))
    data = d.get("data") or {}
    klines = data.get("klines") or []
    rows = [ln.split(",") for ln in klines if ln and ln.count(",") >= len(KLINE_COLS) - 1]
    if not rows:
        raise RuntimeError(f"no kline for secid={secid} (adjust={adjust}); "
                           f"check the secid via resolve_ticker()")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    return str(data.get("name") or ""), df


def _returns_from_raw(raw: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    """原始 frame -> {date: 日收益%}，用于 fqt=2 / fqt=0 交叉验证。"""
    df = raw.copy()
    for c in ("open", "close", "high", "low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0].sort_values("date").drop_duplicates("date")
    s = df.set_index("date")["close"]
    r = (s / s.shift(1) - 1.0) * 100.0
    return {k: float(v) for k, v in r.dropna().items() if np.isfinite(v)}


def _repair_level_shifts(df: pd.DataFrame, max_jump: float,
                         verify: Optional[Dict[pd.Timestamp, float]] = None,
                         tol: float = VERIFY_TOL
                         ) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    """处理"复权因子整体错位"造成的水平位移毛刺。

    现象：东财美股后复权序列在某些日期整体位移（实测 AAPL 2001-01-02 −44%、
    2005-01-03 +91%，与真实拆股/分红日无关），表现为单日 |pct| 巨大。

    为什么不能"删掉这根 bar"：位移是**水平**的，删掉只会把同样的跳变推给下一根
    （实测删掉 2005-01-03 后 2005-01-04 变成 +93.08%），问题原样保留。

    修复方式（等价于一次拆股调整）：把这根 bar **之前的整段**按 k=close[t]/close[t-1]
    缩放。这样 bar t 的收益变 0、价格路径连续、**其余所有 bar 的收益分毫不动**，
    也不产生日期缺口。

    `verify`（fqt=0 原始口径的 {date: 日收益}）不为空时做交叉验证：
      两口径同日收益同方向且相差 <= tol 个百分点 -> **真实行情，不修复**（记 verified）；
      否则 -> 供应商复权因子缺陷 -> 修复。
    依据：单日分红除息对复权因子的影响远小于 2.5pp，超过这个量级的分歧只能是因子错位。

    返回 (df, repaired, verified)。
    """
    repaired: List[dict] = []
    verified: List[dict] = []
    price_cols = ["open", "high", "low", "close"]
    done = set()
    for _ in range(50):
        prev = df["close"].shift(1)
        pct = (df["close"] / prev - 1.0) * 100.0
        cand = [int(i) for i in df.index[pct.abs() > float(max_jump)]
                if df.at[i, "date"] not in done]
        if not cand:
            break
        t = cand[0]
        d = df.at[t, "date"]
        done.add(d)
        if t == 0:
            continue
        r2 = float(pct.at[t])
        r0 = verify.get(d) if verify else None
        if r0 is not None and np.isfinite(r0) and np.sign(r0) == np.sign(r2) \
                and abs(r2 - r0) <= float(tol):
            verified.append({"date": str(d.date()), "pct_chg": round(r2, 4),
                             "raw_pct_chg": round(float(r0), 4)})
            continue                      # 真实极端行情：保留，不动
        k = float(df.at[t, "close"] / df.at[t - 1, "close"])
        if not np.isfinite(k) or k <= 0:
            continue
        for c in price_cols:
            df.loc[df.index[:t], c] = df[c].to_numpy()[:t] * k
        repaired.append({"date": str(d.date()), "pct_chg": round(r2, 4),
                         "factor": round(k, 6),
                         "raw_pct_chg": (round(float(r0), 4) if r0 is not None else None)})
    return df, repaired, verified


def _finalize(df: pd.DataFrame, code: str, repair: "str | bool" = "auto",
              max_jump: Optional[float] = None,
              verify: Optional[Dict[pd.Timestamp, float]] = None) -> pd.DataFrame:
    """数值化 -> 排序去重 -> 修水平位移 -> 重算 pct_chg/change。

    repair:
      "auto"（默认）检出 |收益|>30%，用 fqt=0 交叉验证后才决定是否修复（真实肥尾日保留）
      True          检出 |收益|>50%，一律修复（无脑旁路，会把真实肥尾日一起抹掉）
      False         完全不修复（原始口径）
    """
    for c in KLINE_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"empty frame for {code}")
    df["code"] = str(code)

    mode = "auto" if isinstance(repair, str) and repair.lower() == "auto" else \
           ("on" if repair else "off")
    repaired: List[dict] = []
    verified: List[dict] = []
    if mode != "off":
        thr = float(max_jump) if max_jump is not None else \
              (AUTO_JUMP if mode == "auto" else MAX_JUMP)
        df, repaired, verified = _repair_level_shifts(
            df, thr, verify=(verify if mode == "auto" else None))

    # ⚠️ 东财美股 amount 字段在 2017-07-11 之前**恒为 0**（volume 却有值，实测 9282 根里
    # 6936 根如此）。"有成交量但成交额=0"是自相矛盾的，说明是**缺失**而非零成交。
    # 统一转 NaN：不编数，也不让下游把缺失当成"零流动性"。
    fake_zero = (df["volume"] > 0) & (df["amount"].fillna(0) <= 0)
    if fake_zero.any():
        df.loc[fake_zero, "amount"] = np.nan

    # 用本地 close 重算，不信任接口字段（美股/港股无涨跌停，复权口径下接口字段更不可信）
    df["pct_chg"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    df["change"] = df["close"].diff()
    df.loc[df.index[0], ["pct_chg", "change"]] = 0.0
    for c in ("amplitude", "turnover"):
        if c in df:
            df[c] = df[c].astype(float)
    df.attrs["repaired"] = fixed
    df.attrs["amount_missing"] = int(fake_zero.sum())
    return df[KLINE_COLS + ["code"]]


def _cache_read(path: str) -> Optional[pd.DataFrame]:
    """读缓存 —— 缓存里存的是**原始 fqt 数据**，清洗/修复每次读盘时重跑。

    这样"网络路径"和"缓存路径"产出的 frame 逐元素相同，也避免把修好的结果
    二次落盘后口径悄悄漂移。
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) > 1 else None


def _cache_fresh(path: str, max_age_days: float) -> bool:
    return (os.path.exists(path)
            and (time.time() - os.path.getmtime(path)) / 86400.0 < max_age_days)


def _write_cache(raw: pd.DataFrame, path: str) -> None:
    """原子写：先写 .tmp 再 replace，避免半截文件被下次读成"有效缓存"。"""
    tmp = path + ".tmp"
    raw.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _mode_is_auto(repair) -> bool:
    return isinstance(repair, str) and repair.lower() == "auto"


def _has_jump(raw: pd.DataFrame, thr: float) -> bool:
    """轻量预检：有没有超过阈值的日收益（决定要不要拉 fqt=0 做验证）。"""
    try:
        c = pd.to_numeric(raw["close"], errors="coerce")
        r = (c / c.shift(1) - 1.0).abs() * 100.0
        return bool((r > float(thr)).any())
    except Exception:  # noqa: BLE001 —— 判断不了就当有跳变，宁可多验一次
        return True


def _load_raw_returns(secid: str, cache_path: str, max_age_days: float = 5.0,
                      refresh: bool = False
                      ) -> Tuple[Optional[Dict[pd.Timestamp, float]], int]:
    """fqt=0（原始未复权）日收益 {date: pct}，落盘缓存。

    `repair="auto"` 的交叉验证底座：只在主序列检出跳变时才需要，
    且结果落盘（`{code}_h0.csv`），所以**常态 0 次额外请求**，冷启动最多 1 次。
    返回 (dict|None, 网络请求次数)。
    """
    if not refresh and _cache_fresh(cache_path, max_age_days):
        raw = _cache_read(cache_path)
        if raw is not None:
            try:
                return _returns_from_raw(raw), 0
            except Exception:  # noqa: BLE001 —— 缓存坏了就重拉
                pass
    try:
        _name, raw = _raw_frame(secid, 0, DEFAULT_BEG, DEFAULT_END)
        _write_cache(raw, cache_path)
        return _returns_from_raw(raw), 1
    except Exception:  # noqa: BLE001 —— 验证失败：auto 退化为"一律修复"
        return None, 1


# ---------------------------------------------------------------- resolve
def _load_map() -> Dict[str, str]:
    if not os.path.exists(US_MAP_PATH):
        return {}
    try:
        with open(US_MAP_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {str(k): str(v) for k, v in m.items() if isinstance(v, str)}
    except Exception:  # noqa: BLE001 —— 缓存坏了就当没有，下次重写
        return {}


def _save_map(m: Dict[str, str]) -> None:
    tmp = US_MAP_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, US_MAP_PATH)


def _suggest_secid(ticker: str) -> str:
    """searchapi 建议接口 -> 精确匹配的 US secid。

    只接受 `Code` 与请求**精确相同**的美股条目 —— 东财对相近代码会返回一堆
    无关股票（查 GOOGL 会带出 DEPW/GGLL…），宽松匹配会把 typo 静默映射到错误标的。
    `BRK.B` 这类带点代码在东财写作 `BRK_B`，故做一次分隔符变体重试。
    """
    want = ticker.upper()
    variants = [want]
    if "." in want:
        variants += [want.replace(".", "_"), want.replace(".", "-")]
    seen = 0
    for v in variants:
        d = _get_json(_SUGGEST_URL.format(q=v))
        rows = ((d.get("QuotationCodeTable") or {}).get("Data") or [])
        seen += len(rows)
        hits = [r for r in rows
                if str(r.get("Code", "")).upper() == v.upper()
                and str(r.get("MktNum", "")) in US_MKTS]
        if not hits:
            continue
        order = {m: i for i, m in enumerate(US_MKTS)}
        hits.sort(key=lambda r: order.get(str(r.get("MktNum")), 9))
        r = hits[0]
        return f"{r['MktNum']}.{r['Code']}"
    raise RuntimeError(f"cannot resolve ticker {ticker!r} to a US secid "
                       f"(suggest returned {seen} rows, none with an exact code match)")


def resolve_ticker(ticker: str, refresh: bool = False) -> str:
    """'AAPL' -> '105.AAPL'；'0700.HK'/'00700' -> '116.00700'。带磁盘缓存。

    缓存文件 data/raw/meta/us_map.json；`refresh=True` 强制重新解析。
    """
    t = str(ticker).strip().upper()
    if _is_hk(t):
        return f"{HK_MKT}.{_norm_hk_code(t)}"
    t = _norm_us_ticker(t)
    cache = _load_map()
    if not refresh and t in cache:
        return cache[t]
    secid = _suggest_secid(t)
    cache[t] = secid
    _save_map(cache)
    return secid


# ---------------------------------------------------------------- US kline
def fetch_us_kline(ticker: str, adjust: int = ADJUST, beg: str = DEFAULT_BEG,
                   end: str = DEFAULT_END, repair: "str | bool" = "auto") -> pd.DataFrame:
    """不走缓存，直接拉。"""
    t = _norm_us_ticker(ticker)
    secid = resolve_ticker(t)
    name, raw = _raw_frame(secid, adjust, beg, end)
    ver, net = (None, 0)
    if adjust != 0 and _mode_is_auto(repair) and _has_jump(raw, AUTO_JUMP):
        ver, net = _load_raw_returns(secid, os.path.join(US_KLINE_DIR, f"{_safe_name(t)}_h0.csv"))
    df = _finalize(raw, t, repair=repair, verify=ver)
    df.attrs.update(name=name, secid=secid, adjust=int(adjust),
                    from_cache=False, verify_requests=net)
    return df


def load_us_kline(ticker: str, adjust: int = ADJUST, refresh: bool = False,
                  max_age_days: float = 5.0, repair: "str | bool" = "auto") -> pd.DataFrame:
    """美股日线（后复权）。缓存 data/raw/kline/us/{ticker}_h{adjust}.csv。

    `repair="auto"`（默认）见 `_finalize`：跳变先与 fqt=0 原始口径交叉验证，
    真实肥尾日保留（`attrs["verified_extreme"]`），供应商因子缺陷才修（`attrs["repaired"]`）。
    """
    t = _norm_us_ticker(ticker)
    path = os.path.join(US_KLINE_DIR, f"{_safe_name(t)}_h{int(adjust)}.csv")
    vpath = os.path.join(US_KLINE_DIR, f"{_safe_name(t)}_h0.csv")
    if not refresh and _cache_fresh(path, max_age_days):
        raw = _cache_read(path)
        if raw is not None:
            secid = _load_map().get(t, "")
            ver, net = (None, 0)
            if adjust != 0 and _mode_is_auto(repair) and secid and _has_jump(raw, AUTO_JUMP):
                ver, net = _load_raw_returns(secid, vpath, max_age_days)
            df = _finalize(raw, t, repair=repair, verify=ver)
            df.attrs.update(name="", secid=secid, adjust=int(adjust),
                            from_cache=True, verify_requests=net)
            return df
    secid = resolve_ticker(t)
    name, raw = _raw_frame(secid, adjust, DEFAULT_BEG, DEFAULT_END)
    _write_cache(raw, path)
    ver, net = (None, 0)
    if adjust != 0 and _mode_is_auto(repair) and _has_jump(raw, AUTO_JUMP):
        ver, net = _load_raw_returns(secid, vpath, max_age_days)
    df = _finalize(raw, t, repair=repair, verify=ver)
    df.attrs.update(name=name, secid=secid, adjust=int(adjust),
                    from_cache=False, verify_requests=net)
    return df


# ---------------------------------------------------------------- HK kline
def fetch_hk_kline(code: str, adjust: int = ADJUST, beg: str = DEFAULT_BEG,
                   end: str = DEFAULT_END, repair: "str | bool" = "auto") -> pd.DataFrame:
    """不走缓存，直接拉。"""
    c = _norm_hk_code(code)
    secid = f"{HK_MKT}.{c}"
    name, raw = _raw_frame(secid, adjust, beg, end)
    ver, net = (None, 0)
    if adjust != 0 and _mode_is_auto(repair) and _has_jump(raw, AUTO_JUMP):
        ver, net = _load_raw_returns(secid, os.path.join(HK_KLINE_DIR, f"{c}_h0.csv"))
    df = _finalize(raw, c, repair=repair, verify=ver)
    df.attrs.update(name=name, secid=secid, adjust=int(adjust),
                    from_cache=False, verify_requests=net)
    return df


def load_hk_kline(code: str, adjust: int = ADJUST, refresh: bool = False,
                  max_age_days: float = 5.0, repair: "str | bool" = "auto") -> pd.DataFrame:
    """港股日线（后复权）。'00700' / '0700.HK' / '700' 都能吃。
    缓存 data/raw/kline/hk/{code}_h{adjust}.csv。"""
    c = _norm_hk_code(code)
    secid = f"{HK_MKT}.{c}"
    path = os.path.join(HK_KLINE_DIR, f"{c}_h{int(adjust)}.csv")
    vpath = os.path.join(HK_KLINE_DIR, f"{c}_h0.csv")
    if not refresh and _cache_fresh(path, max_age_days):
        raw = _cache_read(path)
        if raw is not None:
            ver, net = (None, 0)
            if adjust != 0 and _mode_is_auto(repair) and _has_jump(raw, AUTO_JUMP):
                ver, net = _load_raw_returns(secid, vpath, max_age_days)
            df = _finalize(raw, c, repair=repair, verify=ver)
            df.attrs.update(name="", secid=secid, adjust=int(adjust),
                            from_cache=True, verify_requests=net)
            return df
    name, raw = _raw_frame(secid, adjust, DEFAULT_BEG, DEFAULT_END)
    _write_cache(raw, path)
    ver, net = (None, 0)
    if adjust != 0 and _mode_is_auto(repair) and _has_jump(raw, AUTO_JUMP):
        ver, net = _load_raw_returns(secid, vpath, max_age_days)
    df = _finalize(raw, c, repair=repair, verify=ver)
    df.attrs.update(name=name, secid=secid, adjust=int(adjust),
                    from_cache=False, verify_requests=net)
    return df


# ---------------------------------------------------------------- benchmarks
def _load_index_secid(secid: str, code: str, out_dir: str, adjust: int = ADJUST,
                      refresh: bool = False, max_age_days: float = 5.0,
                      min_bars: int = 1000) -> pd.DataFrame:
    path = os.path.join(out_dir, f"{_safe_name(code)}_h{int(adjust)}.csv")
    if not refresh and _cache_fresh(path, max_age_days):
        raw = _cache_read(path)
        if raw is not None:
            df = _finalize(raw, code, repair=False)
            if len(df) >= min_bars:
                df.attrs.update(name="", secid=secid, adjust=int(adjust), from_cache=True)
                return df
    name, raw = _raw_frame(secid, adjust, DEFAULT_BEG, DEFAULT_END)
    _write_cache(raw, path)
    # 指数不做毛刺剔除：指数不会拆股，|pct| 异常大概率是真实暴涨暴跌
    df = _finalize(raw, code, repair=False)
    if len(df) < min_bars:
        raise RuntimeError(f"benchmark {code} only {len(df)} bars (<{min_bars})")
    df.attrs.update(name=name, secid=secid, adjust=int(adjust), from_cache=False)
    return df


def load_us_benchmark(refresh: bool = False) -> pd.DataFrame:
    """SPY 日线（后复权）—— 美股主日历。"""
    return _load_index_secid(BENCH_US_SECID, BENCH_US, US_KLINE_DIR, refresh=refresh)


def load_hk_benchmark(refresh: bool = False) -> pd.DataFrame:
    """恒生指数 HSI 日线 —— 港股主日历。"""
    return _load_index_secid(BENCH_HK_SECID, BENCH_HK, HK_KLINE_DIR, refresh=refresh)


def load_benchmark(market: str = "US") -> pd.DataFrame:
    """统一基准入口（多市场对齐用）：US -> SPY(107.SPY)，HK -> HSI(100.HSI)。

    两者都是**后复权**日线，只用于收益率/日历对齐，价格绝对值不可当指数点位。
    """
    m = str(market).strip().upper()
    if m in ("US", "USA", "NASDAQ", "NYSE", "AMEX"):
        return load_us_benchmark()
    if m in ("HK", "HKG", "HONGKONG", "HKEX"):
        return load_hk_benchmark()
    raise ValueError(f"unknown market {market!r}; expected 'US' or 'HK'")


# ---------------------------------------------------------------- pool
def _clist_us(top_n: int, page_size: int = 100) -> pd.DataFrame:
    pages = max(1, min(20, int(np.ceil(max(top_n, 1) / float(page_size)))))
    frames = []
    for pn in range(1, pages + 1):
        d = _get_json(_CLIST_URL.format(pn=pn, pz=page_size, fs=US_CLIST_FS))
        diff = (d.get("data") or {}).get("diff") or []
        if not diff:
            break
        frames.append(pd.DataFrame(diff))
        if len(diff) < page_size:
            break
    if not frames:
        raise RuntimeError("US clist returned nothing")
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "f12": "code", "f13": "mkt", "f14": "name", "f20": "mktcap",
        "f6": "amount", "f3": "pct_chg", "f100": "sector"})
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["mkt"] = df["mkt"].astype(str).str.strip()
    for c in ("mktcap", "amount", "pct_chg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sector"] = df["sector"].astype(str).replace({"nan": "", "-": ""})
    return df.drop_duplicates("code").reset_index(drop=True)


_CLASS_SUFFIX = re.compile(r"[-_](A|B|C|R|W|S)$")

# 已知双类别/多代码同公司（名称规则兜底用；名称里没有 -A/-B 后缀时靠这张表）
_KNOWN_DUAL = ({"GOOGL", "GOOG"}, {"BRK_A", "BRK_B"}, {"FOX", "FOXA"},
               {"NWS", "NWSA"}, {"LEN", "LEN_B"})


def _probe_listing(secid: str, min_bars: int = POOL_MIN_BARS) -> dict:
    """轻量探测某 secid 是否"正常挂牌 + 历史够长"。

    只取最后 min_bars+1 根（lmt 截断），payload 很小。
    """
    try:
        _n, raw = _raw_frame(secid, ADJUST, DEFAULT_BEG, DEFAULT_END, lmt=min_bars + 1)
    except Exception:  # noqa: BLE001
        return {"ok": False, "bars": 0, "reason": "kline_unavailable"}
    bars = len(raw)
    if bars <= min_bars:
        return {"ok": False, "bars": bars, "reason": f"history_lt_{min_bars}"}
    tail = raw.tail(60)
    amt = pd.to_numeric(tail["amount"], errors="coerce").fillna(0.0)
    if float(amt.max()) <= 0:
        return {"ok": False, "bars": bars, "reason": "amount_all_missing_last60"}
    return {"ok": True, "bars": bars, "reason": ""}


def _dual_class_key(name: str, code: str) -> str:
    n = str(name or "").strip()
    base = _CLASS_SUFFIX.sub("", n)
    return f"name:{base}" if (base and base != n) else f"code:{code}"


def _dedupe_dual_class(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """同公司多类别股（谷歌-A/C、伯克希尔-A/B）去重：保留成交额大的那个。"""
    alias: Dict[str, str] = {}
    for grp in _KNOWN_DUAL:
        rep = sorted(grp)[0]
        for c in grp:
            alias[c] = rep
    d = df.copy()
    d["_grp"] = [alias.get(r["code"]) or _dual_class_key(r["name"], r["code"])
                 for _, r in d.iterrows()]
    d["_amt"] = d["amount"].fillna(-1.0)
    d = d.sort_values(["_grp", "_amt"], ascending=[True, False])
    keep_mask = ~d["_grp"].duplicated(keep="first")
    kept = d[keep_mask].set_index("_grp")["code"]
    dropped = [{"code": r["code"], "kept": str(kept.get(r["_grp"], "")),
                "amount": (None if pd.isna(r["amount"]) else float(r["amount"]))}
               for _, r in d[~keep_mask].iterrows()]
    return d[keep_mask].drop(columns=["_grp", "_amt"]).reset_index(drop=True), dropped


def _screen_anomalies(df: pd.DataFrame, mega_cap: float = MEGA_CAP,
                      min_bars: int = POOL_MIN_BARS) -> Tuple[pd.DataFrame, List[dict]]:
    """剔除"市值异常大但根本不是正常挂牌"的标的（如 SPCX）。

    判据（只看市值 > $1T 的极少数标的，代价可忽略）：
      kline 取不到 / 历史不足 min_bars 根 / 最近 60 根成交额全缺失。
    典型例：东财把 `SPCX`(SpaceX) 挂成 $1.82T 市值，但日线只有 103 根
    （2026-06-12 起），进不了 panel，也撑不起"万亿市值"的流动性假设。
    """
    excluded: List[dict] = []
    keep = np.ones(len(df), dtype=bool)
    for i, r in enumerate(df.itertuples(index=False)):
        cap = getattr(r, "mktcap", np.nan)
        if not (pd.notna(cap) and float(cap) >= mega_cap):
            continue
        secid = f"{getattr(r, 'mkt', '')}.{r.code}".strip(".")
        info = _probe_listing(secid, min_bars=min_bars)
        if not info["ok"]:
            keep[i] = False
            excluded.append({"code": r.code, "mktcap": float(cap),
                             "reason": info["reason"], "bars": info["bars"]})
    return df[keep].reset_index(drop=True), excluded


def _quote_mktcap(secid: str) -> Tuple[float, float]:
    """单只 quote -> (mktcap, amount)；拿不到返回 (nan, nan)。"""
    try:
        d = _get_json(_QUOTE_URL.format(secid=secid))
        q = d.get("data") or {}
        cap = pd.to_numeric(q.get("f116"), errors="coerce")
        amt = pd.to_numeric(q.get("f48"), errors="coerce")
        return float(cap) if pd.notna(cap) else float("nan"), \
               float(amt) if pd.notna(amt) else float("nan")
    except Exception:  # noqa: BLE001 —— 单只失败不影响批量
        return float("nan"), float("nan")


def _build_pool_us(top_n: int, min_amount: float = 0.0) -> pd.DataFrame:
    """clist -> 过滤 -> 双类别去重 -> 异常标的筛查 -> 取前 top_n。"""
    top_n = int(max(1, top_n))
    try:
        df = _clist_us(top_n + 40)          # 多取一页，给去重/剔除留余量
        df = df.sort_values("mktcap", ascending=False, na_position="last")
        if min_amount and min_amount > 0:
            df = df[df["amount"].notna() & (df["amount"] >= float(min_amount))]
        df, deduped = _dedupe_dual_class(df)
        df, excluded = _screen_anomalies(df)
        out = df[["code", "name", "mktcap", "amount", "sector"]].head(top_n)
        out = out.reset_index(drop=True)
        out.attrs["source"] = "eastmoney_clist"
        out.attrs["excluded"] = excluded
        out.attrs["deduped"] = deduped
        out.attrs["mkt_coverage"] = float(out["mktcap"].notna().mean())
        return out
    except Exception as e:  # noqa: BLE001 —— 退回内置表
        print(f"  [warn] US clist failed ({type(e).__name__}), "
              f"falling back to builtin ticker table", flush=True)
    rows = []
    for t in _FALLBACK_US[:top_n]:
        try:
            secid = resolve_ticker(t)
            cap, amt = _quote_mktcap(secid)
        except Exception:  # noqa: BLE001
            cap, amt = float("nan"), float("nan")
        rows.append({"code": t, "name": "", "mktcap": cap, "amount": amt, "sector": ""})
    out = pd.DataFrame(rows, columns=["code", "name", "mktcap", "amount", "sector"])
    out.attrs["source"] = "builtin_fallback"
    out.attrs["excluded"] = []
    out.attrs["deduped"] = []
    out.attrs["mkt_coverage"] = float(out["mktcap"].notna().mean()) if len(out) else 0.0
    return out


def _write_pool_meta(excluded: List[dict], deduped: List[dict]) -> None:
    try:
        with open(US_POOL_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"excluded": excluded, "deduped": deduped,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=1)
    except Exception:  # noqa: BLE001
        pass


def _read_pool_meta() -> dict:
    try:
        with open(US_POOL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def get_pool_us(top_n: int = 80, min_amount: float = 0.0,
                cache_days: float = 3.0, refresh: bool = False) -> pd.DataFrame:
    """美股候选池（与 A 股 `get_pool` 签名对齐）：code, name, mktcap, amount, sector。

    ⚠️⚠️ **美股流动性排序 / 点对点选池必须用 `volume` 或 `volume×close` 代理，
    禁止用 `amount` 排序。**
    原因：东财美股 `amount` 字段在 **2017-07-11 之前恒为 0**（volume 却有值，
    实测 AAPL 9282 根里 6936 根如此）。本模块已把"有量无额"的假 0 转成 `NaN`，
    所以 2017 年前 amount **全部是 NaN**；拿它做排名/分位/滚动均值，NaN 会被
    排名逻辑错误处理（要么整段被丢掉，要么被当 0 排到最后）。
    `min_amount` 参数因此对 2017 年前的美股是"无差别剔除"，请勿依赖。

    其他口径：
      * mktcap / amount 单位都是**美元**（A 股 data.py 里是元/手，别混用）；
      * 同公司多类别股（GOOGL/GOOG、BRK_A/BRK_B）**保留成交额大的那个**，
        被去重的记在 `attrs["deduped"]`；
      * 市值 > $1T 但历史不足 500 根 / 取不到日线 / 成交额长期缺失的标的
        （如 `SPCX`——东财挂成 $1.82T 但只有 103 根日线）**默认剔除**，
        清单在 `attrs["excluded"]`；
      * 拿不到的字段留 NaN —— 不猜、不填零。
    """
    if not refresh and _cache_fresh(US_POOL_PATH, cache_days):
        try:
            df = pd.read_csv(US_POOL_PATH, dtype={"code": str})
            if len(df) >= min(int(top_n), 100):
                df = df.head(int(top_n)).reset_index(drop=True)
                meta = _read_pool_meta()
                df.attrs["source"] = "cache"
                df.attrs["excluded"] = meta.get("excluded", [])
                df.attrs["deduped"] = meta.get("deduped", [])
                df.attrs["mkt_coverage"] = float(df["mktcap"].notna().mean())
                return df
        except Exception:  # noqa: BLE001
            pass
    df = _build_pool_us(top_n=top_n, min_amount=min_amount)
    out = df.copy()
    attrs = dict(df.attrs)
    out.attrs = {}
    out.to_csv(US_POOL_PATH, index=False, encoding="utf-8")
    _write_pool_meta(attrs.get("excluded", []), attrs.get("deduped", []))
    return df


# 向后兼容旧名
get_us_pool = get_pool_us
fetch_us_pool = _build_pool_us


# ---------------------------------------------------------------- batch
def download_us(tickers: List[str], adjust: int = ADJUST, verbose: bool = True) -> int:
    """批量下载（幂等），单只失败不中断。返回成功数。"""
    ok = 0
    for i, t in enumerate(tickers, 1):
        try:
            load_us_kline(t, adjust=adjust)
            ok += 1
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [fail] {t}: {type(e).__name__}: {e}", flush=True)
        if verbose and i % 20 == 0:
            print(f"  ... {i}/{len(tickers)} ok={ok}", flush=True)
    return ok


if __name__ == "__main__":
    print("resolve:", {t: resolve_ticker(t) for t in ("AAPL", "MSFT", "SPY", "0700.HK")})
    for t in ("AAPL", "MSFT", "NVDA"):
        d = load_us_kline(t)
        print(f"{t}: bars={len(d)} {d['date'].iloc[0].date()}..{d['date'].iloc[-1].date()} "
              f"last={d['close'].iloc[-1]:.4g} repaired={len(d.attrs.get('repaired', []))}")
    for c in ("00700", "09988"):
        d = load_hk_kline(c)
        print(f"{c}: bars={len(d)} {d['date'].iloc[0].date()}..{d['date'].iloc[-1].date()} "
              f"last={d['close'].iloc[-1]:.4g}")
    b = load_us_benchmark()
    print(f"SPY: bars={len(b)} {b['date'].iloc[0].date()}..{b['date'].iloc[-1].date()}")
    b = load_hk_benchmark()
    print(f"HSI: bars={len(b)} {b['date'].iloc[0].date()}..{b['date'].iloc[-1].date()}")
    p = get_us_pool(80)
    print(f"pool: n={len(p)} mktcap_cov={p['mktcap'].notna().mean():.2%} "
          f"source={p.attrs.get('source')}")
