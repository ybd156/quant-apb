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

数据质量：US 后复权序列偶见"水平位移"毛刺（复权因子在某日整体错位，形成
单日 |pct_chg| 巨大且被后续反向跳回）。`_finalize` 会丢弃 |pct_chg| > MAX_JUMP 的 bar，
把丢弃清单写进 `df.attrs["dropped_jumps"]` 并打印 ASCII 告警 —— 不静默篡改。

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

MAX_JUMP = 50.0                  # 单日 |pct_chg| 上限（%），超过视为复权毛刺/极端异常
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
    "MMC", "PLTR", "KLAC", "ADI", "INTC", "BA", "MDLZ", "VRTX", "CI", "ELV",
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


def _kline_url(secid: str, adjust: int, beg: str, end: str) -> str:
    return _KLINE_URL.format(secid=secid, fields=KLINE_FIELDS, fqt=int(adjust),
                             beg=beg, end=end)


def _raw_frame(secid: str, adjust: int, beg: str, end: str) -> Tuple[str, pd.DataFrame]:
    """拉原始 kline -> DataFrame（未清洗）。"""
    d = _get_json(_kline_url(secid, adjust, beg, end))
    data = d.get("data") or {}
    klines = data.get("klines") or []
    rows = [ln.split(",") for ln in klines if ln and ln.count(",") >= len(KLINE_COLS) - 1]
    if not rows:
        raise RuntimeError(f"no kline for secid={secid} (adjust={adjust}); "
                           f"check the secid via resolve_ticker()")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    return str(data.get("name") or ""), df


def _finalize(df: pd.DataFrame, code: str, repair: bool = True,
              max_jump: float = MAX_JUMP) -> pd.DataFrame:
    """数值化 -> 排序去重 -> 重算 pct_chg/change -> 剔除复权毛刺 bar。"""
    for c in KLINE_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"empty frame for {code}")
    df["code"] = str(code)

    dropped: List[dict] = []
    if repair:
        pct = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
        bad = df.index[pct.abs() > float(max_jump)]
        if len(bad):
            dropped = [{"date": str(df.at[i, "date"].date()),
                        "pct_chg": round(float(pct.at[i]), 4),
                        "close": float(df.at[i, "close"])} for i in bad]
            df = df.drop(index=bad).reset_index(drop=True)

    # 用本地 close 重算，不信任接口字段（美股/港股无涨跌停，复权口径下接口字段更不可信）
    df["pct_chg"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    df["change"] = df["close"].diff()
    df.loc[df.index[0], ["pct_chg", "change"]] = 0.0
    for c in ("amplitude", "turnover"):
        if c in df:
            df[c] = df[c].astype(float)
    df.attrs["dropped_jumps"] = dropped
    return df[KLINE_COLS + ["code"]]


def _cache_read(path: str, code: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["code"] = str(code)
    df["pct_chg"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    df["change"] = df["close"].diff()
    df.loc[df.index[0], ["pct_chg", "change"]] = 0.0
    df.attrs["dropped_jumps"] = []
    return df[KLINE_COLS + ["code"]]


def _cache_fresh(path: str, max_age_days: float) -> bool:
    return (os.path.exists(path)
            and (time.time() - os.path.getmtime(path)) / 86400.0 < max_age_days)


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
                   end: str = DEFAULT_END, repair: bool = True) -> pd.DataFrame:
    t = _norm_us_ticker(ticker)
    secid = resolve_ticker(t)
    name, raw = _raw_frame(secid, adjust, beg, end)
    df = _finalize(raw, t, repair=repair)
    df.attrs["name"] = name
    df.attrs["secid"] = secid
    df.attrs["adjust"] = int(adjust)
    return df


def load_us_kline(ticker: str, adjust: int = ADJUST, refresh: bool = False,
                  max_age_days: float = 5.0, repair: bool = True) -> pd.DataFrame:
    """美股日线（后复权）。缓存 data/raw/kline/us/{ticker}_h{adjust}.csv。"""
    t = _norm_us_ticker(ticker)
    path = os.path.join(US_KLINE_DIR, f"{_safe_name(t)}_h{int(adjust)}.csv")
    if not refresh and _cache_fresh(path, max_age_days):
        df = _cache_read(path, t)
        if df is not None and len(df) > 1:
            return df
    df = fetch_us_kline(t, adjust=adjust, repair=repair)
    out = df.copy()
    out.attrs = {}
    out.to_csv(path, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------- HK kline
def fetch_hk_kline(code: str, adjust: int = ADJUST, beg: str = DEFAULT_BEG,
                   end: str = DEFAULT_END, repair: bool = True) -> pd.DataFrame:
    c = _norm_hk_code(code)
    secid = f"{HK_MKT}.{c}"
    name, raw = _raw_frame(secid, adjust, beg, end)
    df = _finalize(raw, c, repair=repair)
    df.attrs["name"] = name
    df.attrs["secid"] = secid
    df.attrs["adjust"] = int(adjust)
    return df


def load_hk_kline(code: str, adjust: int = ADJUST, refresh: bool = False,
                  max_age_days: float = 5.0, repair: bool = True) -> pd.DataFrame:
    """港股日线（后复权）。'00700' / '0700.HK' / '700' 都能吃。
    缓存 data/raw/kline/hk/{code}_h{adjust}.csv。"""
    c = _norm_hk_code(code)
    path = os.path.join(HK_KLINE_DIR, f"{c}_h{int(adjust)}.csv")
    if not refresh and _cache_fresh(path, max_age_days):
        df = _cache_read(path, c)
        if df is not None and len(df) > 1:
            return df
    df = fetch_hk_kline(c, adjust=adjust, repair=repair)
    out = df.copy()
    out.attrs = {}
    out.to_csv(path, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------- benchmarks
def _load_index_secid(secid: str, code: str, out_dir: str, adjust: int = ADJUST,
                      refresh: bool = False, max_age_days: float = 5.0,
                      min_bars: int = 1000) -> pd.DataFrame:
    path = os.path.join(out_dir, f"{_safe_name(code)}_h{int(adjust)}.csv")
    if not refresh and _cache_fresh(path, max_age_days):
        df = _cache_read(path, code)
        if df is not None and len(df) >= min_bars:
            return df
    name, raw = _raw_frame(secid, adjust, DEFAULT_BEG, DEFAULT_END)
    # 指数不做毛刺剔除：指数不会拆股，|pct| 异常大概率是真实暴涨暴跌
    df = _finalize(raw, code, repair=False)
    df.attrs["name"] = name
    df.attrs["secid"] = secid
    if len(df) < min_bars:
        raise RuntimeError(f"benchmark {code} only {len(df)} bars (<{min_bars})")
    out = df.copy()
    out.attrs = {}
    out.to_csv(path, index=False, encoding="utf-8")
    return df


def load_us_benchmark(refresh: bool = False) -> pd.DataFrame:
    """SPY 日线（后复权）—— 美股主日历。"""
    return _load_index_secid(BENCH_US_SECID, BENCH_US, US_KLINE_DIR, refresh=refresh)


def load_hk_benchmark(refresh: bool = False) -> pd.DataFrame:
    """恒生指数 HSI 日线 —— 港股主日历。"""
    return _load_index_secid(BENCH_HK_SECID, BENCH_HK, HK_KLINE_DIR, refresh=refresh)


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


def fetch_us_pool(top_n: int = 80) -> pd.DataFrame:
    """美股池：优先东财 clist（按总市值降序），失败退回内置常量表 + 逐只 quote 补数。

    返回列：code, name, mktcap, amount, sector
    ⚠️ mktcap 单位是**美元**，amount 单位是**美元**（A股 data.py 里是元/手，别混用）。
    拿不到的字段留 NaN —— 不猜、不填零。
    """
    top_n = int(max(1, top_n))
    try:
        df = _clist_us(top_n)
        df = df.sort_values("mktcap", ascending=False, na_position="last")
        out = df[["code", "name", "mktcap", "amount", "sector"]].head(top_n)
        out = out.reset_index(drop=True)
        out.attrs["source"] = "eastmoney_clist"
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
    out.attrs["mkt_coverage"] = float(out["mktcap"].notna().mean()) if len(out) else 0.0
    return out


def get_us_pool(top_n: int = 80, cache_days: float = 3.0,
                refresh: bool = False) -> pd.DataFrame:
    """美股候选池（code,name,mktcap,amount,sector），带 CSV 缓存。"""
    if not refresh and _cache_fresh(US_POOL_PATH, cache_days):
        try:
            df = pd.read_csv(US_POOL_PATH, dtype={"code": str})
            if len(df) >= min(int(top_n), 100):
                df = df.head(int(top_n)).reset_index(drop=True)
                df.attrs["source"] = "cache"
                df.attrs["mkt_coverage"] = float(df["mktcap"].notna().mean())
                return df
        except Exception:  # noqa: BLE001
            pass
    df = fetch_us_pool(top_n=top_n)
    out = df.copy()
    out.attrs = {}
    out.to_csv(US_POOL_PATH, index=False, encoding="utf-8")
    return df


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
              f"last={d['close'].iloc[-1]:.4g} dropped={len(d.attrs.get('dropped_jumps', []))}")
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
