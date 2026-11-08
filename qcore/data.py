"""
数据层：东方财富直连（纯标准库 + pandas），本地 CSV 缓存，A股**后复权**日线。

⚠️ 复权口径（血泪教训，勿改）
    东财 `fqt=1`（前复权）是**减法复权**：price = raw − 累计分红。
    对高分红长历史股，累计分红超过历史股价 → 价格被压到 0.01 甚至**负数**，
    接口自带的 pct_chg / amplitude / change 三个字段也随之一并污染
    （实测 601088 最低价 -9.55，单日涨跌幅 +17400%，异常 bar 639 个）。
    因此统一用 `fqt=2`（后复权，乘法复权，永不越零）：601088 最低 +16.63，异常 bar 1 个。
    代价：价格绝对值不能与今价直接比较 —— 本项目只用收益率与相对量，无影响。

其他设计：
  1. 原始数据只落盘一次，文件名带复权标记，避免新旧口径混用；
  2. 限速 + 指数退避，东财对高频请求会直接断连；
  3. 涨跌幅由本地 close 重算，不信任接口字段。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
KLINE_DIR = os.path.join(RAW_DIR, "kline")
INDEX_DIR = os.path.join(RAW_DIR, "index")
META_DIR = os.path.join(RAW_DIR, "meta")
for _d in (KLINE_DIR, INDEX_DIR, META_DIR):
    os.makedirs(_d, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
KLINE_FIELDS = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLINE_COLS = ["date", "open", "close", "high", "low", "volume", "amount",
              "amplitude", "pct_chg", "change", "turnover"]

ADJUST = 2          # 2=后复权（唯一正确口径）；1=前复权（减法，禁用）
_MIN_INTERVAL = 0.22  # 全局最小请求间隔（秒），防被断连
_last_call = [0.0]


def _get_json(url: str, retries: int = 5, timeout: int = 25) -> dict:
    last = None
    for i in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            _last_call[0] = time.time()
            return json.loads(body.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            _last_call[0] = time.time()
            last = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"request failed after {retries}: {url} :: {last}")


def secid(code: str) -> str:
    """代码 -> 东财 secid。1=沪市(6/5/9)，0=深市(0/1/2/3)。"""
    code = str(code).zfill(6)
    if code[0] in "569":
        return f"1.{code}"
    return f"0.{code}"


def index_secid(code: str) -> str:
    code = str(code).zfill(6)
    return f"0.{code}" if code.startswith("399") else f"1.{code}"


# ---------------------------------------------------------------- kline
def fetch_kline(code: str, adjust: int = ADJUST, beg: str = "19900101",
                end: str = "20500101") -> pd.DataFrame:
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid(code)}&fields1=f1,f2,f3,f4,f5,f6&fields2={KLINE_FIELDS}"
           f"&klt=101&fqt={adjust}&beg={beg}&end={end}&lmt=100000")
    d = _get_json(url)
    data = d.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"no kline for {code}")
    df = pd.DataFrame([ln.split(",") for ln in klines], columns=KLINE_COLS)
    for c in KLINE_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = str(code).zfill(6)
    df.attrs["name"] = data.get("name", "")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # 用本地 close 重算涨跌幅，不信任接口字段
    df["pct_chg"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    df.loc[0, "pct_chg"] = 0.0
    return df


def _cache_path(code: str, adjust: int) -> str:
    return os.path.join(KLINE_DIR, f"{code}_h{adjust}.csv")


def load_kline(code: str, adjust: int = ADJUST, refresh: bool = False,
               max_age_days: float = 5.0) -> pd.DataFrame:
    code = str(code).zfill(6)
    path = _cache_path(code, adjust)
    if (not refresh) and os.path.exists(path):
        if (time.time() - os.path.getmtime(path)) / 86400 < max_age_days:
            df = pd.read_csv(path, parse_dates=["date"])
            df["code"] = code
            return df
    df = fetch_kline(code, adjust=adjust)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------- universe
def fetch_universe(max_pages: int = 8, page_size: int = 100) -> pd.DataFrame:
    """
    全 A 股列表。⚠️ 东财 clist 单页最多返回 100 行（pz 设再大也只给 100），
    必须分页，否则只会拿到市值前 100 只 —— 这会让股票池退化成"超级大盘股"。
    """
    base = ("https://push2.eastmoney.com/api/qt/clist/get?pn={pn}&pz={pz}&po=1&np=1"
            "&fltt=2&invt=2&fid=f20&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f12,f14,f20,f6,f3,f100")
    frames = []
    for pn in range(1, max_pages + 1):
        d = _get_json(base.format(pn=pn, pz=page_size))
        diff = (d.get("data") or {}).get("diff") or []
        if not diff:
            break
        frames.append(pd.DataFrame(diff))
        if len(diff) < page_size:
            break
    if not frames:
        raise RuntimeError("universe fetch returned nothing")
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "f12": "code", "f14": "name", "f20": "mktcap", "f6": "amount",
        "f3": "pct_chg", "f100": "industry"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in ("mktcap", "amount", "pct_chg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("code").reset_index(drop=True)


def get_pool(top_n: int = 300, min_amount: float = 1e8, exclude_st: bool = True,
             exclude_bj: bool = True, cache_days: float = 3.0) -> pd.DataFrame:
    """
    候选大池（**只决定"下载哪些股票"，不决定"某天能交易哪些"**）。
    池子按今日市值取前 top_n —— 这本身带生存者偏差，真正的点对点选池
    由 dataset 里的 UNIV 掩码（按当日成交额滚动排名）完成。
    """
    path = os.path.join(META_DIR, "pool.csv")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 86400 < cache_days:
        uni = pd.read_csv(path, dtype={"code": str})
    else:
        uni = fetch_universe()
        uni.to_csv(path, index=False, encoding="utf-8")
    u = uni.copy()
    if exclude_st:
        u = u[~u["name"].astype(str).str.contains("ST|退", na=False)]
    if exclude_bj:
        u = u[~u["code"].str.startswith(("8", "4", "9"))]
    u = u[u["amount"] >= min_amount]
    return u.sort_values("mktcap", ascending=False).head(top_n).reset_index(drop=True)


# 向后兼容
def get_universe(top_n: int = 300, **kw) -> pd.DataFrame:
    return get_pool(top_n=top_n, **kw)


# ---------------------------------------------------------------- index
def fetch_index(code: str = "000300") -> pd.DataFrame:
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={index_secid(code)}&fields1=f1,f2,f3,f4,f5,f6&fields2={KLINE_FIELDS}"
           f"&klt=101&fqt=1&beg=19900101&end=20500101&lmt=100000")
    d = _get_json(url)
    data = d.get("data") or {}
    rows = [ln.split(",") for ln in (data.get("klines") or [])]
    if not rows:
        raise RuntimeError(f"no index kline {code}")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    for c in KLINE_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["pct_chg"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    df.loc[0, "pct_chg"] = 0.0
    return df


def load_index(code: str = "000300") -> pd.DataFrame:
    path = os.path.join(INDEX_DIR, f"{code}.csv")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 86400 < 5:
        return pd.read_csv(path, parse_dates=["date"])
    df = fetch_index(code)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------- panel
def build_panel(codes: Iterable[str], start: str = "2010-01-01",
                min_bars: int = 500, adjust: int = ADJUST,
                clip: bool = False, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    clip=False（默认）：保留**上市以来全部历史**，让指标拿到完整预热期。
        这一点很关键 —— 若先按 start 截断再算 MA250/RSI，增量运行与批量运行
        的信号值会不一致，等于自己给自己制造了"未来函数"式的偏差。
    clip=True：老行为，按 start 截断（仅用于调试/特殊场景）。
    """
    out: Dict[str, pd.DataFrame] = {}
    codes = [str(c).zfill(6) for c in codes]
    for i, c in enumerate(codes, 1):
        try:
            df = load_kline(c, adjust=adjust)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [skip] {c}: {e}")
            continue
        if clip:
            df = df[df["date"] >= pd.Timestamp(start)]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["close"] > 0) & (df["volume"] > 0)]
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        if len(df) < min_bars:
            if verbose:
                print(f"  [skip] {c}: bars={len(df)} < {min_bars}")
            continue
        out[c] = df
        if verbose and i % 50 == 0:
            print(f"  ... {i}/{len(codes)}", flush=True)
    return out


def download(codes: Iterable[str], adjust: int = ADJUST, verbose: bool = True) -> int:
    """批量下载（幂等），返回成功数。"""
    ok = 0
    codes = [str(c).zfill(6) for c in codes]
    for i, c in enumerate(codes, 1):
        try:
            load_kline(c, adjust=adjust)
            ok += 1
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [fail] {c}: {e}", flush=True)
        if verbose and i % 50 == 0:
            print(f"  ... {i}/{len(codes)} ok={ok}", flush=True)
    return ok


if __name__ == "__main__":
    pool = get_pool(top_n=300)
    print(f"pool: {len(pool)}")
    n = download(pool["code"].tolist())
    print(f"downloaded {n}/{len(pool)}")
