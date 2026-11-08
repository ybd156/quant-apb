#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qcore/flow_factor.py —— 资金流因子（东方财富 fflow daykline，字段口径已真实请求核实）

## 数据接口（实测 2026-11-08）

    https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get
      ?secid=1.600519 &fields1=f1,f2,f3,f7
      &fields2=f51..f65 &lmt=1000

`data.klines` 是逗号分隔字符串，**实测 15 个字段**，顺序为：

| 位置 | 字段 | 含义 | 单位 |
|---|---|---|---|
| 0  | date        | 日期 | - |
| 1  | main_net    | 主力净流入 | 元 |
| 2  | small_net   | 小单净流入（散户） | 元 |
| 3  | mid_net     | 中单净流入 | 元 |
| 4  | big_net     | 大单净流入 | 元 |
| 5  | super_net   | 超大单净流入 | 元 |
| 6  | main_ratio  | 主力净占比 | % |
| 7  | small_ratio | 小单净占比 | % |
| 8  | mid_ratio   | 中单净占比 | % |
| 9  | big_ratio   | 大单净占比 | % |
| 10 | super_ratio | 超大单净占比 | % |
| 11 | close       | 收盘价（**不复权**） | 元 |
| 12 | chg_pct     | 涨跌幅 | % |
| 13 | f64         | 恒为 0（未使用） | - |
| 14 | f65         | 恒为 0（未使用） | - |

**字段顺序不是照抄文档，是实测反推的**：两条独立样本上
`main_net == big_net + super_net` 逐位相等
（600519 2026-05-14: −12012832 + −368047328 = −380060160 ✓），
且 `main_ratio == big_ratio + super_ratio`（−0.16 + −4.95 = −5.11 ≈ −5.12 ✓），
四层净流入之和 ≈ 0（资金守恒）。这三条同时成立，字段顺序才可判定为正确。

## 两个必须知道的口径陷阱

1. **历史只有 120 个交易日**（`lmt` 给 1000 也只有 120）。这是接口硬限制，
   所以所有因子的样本量上限就是 ~100（扣掉 20 日预热）。不要把它当长历史因子用。
2. **`close` 是不复权价**（600519 实测 1453.00），而本项目行情主口径是后复权
   （同日 8906.67）。因此 `flow_divergence` 里的 5 日收益**不用这里的 close**，
   而是从日线 K 线接口取 `fqt=2` 的后复权收盘价（`_daily_aux`），避免除权日被
   误判成"下跌"。这是 INV-5 的直接要求。

## 因子的经济含义

资金流是**按单笔成交金额的事后切分**，不是账户身份。"小单=散户"是统计推断，
拆单算法会让大资金伪装成小单。主力净流入与当日涨跌高度同期相关（买得多所以涨），
它是**同期描述**而非领先信号；本项目历史实测个股主力净流入 IC ≈ −0.003（t=−0.10），
几乎无预测力。真正可能带信息的是**背离**与**持续性**，即下面第 4、5 号因子。

## 因果性

第 t 行只用 ≤t 的数据：全部统计量挂在 `rolling(...)` 上，`min_periods=窗口长度`
（不足即 NaN，不填 0）。无 `shift(-n)`、无 `center=True`、无全样本 mean/std。
`audit/leak_scan.py` HIGH=0。

用法：
    from qcore.flow_factor import fetch_flow, flow_factors
    raw = fetch_flow("600519")        # DataFrame[date, main_net, ...]
    f   = flow_factors("600519")      # index=date, 6 个因子列
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

__all__ = ["fetch_flow", "flow_factors", "FLOW_FACTORS", "RAW_COLUMNS", "PUBLIC_COLUMNS"]

# ------------------------------------------------------------------ 常量

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://data.eastmoney.com/",
}

_FLOW_URL = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
             "?secid={secid}&fields1=f1,f2,f3,f7"
             "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
             "&lmt={lmt}")

_KLINE_URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
              "?secid={secid}&fields1=f1,f2,f3"
              "&fields2=f51,f52,f53,f54,f55,f56,f57"
              "&klt=101&fqt=2&beg=0&end=20500101&lmt={lmt}")

#: 接口原始 15 字段（顺序见模块 docstring 的实测表）
RAW_COLUMNS = [
    "date", "main_net", "small_net", "mid_net", "big_net", "super_net",
    "main_ratio", "small_ratio", "mid_ratio", "big_ratio", "super_ratio",
    "close", "chg_pct", "_f64", "_f65",
]

#: 对外承诺的 fetch_flow 列
PUBLIC_COLUMNS = ["date", "main_net", "small_net", "mid_net", "big_net",
                  "super_net", "main_ratio", "close", "chg_pct"]

#: 本模块产出的 6 个因子
FLOW_FACTORS = [
    "flow_main_5d",       # 主力 5 日累计净流入 / 20 日均成交额
    "flow_main_ratio",    # 主力净占比 5 日均值
    "flow_super_10d",     # 超大单 10 日累计净流入 / 20 日均成交额
    "flow_divergence",    # 价量资金背离
    "flow_persistence",   # 主力连续净流入天数 / 5
    "flow_retail_vs_main",# 散户与主力的对抗程度
]

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "flow"
_CACHE_TTL = 86400.0          # 1 天
_RETRIES = 3                  # 重试 3 次
_TIMEOUT = 25
_MIN_INTERVAL = 0.25          # 请求节流，避免被限流
_last_call = [0.0]


# ------------------------------------------------------------------ 基础设施

def _norm(code: str) -> str:
    """'600519' / '600519.SH' / 'sh600519' -> '600519'。"""
    s = str(code).strip().upper()
    for suf in (".SH", ".SZ", ".BJ"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = s.replace("SH", "").replace("SZ", "").replace("BJ", "")
    s = s.strip()
    if not s.isdigit():
        raise ValueError(f"无法识别的股票代码: {code!r}")
    return s.zfill(6)


def _secid(code: str) -> str:
    """东财 secid：沪市（6/9 开头，含 688 科创板、900 B 股）用 1.，深市（0/2/3）与北交所（4/8）用 0.。"""
    c = _norm(code)
    return f"1.{c}" if c[0] in "69" else f"0.{c}"


def _get(url: str) -> str:
    """
    HTTP GET，失败重试 3 次，间隔递增（1s / 2s / 3s），最终失败抛异常。

    只依赖标准库 urllib —— 本项目禁止 requests。
    """
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
                time.sleep(1.0 * (i + 1))             # 间隔递增：1s, 2s
    raise RuntimeError(f"请求失败（已重试 {_RETRIES} 次）: {url} :: {last!r}")


def _cache_paths(code: str):
    stem = _CACHE_DIR / _norm(code)
    return Path(str(stem) + ".parquet"), Path(str(stem) + ".pkl")


def _cache_read(code: str) -> Optional[pd.DataFrame]:
    """读缓存；过期（>1 天）或损坏返回 None。"""
    pq, pk = _cache_paths(code)
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


def _cache_write(code: str, df: pd.DataFrame) -> Path:
    """优先 parquet；没有 pyarrow 时自动降级 pickle（本机实测无 pyarrow）。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pq, pk = _cache_paths(code)
    try:
        df.to_parquet(pq, index=False)
        if pk.exists():
            pk.unlink()
        return pq
    except Exception:                                 # noqa: BLE001 - 无 pyarrow / 引擎缺失
        df.to_pickle(pk)
        if pq.exists():
            pq.unlink()
        return pk


# ------------------------------------------------------------------ 抓取

def _fetch_flow_api(code: str, lmt: int = 1000) -> pd.DataFrame:
    """抓资金流原始表（15 字段），已转数值、按日期升序。失败抛 RuntimeError。"""
    url = _FLOW_URL.format(secid=_secid(code), lmt=int(lmt))
    payload = json.loads(_get(url))
    data = payload.get("data") or {}
    rows = data.get("klines") or []
    if not rows:
        raise RuntimeError(
            f"{code}: 资金流接口返回空（data={data!r}）。"
            f"该代码可能已退市/停牌，或 secid 判定错误。"
        )

    split = [r.split(",") for r in rows]
    width = len(split[0])
    if width < len(RAW_COLUMNS):
        raise RuntimeError(
            f"{code}: 资金流字段数 {width} < 期望 {len(RAW_COLUMNS)}，"
            f"接口口径可能变了，请重新核实字段顺序。首行={rows[0]!r}"
        )

    df = pd.DataFrame([r[: len(RAW_COLUMNS)] for r in split], columns=RAW_COLUMNS)
    for col in RAW_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _daily_aux(code: str, lmt: int = 400) -> pd.DataFrame:
    """
    取日线的 **后复权收盘价** 与 **成交额**（fqt=2），用于两个分母/收益计算。

    为什么要单独取：
      - `amount`（成交额，元，不复权口径）是 `flow_*_d` 分母的基准；
      - 资金流接口的 `close` 不复权，除权日会被误算成暴跌，
        所以 5 日收益必须用这里的后复权收盘价（INV-5）。
    """
    url = _KLINE_URL.format(secid=_secid(code), lmt=int(lmt))
    payload = json.loads(_get(url))
    data = payload.get("data") or {}
    rows = data.get("klines") or []
    if not rows:
        raise RuntimeError(f"{code}: 日线接口返回空，无法取得成交额/后复权收盘价。")

    split = [r.split(",") for r in rows]
    if len(split[0]) < 7:
        raise RuntimeError(f"{code}: 日线字段数 {len(split[0])} < 7，接口口径可能变了。")

    df = pd.DataFrame([r[:7] for r in split],
                      columns=["date", "open", "close_adj", "high", "low",
                               "volume", "amount"])
    for col in ["open", "close_adj", "high", "low", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])[["date", "amount", "close_adj"]]
    return df.sort_values("date").reset_index(drop=True)


def _load(code: str, with_aux: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """
    取「资金流 + 成交额 + 后复权收盘价」的合并表（按 date 左连接）。

    with_aux=False 时只取资金流（fetch_flow 用，不需要成交额，也就不会被
    日线接口的故障连累）。
    """
    if use_cache:
        cached = _cache_read(code)
        if cached is not None:
            if not with_aux or "amount" in cached.columns:
                return cached

    flow = _fetch_flow_api(code, lmt=1000)
    if not with_aux:
        _cache_write(code, flow)
        return flow

    aux = _daily_aux(code)
    merged = flow.merge(aux, on="date", how="left")
    _cache_write(code, merged)
    return merged


def fetch_flow(code: str, lmt: int = 1000, use_cache: bool = True) -> pd.DataFrame:
    """
    返回 DataFrame[date, main_net, small_net, mid_net, big_net, super_net,
    main_ratio, close, chg_pct]，按日期升序。

    金额单位：元；`main_ratio` 与 `chg_pct` 单位：%。
    注意 `close` 是**不复权**价（接口原生口径）。

    lmt 是「取最近 lmt 个交易日」的切片上限；接口本身最多只给约 120 天。
    失败抛异常由调用方处理（已内置 3 次重试）。
    """
    df = _load(code, with_aux=False, use_cache=use_cache)
    out = df[PUBLIC_COLUMNS].copy()
    if lmt and lmt > 0:
        out = out.tail(int(lmt)).reset_index(drop=True)
    return out


# ------------------------------------------------------------------ 因子

def _flow_factors_from_frame(d: pd.DataFrame) -> pd.DataFrame:
    """
    从合并表（资金流 + amount + close_adj）算 6 个因子。

    全部因果：只用 rolling（窗口内向后看），min_periods=窗口长度 → 预热期 NaN。
    """
    if len(d) == 0:
        return pd.DataFrame(columns=FLOW_FACTORS, dtype=float)

    d = d.sort_values("date").reset_index(drop=True)
    idx = pd.DatetimeIndex(d["date"].to_numpy())

    main = pd.Series(d["main_net"].to_numpy(float), index=idx)
    small = pd.Series(d["small_net"].to_numpy(float), index=idx)
    super_ = pd.Series(d["super_net"].to_numpy(float), index=idx)
    ratio = pd.Series(d["main_ratio"].to_numpy(float), index=idx)

    # 成交额：0 或缺失视为无效（停牌），换成 NaN 避免除零得到 inf
    amount = pd.Series(d["amount"].to_numpy(float), index=idx) if "amount" in d.columns \
        else pd.Series(np.nan, index=idx)
    amount = amount.where(amount > 0)

    # 后复权收盘价；缺失时退回资金流的不复权 close（仅作兜底，会在注释里标注）
    if "close_adj" in d.columns:
        close = pd.Series(d["close_adj"].to_numpy(float), index=idx)
    else:
        close = pd.Series(d["close"].to_numpy(float), index=idx)
    close = close.where(close > 0)

    # 20 日均成交额 = 分母基准（滚动，因果）
    amount20 = amount.rolling(20, min_periods=20).mean()

    out = pd.DataFrame(index=idx)

    # ---- 1. flow_main_5d：主力 5 日累计净流入 / 20 日均成交额 -------------
    # 经济含义：把「5 天里主力净买了多少钱」除以「平均一天成交多少钱」，
    # 得到无量纲的"主力 5 日净买入相当于几天的成交额"。
    # 用成交额做分母而不是市值/绝对值，是为了跨股票可比（大票小票同一尺度）。
    main_5 = main.rolling(5, min_periods=5).sum()
    out["flow_main_5d"] = main_5 / amount20

    # ---- 2. flow_main_ratio：主力净占比 5 日均值 --------------------------
    # 经济含义：主力净流入占当日成交额的百分比（接口原生 % 口径）。
    # 取 5 日均值是为了压掉单日的噪声与拆单误差；单位保持 %。
    out["flow_main_ratio"] = ratio.rolling(5, min_periods=5).mean()

    # ---- 3. flow_super_10d：超大单 10 日累计净流入 / 20 日均成交额 --------
    # 经济含义：超大单（单笔 ≥100 万）最接近"机构/真主力"，
    # 比大单更少受游资快进快出干扰；10 日窗口捕捉中期建仓/派发。
    super_10 = super_.rolling(10, min_periods=10).sum()
    out["flow_super_10d"] = super_10 / amount20

    # ---- 4. flow_divergence：价量资金背离 --------------------------------
    # 经济含义：sign(5日涨幅) × (−主力5日净流入强度)。
    #   价格涨 + 主力流出  → 正值（拉高出货，危险）
    #   价格跌 + 主力流入  → 负值（逆势吸筹，机会）
    # 这是资金流里唯一可能带领先性的构造：把"同期相关"的那部分抵掉。
    ret_5 = close / close.shift(5) - 1.0
    out["flow_divergence"] = np.sign(ret_5.to_numpy(float)) * (-out["flow_main_5d"].to_numpy(float))

    # ---- 5. flow_persistence：主力连续净流入天数 / 5 ----------------------
    # 经济含义：主力净流入"连续"了几天（今天为止往前数）。
    # 持续净流入比单日大额流入更像真实建仓（单日大额可能是对倒/ETF 申赎）。
    # 实现：pos 的连续段编号 = cumsum(状态变化)，再取组内序号；
    # 非净流入日记 0。该因子不依赖跨日窗口，故无预热期。
    pos = main.to_numpy(float) > 0
    n = len(pos)
    if n:
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = pos[1:] != pos[:-1]
        seg = np.cumsum(change)
        run = pd.Series(pos.astype(np.int64)).groupby(seg).cumcount().to_numpy() + 1
        streak = np.where(pos, run, 0).astype(float)
    else:
        streak = np.zeros(0, dtype=float)
    out["flow_persistence"] = pd.Series(streak, index=idx) / 5.0

    # ---- 6. flow_retail_vs_main：散户与主力的对抗程度 ---------------------
    # 经济含义：小单净流入 / (|主力净流入| + eps)。
    #   正值 = 散户在买（主力在卖时尤其显著 → 散户接盘）
    #   负值 = 散户在卖（主力在买时尤其显著 → 散户割肉给主力）
    # eps 取 20 日均成交额的 1%：主力净流入≈0 时若用 1e-9，比值会爆炸成 1e6 级
    # 噪声。代价是该因子同样需要 20 日预热。
    eps = 0.01 * amount20
    out["flow_retail_vs_main"] = small / (main.abs() + eps)

    return out[FLOW_FACTORS].astype(float)


def flow_factors(code: str, use_cache: bool = True) -> pd.DataFrame:
    """
    返回 index=date、columns=FLOW_FACTORS 的日频因子表（6 列）。

    预热期（20 日）输出 NaN，不填 0。
    成交额/后复权收盘价取不到时抛异常（已重试 3 次）——不编造数据。
    """
    d = _load(code, with_aux=True, use_cache=use_cache)
    out = _flow_factors_from_frame(d)
    out.index.name = "date"
    return out


# ------------------------------------------------------------------ 自检

def _synthetic_flow_frame(n: int = 2600, seed: int = 7) -> pd.DataFrame:
    """合成 2600 行资金流合并表，仅用于性能与截断不变性测试（不进任何生产路径）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=n)
    amount = rng.lognormal(18.6, 0.55, n)                 # 成交额 ~1e8 量级
    main = rng.normal(0.0, 0.06, n) * amount             # 主力净流入
    super_ = main * rng.uniform(0.5, 0.9, n)
    big = main - super_
    small = -main * rng.uniform(0.4, 1.2, n)
    mid = -(main + small)                                # 资金守恒：四层合计≈0
    close_adj = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, n))
    return pd.DataFrame({
        "date": dates,
        "main_net": main, "small_net": small, "mid_net": mid,
        "big_net": big, "super_net": super_,
        "main_ratio": main / amount * 100.0,
        "small_ratio": small / amount * 100.0,
        "mid_ratio": mid / amount * 100.0,
        "big_ratio": big / amount * 100.0,
        "super_ratio": super_ / amount * 100.0,
        "close": close_adj,                              # 不复权价的替身（仅测试）
        "chg_pct": np.zeros(n),
        "_f64": 0.0, "_f65": 0.0,
        "amount": amount,
        "close_adj": close_adj,
    })


def _print_stats(f: pd.DataFrame, label: str) -> None:
    """打印每个因子的 非NaN行数 / 均值 / 标准差 / 最大值 / 最小值。"""
    print(f"\n--- {label} 因子统计 (n_rows={len(f)}, 区间 {f.index.min()} .. {f.index.max()}) ---")
    if len(f) == 0:
        print("  （空表，无法统计）")
        return
    # describe() 一次给出 count/mean/std/min/max（诊断用途，非特征计算）
    desc = f.describe().T
    head = f"{'因子':<22}{'非NaN':>7}{'均值':>13}{'标准差':>13}{'最大值':>13}{'最小值':>13}"
    print(head)
    print("-" * len(head))
    for name in FLOW_FACTORS:
        if name not in desc.index:
            print(f"{name:<22}{'不可用':>7}")
            continue
        r = desc.loc[name]
        print(f"{name:<22}{int(r['count']):>7}{r['mean']:>13.6f}"
              f"{r['std']:>13.6f}{r['max']:>13.6f}{r['min']:>13.6f}")


def _truncation_test(frame: pd.DataFrame, n_trunc: int, label: str) -> bool:
    """
    截断不变性：只用前 n_trunc 行算 vs 用全部行算，前 n_trunc 行必须逐位相同。

    这是 INV-1（无未来函数）的运行时验证：任何依赖未来行的构造都会在这里炸掉。
    """
    n_total = len(frame)
    n_trunc = int(min(n_trunc, n_total))
    full = _flow_factors_from_frame(frame)
    part = _flow_factors_from_frame(frame.iloc[:n_trunc].copy())
    a = full.iloc[: len(part)].to_numpy(float)
    b = part.to_numpy(float)
    if a.shape != b.shape:
        print(f"  [FAIL] {label}: shape 不一致 full[:{len(part)}]={a.shape} part={b.shape}")
        return False
    ok = bool(np.array_equal(a, b, equal_nan=True))
    nan_a = int(np.isnan(a).sum())
    nan_b = int(np.isnan(b).sum())
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: 前 {n_trunc}/{n_total} 行逐位相同="
          f"{ok}  (NaN 个数 {nan_a} vs {nan_b})")
    return ok


def _corr_report(f: pd.DataFrame, label: str) -> None:
    """打印因子间相关系数矩阵上三角的最大绝对相关；>0.9 说明因子重复。"""
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
          f"（{FLOW_FACTORS[i]} vs {FLOW_FACTORS[j]}）{flag}")


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
        print(f"[资金流因子] {code}")
        print("=" * 92)

        t0 = time.perf_counter()
        try:
            raw = fetch_flow(code)
        except Exception as exc:                      # noqa: BLE001
            print(f"  不可用：资金流接口取数失败 -> {exc}")
            all_ok = False
            continue
        t_fetch = time.perf_counter() - t0

        print(f"  原始资金流: {len(raw)} 行, {raw['date'].min()} .. {raw['date'].max()} "
              f"(耗时 {t_fetch:.2f}s)")

        # 口径自检（两条独立恒等式，同时成立才能确认字段顺序）：
        #   ① main_net == big_net + super_net
        #   ② small + mid + big + super ≈ 0（资金守恒；注意**不能再加 main_net**，
        #      因为 main 本身就是 big+super，加进去会重复计算 2 倍）
        chk = float((raw["main_net"] - raw["big_net"] - raw["super_net"]).abs().max())  # leak-scan: ok 口径自检
        tot = float((raw["small_net"] + raw["mid_net"] + raw["big_net"]
                     + raw["super_net"]).abs().max())                                   # leak-scan: ok 口径自检
        rel = tot / float(raw["main_net"].abs().mean())                                 # leak-scan: ok 口径自检
        print(f"  口径校验: |main-(big+super)|max={chk:.2f} 元; "
              f"|small+mid+big+super|max={tot:.2f} 元 (占日均|主力| {rel * 100:.2f}%)")

        t1 = time.perf_counter()
        try:
            f = flow_factors(code)                    # 公开接口（含取数/缓存）
        except Exception as exc:                      # noqa: BLE001
            print(f"  不可用：因子计算失败（成交额/后复权价取不到）-> {exc}")
            all_ok = False
            continue
        t_api = time.perf_counter() - t1

        # 纯计算耗时（不含任何网络 IO），这才是"计算性能"该看的数字
        merged = _load(code)
        t2 = time.perf_counter()
        f2 = _flow_factors_from_frame(merged)
        t_calc = time.perf_counter() - t2
        same = bool(np.array_equal(f.to_numpy(float), f2.to_numpy(float), equal_nan=True))
        if not same:
            print("  [FAIL] flow_factors() 与 _flow_factors_from_frame() 结果不一致")
            all_ok = False

        _print_stats(f, f"{code} 真实数据")
        print(f"  耗时: flow_factors() 全流程 {t_api:.2f}s（含取数/缓存）; "
              f"纯计算 {t_calc * 1000:.2f} ms / {len(merged)} 行")

        print("\n  截断不变性（真实数据）:")
        n_trunc = 500 if len(merged) > 500 else max(2, len(merged) // 2)
        all_ok &= _truncation_test(merged, n_trunc, f"{code} 真实 {len(merged)} 行")

        print("\n  截断不变性（合成 2600 行，按要求用前 500 行）:")
        syn = _synthetic_flow_frame(2600)
        all_ok &= _truncation_test(syn, 500, "合成 2600 行")

        print("\n  性能（合成 2600 行纯计算，要求 < 2s）:")
        t2 = time.perf_counter()
        _flow_factors_from_frame(syn)
        dt = time.perf_counter() - t2
        print(f"    flow_factors(2600 行) = {dt * 1000:.1f} ms  "
              f"[{'PASS' if dt < 2.0 else 'FAIL'}]")

        print("\n  因子间相关性:")
        _corr_report(f, code)

    print("\n" + "=" * 92)
    print(f"总判定: {'全部通过' if all_ok else '存在失败项（见上）'}")
    print("=" * 92)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
