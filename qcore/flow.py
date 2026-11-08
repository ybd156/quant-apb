"""
资金分层 —— 把"主力 / 游资 / 散户"从成交里拆出来。

## 数据基础

东方财富资金流按单笔金额分四层（每层都有净流入，单位元）：

| 层 | 单笔金额 | 通常被解读为 |
|---|---|---|
| 超大单 | ≥ 100 万 | 机构 / 主力 |
| 大单 | 20–100 万 | 机构或**游资**（快进快出） |
| 中单 | 4–20 万 | 半专业 |
| 小单 | < 4 万 | **散户** |

主力净流入 = 超大单 + 大单；散户净流入 = 小单。

## 关键提醒（不要神化）

1. **这是按成交金额的事后切分，不是账户身份**。"小单=散户"是统计推断，
   不是事实；拆单算法会让大资金伪装成小单。
2. 主力净流入与当日涨跌**高度同期相关**（买得多所以涨），
   真正有预测力的是**背离**：价涨但主力流出、价跌但主力流入。
3. 游资没有独立字段，只能从"大单占比高 + 换手高 + 持仓周期短"间接推断。

## 本模块做什么

只做三件事：
  1. 抓取并规范化四层资金流
  2. 构造派生特征（主力占比、散户占比、背离、游资强度）
  3. 提供可检验的方向信号（方向由实测决定，不预设）
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}
_MIN_INTERVAL = 0.25
_last = [0.0]

FLOW_COLS = ["date", "main_net", "small_net", "mid_net", "big_net", "huge_net",
             "main_pct", "small_pct", "mid_pct", "big_pct", "huge_pct",
             "close", "pct_chg", "x1", "x2"]


def _get(url: str, retries: int = 3, timeout: int = 25) -> str:
    last = None
    for i in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
            _last[0] = time.time()
            return body
        except Exception as e:  # noqa: BLE001
            _last[0] = time.time()
            last = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"flow fetch failed: {url} :: {last}")


def _secid(code: str) -> str:
    c = str(code).zfill(6)
    return f"1.{c}" if c[0] in "569" else f"0.{c}"


def fetch_flow(code: str) -> pd.DataFrame:
    """日频资金流（东财最多约 120 个交易日）。"""
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           f"?secid={_secid(code)}&fields1=f1,f2,f3,f7"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           "&klt=101&lmt=0")
    d = json.loads(_get(url))
    rows = (d.get("data") or {}).get("klines") or []
    if not rows:
        raise RuntimeError(f"no flow for {code}")
    df = pd.DataFrame([r.split(",") for r in rows], columns=FLOW_COLS)
    for c in FLOW_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = str(code).zfill(6)
    return df.sort_values("date").reset_index(drop=True)


def features(df: pd.DataFrame, win: int = 5) -> pd.DataFrame:
    """
    派生特征（全部因果）：
      main_ratio   主力净流入 / 总成交额（用净额占比字段的近似）
      retail_ratio 小单净流入占比
      divergence   量价背离：涨跌幅 > 0 但主力净流出（或反之）
      hot_money    游资强度：大单占比高 + 换手高
      accum_{w}    近 w 日主力净流入累计
    """
    d = df.copy()
    out = pd.DataFrame(index=pd.RangeIndex(len(d)))
    out["date"] = d["date"].values
    out["main_pct"] = d["main_pct"]
    out["retail_pct"] = d["small_pct"]
    out["pct_chg"] = d["pct_chg"]

    out["main_ratio"] = d["main_pct"] / 100.0
    out["retail_ratio"] = d["small_pct"] / 100.0
    out["hot_money"] = (d["big_pct"].abs() / 100.0) * (d["main_pct"].abs() / 100.0)
    out["divergence"] = np.where(
        (d["pct_chg"] > 0) & (d["main_net"] < 0), -1.0,
        np.where((d["pct_chg"] < 0) & (d["main_net"] > 0), 1.0, 0.0))
    for w in (5, 20):
        out[f"accum_{w}"] = d["main_net"].rolling(w, min_periods=max(2, w // 2)).sum() / 1e8
        out[f"retail_accum_{w}"] = d["small_net"].rolling(w, min_periods=max(2, w // 2)).sum() / 1e8
    return out


def flow_signal(df: pd.DataFrame, win: int = 5) -> np.ndarray:
    """
    资金流方向信号 ∈ [-1,1]：
      主力累计净流入（标准化）为正 → 看多；散户累计净流入为正 → 看空（反向）
    权重 0.7 给主力、0.3 给散户反向，是**先验**，不是实测结论。
    """
    f = features(df, win)
    m = f[f"accum_{win}"].to_numpy(dtype=float)
    r = f[f"retail_accum_{win}"].to_numpy(dtype=float)
    def z(x):
        s = pd.Series(x)
        mu = s.rolling(60, min_periods=20).mean()
        sd = s.rolling(60, min_periods=20).std()
        return ((s - mu) / sd.replace(0, np.nan)).clip(-3, 3).to_numpy()
    zm, zr = z(m), z(r)
    sig = 0.7 * np.tanh(zm) - 0.3 * np.tanh(zr)
    return np.clip(sig, -1, 1)


def classify_participant(df: pd.DataFrame) -> pd.Series:
    """粗分类当日主导参与者（仅作解释用，不作信号）。"""
    d = df
    main = d["main_net"].to_numpy(float)
    big = d["big_net"].to_numpy(float)
    huge = d["huge_net"].to_numpy(float)
    small = d["small_net"].to_numpy(float)
    total = np.abs(main) + np.abs(small) + 1e-9
    main_share = np.abs(main) / total
    out = np.where(main_share > 0.7, np.where(main > 0, "主力买入", "主力卖出"),
                   np.where(small > 0, "散户接盘", "散户离场"))
    hot = (np.abs(big) > np.abs(huge) * 1.5) & (np.abs(big) / (np.abs(main) + 1e-9) > 0.6)
    out = np.where(hot, "游资主导", out)
    return pd.Series(out, index=d.index, name="participant")
