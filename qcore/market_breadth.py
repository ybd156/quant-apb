"""
市场宽度 / 情绪：从**本地行情**自算，因此有完整历史。

## 为什么不用现成接口

`data/market_history.jsonl` 只有 1 行（`market_live` 是前向采集，没有历史），
和散户情绪一样只能从今天起累积。但市场宽度本质上是**横截面统计** ——
只要手上有 300 只票的日线，就能把历史上每一天的宽度都算出来。

所以这里不联网、不依赖外部接口，直接从 `data/raw/kline/*_h2.csv` 算：

    up_ratio    当日上涨家数 / 有效家数
    limit_up    涨幅 >= 9.8% 的家数（近似涨停）
    limit_down  跌幅 <= -9.8% 的家数
    breadth_z   up_ratio 相对过去 20 日均值的 z 值
    amount_ratio 当日全池成交额 / 过去 20 日均值

## 缓存

算一次落盘 `data/state/market_breadth.csv`（全历史，300 只约 1~2 秒）。
行情更新后重跑 `python -m qcore.market_breadth --refresh`。
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KLINE_DIR = os.path.join(ROOT, "data", "raw", "kline")
STATE = os.path.join(ROOT, "data", "state")
CACHE = os.path.join(STATE, "market_breadth.csv")

COLUMNS = ["date", "up_ratio", "limit_up", "limit_down", "breadth_z", "amount_ratio"]

#: 涨停/跌停近似阈值（A 股主板 ±10%，留 0.2pp 余量避免复权误差）
LIMIT_TH = 0.098


def build(max_files: int = 400, verbose: bool = True) -> pd.DataFrame:
    """从本地日线算全历史市场宽度。返回 [date, up_ratio, limit_up, ...]。"""
    files = sorted(glob.glob(os.path.join(KLINE_DIR, "*_h2.csv")))[:max_files]
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, usecols=["date", "close", "amount"])
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 30:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change()
        frames.append(pd.DataFrame({
            "date": df["date"].to_numpy(),
            "up": (ret > 0).astype(float).to_numpy(),
            "valid": ret.notna().astype(float).to_numpy(),
            "lu": (ret >= LIMIT_TH).astype(float).to_numpy(),
            "ld": (ret <= -LIMIT_TH).astype(float).to_numpy(),
            "amt": pd.to_numeric(df["amount"], errors="coerce").to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    allf = pd.concat(frames, ignore_index=True)
    # 按日横截面聚合：这就是「宽度」的定义，天然因果（只用当日已收盘数据）
    g = allf.groupby("date", as_index=False).agg(
        up=("up", "sum"), valid=("valid", "sum"),
        lu=("lu", "sum"), ld=("ld", "sum"), amt=("amt", "sum"))
    g = g.sort_values("date").reset_index(drop=True)
    g["up_ratio"] = np.where(g["valid"] > 0, g["up"] / g["valid"], np.nan)
    g["limit_up"] = g["lu"]
    g["limit_down"] = g["ld"]
    # z 值用滚动窗口（禁止全样本统计，INV-3）
    ma = g["up_ratio"].rolling(20, min_periods=10).mean()
    sd = g["up_ratio"].rolling(20, min_periods=10).std()
    g["breadth_z"] = np.where(sd > 1e-9, (g["up_ratio"] - ma) / sd, np.nan)
    amt_ma = g["amt"].rolling(20, min_periods=10).mean()
    g["amount_ratio"] = np.where(amt_ma > 0, g["amt"] / amt_ma, np.nan)
    out = g[COLUMNS].copy()
    if verbose:
        print(f"[breadth] {len(files)} 只 / {len(out)} 天  "
              f"{out['date'].iloc[0].date()} → {out['date'].iloc[-1].date()}", flush=True)
    return out


def load(refresh: bool = False) -> pd.DataFrame:
    """读缓存；没有就现算一次并落盘。"""
    if (not refresh) and os.path.isfile(CACHE):
        try:
            df = pd.read_csv(CACHE, parse_dates=["date"])
            if len(df):
                return df
        except Exception:  # noqa: BLE001
            pass
    df = build()
    if len(df):
        try:
            os.makedirs(STATE, exist_ok=True)
            tmp = CACHE + ".tmp"
            df.to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, CACHE)
        except Exception:  # noqa: BLE001
            pass
    return df


def coverage() -> Dict:
    df = load()
    if df is None or df.empty:
        return {"ok": False, "reason": "no_breadth"}
    return {"ok": True, "rows": int(len(df)),
            "first": str(df["date"].iloc[0].date()),
            "last": str(df["date"].iloc[-1].date()),
            "mean_up_ratio": round(float(df["up_ratio"].mean()), 4)}


if __name__ == "__main__":
    import json
    import sys
    if "--refresh" in sys.argv:
        load(refresh=True)
    print(json.dumps(coverage(), ensure_ascii=False, indent=1))
