"""
成交价口径探测（评审 R22）—— 回测"次日开盘"vs 模拟盘"当日收盘"差多少？

## 为什么要量它

回测（`lowturnover_bt` / `backtest`）按 **T+1 开盘**成交；
模拟盘 `paper_auto._px_for()` 却按 **T 日收盘**（as_of）成交。
同一份名单、同一套规则，两个基准 → 净值不可直接比较。

改成"次日开盘"有两个后果，必须先量清：
1. **净值不可比**：历史成交要重算，否则旧净值与新口径混在一条曲线上；
2. **偏差多大**：如果只差几 bps，那这是"口径洁癖"；如果差几十 bps，
   那模拟盘一直在系统性地占（或吃亏）一个隔夜跳空。

## 本脚本量什么

对池内股票、最近 N 个交易日，算**隔夜跳空**：

    gap_t = open[t+1] / close[t] - 1

用收盘价成交 = 拿到 `close[t]`；用次日开盘成交 = 拿到 `close[t] * (1+gap_t)`。
所以「按收盘成交」相对「按次日开盘成交」的系统性偏差 ≈ `-gap` 的均值。

再对**当日 top-k 名单**单独算一遍：只关心真正会买的那些票的 gap。

## 用法

    python -m tools.fill_basis_probe            # 默认池内 100 只、最近 60 个交易日
    python -m tools.fill_basis_probe --stocks 200 --days 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore.data import get_pool, load_kline  # noqa: E402


def _gaps(code: str, days: int) -> np.ndarray:
    df = load_kline(code)
    if df is None or len(df) < days + 3:
        return np.array([])
    d = df.tail(days + 2).reset_index(drop=True)
    c = d["close"].to_numpy(dtype=float)
    o = d["open"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = o[1:] / c[:-1] - 1.0
    g = g[np.isfinite(g)]
    return g[-days:] if g.size else np.array([])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="成交价口径探测")
    ap.add_argument("--stocks", type=int, default=100)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    codes = get_pool(top_n=max(a.stocks, 20))["code"].tolist()[: a.stocks]
    all_gaps: List[np.ndarray] = []
    per_code = {}
    for c in codes:
        g = _gaps(c, a.days)
        if g.size:
            all_gaps.append(g)
            per_code[c] = float(np.mean(g))
    if not all_gaps:
        print("没有可用数据")
        return 1
    G = np.concatenate(all_gaps)

    # 当日名单的 gap（如有 picks）
    pick_gap = None
    try:
        p = os.path.join(ROOT, "data", "state", "picks.json")
        if os.path.isfile(p):
            d = json.load(open(p, encoding="utf-8"))
            rows = (d.get("rows") or [])[:10]
            gs = [_gaps(str(r.get("code")), 5).mean() for r in rows
                  if r.get("code") and _gaps(str(r.get("code")), 5).size]
            if gs:
                pick_gap = float(np.mean(gs))
    except Exception:  # noqa: BLE001
        pass

    out = {
        "n_codes": len(per_code), "n_days": int(a.days), "n_obs": int(G.size),
        "mean_gap_bps": round(float(np.mean(G)) * 1e4, 2),
        "median_gap_bps": round(float(np.median(G)) * 1e4, 2),
        "std_gap_bps": round(float(np.std(G)) * 1e4, 1),
        "p05_bps": round(float(np.percentile(G, 5)) * 1e4, 1),
        "p95_bps": round(float(np.percentile(G, 95)) * 1e4, 1),
        "pick_gap_bps": None if pick_gap is None else round(pick_gap * 1e4, 2),
        "note": ("gap = open[t+1]/close[t]-1。模拟盘按 close[t] 成交 = 相对"
                 "「次日开盘」口径系统性偏差约 -mean_gap_bps（单边、未计成本）"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n读法：池内 {out['n_codes']} 只 × 最近 {a.days} 天（{out['n_obs']} 个观测）")
    print(f"  隔夜跳空均值 {out['mean_gap_bps']:+.2f} bps，中位 {out['median_gap_bps']:+.2f} bps，"
          f"5–95 分位 {out['p05_bps']:+.1f} ~ {out['p95_bps']:+.1f} bps")
    print(f"  → 按收盘成交相对次日开盘成交，系统性偏差 ≈ {-out['mean_gap_bps']:+.2f} bps/笔（单边）")
    if out["pick_gap_bps"] is not None:
        print(f"  → 当日 top-10 名单的隔夜跳空均值 {out['pick_gap_bps']:+.2f} bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
