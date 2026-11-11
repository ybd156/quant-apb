"""
成交口径审计（评审 R22）—— 把"模拟盘按收盘成交"的口径差，逐笔记在账上。

## 背景

回测（`lowturnover_bt` / `backtest`）用 **T+1 开盘**成交；
模拟盘 `paper_auto._px_for()` 用 **T 日收盘**（或盘中实时价）成交。
两者不是同一个基准，所以：

- 模拟盘的净值**不能**直接和回测净值比；
- 偏差有多大，得用真实成交记录逐笔量，而不是拍一个常数。

## 本模块做什么

读 `data/paper/trades.jsonl`（模拟盘自动成交），对每一笔：

    gap_bps = (次日开盘价 / 成交价 - 1) × 1e4      # 买入视角
    gap_bps = (成交价 / 次日开盘价 - 1) × 1e4      # 卖出视角（方向反过来）

正的 gap_bps 表示"按收盘成交"相对"按次日开盘成交"**占了便宜**（买得更便宜/卖得更贵）。
聚合后写 `data/state/fill_audit.json`，供 UI 显示与后续决策使用。

## 边界（不许当结论用）

1. 只覆盖**有成交记录**的票；样本少时均值不可靠（同样受 `MIN_TRADES` 精神约束）；
2. 次日开盘价取本地日线（后复权口径）—— 若成交价来自实时报价（`price_basis="live"`），
   两者口径不完全一致，会额外引入误差，故 `live` 的笔单独统计；
3. 它**不改任何成交**，只做归因。真要改成交模型，是另一件事（需重算历史净值）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES = os.path.join(ROOT, "data", "paper", "trades.jsonl")
STATE = os.path.join(ROOT, "data", "state")
OUT = os.path.join(STATE, "fill_audit.json")

#: 少于这么多笔就只报数、不下"平均偏差"的结论（与 thresholds.MIN_TRADES 同精神）
MIN_TRADES = 30


def _read_trades() -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isfile(TRADES):
        return rows
    with open(TRADES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return rows


def _gap_from_kline(code: str, day: str) -> Optional[Dict]:
    """用**同一条日线**算隔夜跳空：gap = open[t+1]/close[t] - 1。

    为什么不能拿 `次日开盘(后复权) / 成交价(原始)`：那是两个价格口径相除。
    实测第一次跑出 **+78,023 bps（780%）** —— 后复权价是原始价的十几倍，
    除出来的东西没有意义。比率必须取自同一序列。
    """
    try:
        from .data import load_kline
        df = load_kline(str(code))
        if df is None or not len(df):
            return None
        d = pd.to_datetime(df["date"]).dt.normalize().to_numpy()
        t = np.datetime64(pd.Timestamp(day).normalize())
        pos = int(np.searchsorted(d, t, side="left"))
        if pos >= len(d) or d[pos] != t or pos + 1 >= len(d):
            return None
        c0 = float(df["close"].to_numpy()[pos])
        o1 = float(df["open"].to_numpy()[pos + 1])
        if c0 <= 0 or o1 <= 0:
            return None
        return {"close_t": c0, "open_next": o1, "gap": o1 / c0 - 1.0,
                "has_next": True}
    except Exception:  # noqa: BLE001
        return None


def compute(market: str = "CN", horizon: Optional[int] = None,
            verbose: bool = False) -> Dict:
    rows = [r for r in _read_trades()
            if str(r.get("market") or "").upper() == str(market).upper()
            and (horizon is None or int(r.get("horizon") or 0) == int(horizon))]
    recs = []
    for r in rows:
        code, day, px = r.get("code"), str(r.get("date") or "")[:10], r.get("price")
        if not code or not day or not px:
            continue
        g = _gap_from_kline(str(code), day)
        if not g or float(px) <= 0:
            continue
        buy = str(r.get("side") or "buy").lower() == "buy"
        gap = g["gap"] if buy else -g["gap"]          # 卖出：次日开盘更高才占便宜 → 符号相反
        # 成交价与当日收盘价的偏离（>2% 说明成交不是按收盘价，口径可疑）
        dev = float(px) / g["close_t"] - 1.0
        recs.append({"code": str(code), "date": day, "side": r.get("side"),
                     "price": float(px), "close_t": round(g["close_t"], 4),
                     "open_next": round(g["open_next"], 4),
                     "gap_bps": round(gap * 1e4, 2),
                     "price_vs_close_pct": round(dev * 100, 2),
                     "price_basis": r.get("price_basis") or "unknown",
                     # 只按**记录的** price_basis 分类；不能用"价格偏离收盘"
                     # 去猜 —— 旧记录的成交价是原始价、close_t 是后复权价，
                     # 偏离 10 倍以上是口径差，不代表当时用的是实时价。
                     "live_priced": str(r.get("price_basis") or "").lower() == "live",
                     "price_basis_unknown": not r.get("price_basis")})

    n = len(recs)
    if n == 0:
        return {"ok": False, "reason": "no_trades_with_next_open",
                "market": market, "horizon": horizon,
                "n_trades_total": len(rows),
                "note": "没有可归因的成交（缺次日开盘价或该票无本地日线）"}

    # gap 本身与成交价口径无关（取自同一条日线），所以主口径用全部可归因成交；
    # 实时价成交单列一组，因为它的"成交价"确实不是 close_T。
    g = np.array([x["gap_bps"] for x in recs if not x["live_priced"]], dtype=float)
    gl = np.array([x["gap_bps"] for x in recs if x["live_priced"]], dtype=float)
    out = {
        "ok": True, "market": str(market).upper(), "horizon": horizon,
        "n_trades_total": len(rows), "n_attributed": n,
        "n_close_priced": int(g.size), "n_live_priced": int(gl.size),
        "n_price_basis_unknown": int(sum(1 for x in recs if x.get("price_basis_unknown"))),
        "mean_gap_bps": round(float(np.mean(g)), 2) if g.size else None,
        "median_gap_bps": round(float(np.median(g)), 2) if g.size else None,
        "stdev_gap_bps": round(float(np.std(g)), 1) if g.size else None,
        "sum_gap_bps": round(float(np.sum(g)), 1) if g.size else None,
        "mean_gap_bps_live": round(float(np.mean(gl)), 2) if gl.size else None,
        "insufficient": bool(g.size < MIN_TRADES),
        "min_trades": MIN_TRADES,
        "basis": "close_T vs open_T+1",
        "note": ("gap>0 表示按收盘成交相对次日开盘「占便宜」。"
                 f"样本 < {MIN_TRADES} 时只看数、不下结论。"),
    }
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    try:
        os.makedirs(STATE, exist_ok=True)
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({**out, "items": recs[-200:]}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, OUT)
    except Exception:  # noqa: BLE001
        pass
    return out


def summary() -> Dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "not_ready",
                "hint": "python -m qcore.fill_audit --run"}
    return d


if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        print(json.dumps(compute(verbose=True), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1)[:1500])
