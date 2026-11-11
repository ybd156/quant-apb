#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_fill_audit.py —— 成交口径审计（评审 R22）契约

守住四件事：
  1. gap 必须取自**同一条日线**（`open[t+1]/close[t]-1`），不能拿"成交价"去除 ——
     第一次实现就是这么错的，跑出 **+78,023 bps（780%）**，因为成交价是原始价、
     日线是后复权价；
  2. 卖出方向的符号要反过来（次日开盘更高 → 按收盘卖出是吃亏）；
  3. 样本 < MIN_TRADES 时必须标 `insufficient=True`，不许拿 2 笔当结论；
  4. `price_basis` 只能按**记录值**分类，不能靠"价格偏离收盘"去猜。

用法：python -m tests.test_fill_audit
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import fill_audit as FA  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("fill_audit —— 成交口径（close_T vs open_T+1）")
    print("=" * 70)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", f"fill_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    # 造一条日线：close [10,10,11,12,13]，open [10,10.5,11.2,12.1,13.1]
    days = pd.date_range("2026-01-05", periods=5, freq="D")
    df = pd.DataFrame({"date": days,
                       "open": [10.0, 10.5, 11.2, 12.1, 13.1],
                       "high": [10.0, 10.6, 11.3, 12.2, 13.2],
                       "low": [9.9, 10.4, 11.1, 12.0, 13.0],
                       "close": [10.0, 10.0, 11.0, 12.0, 13.0],
                       "volume": [1e6] * 5})

    import qcore.data as D
    old_load, old_trades, old_out = D.load_kline, FA.TRADES, FA.OUT
    D.load_kline = lambda code, *a, **k: df.copy()
    FA.TRADES = os.path.join(tmp, "trades.jsonl")
    FA.OUT = os.path.join(tmp, "fill_audit.json")
    try:
        rows = [
            # 买入：d0 收盘 10 → d1 开盘 10.5 ⇒ gap +5%
            {"code": "000001", "date": "2026-01-05", "side": "buy", "price": 10.0,
             "market": "CN", "horizon": 1},
            # 卖出：d2 收盘 11 → d3 开盘 12.1 ⇒ 按收盘卖出吃亏 10% ⇒ gap −10%
            {"code": "000001", "date": "2026-01-07", "side": "sell", "price": 11.0,
             "market": "CN", "horizon": 1},
            # 别的市场，不该被计入 CN
            {"code": "000001", "date": "2026-01-05", "side": "buy", "price": 10.0,
             "market": "US", "horizon": 1},
        ]
        with open(FA.TRADES, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        out = FA.compute(market="CN")
        check("只统计目标市场", out["n_attributed"] == 2, f"n={out.get('n_attributed')}")
        check("买入 gap = +500 bps（10→10.5）",
              abs([i for i in json.load(open(FA.OUT, encoding="utf-8"))["items"]
                   if i["side"] == "buy"][0]["gap_bps"] - 500.0) < 0.01)
        check("卖出符号相反（gap = −1000 bps）",
              abs([i for i in json.load(open(FA.OUT, encoding="utf-8"))["items"]
                   if i["side"] == "sell"][0]["gap_bps"] + 1000.0) < 0.01)
        check("gap 不拿成交价去除（数量级合理，|mean| < 5000bps）",
              abs(out["mean_gap_bps"]) < 5000, f"mean={out['mean_gap_bps']}")
        check("样本不足要标 insufficient", out["insufficient"] is True)
        check("price_basis 未记录 → 归入 unknown",
              out["n_price_basis_unknown"] == 2, str(out["n_price_basis_unknown"]))
        check("basis 文案明确", out.get("basis") == "close_T vs open_T+1")

        # 同一天没有下一根 → 不可归因
        one = [{"code": "000001", "date": "2026-01-09", "side": "buy",
                "price": 13.0, "market": "CN", "horizon": 1}]
        with open(FA.TRADES, "w", encoding="utf-8") as f:
            for r in one:
                f.write(json.dumps(r) + "\n")
        out2 = FA.compute(market="CN")
        check("末日无次日开盘 → 不归因", out2.get("ok") is False, str(out2.get("reason")))
    finally:
        D.load_kline = old_load
        FA.TRADES, FA.OUT = old_trades, old_out
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
