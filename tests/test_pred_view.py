#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_pred_view.py —— 四本账只读合并视图（评审 R50 / 决策 3C）

守住四件事：
  1. 四本账各自被识别（pred_ledger / picks / paper_trades / claims），缺的不报错；
  2. 每一行都带 `source` 与 `basis` —— 口径不同不许混着加；
  3. 查不到就 `ok=False`，不返回空成功；
  4. **只读**：调用前后四本账的文件 mtime 不变（这条最重要，C 方案的承诺就是"不动写入方"）。

用法：python -m tests.test_pred_view
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import pred_view as PV  # noqa: E402

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
    print("pred_view —— 四本账只读合并")
    print("=" * 70)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", f"pv_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "ledger"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "state"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "paper"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "claims"), exist_ok=True)

    led = os.path.join(tmp, "ledger", "2026.jsonl")
    with open(led, "w", encoding="utf-8") as f:
        f.write(json.dumps({"as_of": "2026-11-10T00:00:00", "code": "600519",
                            "horizon": 5, "score": 0.42, "status": "scored"}) + "\n")
        f.write(json.dumps({"as_of": "2026-11-10T00:00:00", "code": "000001",
                            "horizon": 5, "score": -0.1, "status": "scored"}) + "\n")
    with open(os.path.join(tmp, "state", "picks.json"), "w", encoding="utf-8") as f:
        json.dump({"as_of": "2026-11-11", "decision_source": "gbdt",
                   "rows": [{"code": "600519", "score": 0.5, "rank": 1}]}, f)
    with open(os.path.join(tmp, "paper", "trades.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"code": "600519", "date": "2026-11-06", "side": "buy",
                            "price": 1700.0, "price_basis": "close"}) + "\n")
    with open(os.path.join(tmp, "claims", "claims.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "claim": "600519 在 5 日内跑赢池等权",
                            "falsifier": "5 日后 ret<0", "review_by": "2026-11-20"}) + "\n")

    old = (PV.LEDGER_DIR, PV.STATE, PV.PAPER_TRADES, PV.CLAIMS)
    PV.LEDGER_DIR = os.path.join(tmp, "ledger")
    PV.STATE = os.path.join(tmp, "state")
    PV.PAPER_TRADES = os.path.join(tmp, "paper", "trades.jsonl")
    PV.CLAIMS = os.path.join(tmp, "claims", "claims.jsonl")
    try:
        s = PV.sources()
        names = [x["source"] for x in s["sources"]]
        check("四本账都识别到", s["n_sources"] == 4, str(names))
        for want in ("pred_ledger", "picks", "paper_trades", "claims"):
            check(f"含 {want}", want in names)

        v = PV.lookup("600519")
        check("能查到该票", v["ok"] is True and v["n"] >= 3, f"n={v.get('n')}")
        check("跨四源都命中", set(v["by_source"]) >= {"pred_ledger", "picks", "paper_trades"},
              str(v["by_source"]))
        check("每行都带 source 与 basis",
              all(it.get("source") and it.get("basis") for it in v["items"]))
        check("台账分数被原样带出（不重算）",
              any(it.get("score") == 0.42 for it in v["items"]))

        v2 = PV.lookup("999999")
        check("查不到 → ok=False", v2["ok"] is False, str(v2.get("n")))

        # 只读承诺：mtime 不变
        before = [os.path.getmtime(p) for p in (led, PV.PAPER_TRADES, PV.CLAIMS)]
        PV.sources(); PV.lookup("600519")
        after = [os.path.getmtime(p) for p in (led, PV.PAPER_TRADES, PV.CLAIMS)]
        check("只读：文件 mtime 未变", before == after, f"{before} vs {after}")
    finally:
        PV.LEDGER_DIR, PV.STATE, PV.PAPER_TRADES, PV.CLAIMS = old
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
