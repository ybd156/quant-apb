#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_tick_chunk.py —— 快照分块与"静默截断"补偿（评审 R46）

实测事实（tools/tick_capacity_probe.py，2026-11-11）：
    300 只 → 100% 覆盖 / 0.09s
    600 只 → 100% 覆盖 / 0.12s
    800 只 → **只回 736 行（92%）** —— 服务端静默截断，请求本身还报成功

所以 `snapshot()` 必须：① 按 `TICK_CHUNK` 分块；② 对缺的票重试一次。
本测试用假请求模拟"每块最多回 736 行"，验证这两条真的生效。

用法：python -m tests.test_tick_chunk
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import tick as T  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _fake_row(code: str) -> dict:
    return {"code": code, "name": "X", "price": 10.0, "chg_pct": 0.0,
            "ts": 1.0e9, "ts_raw": "15:00:00", "source": "fake"}


def main() -> int:
    print("=" * 70)
    print("tick 分块 —— 静默截断必须被补偿")
    print("=" * 70)

    old_one, old_chunk = T._snapshot_one, T.TICK_CHUNK
    calls = {"n": 0, "sizes": []}

    def _truncating(codes):
        """模拟服务端"静默截断"：每次最多回 300 行（比 TICK_CHUNK=400 更小）。

        注意：**分块本身已经让截断不会发生**（400 < 736 的真实上限）。
        这里故意把假上限压到 300，是为了验证"万一某块少了，重试能不能补回来"。
        """
        calls["n"] += 1
        calls["sizes"].append(len(codes))
        take = codes[:300]
        return {T._norm(c): _fake_row(c) for c in take}, "fake"

    try:
        T._snapshot_one = _truncating
        T.TICK_CHUNK = 400                       # 900 只 → 3 块

        codes = [f"{600000 + i:06d}" for i in range(900)]
        r = T.snapshot(codes)

        check("全部 900 只都拿到了", r.get("n") == 900, f"rows={r.get('n')}")
        # 3 块各少 100 只（第一/二块 400→300），缺失 200 只 → 重试 1 次补齐
        check("分了 3 块并对缺失重试 1 次", r.get("chunks") == 3 and calls["n"] == 4,
              f"chunks={r.get('chunks')} 请求次数={calls['n']}")
        check("没有漏（missing 为空）", not r.get("missing"), str(r.get("missing"))[:80])
        check("partial=False", r.get("partial") is False)
        check("每块不超过 TICK_CHUNK", max(calls["sizes"]) <= 400,
              f"最大块 {max(calls['sizes'])}")

        # 单块足够小的时候不该多请求
        calls["n"] = 0
        r2 = T.snapshot(codes[:50])
        check("50 只只发 1 次", calls["n"] == 1 and r2.get("n") == 50,
              f"请求 {calls['n']} 次")

        # 源全失败 → 明确失败，不要把空当成成功
        T._snapshot_one = lambda cs: ({}, None)
        r3 = T.snapshot(codes[:10])
        check("全源失败 → ok=False", r3.get("ok") is False
              and r3.get("reason") == "all_sources_failed", str(r3.get("reason")))
    finally:
        T._snapshot_one, T.TICK_CHUNK = old_one, old_chunk

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
