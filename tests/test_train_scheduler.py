#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_train_scheduler.py —— 训练任务调度器契约（不联网、不落真实任务）

守住六件事：
  1. 没有模型 → 立刻排（priority 1）
  2. 队列里已有该周期 → 不重复排
  3. 冷却期内 → 不重复排
  4. 模型新鲜且数据未落后 → 不排
  5. dry-run **绝不**调 create_task
  6. 单轮不超过 MAX_ENQUEUE

用法：python -m tests.test_train_scheduler
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import train_scheduler as TS  # noqa: E402

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
    print("train_scheduler —— 谁来判断该不该训")
    print("=" * 70)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", f"tsched_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    old = {k: getattr(TS, k) for k in ("_model_meta", "_pending_horizons",
                                       "_last_enqueued", "_trading_lag",
                                       "_atomic", "PLAN_PATH", "HORIZONS")}
    created = []

    class _SY:
        @staticmethod
        def create_task(horizons, trees=250, stocks=300, note="", created_by="", job=""):
            created.append({"horizons": list(horizons), "note": note})
            return {"ok": True, "task": {"id": "T%d" % len(created)}}

    import qcore.sync as SY
    old_create = SY.create_task
    SY.create_task = _SY.create_task

    try:
        TS.PLAN_PATH = os.path.join(tmp, "train_plan.json")
        TS._atomic = lambda p, o: open(p, "w", encoding="utf-8").write(json.dumps(o))
        TS.HORIZONS = (1, 5)

        # 1) 没有模型 → train, priority 1
        TS._model_meta = lambda h: {}
        TS._pending_horizons = lambda: {}
        TS._last_enqueued = lambda: {}
        TS._trading_lag = lambda a, b: None
        p = TS.plan()
        _h_items = [i for i in p["items"] if "horizon" in i]     # ab 条目没有 horizon
        check("无模型 → 两个周期都要训",
              len([i for i in _h_items if i["action"] == "train"]) == 2,
              str([(i["horizon"], i["action"]) for i in _h_items]))
        check("无模型优先级为 1",
              all(i["priority"] == 1 for i in _h_items if i["action"] == "train"))
        # 双轨复核（决策 1C）：从未跑过时该排；但它不占 MAX_ENQUEUE 名额
        check("含双轨复核条目",
              any(i.get("kind") == "train_ab" for i in p["items"]),
              str([i.get("kind") for i in p["items"]]))

        # 2) 队列已有 → skip
        TS._pending_horizons = lambda: {1: "T-EXIST"}
        p = TS.plan()
        h1 = next(i for i in p["items"] if i.get("horizon") == 1)
        check("队列已有该周期 → 不重复排", h1["action"] == "skip" and "T-EXIST" in h1["reason"],
              h1["reason"])

        # 3) 冷却期 → skip
        TS._pending_horizons = lambda: {}
        TS._last_enqueued = lambda: {1: datetime.now().astimezone().isoformat(timespec="seconds")}
        p = TS.plan()
        h1 = next(i for i in p["items"] if i.get("horizon") == 1)
        check("12h 内排过 → 冷却跳过", h1["action"] == "skip" and "冷却" in h1["reason"], h1["reason"])

        # 4) 新鲜 + 数据不落后 → skip
        TS._last_enqueued = lambda: {}
        TS._model_meta = lambda h: {"trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                    "train_end": "2026-11-10"}
        TS._trading_lag = lambda a, b: 0
        p = TS.plan()
        _h_only = [i for i in p["items"] if "horizon" in i]
        check("模型新鲜且数据同步 → 不排", all(i["action"] == "skip" for i in _h_only),
              str([i["reason"] for i in _h_only]))

        # 5) dry-run 不落任务
        TS._model_meta = lambda h: {}
        created.clear()
        r = TS.run(dry_run=True, verbose=False)
        check("dry-run 不调用 create_task", len(created) == 0, f"created={len(created)}")
        check("dry-run 给出 would_enqueue", any(i.get("would_enqueue") for i in r["items"]))

        # 6) 真跑：最多 MAX_ENQUEUE 个
        created.clear()
        r = TS.run(dry_run=False, verbose=False)
        check("真跑会排任务", len(created) >= 1, f"created={len(created)}")
        check("单轮不超过 MAX_ENQUEUE（ab 单列，不计入）", len(created) <= TS.MAX_ENQUEUE + 1,
              f"{len(created)} <= {TS.MAX_ENQUEUE}")
        check("任务备注带原因（可追溯）", all("[auto]" in c["note"] for c in created),
              str(created[:1]))

    finally:
        for k, v in old.items():
            setattr(TS, k, v)
        SY.create_task = old_create
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
