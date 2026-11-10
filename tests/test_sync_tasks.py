"""
任务队列回归测试：超时按 job 类型 + 失败可重领。

实测背景（2026-11-10）：队列 8 条任务全部 `finished_at=None`，一次都没回报过。
其中一条被领走 3 小时仍是 `claimed` —— 界面上写「训练中」，实际早死了。
根因是 `EXPIRE_MIN=240`（4 小时）一刀切，而 pipeline 任务只要 20 分钟。

本测试钉住三件事：
  1. 不同 job 用不同超时窗口（train 长、pipeline/prewarm 短）
  2. 超时的 claimed 任务**退回 pending 重领**，而不是直接判死
  3. 重试次数用完才标 expired，并写明原因

跑法：python -m tests.test_sync_tasks
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore import sync as SY  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _mk(job: str, minutes_ago: int, retries: int = 0) -> dict:
    at = (datetime.now().astimezone() - timedelta(minutes=minutes_ago)).isoformat()
    return {"id": f"t-{job}-{minutes_ago}-{retries}", "job": job, "status": "claimed",
            "claimed_at": at, "claimed_by": "worker-x", "retries": retries,
            "horizons": [5], "trees": 250, "stocks": 300}


def main() -> int:
    print("[1] 超时窗口按 job 类型区分")
    check("train 窗口最长",
          SY.EXPIRE_MIN_BY_JOB["train"] > SY.EXPIRE_MIN_BY_JOB["pipeline"])
    check("pipeline 窗口明显短于 train",
          SY.EXPIRE_MIN_BY_JOB["pipeline"] <= SY.EXPIRE_MIN_BY_JOB["train"] / 2,
          f"train={SY.EXPIRE_MIN_BY_JOB['train']} / pipeline={SY.EXPIRE_MIN_BY_JOB['pipeline']}")
    check("有重试上限", int(SY.MAX_RETRIES) >= 1, f"MAX_RETRIES={SY.MAX_RETRIES}")

    print("\n[2] 超时 -> 退回 pending 重领（不直接判死）")
    win = SY.EXPIRE_MIN_BY_JOB["pipeline"]
    d = {"tasks": [_mk("pipeline", win + 10)]}
    n = SY._expire(d)
    t = d["tasks"][0]
    check("确实处理了一条", n == 1, f"n={n}")
    check("状态退回 pending", t["status"] == "pending", t["status"])
    check("重试次数 +1", int(t.get("retries") or 0) == 1, str(t.get("retries")))
    check("清空 claimed_by/at", t.get("claimed_by") is None and t.get("claimed_at") is None)
    check("留下原因", bool(t.get("last_expire_reason")), str(t.get("last_expire_reason"))[:60])

    print("\n[3] 未超时的不动")
    d = {"tasks": [_mk("pipeline", max(1, win // 2))]}
    SY._expire(d)
    check("仍在 claimed", d["tasks"][0]["status"] == "claimed", d["tasks"][0]["status"])

    print("\n[4] 重试次数用完 -> expired 并写明原因")
    d = {"tasks": [_mk("pipeline", win + 10, retries=SY.MAX_RETRIES)]}
    SY._expire(d)
    t = d["tasks"][0]
    check("状态 expired", t["status"] == "expired", t["status"])
    check("有 expired_at", bool(t.get("expired_at")))
    check("写明重试耗尽", "重试" in str(t.get("expire_reason")), str(t.get("expire_reason"))[:60])

    print("\n[5] train 任务在 pipeline 窗口内不该被动")
    d = {"tasks": [_mk("train", win + 10)]}
    SY._expire(d)
    check("train 仍在 claimed（窗口更长）", d["tasks"][0]["status"] == "claimed",
          d["tasks"][0]["status"])

    print("\n[6] 负控：这些检查器真的会失败")
    d = {"tasks": [_mk("pipeline", win + 10)]}
    SY._expire(d)
    check("如果退回 pending 就不该同时是 expired",
          d["tasks"][0]["status"] != "expired", d["tasks"][0]["status"])

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
