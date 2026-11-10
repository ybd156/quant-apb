"""
训练机守护进程回归测试：静默后台、持续跟随版本、崩了自动拉起。

背景（2026-11-10）：`sync_agent --auto-update` 更新完只打印
「重启后生效」，然后**继续用内存里的旧代码跑** —— 版本永远追不上。
训练机也因此错过好几版。

本测试钉住：
  1. 更新后以 EXIT_UPDATED(42) 退出，而不是继续跑旧代码
  2. 守护进程认 42 → 立刻重启（不退避）；崩溃 → 指数退避
  3. 停止标记能生效，状态可查
  4. 日志轮转阈值存在
  5. bat/vbs 启动器符合陷阱 14（纯 ASCII + CRLF + 无控制字符）

跑法：python -m tests.test_agent_daemon
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

import agent_daemon as AD  # noqa: E402

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


def main() -> int:
    print("[1] 退出码契约")
    check("EXIT_UPDATED == 42", AD.EXIT_UPDATED == 42, str(AD.EXIT_UPDATED))
    src = open(os.path.join(ROOT, "tools", "sync_agent.py"), encoding="utf-8").read()
    check("sync_agent 更新后返回 42", "return 42" in src)
    check("sync_agent 不再只打印「重启后生效」",
          "（重启后生效）" not in src)

    print("\n[2] 退避策略")
    check("退避是递增的", AD.BACKOFF == sorted(AD.BACKOFF), str(AD.BACKOFF[:4]))
    check("退避有上限（不会无限翻倍）", AD.BACKOFF[-1] <= 600, str(AD.BACKOFF[-1]))
    check("首次退避不太长", AD.BACKOFF[0] <= 15, str(AD.BACKOFF[0]))

    print("\n[3] 日志轮转")
    check("有轮转阈值", AD.LOG_MAX_BYTES >= 1024 * 1024, f"{AD.LOG_MAX_BYTES/1024/1024:.0f} MB")

    print("\n[4] 停止 / 状态")
    # 用临时目录，别碰真实 logs
    tmp = os.path.join(ROOT, "tmp", "daemon_test")
    os.makedirs(tmp, exist_ok=True)
    AD.LOGS = tmp
    AD.LOG_PATH = os.path.join(tmp, "agent.log")
    AD.STOP_FLAG = os.path.join(tmp, "agent.stop")
    AD.PID_PATH = os.path.join(tmp, "agent.pid")
    AD.STATE_PATH = os.path.join(tmp, "agent_state.json")
    for p in (AD.STOP_FLAG, AD.PID_PATH):
        if os.path.isfile(p):
            os.remove(p)
    st = AD.status()
    check("初始未运行", st["alive"] is False)
    check("初始未请求停止", st["stop_requested"] is False)
    AD.stop()
    check("stop() 写了标记", os.path.isfile(AD.STOP_FLAG))
    check("status 反映已请求停止", AD.status()["stop_requested"] is True)
    if os.path.isfile(AD.STOP_FLAG):
        os.remove(AD.STOP_FLAG)

    print("\n[5] 子进程命令拼装")
    cmd = AD._child_cmd(60, "http://nas:8765", "apb_tok", "")
    joined = " ".join(cmd)
    check("带 --loop", "--loop 60" in joined)
    check("带 --auto-update", "--auto-update" in joined)
    check("带 --nas", "http://nas:8765" in joined)
    check("带 --token", "apb_tok" in joined)
    cmd2 = AD._child_cmd(30, "", "", "pw123")
    check("没有 token 时用 --password", "--password pw123" in " ".join(cmd2))
    check("token 优先于 password",
          "--password" not in " ".join(AD._child_cmd(60, "", "tok", "pw")))

    print("\n[6] 启动器符合陷阱 14")
    for f in ("start_agent.bat", "start_agent_hidden.vbs"):
        p = os.path.join(ROOT, f)
        if not os.path.isfile(p):
            check(f"{f} 存在", False)
            continue
        raw = open(p, "rb").read()
        check(f"{f} 纯 ASCII", all(b < 0x80 for b in raw))
        check(f"{f} 用 CRLF", b"\r\n" in raw)
        ctrl = [b for b in raw if b < 0x09 or 0x0b <= b <= 0x0c or 0x0e <= b <= 0x1f]
        check(f"{f} 无控制字符", not ctrl, str(ctrl[:3]))

    print("\n[7] 负控：这些检查器真的会失败")
    check("乱写的退出码不被当成更新",
          AD.EXIT_UPDATED != 0 and AD.EXIT_UPDATED != 1, str(AD.EXIT_UPDATED))

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
