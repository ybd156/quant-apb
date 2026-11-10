"""
训练机菜单：唯一入口 `APB.bat` 的实际界面。

## 为什么把菜单放在 Python 里

`.bat` 里的中文在简体中文 Windows 的 GBK 控制台会变成乱码
（本项目陷阱 14）。所以 `APB.bat` 保持纯 ASCII、只负责找 Python，
所有给人看的文字都在这里渲染 —— Python 能自己控制 stdout 编码。

## 用法

    APB.bat                # 交互菜单（顶部先显示一屏状态）
    APB.bat status         # 只看状态
    APB.bat start / stop   # 启停后台守护
    APB.bat setup          # 配置向导（窗口）
    APB.bat train          # 前台训练一次
    APB.bat selftest       # 能否承接所有任务
    APB.bat update         # 检查并应用更新
    APB.bat hidden         # 静默后台启动（无窗口，等价旧 vbs）
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for p in (ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows 控制台默认 GBK，中文会乱码 —— 强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

MENU = """
  ────────────────────────────────────────────────────────────
   1  配置（填 NAS 地址 / 令牌）           setup
   2  启动后台运行（无窗口，推荐）          start
   3  停止                                  stop
   4  刷新状态                              status
   5  立即训练一次（前台，能看到过程）       train
   6  自检（能否承接所有任务）              selftest
   7  检查更新                              update
   0  退出
  ────────────────────────────────────────────────────────────"""


def _py() -> str:
    return sys.executable


def _run(args, pause=False) -> int:
    """跑一个子命令，失败不炸整个菜单。"""
    try:
        rc = subprocess.call([_py()] + args, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n  已中断")
        rc = 130
    except Exception as e:  # noqa: BLE001
        print(f"  执行失败：{type(e).__name__}: {e}")
        rc = 1
    if pause:
        try:
            input("\n  回车返回菜单…")
        except (EOFError, KeyboardInterrupt):
            pass
    return rc


def show_status() -> None:
    _run([os.path.join(TOOLS, "kit_status.py")])


def do_start(hidden: bool = False) -> None:
    print("\n  [..] 启动后台守护" + ("（无窗口）" if hidden else ""))
    if os.name == "nt":
        # /b 不新开窗口；pythonw 由 agent_daemon 自己选不到，这里用 python + /b
        subprocess.Popen([_py(), os.path.join(TOOLS, "agent_daemon.py")],
                         cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        subprocess.Popen([_py(), os.path.join(TOOLS, "agent_daemon.py")], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import time
    time.sleep(3)
    show_status()
    print("  [OK] 已启动。守护进程会持续跟随版本更新，崩了自动重拉。")
    print("       日志：logs/agent.log")


def do_stop() -> None:
    _run([os.path.join(TOOLS, "agent_daemon.py"), "--stop"])


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    cmd = (argv[0].lower() if argv else "")

    if cmd == "status":
        show_status()
        return 0
    if cmd == "start":
        do_start()
        return 0
    if cmd == "hidden":
        do_start(hidden=True)
        return 0
    if cmd == "stop":
        do_stop()
        return 0
    if cmd == "setup":
        return _run([os.path.join(TOOLS, "kit_setup.py")])
    if cmd == "train":
        return _run([os.path.join(TOOLS, "train_one_click.py")])
    if cmd == "selftest":
        return _run([os.path.join(TOOLS, "kit_selftest.py")])
    if cmd == "update":
        return _run([os.path.join(TOOLS, "sync_agent.py"), "--once", "--auto-update"])
    if cmd and cmd not in ("menu",):
        print(f"  未知命令：{cmd}")
        print("  可用：status start stop setup train selftest update hidden")
        return 2

    # 交互菜单
    while True:
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:  # noqa: BLE001
            pass
        show_status()
        print(MENU)
        try:
            ch = input("\n  请输入序号后回车: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  已退出")
            return 0
        if ch == "0":
            return 0
        if ch == "1":
            _run([os.path.join(TOOLS, "kit_setup.py")], pause=True)
        elif ch == "2":
            do_start()
            try:
                input("\n  回车返回菜单…")
            except (EOFError, KeyboardInterrupt):
                return 0
        elif ch == "3":
            do_stop()
            try:
                input("\n  回车返回菜单…")
            except (EOFError, KeyboardInterrupt):
                return 0
        elif ch == "4":
            continue
        elif ch == "5":
            _run([os.path.join(TOOLS, "train_one_click.py")], pause=True)
        elif ch == "6":
            _run([os.path.join(TOOLS, "kit_selftest.py")], pause=True)
        elif ch == "7":
            _run([os.path.join(TOOLS, "sync_agent.py"), "--once", "--auto-update"], pause=True)
        else:
            print("  请输入 0-7")
            try:
                input("  回车继续…")
            except (EOFError, KeyboardInterrupt):
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
