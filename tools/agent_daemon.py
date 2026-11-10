"""
训练机守护进程：静默后台运行，持续跟随版本更新，崩了自动拉起。

## 为什么需要它

`sync_agent.py --auto-update` 原来更新完只打印一句「重启后生效」，
然后**继续用内存里的旧代码跑** —— 版本永远追不上，除非人手动重启。
实测就是这么错过好几版的。

而且训练机是长时间无人值守的机器：
  · 不能弹黑窗口占着桌面
  · 不能崩了就静默死掉（队列里任务全变成 expired）
  · 日志要留，但不能无限涨

## 它做什么

  1. 以子进程方式跑 `sync_agent.py --loop N --auto-update`
  2. 子进程**因更新而退出**（exit 42）→ 立刻用新代码重启
  3. 子进程崩溃 → 指数退避重启（10s → 20s → ... → 上限 5 分钟）
  4. 所有输出重定向到 `logs/agent.log`，超过 5MB 自动轮转
  5. 支持 `--stop`（写停止标记，子进程下一轮退出）

## 静默启动（Windows）

    start_agent_hidden.vbs        # 双击即后台运行，无窗口
    pythonw tools/agent_daemon.py # 等价

## 停止

    python tools/agent_daemon.py --stop
    stop_agent.bat
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
LOG_PATH = os.path.join(LOGS, "agent.log")
STOP_FLAG = os.path.join(LOGS, "agent.stop")
PID_PATH = os.path.join(LOGS, "agent.pid")
#: 心跳文件：每轮循环刷新 mtime。
#: 不用「PID 是否存在」判活 —— PID 会被系统复用，会误报 alive（P2-12）。
HEART_PATH = os.path.join(LOGS, "agent.heartbeat")
#: 心跳超过这么久没刷新就认为守护进程已死（秒）
HEART_TIMEOUT = 180
STATE_PATH = os.path.join(LOGS, "agent_state.json")

#: 子进程「已更新，请重启我」的退出码
EXIT_UPDATED = 42
#: 日志轮转阈值
LOG_MAX_BYTES = 5 * 1024 * 1024
#: 崩溃重启退避（秒）
BACKOFF = [10, 20, 40, 80, 160, 300]
#: 同一个错误连续出现这么多次就判定「这台机器需要人管」，停止重试
SAME_ERR_LIMIT = 10


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    try:
        os.makedirs(LOGS, exist_ok=True)
        if os.path.isfile(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
            bak = LOG_PATH + ".1"
            try:
                if os.path.isfile(bak):
                    os.remove(bak)
                os.replace(LOG_PATH, bak)
            except Exception:  # noqa: BLE001
                pass
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def _beat() -> None:
    """刷新心跳文件。"""
    try:
        os.makedirs(LOGS, exist_ok=True)
        with open(HEART_PATH, "w", encoding="utf-8") as f:
            f.write(_now())
    except Exception:  # noqa: BLE001
        pass


def _own_stamp() -> str:
    """本文件的 mtime+size 指纹，用于检测「自己被更新了」。"""
    try:
        p = os.path.abspath(__file__)
        st = os.stat(p)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except Exception:  # noqa: BLE001
        return ""


def _write_state(d: dict) -> None:
    try:
        os.makedirs(LOGS, exist_ok=True)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_PATH)
    except Exception:  # noqa: BLE001
        pass


def _child_cmd(loop_sec: int, nas: str, token: str, password: str) -> list:
    cmd = [sys.executable, os.path.join(ROOT, "tools", "sync_agent.py"),
           "--loop", str(loop_sec), "--auto-update"]
    if nas:
        cmd += ["--nas", nas]
    if token:
        cmd += ["--token", token]
    elif password:
        cmd += ["--password", password]
    return cmd


def run(loop_sec: int = 60, nas: str = "", token: str = "",
        password: str = "", max_restarts: int = 0) -> int:
    """守护主循环。max_restarts=0 表示无限重启。"""
    os.makedirs(LOGS, exist_ok=True)
    try:
        if os.path.isfile(STOP_FLAG):
            os.remove(STOP_FLAG)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(PID_PATH, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:  # noqa: BLE001
        pass

    _own = _own_stamp()
    _log(f"daemon start pid={os.getpid()} loop={loop_sec}s nas={nas or '(default)'} "
         f"own={_own}")
    _beat()
    restarts = 0
    idx = 0
    #: 连续在同一类错误上失败几次后停止重试（P2-11）
    same_err = 0
    last_err = None
    while True:
        if os.path.isfile(STOP_FLAG):
            _log("stop flag found -> exit")
            break
        _beat()
        # P1-7：守护进程**自己**的文件被 kit_update 覆盖了。
        # 子进程会拿到新代码，但本进程仍在跑内存里的旧逻辑 ——
        # 不自我重启的话，以后修 agent_daemon 的 bug 永远不生效。
        _now_own = _own_stamp()
        if _own and _now_own and _now_own != _own:
            _log(f"self updated ({_own} -> {_now_own}) -> execv self-restart")
            _write_state({"self_restart": True, "at": _now(),
                          "old": _own, "new": _now_own})
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:  # noqa: BLE001
                _log(f"execv failed: {type(e).__name__}: {str(e)[:100]}")
                _own = _now_own
        cmd = _child_cmd(loop_sec, nas, token, password)
        _log("spawn: " + " ".join(cmd[:3]) + " ...")
        t0 = time.time()
        try:
            rc = subprocess.call(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            _log("interrupted by user")
            break
        except Exception as e:  # noqa: BLE001
            _log(f"spawn failed: {type(e).__name__}: {str(e)[:120]}")
            rc = -1
        alive = time.time() - t0

        if rc == EXIT_UPDATED:
            # 更新完成：立刻用新代码重启，不退避
            _log(f"child exited {EXIT_UPDATED} (updated) after {alive:.0f}s -> restart now")
            restarts += 1
            idx = 0
            continue

        restarts += 1
        # P2-11：同一个错误连续出现太多次，说明不是偶发崩溃，是环境问题
        # （缺依赖、磁盘满、权限）。继续每 5 分钟重试只会刷日志，
        # 用户永远不知道这台机器已经废了。
        sig = f"{rc}:{int(alive) // 5}"
        if sig == last_err:
            same_err += 1
        else:
            same_err = 1
            last_err = sig
        if same_err >= SAME_ERR_LIMIT:
            _log(f"same failure {same_err}x (rc={rc}) -> needs_attention, stop retrying")
            _write_state({"needs_attention": True, "reason": f"rc={rc} 连续 {same_err} 次",
                          "restarts": restarts, "at": _now()})
            break
        if max_restarts and restarts >= max_restarts:
            _log(f"max_restarts={max_restarts} reached -> exit")
            break
        wait = BACKOFF[min(idx, len(BACKOFF) - 1)]
        idx += 1
        _log(f"child exited rc={rc} after {alive:.0f}s -> restart in {wait}s "
             f"(restart #{restarts})")
        _write_state({"restarts": restarts, "last_rc": rc,
                      "last_alive_s": round(alive, 1), "at": _now()})
        # 等待期间也响应停止标记
        for _ in range(wait):
            if os.path.isfile(STOP_FLAG):
                _log("stop flag during backoff -> exit")
                return 0
            time.sleep(1)

    _write_state({"stopped": True, "restarts": restarts, "at": _now()})
    _log("daemon exit")
    return 0


def stop() -> int:
    """请求停止（写标记文件，子进程与守护进程都会退出）。"""
    try:
        os.makedirs(LOGS, exist_ok=True)
        with open(STOP_FLAG, "w", encoding="utf-8") as f:
            f.write(_now())
    except Exception as e:  # noqa: BLE001
        print(f"写停止标记失败：{e}")
        return 1
    _log("stop requested")
    print("已请求停止。守护进程会在下一轮检查时退出（最多等一个轮询周期）。")
    return 0


def status() -> dict:
    pid = None
    try:
        pid = int(open(PID_PATH, encoding="utf-8").read().strip())
    except Exception:  # noqa: BLE001
        pid = None
    # 用**心跳**判活，不用 PID：PID 会被复用，os.kill(pid,0) 只能说明
    # 「这个号有人占着」，可能完全是另一个进程（P2-12）。
    alive = False
    heart_age = None
    try:
        if os.path.isfile(HEART_PATH):
            heart_age = time.time() - os.path.getmtime(HEART_PATH)
            alive = heart_age <= HEART_TIMEOUT
    except Exception:  # noqa: BLE001
        alive = False
    st = {}
    try:
        st = json.load(open(STATE_PATH, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        st = {}
    log_tail = ""
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            log_tail = "".join(f.readlines()[-8:])
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "pid": pid, "alive": alive,
            "heartbeat_age_s": round(heart_age, 1) if heart_age is not None else None,
            "stop_requested": os.path.isfile(STOP_FLAG),
            "needs_attention": bool(st.get("needs_attention")),
            "state": st, "log_tail": log_tail}


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--stop" in a:
        raise SystemExit(stop())
    if "--status" in a:
        print(json.dumps(status(), ensure_ascii=False, indent=1))
        raise SystemExit(0)
    loop = 60
    if "--loop" in a:
        try:
            loop = int(a[a.index("--loop") + 1])
        except Exception:  # noqa: BLE001
            pass
    nas = token = password = ""
    for flag, key in (("--nas", "nas"), ("--token", "token"), ("--password", "password")):
        if flag in a:
            v = a[a.index(flag) + 1] if len(a) > a.index(flag) + 1 else ""
            if key == "nas":
                nas = v
            elif key == "token":
                token = v
            else:
                password = v
    # 没显式给就走 apb_defaults.json
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import defaults as DEF
        nas = nas or DEF.nas()
        token = token or DEF.token()
        password = password or DEF.password()
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(run(loop_sec=loop, nas=nas, token=token, password=password))
