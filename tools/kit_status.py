"""
训练机一屏状态：这台机器现在什么情况，一屏看完。

## 为什么要有它

用户原话：「训练机那么多文件，都不知点哪个，连看当前状态都没有」。

原来要看状态得记三条命令（`agent_daemon.py --status`、`sync_agent.py --once`、
翻 `logs/agent.log`），而且看不出「数据是哪天的」「我领到任务了吗」
「传到哪了」。这个脚本把所有这些拼成一屏。

## 输出

    ┌ 版本 / 数据 / 鉴权 / 守护进程 / 队列 / 最近日志 ┐

每项都带**判断**而不只是数字：数据落后几天、令牌有没有配、
守护进程心跳是否新鲜、任务进行到第几个周期。

## 用法

    python tools/kit_status.py            # 一屏
    python tools/kit_status.py --json     # 机器可读
    python tools/kit_status.py --no-net   # 不联网（只看本地）
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

_UA = "APB-KitStatus"


def _read_version() -> str:
    try:
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except Exception:  # noqa: BLE001
        return "?"


def _http(nas: str, path: str, token: str = "", timeout: int = 10,
          method: str = "GET", body: dict | None = None):
    import urllib.error
    import urllib.request
    hdr = {"User-Agent": _UA, "Content-Type": "application/json"}
    if token:
        hdr["X-APB-Token"] = token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(nas.rstrip("/") + path, data=data,
                                 method=method, headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:70]}"}


def collect(no_net: bool = False) -> dict:
    out = {"ok": True, "version": _read_version(), "problems": [], "warnings": []}

    # ---- 鉴权配置 ----
    nas = token = password = ""
    try:
        import defaults as DEF
        nas, token, password = DEF.nas(), DEF.token(), DEF.password()
    except Exception as e:  # noqa: BLE001
        out["warnings"].append(f"读不到 apb_defaults.json：{e}")
    out["nas"] = nas
    out["auth"] = ("token" if token else ("password" if password else "none"))
    if not token and not password:
        out["problems"].append("没配令牌也没配密码 —— 上传会 login_required，"
                               "双击 APB.bat 选「1 配置」填一下")

    # ---- 依赖 ----
    deps = {}
    for m in ("numpy", "pandas"):
        try:
            deps[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            deps[m] = None
            out["problems"].append(f"缺少 {m}")
    out["deps"] = deps

    # ---- 本地数据日期 ----
    try:
        import qcore.data_sync as DS
        out["data_date"] = DS.local_date()
    except Exception as e:  # noqa: BLE001
        out["data_date"] = None
        out["warnings"].append(f"读不到数据日期：{str(e)[:60]}")

    # ---- 守护进程 ----
    try:
        import agent_daemon as AD
        st = AD.status()
        out["daemon"] = {
            "alive": st.get("alive"),
            "pid": st.get("pid"),
            "heartbeat_age_s": st.get("heartbeat_age_s"),
            "needs_attention": st.get("needs_attention"),
            "state": st.get("state") or {},
        }
        if not st.get("alive"):
            out["warnings"].append("守护进程没在跑 —— 双击 APB.bat 选「启动后台」")
        if st.get("needs_attention"):
            out["problems"].append(
                "守护进程判定「需要人工处理」并已停止重试："
                + str((st.get("state") or {}).get("reason") or ""))
    except Exception as e:  # noqa: BLE001
        out["daemon"] = {"alive": None, "error": str(e)[:80]}

    # ---- 日志尾部 ----
    try:
        lp = os.path.join(ROOT, "logs", "agent.log")
        if os.path.isfile(lp):
            with open(lp, encoding="utf-8", errors="replace") as f:
                out["log_tail"] = [x.rstrip() for x in f.readlines()[-6:]]
        else:
            out["log_tail"] = []
    except Exception:  # noqa: BLE001
        out["log_tail"] = []

    # ---- 联网：服务端版本 / 数据日期 / 我的任务 ----
    if no_net or not nas:
        out["net"] = {"skipped": True}
        return out
    ver = _http(nas, "/api/version", token)
    out["server_version"] = ((ver.get("data") or {}) if ver.get("ok") else {}).get("version")
    if not ver.get("ok"):
        out["problems"].append(f"连不上 NAS：{ver.get('error')}")
        out["net"] = {"ok": False}
        return out
    if out["server_version"] and out["server_version"] != out["version"]:
        out["warnings"].append(
            f"版本不一致：本机 {out['version']} / 服务端 {out['server_version']} "
            f"（守护进程会自动更新，也可以选「检查更新」）")

    dv = _http(nas, "/api/data/version", token)
    srv_date = ((dv.get("data") or {}) if dv.get("ok") else {}).get("date")
    out["server_data_date"] = srv_date
    if srv_date and out["data_date"] and str(srv_date) > str(out["data_date"]):
        out["warnings"].append(
            f"本地数据落后：本机 {out['data_date']} / 服务端 {srv_date}")

    # 队列里属于这台机器的任务
    tk = _http(nas, "/api/sync/task", token)
    tasks = ((tk.get("data") or {}) if tk.get("ok") else {}).get("tasks") or []
    mine = []
    host = os.environ.get("COMPUTERNAME") or os.uname().nodename if hasattr(os, "uname") \
        else os.environ.get("COMPUTERNAME", "")
    for t in tasks:
        if str(t.get("claimed_by") or "").startswith(str(host)[:12]):
            mine.append(t)
    out["tasks"] = tasks[:8]
    out["my_tasks"] = mine[:3]
    out["net"] = {"ok": True}
    return out


def render(s: dict) -> str:
    L = []
    bar = "=" * 62
    L.append(bar)
    L.append("  APB 训练机状态")
    L.append(bar)

    def row(k, v, flag=""):
        L.append(f"  {k:<10s} {v}{flag}")

    row("版本", s.get("version") or "?")
    d = s.get("deps") or {}
    row("依赖", f"numpy {d.get('numpy') or '缺'} / pandas {d.get('pandas') or '缺'}")
    auth = s.get("auth")
    row("鉴权", {"token": "已配令牌", "password": "已配密码",
                 "none": "**没配**（上传会失败）"}.get(auth, str(auth)),
        "" if auth in ("token", "password") else "  ← 双击 APB.bat 选 1")
    row("数据", s.get("data_date") or "未知")
    if s.get("server_data_date"):
        row("服务端数据", s["server_data_date"])
    if s.get("server_version"):
        row("服务端版本", s["server_version"])

    dm = s.get("daemon") or {}
    if dm.get("alive") is None:
        row("守护进程", "未知 " + str(dm.get("error") or ""))
    elif dm.get("alive"):
        age = dm.get("heartbeat_age_s")
        row("守护进程", f"运行中（PID {dm.get('pid')}，心跳 {age}s 前）")
    else:
        row("守护进程", "**没在跑**  ← 双击 APB.bat 选「启动后台」")
    if dm.get("needs_attention"):
        row("", "!! 需要人工处理：" + str((dm.get("state") or {}).get("reason") or ""))

    mine = s.get("my_tasks") or []
    if mine:
        L.append("")
        L.append("  我这台机器正在做的任务：")
        for t in mine:
            p = t.get("progress") or {}
            bits = [str(t.get("status"))]
            if p.get("phase"):
                bits.append(str(p["phase"]))
            if p.get("horizon") is not None:
                bits.append(f"h{p['horizon']}")
            if p.get("done") is not None and p.get("total"):
                bits.append(f"{p['done']}/{p['total']}")
            if p.get("pct") is not None:
                bits.append(f"{p['pct']}%")
            if p.get("eta_s"):
                bits.append(f"约{round(p['eta_s'] / 60)}分")
            L.append(f"    {str(t.get('id'))[:8]}  " + " · ".join(bits))
    else:
        L.append("")
        L.append("  我这台机器：当前没有领取任务")

    if s.get("log_tail"):
        L.append("")
        L.append("  最近日志：")
        for ln in s["log_tail"]:
            L.append("    " + ln[:110])

    probs = s.get("problems") or []
    warns = s.get("warnings") or []
    if probs or warns:
        L.append("")
        L.append("-" * 62)
        for p in probs:
            L.append("  [问题] " + p)
        for w in warns:
            L.append("  [注意] " + w)
    L.append(bar)
    return "\n".join(L)


if __name__ == "__main__":
    a = sys.argv[1:]
    st = collect(no_net=("--no-net" in a))
    if "--json" in a:
        print(json.dumps(st, ensure_ascii=False, indent=1))
    else:
        print(render(st))
    raise SystemExit(0 if not (st.get("problems")) else 0)
