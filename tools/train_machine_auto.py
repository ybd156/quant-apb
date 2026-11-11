#!/usr/bin/env python3
"""
训练机「一键」入口 —— 双击/一条命令，把这台机器变成训练机。

## 为什么还要一个入口

仓库里已经有三块能用的东西，但它们各自为政：

| 已有 | 干什么 | 缺什么 |
|---|---|---|
| `tools/kit_setup.py --cli` | 填 NAS 地址 + 令牌，测通后保存 | 用户得知道该先跑它 |
| `tools/agent_daemon.py` | 把 `sync_agent` 挂到后台 | 没配置时启动会失败退出 |
| `tools/train_one_click.py` | 本地训完打包、推模型 | **不领队列任务** |

于是新机器的实际流程是：先猜着跑哪个、报错了再翻文档。本脚本把这条链路收成一步：

    环境自检 → 缺配置就引导配置 → 测通 NAS → 起后台守护 → 打印状态与"接下来会怎样"

## 用法

    python tools/train_machine_auto.py            # 一键（缺配置会就地引导）
    python tools/train_machine_auto.py --check    # 只看环境与配置，不动手
    python tools/train_machine_auto.py --once     # 领一个任务就跑（不常驻）
    python tools/train_machine_auto.py --no-daemon  # 前台常驻（看得见日志）

Windows 双击 `APB.bat auto`；Linux 跑 `bash apb.sh auto` 或直接本命令。

## 它不会做什么

- **不替你装 Python**（装 Python 需要管理员权限，脚本不该偷偷做）；
- **不下载数据包**（训练数据 545MB，要不要下由 `--data` 显式触发）；
- **不改 NAS 上的任何东西**（除了按队列领任务、按结果回报）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
DEFAULTS = os.path.join(ROOT, "apb_defaults.json")
PY = sys.executable


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int, total: int, title: str) -> None:
    say(f"\n[{n}/{total}] {title}")


def run(args, timeout=180, capture=True):
    try:
        r = subprocess.run([PY] + args, cwd=ROOT, timeout=timeout,
                           capture_output=capture, text=True,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or ""), (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:  # noqa: BLE001
        return 1, "", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- 1 环境
def check_env() -> dict:
    v = sys.version_info
    out = {"python": f"{v.major}.{v.minor}.{v.micro}", "ok": v >= (3, 10), "missing": []}
    for mod in ("numpy", "pandas"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            out["missing"].append(mod)
    out["ok"] = out["ok"] and not out["missing"]
    try:
        import shutil
        out["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    except Exception:  # noqa: BLE001
        out["disk_free_gb"] = None
    data = os.path.join(ROOT, "data", "qlib")
    out["qlib_data"] = os.path.isdir(os.path.join(data, "x")) or \
        os.path.isdir(os.path.join(data, "qlib_bin"))
    return out


# ---------------------------------------------------------------- 2 配置
def load_cfg() -> dict:
    for p in (os.path.join(ROOT, "apb_defaults.json"),
              os.path.join(os.path.dirname(ROOT), "apb_defaults.json")):
        try:
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:  # noqa: BLE001
            continue
    return {}


def cfg_ready(cfg: dict) -> bool:
    return bool(str(cfg.get("nas") or "").strip()
                and (str(cfg.get("token") or "").strip()
                     or str(cfg.get("password") or "").strip()))


def redeem_pair(text: str) -> dict:
    """用授权链接/授权码向 NAS 换令牌，并写进 apb_defaults.json。"""
    sys.path.insert(0, ROOT)
    from qcore import pairing as PR
    parsed = PR.parse_link(text)
    if not parsed.get("ok"):
        return {"ok": False, "reason": parsed.get("reason"),
                "hint": parsed.get("hint") or "需要形如 K7M2-QP4R 的 8 位授权码"}
    nas, code = parsed.get("nas") or "", parsed["code"]
    if not nas:
        cfg = load_cfg()
        nas = str(cfg.get("nas") or "").rstrip("/")
    if not nas:
        return {"ok": False, "reason": "need_nas",
                "hint": "只给了授权码，缺 NAS 地址：请用完整链接，或先填一次 NAS 地址"}

    import urllib.request
    payload = json.dumps({"code": code}).encode()
    req = urllib.request.Request(nas.rstrip("/") + "/api/pair/redeem", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "network", "detail": f"{type(e).__name__}: {e}"}
    if not d.get("ok"):
        # 服务端的错误形态是 {"ok":false,"error":"already_used","hint":...}
        # （_fail 把 code 放在 error 字段里），不要去找不存在的 reason。
        _err = d.get("error")
        if isinstance(_err, dict):
            _err = _err.get("code")
        return {"ok": False, "reason": str(_err or d.get("reason") or "redeem_failed"),
                "hint": str(d.get("hint") or d.get("detail") or "")}
    data = d.get("data") or {}
    tok = str(data.get("token") or "")
    if not tok:
        return {"ok": False, "reason": "no_token_in_response"}
    try:
        with open(DEFAULTS, "w", encoding="utf-8") as f:
            json.dump({"nas": nas, "token": tok,
                       "paired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "paired_code": code}, f, ensure_ascii=False, indent=1)
        try:
            os.chmod(DEFAULTS, 0o600)
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "save_failed", "detail": str(e)[:120]}
    return {"ok": True, "nas": nas, "token_prefix": tok[:12], "code": code}


def do_setup() -> int:
    """就地跑配置向导（终端版）。返回码直接透传。"""
    say("  没有找到可用配置 → 启动配置向导（填 NAS 地址 + 令牌，测通才保存）")
    return run([os.path.join(TOOLS, "kit_setup.py"), "--cli"], timeout=900,
               capture=False)[0]


# ---------------------------------------------------------------- 3 连通
def check_nas(cfg: dict) -> dict:
    nas = str(cfg.get("nas") or "").rstrip("/")
    token = str(cfg.get("token") or "")
    pwd = str(cfg.get("password") or "")
    code, out, err = run([os.path.join(TOOLS, "kit_status.py")], timeout=90)
    return {"ok": code == 0, "tail": (out or err or "")[-800:]}


# ---------------------------------------------------------------- 4 守护
def daemon_running() -> bool:
    try:
        code, out, _ = run([os.path.join(TOOLS, "kit_status.py"), "--json"], timeout=60)
        if code == 0 and out.strip().startswith("{"):
            d = json.loads(out)
            return bool(d.get("daemon_running") or d.get("running"))
    except Exception:  # noqa: BLE001
        pass
    # 兜底：看 pid 文件
    for p in (os.path.join(ROOT, "logs", "agent.pid"),
              os.path.join(ROOT, "agent.pid")):
        if os.path.isfile(p):
            return True
    return False


def start_daemon() -> dict:
    code, out, err = run([os.path.join(TOOLS, "agent_daemon.py"), "start"], timeout=180)
    return {"ok": code == 0, "tail": (out or err or "")[-600:]}


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练机一键入口")
    ap.add_argument("--check", action="store_true", help="只体检，不启动")
    ap.add_argument("--once", action="store_true", help="领一个任务就退出（不常驻）")
    ap.add_argument("--no-daemon", action="store_true", help="前台常驻（日志直接打印）")
    ap.add_argument("--data", action="store_true", help="显式触发数据包下载（545MB）")
    ap.add_argument("--pair", default="", metavar="LINK_OR_CODE",
                    help="用授权链接/授权码一步配置（NAS 上生成，一次性）")
    a = ap.parse_args(argv)

    total = 5 if not a.check else 2
    say("=" * 66)
    say("  APB 训练机 · 一键启动")
    say("=" * 66)

    # 1) 环境
    step(1, total, "环境自检")
    env = check_env()
    say(f"  Python {env['python']}" + ("" if env["ok"] else "  ← 需要 3.10+"))
    if env["missing"]:
        say(f"  缺依赖：{', '.join(env['missing'])}")
        say("  安装：python -m pip install " + " ".join(env["missing"]))
    say(f"  磁盘剩余 {env.get('disk_free_gb')} GB · 训练数据"
        + ("已就位" if env.get("qlib_data") else "**未下载**（训练前需要）"))
    if a.check:
        return 0 if env["ok"] else 2
    if not env["ok"]:
        say("\n  环境不满足，先解决上面两项再双击一次。")
        return 2

    # 2) 配置
    step(2, total, "训练机配置")
    if a.pair:
        say(f"  用授权码配置：{a.pair[:40]}{'…' if len(a.pair) > 40 else ''}")
        r = redeem_pair(a.pair)
        if not r.get("ok"):
            say(f"  兑换失败：{r.get('reason')}  {r.get('hint') or r.get('detail') or ''}")
            return 5
        say(f"  已配置 NAS {r['nas']}（令牌 {r['token_prefix']}…）")
    cfg = load_cfg()
    if not cfg_ready(cfg):
        rc = do_setup()
        cfg = load_cfg()
        if rc != 0 or not cfg_ready(cfg):
            say("\n  配置未完成（取消或测试未通过）。再双击一次即可继续。")
            return 3
    say(f"  NAS：{cfg.get('nas')}")
    say("  凭据：" + ("令牌" if str(cfg.get("token") or "").strip() else "密码"))

    # 3) 连通
    step(3, total, "连通性")
    st = check_nas(cfg)
    for line in (st["tail"] or "").strip().splitlines()[-6:]:
        say("  " + line)
    if not st["ok"]:
        say("\n  连不上 NAS：先确认地址/凭据，或 NAS 是否在线。")
        return 4

    # 4) 数据（只在显式要求时下载）
    if a.data and not env.get("qlib_data"):
        step(4, total, "下载训练数据（约 545MB）")
        code, out, err = run([os.path.join(TOOLS, "bootstrap.ps1")], timeout=3600)
        say("  " + (out or err or "")[-400:])
    else:
        step(4, total, "训练数据")
        say("  已就位" if env.get("qlib_data") else
            "  未下载 —— 领到 train 任务时会失败；需要的话加 --data")

    # 5) 启动
    step(5, total, "启动")
    if a.once:
        say("  单次模式：领一个任务 → 执行 → 上传 → 退出")
        code, out, err = run([os.path.join(TOOLS, "sync_agent.py"), "--once"],
                             timeout=7200, capture=False)
        return code
    if a.no_daemon:
        say("  前台常驻：日志直接打印（Ctrl+C 退出）")
        code, out, err = run([os.path.join(TOOLS, "sync_agent.py"), "--loop", "60"],
                             timeout=86400 * 30, capture=False)
        return code
    if daemon_running():
        say("  后台守护已在运行（不重复启动）")
    else:
        r = start_daemon()
        say("  已启动后台守护" if r["ok"] else "  启动失败：\n  " + r["tail"])

    say("\n" + "-" * 66)
    say("  接下来会发生什么：")
    say("   · 每 60 秒问一次 NAS「有没有任务」（train / train_ab / pipeline / prewarm）")
    say("   · 领到就训练，训完自动上传；失败会回报原因，不静默")
    say("   · 想看状态：APB.bat status       停止：APB.bat stop")
    say("-" * 66)
    code, out, _ = run([os.path.join(TOOLS, "kit_status.py")], timeout=90)
    for line in (out or "").strip().splitlines()[:14]:
        say("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
