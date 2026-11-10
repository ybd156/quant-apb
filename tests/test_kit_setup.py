"""
配置向导回归测试：窗口填令牌 → 测连接 → 保存。

背景：原来配令牌要手工编辑 `apb_defaults.json` —— 得知道 JSON 语法、
字段名、引号位置，写错一个字符就 `login_required`，而报错信息里
根本看不出是配置问题。

本测试钉住：
  1. `probe()` 能把「地址不通 / 没凭据 / 令牌无效 / 凭据有效」分开
  2. `save_cfg()` **保留其它字段**（不整体覆盖 apb_defaults.json）
  3. 保存后能读回来（round-trip）
  4. setup.bat 符合陷阱 14（纯 ASCII + CRLF + 无控制字符）

不联网：全部用本地起的最小 HTTP 桩，或断言错误分支。
跑法：python -m tests.test_kit_setup
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

import kit_setup as KS  # noqa: E402

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


GOOD_TOKEN = "apb_good"


class _Stub(BaseHTTPRequestHandler):
    """最小 NAS 桩：/api/version 公开；/api/sync/task 要令牌。"""

    def log_message(self, *a):
        return

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/version"):
            return self._send(200, {"ok": True, "data": {"version": "9.9.9"}})
        return self._send(404, {"ok": False})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n) if n else None
        if self.path.startswith("/api/sync/task"):
            # 真实 NAS 认两种：机器令牌 或 登录会话（allow_session=True）。
            # 桩只认 token 的话，「密码换会话」这条路径测不出来（实测踩过）。
            tok_ok = self.headers.get("X-APB-Token") == GOOD_TOKEN
            sess_ok = "apb_session=" in (self.headers.get("Cookie") or "")
            if tok_ok or sess_ok:
                return self._send(200, {"ok": True, "data": {"task": None}})
            return self._send(401, {"ok": False, "error": "login_required",
                                    "hint": "请先登录"})
        if self.path.startswith("/api/login"):
            body = json.dumps({"ok": True, "data": {"ok": True}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", "apb_session=stub; Path=/; HttpOnly")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self._send(404, {"ok": False})


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    print("[1] probe：四类结果要能分开")
    r = KS.probe("", "x", "")
    check("空地址 -> net", (not r["ok"]) and r["stage"] == "net", r["detail"][:40])
    r = KS.probe("http://127.0.0.1:1", "x", "", timeout=3)
    check("端口不通 -> net", (not r["ok"]) and r["stage"] == "net", r["detail"][:40])
    r = KS.probe(base, "", "")
    check("没凭据 -> auth", (not r["ok"]) and r["stage"] == "auth", r["detail"][:40])
    r = KS.probe(base, "apb_wrong", "")
    check("令牌无效 -> auth", (not r["ok"]) and r["stage"] == "auth", r["detail"][:50])
    check("令牌无效的提示是具体的（不是通用 login_required）",
          "令牌" in r["detail"], r["detail"][:40])
    r = KS.probe(base, GOOD_TOKEN, "")
    check("有效令牌 -> ok", r["ok"] and r["stage"] == "ok", r["detail"])
    check("带回服务端版本", r.get("version") == "9.9.9", str(r.get("version")))
    r = KS.probe(base, "", "anypassword")
    check("密码也能过（会话）", r["ok"], r["detail"][:40])

    print("\n[2] save_cfg 保留其它字段")
    tmp = os.path.join(ROOT, "tmp", "kit_setup_test")
    os.makedirs(tmp, exist_ok=True)
    KS.DEFAULTS = os.path.join(tmp, "apb_defaults.json")
    with open(KS.DEFAULTS, "w", encoding="utf-8") as f:
        json.dump({"nas_url": "http://old", "token": "", "password": "",
                   "default_horizons": [1, 5], "default_trees": 999,
                   "custom_field": "keep-me"}, f)
    KS.save_cfg("http://new:8765", "apb_tok", "")
    d = json.load(open(KS.DEFAULTS, encoding="utf-8"))
    check("nas_url 已更新", d["nas_url"] == "http://new:8765", d["nas_url"])
    check("token 已写入", d["token"] == "apb_tok")
    check("**保留了** default_horizons", d.get("default_horizons") == [1, 5],
          str(d.get("default_horizons")))
    check("**保留了** default_trees", d.get("default_trees") == 999, str(d.get("default_trees")))
    check("**保留了**自定义字段", d.get("custom_field") == "keep-me")
    check("load_cfg 能读回", KS.load_cfg()["token"] == "apb_tok")

    print("\n[3] setup.bat 符合陷阱 14")
    p = os.path.join(ROOT, "setup.bat")
    check("setup.bat 存在", os.path.isfile(p))
    if os.path.isfile(p):
        raw = open(p, "rb").read()
        check("纯 ASCII", all(b < 0x80 for b in raw))
        check("用 CRLF", b"\r\n" in raw)
        ctrl = [b for b in raw if b < 0x09 or 0x0b <= b <= 0x0c or 0x0e <= b <= 0x1f]
        check("无控制字符", not ctrl, str(ctrl[:3]))

    print("\n[4] 无图形环境时能退回控制台")
    src = open(os.path.join(ROOT, "tools", "kit_setup.py"), encoding="utf-8").read()
    check("有 --cli 分支", "--cli" in src)
    check("tkinter 不可用时自动退回", "run_cli()" in src)

    print("\n[5] 负控：检测器真的会拒绝")
    check("错误令牌不会被当成成功",
          KS.probe(base, "apb_wrong", "")["ok"] is False)

    srv.shutdown()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
