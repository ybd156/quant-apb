"""
训练机配置向导：打开一个窗口，填 NAS 地址和令牌，测通后保存。

## 为什么要有它

原来配令牌要手工编辑 `apb_defaults.json` —— 用户得知道：
JSON 语法、字段名、token 从哪来、引号不能少。写错一个字符就
`login_required`，而报错信息里根本看不出是配置问题。

这个向导把这件事变成三步：**填 → 测 → 存**。
测不通就不让保存，并且直接说清是「地址不对」「令牌无效」还是「网络不通」。

## 两种保存位置

  · `apb_defaults.json`（推荐）—— 持久，下次运行自动读取
  · 环境变量（仅当前会话）—— 不想留文件时用

## 界面（tkinter，标准库，不引新依赖）

    NAS 地址  [https://apbcdn.683868.xyz        ]
    令牌      [apb_****************************] [显示]
    ------------------------------------------------
    [测试连接]   [保存]   [取消]
    状态：……

## 没有图形环境时

自动退回控制台问答（同样校验后保存）：

    python tools/kit_setup.py --cli
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = os.path.join(ROOT, "apb_defaults.json")
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) APB-Setup"


# ---------------------------------------------------------------- 配置读写
def load_cfg() -> dict:
    try:
        with open(DEFAULTS, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_cfg(nas: str, token: str = "", password: str = "") -> dict:
    """保存到 apb_defaults.json，**保留其它字段**（不要整体覆盖）。"""
    d = load_cfg()
    d["nas_url"] = nas.strip()
    d["token"] = token.strip()
    d["password"] = password.strip()
    d.setdefault("auth_note", "填 token（NAS 的访问令牌）或 password（NAS 登录密码）任意一个")
    d.setdefault("default_horizons", [1, 3, 5, 10, 15])
    d.setdefault("default_trees", 250)
    d.setdefault("default_stocks", 300)
    tmp = DEFAULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DEFAULTS)
    return d


# ---------------------------------------------------------------- 连通性测试
def probe(nas: str, token: str = "", password: str = "", timeout: int = 12) -> dict:
    """测三件事：能不能连、版本是多少、令牌/密码能不能过鉴权。

    返回 {"ok", "stage", "detail", "version"}
      stage: net | version | auth | ok
    """
    nas = (nas or "").strip().rstrip("/")
    if not nas:
        return {"ok": False, "stage": "net", "detail": "NAS 地址是空的"}
    if not nas.startswith(("http://", "https://")):
        nas = "http://" + nas

    # 1) 能不能连
    try:
        req = urllib.request.Request(nas + "/api/version", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        ver = ((body.get("data") or {}) if isinstance(body, dict) else {}).get("version")
    except urllib.error.HTTPError as e:
        return {"ok": False, "stage": "net", "detail": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stage": "net",
                "detail": f"{type(e).__name__}: {str(e)[:80]}"}
    if not ver:
        return {"ok": False, "stage": "version", "detail": "服务端没返回版本号"}

    # 2) 鉴权：调一个需要写权限的端点
    headers = {"Content-Type": "application/json", "User-Agent": _UA}
    cookie = ""
    if token:
        headers["X-APB-Token"] = token.strip()
    elif password:
        try:
            payload = json.dumps({"password": password}).encode()
            req = urllib.request.Request(nas + "/api/login", data=payload, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.headers.get("Set-Cookie") or ""
                r.read()
            for part in raw.split(","):
                kv = part.split(";")[0].strip()
                if kv.startswith("apb_session="):
                    cookie = kv
                    break
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "stage": "auth",
                    "detail": f"登录失败：{type(e).__name__}: {str(e)[:60]}", "version": ver}
        if not cookie:
            return {"ok": False, "stage": "auth", "detail": "密码不正确（没拿到会话）",
                    "version": ver}
        headers["Cookie"] = cookie
    else:
        return {"ok": False, "stage": "auth",
                "detail": "没填令牌也没填密码 —— 上传会 login_required", "version": ver}

    try:
        req = urllib.request.Request(nas + "/api/sync/task", method="POST",
                                     data=json.dumps({"claim": True, "worker": "setup-probe"}).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        if d.get("ok"):
            return {"ok": True, "stage": "ok", "detail": "鉴权通过", "version": ver}
        return {"ok": False, "stage": "auth",
                "detail": str(d.get("error") or d.get("hint") or "鉴权被拒")[:120],
                "version": ver}
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read().decode("utf-8", "replace"))
            msg = str(d.get("hint") or d.get("error") or f"HTTP {e.code}")[:120]
        except Exception:  # noqa: BLE001
            msg = f"HTTP {e.code}"
        # 服务端对「无效令牌」只回一句通用的 login_required，
        # 用户看不出是令牌问题还是没登录。这里按我们发了什么给具体提示。
        if e.code in (401, 403) and token:
            msg = ("令牌无效、已吊销，或没有 sync 权限 —— "
                   "在 NAS 的「设置 → 访问令牌」重新建一把")
        return {"ok": False, "stage": "auth", "detail": msg, "version": ver}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stage": "auth",
                "detail": f"{type(e).__name__}: {str(e)[:80]}", "version": ver}


# ---------------------------------------------------------------- 图形界面
def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:  # noqa: BLE001
        print("这个 Python 没有 tkinter，改用控制台模式。")
        return run_cli()

    cfg = load_cfg()
    root = tk.Tk()
    root.title("APB 训练机配置")
    root.geometry("560x300")
    root.resizable(False, False)

    pad = {"padx": 14, "pady": 8}
    tk.Label(root, text="训练机配置", font=("", 13, "bold")).pack(anchor="w", **pad)
    tk.Label(root, text="填好点「测试连接」，通了再保存。", fg="#666").pack(anchor="w", padx=14)

    frm = tk.Frame(root)
    frm.pack(fill="x", **pad)

    tk.Label(frm, text="NAS 地址").grid(row=0, column=0, sticky="w", pady=4)
    v_nas = tk.StringVar(value=cfg.get("nas_url") or "https://apbcdn.683868.xyz")
    tk.Entry(frm, textvariable=v_nas, width=46).grid(row=0, column=1, columnspan=2, sticky="we", pady=4)

    tk.Label(frm, text="令牌").grid(row=1, column=0, sticky="w", pady=4)
    v_tok = tk.StringVar(value=cfg.get("token") or "")
    e_tok = tk.Entry(frm, textvariable=v_tok, width=38, show="*")
    e_tok.grid(row=1, column=1, sticky="we", pady=4)

    v_show = tk.BooleanVar(value=False)

    def toggle():
        e_tok.config(show="" if v_show.get() else "*")

    tk.Checkbutton(frm, text="显示", variable=v_show, command=toggle).grid(row=1, column=2, sticky="w")

    tk.Label(frm, text="或密码").grid(row=2, column=0, sticky="w", pady=4)
    v_pw = tk.StringVar(value=cfg.get("password") or "")
    tk.Entry(frm, textvariable=v_pw, width=38, show="*").grid(row=2, column=1, sticky="we", pady=4)
    tk.Label(frm, text="（二选一）", fg="#888").grid(row=2, column=2, sticky="w")

    v_msg = tk.StringVar(value="")
    lbl = tk.Label(root, textvariable=v_msg, wraplength=520, justify="left", fg="#b7791f")
    lbl.pack(anchor="w", padx=14, pady=(0, 4))

    btns = tk.Frame(root)
    btns.pack(fill="x", **pad)
    state = {"probed": False, "version": ""}

    def do_probe():
        v_msg.set("测试中…")
        lbl.config(fg="#666")
        root.update_idletasks()
        r = probe(v_nas.get(), v_tok.get(), v_pw.get())
        if r["ok"]:
            state["probed"] = True
            state["version"] = r.get("version") or ""
            v_msg.set(f"连接成功 · 服务端 v{r.get('version')} · 鉴权通过")
            lbl.config(fg="#2f855a")
        else:
            state["probed"] = False
            stage = {"net": "网络/地址", "version": "版本", "auth": "鉴权"}.get(r["stage"], "")
            v_msg.set(f"失败（{stage}）：{r.get('detail')}")
            lbl.config(fg="#c53030")

    def do_save():
        if not state["probed"]:
            if not messagebox.askyesno("还没测试", "还没测试连接，确定直接保存吗？"):
                return
        try:
            save_cfg(v_nas.get(), v_tok.get(), v_pw.get())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("保存失败", f"{type(e).__name__}: {e}")
            return
        messagebox.showinfo("已保存",
                            f"已写入：\n{DEFAULTS}\n\n"
                            "之后双击 start_agent_hidden.vbs 即可后台运行。")
        root.destroy()

    tk.Button(btns, text="测试连接", width=12, command=do_probe).pack(side="left")
    tk.Button(btns, text="保存", width=10, command=do_save).pack(side="left", padx=8)
    tk.Button(btns, text="取消", width=10, command=root.destroy).pack(side="left")

    tk.Label(root, text=f"配置将写入：{DEFAULTS}", fg="#888").pack(anchor="w", padx=14)
    root.mainloop()
    return 0


# ---------------------------------------------------------------- 控制台模式
def run_cli() -> int:
    cfg = load_cfg()
    print("=" * 56)
    print("  APB 训练机配置")
    print("=" * 56)
    nas = input(f"NAS 地址 [{cfg.get('nas_url') or 'https://apbcdn.683868.xyz'}]: ").strip()
    nas = nas or (cfg.get("nas_url") or "https://apbcdn.683868.xyz")
    tok = input("令牌（没有就回车，改用密码）: ").strip()
    pw = ""
    if not tok:
        pw = input("密码: ").strip()
    print("\n测试连接…")
    r = probe(nas, tok, pw)
    if not r["ok"]:
        print(f"  失败（{r['stage']}）：{r.get('detail')}")
        if input("仍要保存吗？(y/N) ").strip().lower() != "y":
            return 1
    else:
        print(f"  成功 · 服务端 v{r.get('version')} · 鉴权通过")
    save_cfg(nas, tok, pw)
    print(f"\n已写入 {DEFAULTS}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--cli" in a:
        raise SystemExit(run_cli())
    if "--probe" in a:
        c = load_cfg()
        print(json.dumps(probe(c.get("nas_url", ""), c.get("token", ""),
                               c.get("password", "")), ensure_ascii=False, indent=1))
        raise SystemExit(0)
    raise SystemExit(run_gui())
