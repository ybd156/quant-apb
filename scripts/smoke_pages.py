"""
页面冒烟测试 —— 用 headless Chrome 打开每个页面，抓 console 错误。

## 为什么需要

`node --check` 只查语法，查不出「调用了不存在的函数」这类运行时错误。
实测踩过：`reasonText is not defined` —— 语法完全合法，
但页面一打开就是「渲染失败：页面渲染出错」。

这类问题的唯一可靠检测方式是**真的跑一遍**。

## 做法

1. 用 Chrome headless 打开每个 hash 路由（#/overview、#/paper...）
2. 注入 console 错误 / 未捕获异常收集器
3. 只要有任何一条错误，就算失败

## 依赖

需要本机有 Chrome。没有就跳过（返回 skipped，不阻塞发版）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAGES = [
    ("overview", "总览"), ("market", "市场"), ("chart", "个股"),
    ("hotspot", "风口"), ("watch", "自选"), ("signals", "信号"),
    ("model", "模型"), ("ledger", "预测台账"), ("paper", "模拟实盘"),
    ("behavior", "行为因子"), ("events", "政策事件"), ("risk", "风险体检"),
    ("settings", "设置"),
]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome():
    for p in CHROME_CANDIDATES:
        if os.path.isfile(p):
            return p
    for name in ("chrome", "google-chrome", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    return None


PROBE_JS = r"""
const errors = [];
const origErr = console.error;
console.error = function(...a) {
  errors.push('console.error: ' + a.map(String).join(' '));
  origErr.apply(console, a);
};
window.addEventListener('error', (e) => {
  errors.push('uncaught: ' + (e.message || '') + ' @' + (e.filename || '') + ':' + (e.lineno || ''));
});
window.addEventListener('unhandledrejection', (e) => {
  errors.push('rejection: ' + String(e.reason));
});
setTimeout(() => {
  const el = document.createElement('pre');
  el.id = '__apb_smoke';
  el.textContent = JSON.stringify(errors);
  document.body.appendChild(el);
}, 6000);
"""


def check_page(chrome: str, base: str, page: str, timeout: int = 40,
               prof: str = "", warm: bool = True) -> dict:
    """打开一个页面，抓 console 错误。"""
    url = f"{base}/#{page}"
    own_prof = not prof
    if own_prof:
        prof = tempfile.mkdtemp(prefix="apb_smoke_")
    try:
        # 用 --dump-dom 拿到最终 DOM，检查有没有错误收集节点
        cmd = [
            chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--disable-dev-shm-usage", f"--user-data-dir={prof}",
            "--virtual-time-budget=9000", "--dump-dom", url,
        ]
        # 必须显式指定 utf-8：Windows 下 text=True 会用 locale(GBK) 解码，
        # 而 Chrome 输出 UTF-8 -> 中文变乱码 -> 下面的关键字判断永远 False
        # -> 冒烟测试永远报 OK（实测踩过：页面明明报错，13/13 却全过）
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        dom = r.stdout or ""
        # 双重判据：
        #   1) 关键字（编码修好后可靠）
        #   2) 结构特征：<h2>渲染失败</h2> —— 即使关键字被编码搞坏也能抓到
        _bad = ("页面渲染出错" in dom) or ("渲染失败</h2>" in dom)             or ("渲染失败<" in dom)
        if _bad:
            m = re.search(r"页面渲染出错[^<]*<[^>]*>([^<]{0,160})", dom)
            return {"page": page, "ok": False,
                    "reason": "render_error",
                    "detail": (m.group(1).strip() if m else "")[:160]}
        # 抓我们注入的错误节点
        m = re.search(r'id="__apb_smoke"[^>]*>(.*?)</pre>', dom, re.S)
        if m:
            try:
                errs = json.loads(m.group(1))
            except Exception:
                errs = []
            real = [e for e in errs if "favicon" not in e.lower()]
            if real:
                return {"page": page, "ok": False, "reason": "js_error",
                        "detail": "; ".join(real[:3])[:200]}
        return {"page": page, "ok": True}
    except subprocess.TimeoutExpired:
        return {"page": page, "ok": False, "reason": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"page": page, "ok": False, "reason": f"{type(e).__name__}: {str(e)[:80]}"}
    finally:
        if own_prof:
            shutil.rmtree(prof, ignore_errors=True)


def main(argv=None) -> int:
    base = os.environ.get("APB_BASE", "http://127.0.0.1:8765")
    for a in (argv or sys.argv[1:]):
        if a.startswith("--base="):
            base = a.split("=", 1)[1]

    chrome = find_chrome()
    if not chrome:
        print("[smoke] 未找到 Chrome，跳过（不影响发版）")
        return 0
    print(f"[smoke] Chrome: {chrome}")
    print(f"[smoke] 目标: {base}")
    print()

    # 共用一个 profile，先登录一次（?token= 一次性的），后续页面带 cookie
    prof = tempfile.mkdtemp(prefix="apb_smoke_shared_")
    token = os.environ.get("APB_TOKEN", "")
    if token:
        warmup = f"{base}/login?token={token}&continue=/"
        try:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                            f"--user-data-dir={prof}", "--virtual-time-budget=6000",
                            "--dump-dom", warmup],
                           capture_output=True, text=True, timeout=40, errors="replace")
            print("[smoke] 已用 token 登录（cookie 复用）")
        except Exception as e:  # noqa: BLE001
            print(f"[smoke] 预热登录失败: {type(e).__name__}")
    else:
        print("[smoke] 未提供 APB_TOKEN，可能需要登录才能测")

    bad = 0
    for page, label in PAGES:
        r = check_page(chrome, base, page, prof=prof)
        if r["ok"]:
            print(f"  [OK]   {label:<10} #{page}")
        else:
            bad += 1
            print(f"  [FAIL] {label:<10} #{page}  {r['reason']}: {r.get('detail', '')}")

    shutil.rmtree(prof, ignore_errors=True)
    print()
    print(f"[smoke] {len(PAGES) - bad}/{len(PAGES)} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
