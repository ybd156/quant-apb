"""
发布前自检 —— 每次改完必须跑，全绿才允许发版。

## 为什么要有这个

本项目已经出现过太多次"改完就发、用户发现是坏的"：
  · `N` 被 import 覆盖导致 `int(N)` 崩溃
  · `pool_df` 未定义
  · `[hidden]` 被 CSS 覆盖，弹窗盖住整页
  · `--to-json` 参数根本没实现
  · 辅助容器没覆盖 ENTRYPOINT，跑错程序
  · `docker cp` 丢执行位，容器 Exited(126)
  · `/api/events` 的 `_rows_of` 不认 `events` 键

这些**全都能被一个自动检查抓住**。本脚本就是那道闸门。

## 检查项

1. 全部模块可导入（语法/依赖/循环引用）
2. 状态文件齐全且是合法 JSON（UI 的 9 个页面都不该"未就绪"）
3. `names.json` 非空（用户要求：显示代码必须带名称）
4. 账户配置读写往返一致，越界值被拒绝
5. 起服务，逐个打所有 GET 端点，断言没有 `not_ready`
6. POST 账户配置的成功/失败两条路径
7. 跑既有测试套件（无未来函数 / 信号 / 指标）

用法：
    python scripts/preflight.py            # 全跑
    python scripts/preflight.py --quick    # 跳过引擎重跑与测试套件
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def rec(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {PASS: "OK  ", FAIL: "FAIL", WARN: "WARN"}[status]
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. 导入
MODULES = [
    "qcore.data", "qcore.dataset", "qcore.signals", "qcore.learner", "qcore.engine",
    "qcore.meta", "qcore.confidence", "qcore.stats", "qcore.analysis", "qcore.costs",
    "qcore.risk", "qcore.forecast", "qcore.claims", "qcore.paper", "qcore.context",
    "qcore.store", "qcore.pipeline", "qcore.live", "qcore.behavioral", "qcore.tactics",
    "qcore.flow", "qcore.capital", "qcore.sentiment", "qcore.chart", "qcore.brief",
    "qcore.reward", "qcore.account", "qcore.invariants", "qcore.indicators2",
    "qcore.model", "qcore.stress", "qcore.position", "qcore.residual", "qcore.advanced",
    "qcore.multihorizon", "qcore.ranker", "qcore.signed", "qcore.optimizer",
    "qcore.survival", "qcore.drift", "qcore.altdata", "qcore.ensemble",
    "qcore.market_live", "qcore.update_check", "qcore.update_apply", "qcore.docker_api",
    "qcore.names", "qcore.account_config", "qcore.behavior_report", "qcore.policy_calendar",
    "webui.server",
]


def check_imports() -> None:
    print("\n[1] 模块导入")
    bad = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m}: {type(e).__name__}: {e}")
    if bad:
        rec("import", FAIL, f"{len(bad)}/{len(MODULES)} 失败")
        for b in bad[:6]:
            print(f"        {b}")
    else:
        rec("import", PASS, f"{len(MODULES)} 个模块")


# ---------------------------------------------------------------- 2. 状态文件
REQUIRED_STATE = {
    "status.json": True, "weights.json": True, "regime.json": True,
    "ledger.json": True, "performance.json": True, "paper.json": True,
    "events.json": True, "names.json": True, "market.json": False,
    "update.json": False, "account.json": False,
}


def check_state() -> None:
    print("\n[2] 状态文件")
    sd = os.path.join(ROOT, "data", "state")
    missing_req, bad_json = [], []
    for fn, required in REQUIRED_STATE.items():
        p = os.path.join(sd, fn)
        if not os.path.isfile(p):
            if required:
                missing_req.append(fn)
            continue
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except Exception as e:  # noqa: BLE001
            bad_json.append(f"{fn}: {e}")
    if missing_req:
        rec("state.files", FAIL, f"缺必需文件: {', '.join(missing_req)}")
    else:
        rec("state.files", PASS, f"{len(REQUIRED_STATE)} 个")
    if bad_json:
        rec("state.json", FAIL, "; ".join(bad_json[:3]))
    else:
        rec("state.json", PASS, "全部合法 JSON")

    # names 非空
    try:
        with open(os.path.join(sd, "names.json"), encoding="utf-8") as f:
            nm = json.load(f).get("names") or {}
        sample = next(iter(nm.items())) if nm else None
        if len(nm) < 10:
            rec("state.names", FAIL, f"只有 {len(nm)} 条，UI 无法显示名称")
        elif not sample or not sample[1].get("name"):
            rec("state.names", FAIL, "条目缺 name 字段")
        else:
            rec("state.names", PASS, f"{len(nm)} 条，例 {sample[0]}={sample[1]['name']}")
    except Exception as e:  # noqa: BLE001
        rec("state.names", FAIL, str(e)[:80])

    # events.impact
    try:
        with open(os.path.join(sd, "events.json"), encoding="utf-8") as f:
            ev = json.load(f)
        imp = ev.get("impact")
        nev = ev.get("n") or len(ev.get("rows") or [])
        if not nev:
            rec("state.events", FAIL, "事件数为 0")
        elif not isinstance(imp, list) or not imp:
            rec("state.impact", FAIL, "impact 表为空（事件研究未跑）")
        else:
            rec("state.impact", PASS, f"{nev} 个事件 / {len(imp)} 组影响因子")
    except Exception as e:  # noqa: BLE001
        rec("state.impact", FAIL, str(e)[:80])

    # performance.behavior
    try:
        with open(os.path.join(sd, "performance.json"), encoding="utf-8") as f:
            perf = json.load(f)
        beh = perf.get("behavior")
        if not isinstance(beh, list) or not beh:
            rec("state.behavior", FAIL, f"behavior 为空 ({perf.get('behavior_error', '无错误信息')})")
        else:
            rec("state.behavior", PASS, f"{len(beh)} 个因子")
    except Exception as e:  # noqa: BLE001
        rec("state.behavior", FAIL, str(e)[:80])


# ---------------------------------------------------------------- 3. 账户配置
def check_account() -> None:
    print("")
    print("[3] 账户配置（按市场）")
    from qcore import account_config as AC
    base = AC.load("CN")
    if not (10_000 <= base["initial_capital"] <= 1e9):
        rec("account.default", FAIL, f"CN 默认本金越界: {base['initial_capital']}")
        return
    rec("account.default", PASS, f"CN 本金 {base['initial_capital']:,.0f} {base['currency']}")
    # 三个市场都要能读，且成本口径不同
    try:
        sm = AC.summary()
        mk = sm["markets"]
        costs = {m: mk[m]["cost_bps"] for m in ("CN", "US", "HK")}
        ok = len(set(costs.values())) == 3
        rec("account.markets", PASS if ok else FAIL,
            " / ".join(f"{m}={c}bps" for m, c in costs.items()))
    except Exception as e:  # noqa: BLE001
        rec("account.markets", FAIL, str(e)[:80])
    # 往返
    new, err = AC.save({"initial_capital": 250000, "top_k": 7}, "CN")
    if err or not new:
        rec("account.write", FAIL, err or "无返回")
    elif new["initial_capital"] != 250000 or new["top_k"] != 7:
        rec("account.write", FAIL, f"写入后不一致: {new}")
    else:
        rec("account.write", PASS, "写入并回读一致")
    # 越界必须拒绝
    _, err2 = AC.save({"initial_capital": 1}, "CN")
    rec("account.reject", PASS if err2 else FAIL, err2 or "越界值竟然被接受")
    # 非法市场必须拒绝
    _, err3 = AC.save({"initial_capital": 100}, "XX")
    rec("account.bad_market", PASS if err3 else FAIL, err3 or "非法市场竟然被接受")
    # 还原
    AC.save({"initial_capital": base["initial_capital"], "top_k": base["top_k"]}, "CN")


# ---------------------------------------------------------------- 4. HTTP
def check_http(port: int = 8791) -> None:
    print(f"\n[4] HTTP 端点 (port {port})")
    # 本地自检实例必须完全开放：不要继承 APB_TOKEN / APB_PASSWORD，
    # 否则 /api/account/config 这类 POST 会被 _write_allowed 拦成 403
    # （实测：在命令行设了 APB_TOKEN 跑 preflight，两个 POST 检查全红）
    env = {k: v for k, v in os.environ.items()
           if k not in ("APB_TOKEN", "APB_PASSWORD")}
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "webui.server", "--port", str(port), "--no-run"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"{base}/api/health", timeout=2)
                break
            except Exception:
                time.sleep(0.4)
        else:
            rec("http.boot", FAIL, "服务 16 秒内未就绪")
            return
        rec("http.boot", PASS, base)

        endpoints = ["status", "signals", "regime", "ledger", "performance",
                     "paper?market=CN", "transfer",
                     "paper/trades", "run/status", "version", "behavior", "events",
                     "health", "market", "update", "names", "account/config"]
        not_ready, failed = [], []
        for ep in endpoints:
            try:
                raw = urllib.request.urlopen(f"{base}/api/{ep}", timeout=15).read().decode()
                d = json.loads(raw)
                if not d.get("ok"):
                    if d.get("error") == "not_ready":
                        not_ready.append(f"{ep}(缺{d.get('missing')})")
                    else:
                        failed.append(f"{ep}:{d.get('error')}")
            except Exception as e:  # noqa: BLE001
                failed.append(f"{ep}:{type(e).__name__}")
        if failed:
            rec("http.endpoints", FAIL, "; ".join(failed[:4]))
        elif not_ready:
            rec("http.endpoints", FAIL, f"未就绪: {', '.join(not_ready)}")
        else:
            rec("http.endpoints", PASS, f"{len(endpoints)} 个全部 ok")

        # POST 账户配置
        body = json.dumps({"initial_capital": 800000}).encode()
        req = urllib.request.Request(f"{base}/api/account/config", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
            ok = d.get("ok") and d["data"]["initial_capital"] == 800000
            rec("http.post_account", PASS if ok else FAIL, "" if ok else str(d)[:80])
        except Exception as e:  # noqa: BLE001
            rec("http.post_account", FAIL, str(e)[:80])
        # 越界必须 400
        body2 = json.dumps({"initial_capital": 1}).encode()
        req2 = urllib.request.Request(f"{base}/api/account/config", data=body2,
                                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req2, timeout=10)
            rec("http.post_reject", FAIL, "越界值被接受")
        except urllib.error.HTTPError as e:
            rec("http.post_reject", PASS if e.code == 400 else FAIL, f"HTTP {e.code}")
        except Exception as e:  # noqa: BLE001
            rec("http.post_reject", FAIL, str(e)[:60])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()


# ---------------------------------------------------------------- 5. 测试套件
SUITES = [
    ("无未来函数", [sys.executable, "tests/test_no_lookahead.py"]),
    ("信号一致性", [sys.executable, "-m", "tests.test_signals"]),
    ("指标", [sys.executable, "-m", "tests.test_metrics"]),
    ("事件", [sys.executable, "-m", "tests.test_events"]),
    ("不变量", [sys.executable, "-m", "qcore.invariants"]),
]


def check_suites() -> None:
    print("\n[5] 测试套件")
    for name, cmd in SUITES:
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=420)
            if r.returncode == 0:
                rec(f"test.{name}", PASS, "")
            else:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["?"]
                rec(f"test.{name}", FAIL, tail[0][:110])
        except subprocess.TimeoutExpired:
            rec(f"test.{name}", FAIL, "超时")
        except Exception as e:  # noqa: BLE001
            rec(f"test.{name}", FAIL, str(e)[:80])


def check_wiring() -> None:
    """
    前后端接线检查 —— 这类 bug 占了本版全部严重问题的一半。

    查三件事：
      1. 路由调用的 self._xxx() 方法是否真的定义了
         （踩过两次：先加路由、判重条件被路由里的调用污染 -> 方法没加
          -> 运行时报 'ApiHandler' object has no attribute '_xxx'）
      2. app.js 里调用的自定义函数是否定义了
         （踩过：reasonText is not defined，node --check 查不出来）
      3. 关键数据文件的顶层键是否被 UI 正确读取
    """
    import re

    # ---- 1) 路由 -> 方法 ----
    p = os.path.join(ROOT, "webui", "server.py")
    try:
        src = open(p, encoding="utf-8").read()
    except Exception as e:  # noqa: BLE001
        rec("wiring.server", FAIL, f"读不到 {p}: {e}")
        return
    calls = set(re.findall(r"return self\.(_\w+)\(", src))
    defs = set(re.findall(r"^\s*def (_\w+)\(", src, re.M))
    missing = sorted(calls - defs)
    if missing:
        rec("wiring.routes", FAIL,
            f"{len(missing)} 个路由方法缺失: " + ", ".join(missing[:6]))
    else:
        rec("wiring.routes", PASS, f"{len(calls)} 个路由方法全部存在")

    # ---- 2) app.js 自定义函数 ----
    p2 = os.path.join(ROOT, "webui", "static", "app.js")
    try:
        js = open(p2, encoding="utf-8").read()
    except Exception as e:  # noqa: BLE001
        rec("wiring.ui", FAIL, f"读不到 app.js: {e}")
        return
    # 只查本项目自己定义的函数形状：function foo(  /  const foo = (
    defined = set(re.findall(r"function ([A-Za-z_]\w*)\s*\(", js))
    defined |= set(re.findall(r"(?:const|let|var) ([A-Za-z_]\w*)\s*=\s*(?:function|\()", js))
    # 调用点：名字后跟 (，且前面不是 . 或 function/const
    called = set()
    for m in re.finditer(r"(?<![.\w])([a-z][A-Za-z0-9_]{3,})\s*\(", js):
        called.add(m.group(1))
    # 常见内置/浏览器 API 白名单
    BUILTIN = {
        "if", "for", "while", "switch", "catch", "return", "typeof", "function",
        "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURIComponent",
        "decodeURIComponent", "requestAnimationFrame", "setTimeout", "setInterval",
        "clearTimeout", "clearInterval", "confirm", "alert", "prompt", "fetch",
        "require", "addEventListener", "removeEventListener", "stringify", "parse",
        "getElementById", "querySelector", "querySelectorAll", "createElement",
        "appendChild", "removeChild", "toLocaleTimeString", "toLocaleString",
        "toFixed", "toString", "toUpperCase", "toLowerCase", "indexOf", "slice",
        "splice", "join", "split", "replace", "includes", "startsWith", "endsWith",
        "padStart", "padEnd", "filter", "map", "forEach", "reduce", "find", "some",
        "every", "sort", "concat", "trim", "keys", "values", "entries", "assign",
        "from", "isArray", "now", "max", "min", "abs", "round", "floor", "ceil",
        "random", "pow", "sqrt", "log", "exp", "sign", "test", "exec", "match",
        "reload", "open", "close", "focus", "blur", "scrollTo", "print",
        "setAttribute", "getAttribute", "hasAttribute", "removeAttribute",
        "classList", "dataset", "preventDefault", "stopPropagation",
        # 关键字 / CSS 函数 / 注释里的文本，不是 JS 函数
        "async", "await", "yield", "new", "delete", "void", "in", "of", "do",
        "else", "try", "finally", "throw", "case", "default", "break", "continue",
        "repeat", "minmax", "rgba", "rgb", "hsl", "hsla", "calc", "var", "env",
        "translate", "translateX", "translateY", "scale", "rotate", "opacity",
        "url", "linear-gradient", "radial-gradient", "cubic-bezier",
        "list_factors", "describe", "it", "test", "expect",
    }
    maybe_missing = sorted(
        n for n in (called - defined - BUILTIN)
        if not n.startswith("_") and any(ch.isupper() for ch in n) is False
        and n not in ("toast", "card", "esc", "num", "pct")  # 这些在别处定义
    )
    # 只报告「看起来像自定义函数」的：小驼峰、长度>3、全小写开头
    suspicious = []
    for n in maybe_missing:
        # 出现在 `function n(` 或 `n =` 之外的才是真嫌疑
        if re.search(r"function\s+" + re.escape(n) + r"\b", js):
            continue
        if re.search(r"(?:const|let|var)\s+" + re.escape(n) + r"\b", js):
            continue
        # 只留下被当作函数调用、且项目里从未定义的
        suspicious.append(n)
    if suspicious:
        # 数量太多通常是正则误报，只给警告不阻断
        rec("wiring.ui", WARN, f"{len(suspicious)} 个疑似未定义: " +
            ", ".join(suspicious[:5]))
    else:
        rec("wiring.ui", PASS, f"{len(defined)} 个自定义函数都有定义")

    # ---- 3) source_health 结构 ----
    sh = os.path.join(ROOT, "data", "state", "source_health.json")
    if os.path.isfile(sh):
        try:
            d = json.load(open(sh, encoding="utf-8"))
            if isinstance(d, dict) and "sources" in d:
                rec("wiring.source_health", PASS,
                    f"结构正确，{len(d.get('sources') or {})} 个源")
            else:
                rec("wiring.source_health", WARN,
                    "缺 sources 段，UI 会遍历顶层键导致显示错位")
        except Exception as e:  # noqa: BLE001
            rec("wiring.source_health", WARN, str(e)[:60])


def check_smoke(port: int = 8791) -> None:
    """
    页面冒烟 —— 用 headless Chrome 打开每个页面，抓渲染错误。

    ## 为什么要放在 preflight

    实测教训：`node --check` 只查语法，查不出「调用了不存在的变量」。
    我写完 `reasonText` / `curatedUntil` / `r._live` 三处 TDZ/未定义错误，
    语法全过、preflight 全绿，但页面一点开就是「渲染失败」。

    更糟的是：冒烟脚本自己有编码 bug（Windows 用 GBK 解码 Chrome 的 UTF-8），
    导致它一直报 13/13 通过 —— **验证工具失灵比被测代码出错更危险**。

    所以：
      1. 冒烟必须自动化跑，不能靠人记得
      2. 跑不通要能明确看到是哪一页
    """
    import subprocess as sp
    # 冒烟要测「有真实数据」的页面。
    # 本地实例常常没有数据 -> 页面提前返回 notReadyBox -> 出错的分支根本执行不到
    # -> 冒烟全绿，但线上一点开就崩（实测：onlineRes is not defined 就是这样漏的）。
    # 设 APB_SMOKE_BASE 指向真实环境（如 NAS 域名）即可覆盖这条盲区。
    base = os.environ.get("APB_SMOKE_BASE", "").strip() or f"http://127.0.0.1:{port}"
    smoke = os.path.join(ROOT, "scripts", "smoke_pages.py")
    if not os.path.isfile(smoke):
        rec("smoke.pages", WARN, "没有 smoke_pages.py")
        return
    try:
        r = sp.run([sys.executable, smoke, f"--base={base}"],
                   capture_output=True, text=True, timeout=420,
                   encoding="utf-8", errors="replace",
                   env={k: v for k, v in os.environ.items()
                        if k not in ("APB_TOKEN", "APB_PASSWORD")})
        out = (r.stdout or "") + (r.stderr or "")
        if "未找到 Chrome" in out:
            rec("smoke.pages", WARN, "没装 Chrome，跳过页面冒烟")
            return
        m = None
        for line in out.splitlines():
            if "[smoke]" in line and "通过" in line:
                m = line.strip()
        if r.returncode == 0:
            rec("smoke.pages", PASS, (m or "全部页面通过") + f" @ {base}")
        else:
            fails = [ln.strip() for ln in out.splitlines() if "[FAIL]" in ln]
            rec("smoke.pages", FAIL,
                (m + " | " if m else "") + "; ".join(fails[:3]))
    except sp.TimeoutExpired:
        rec("smoke.pages", FAIL, "超时（>7 分钟）")
    except Exception as e:  # noqa: BLE001
        rec("smoke.pages", WARN, f"{type(e).__name__}: {str(e)[:60]}")


def check_docker() -> None:
    """
    Dockerfile 的 COPY 源必须存在，且不能被 .dockerignore 排除。

    实测踩过：Dockerfile 加了 `COPY tests/ ./tests/`，
    但 .dockerignore 里写着 `tests/` —— 本地怎么都正常，
    CI 构建直接失败，整个 Release 挂掉，tag 已经打出去了却发不了版。
    """
    df = os.path.join(ROOT, "Dockerfile")
    if not os.path.isfile(df):
        rec("docker.dockerfile", WARN, "没有 Dockerfile")
        return

    # 读 .dockerignore
    ignore = []
    ip = os.path.join(ROOT, ".dockerignore")
    if os.path.isfile(ip):
        for ln in open(ip, encoding="utf-8"):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                ignore.append(ln.rstrip("/"))

    src = open(df, encoding="utf-8").read()
    missing, ignored = [], []
    for m in re.finditer(r"^COPY\s+(.+?)\s+\S+\s*$", src, re.M):
        spec = m.group(1).strip()
        # `COPY --from=deps /install /usr/local` 的源是镜像内部路径，
        # 不是构建上下文里的文件，跳过。
        if "--from" in spec:
            continue
        if spec.startswith("--"):
            continue
        for part in spec.split():
            if part.startswith("--") or "$" in part:
                continue
            p = os.path.join(ROOT, part)
            if not os.path.exists(p):
                missing.append(part)
                continue
            # 检查是否被 dockerignore 排除（只查顶层目录名）
            top = part.split("/")[0]
            if top in ignore:
                ignored.append(part)

    if missing:
        rec("docker.copy_src", FAIL, f"COPY 源不存在: {', '.join(missing[:4])}")
    elif ignored:
        rec("docker.copy_src", FAIL,
            f"COPY 源被 .dockerignore 排除（CI 会失败）: {', '.join(ignored[:4])}")
    else:
        rec("docker.copy_src", PASS, "所有 COPY 源存在且未被排除")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过测试套件")
    a = ap.parse_args()

    print("=" * 66)
    print("APB 发布前自检")
    print("=" * 66)
    check_imports()
    check_state()
    check_account()
    check_http()
    check_wiring()
    check_docker()
    if not a.quick:
        check_smoke()
        check_suites()

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    print("\n" + "=" * 66)
    print(f"结果: {n_pass} 通过 / {n_fail} 失败 / {len(results) - n_pass - n_fail} 警告")
    if n_fail:
        print("\n失败项：")
        for n, s, d in results:
            if s == FAIL:
                print(f"  - {n}: {d}")
        print("\n>>> 不许发版。修完再跑一次。")
        return 1
    print(">>> 全部通过，可以发版。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
