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
    print("\n[3] 账户配置")
    from qcore import account_config as AC
    base = AC.load()
    if not (10_000 <= base["initial_capital"] <= 1e9):
        rec("account.default", FAIL, f"默认本金越界: {base['initial_capital']}")
        return
    rec("account.default", PASS, f"本金 {base['initial_capital']:,.0f}")
    # 往返
    new, err = AC.save({"initial_capital": 250000, "top_k": 7})
    if err or not new:
        rec("account.write", FAIL, err or "无返回")
    elif new["initial_capital"] != 250000 or new["top_k"] != 7:
        rec("account.write", FAIL, f"写入后不一致: {new}")
    else:
        rec("account.write", PASS, "写入并回读一致")
    # 越界必须拒绝
    _, err2 = AC.save({"initial_capital": 1})
    rec("account.reject", PASS if err2 else FAIL, err2 or "越界值竟然被接受")
    # 还原
    AC.save({"initial_capital": base["initial_capital"], "top_k": base["top_k"]})


# ---------------------------------------------------------------- 4. HTTP
def check_http(port: int = 8791) -> None:
    print(f"\n[4] HTTP 端点 (port {port})")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
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

        endpoints = ["status", "signals", "regime", "ledger", "performance", "paper",
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
    if not a.quick:
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
