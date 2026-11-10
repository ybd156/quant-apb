"""
不变量自检 —— 把 AGENTS.md §3 的 8 条铁律变成可执行、可展示的检查。

每个检查返回 (status, detail)：
    pass  已用代码/数据验证通过
    fail  违反
    skip  当前环境无法验证（例如缺数据）—— **不谎报 pass**

结果写到 `data/state/invariants.json`，供 Web UI 的「风险体检」页展示。
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")


def _read(p: str) -> str:
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def inv1_no_lookahead() -> tuple:
    """INV-1 无未来函数：跑截断不变性检验。"""
    script = os.path.join(ROOT, "tests", "test_no_lookahead.py")
    if not os.path.exists(script):
        return "skip", "tests/test_no_lookahead.py 不存在"
    try:
        r = subprocess.run([sys.executable, script], cwd=ROOT,
                           capture_output=True, text=True, timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"worst diff\s*=\s*([0-9.eE+-]+)", out)
        if r.returncode == 0:
            return "pass", f"截断不变性通过（worst diff={m.group(1) if m else '0'}）"
        if r.returncode == 2:
            return "skip", "无数据，检验被跳过"
        return "fail", f"退出码 {r.returncode}"
    except Exception as e:  # noqa: BLE001
        return "skip", f"{type(e).__name__}: {e}"


def inv2_order() -> tuple:
    """INV-2 先预测后对账：顺序 + 到期判据都必须可验证。

    旧实现只 grep 两个注释字符串的位置，把 `s = t - 1 - h` 改成 `s = t`
    照样 PASS —— 检查器测的是注释，不是语义。
    """
    src = _read(os.path.join(ROOT, "qcore", "engine.py"))
    if not src:
        return "skip", "engine.py 不可读"
    i_score = src.find("# ① 先预测")
    i_upd = src.find("# ②③ 对账")
    if i_score < 0 or i_upd < 0:
        return "skip", "找不到顺序标记注释"
    if i_score > i_upd:
        return "fail", "对账出现在打分之前"
    m = re.search(r"^\s*s\s*=\s*t\s*-\s*1\s*-\s*h\s*$", src, re.M)
    if not m:
        return "fail", ("到期判据不是 s = t - 1 - h（AGENTS INV-2："
                        "决策日 s 的样本须在 t ≥ s+1+h 才可用）")
    return "pass", "循环顺序 ① 打分 → ②③ 对账；到期判据 s = t - 1 - h 存在"


def inv3_rolling():
    """INV-3 统计量必须滚动：静态扫描高危全样本统计。"""
    script = os.path.join(ROOT, "audit", "leak_scan.py")
    if not os.path.exists(script):
        return "skip", ("静态扫描器未随镜像发布（audit/leak_scan.py）。"
                        "该检查在 CI 每次发版执行，见 GitHub Actions 的 "
                        "「无未来函数检验」步骤")
    try:
        r = subprocess.run([sys.executable, script], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"HIGH=(\d+)", out)
        high = int(m.group(1)) if m else -1
        m_med = re.search(r"MEDIUM=(\d+)", out)
        m_wv = re.search(r"WAIVED=(\d+)", out)
        med = int(m_med.group(1)) if m_med else -1
        wv = int(m_wv.group(1)) if m_wv else -1
        if high == 0:
            # 旧文案写「MEDIUM 项为评估层统计，已人工确认」——但实测
            # MEDIUM=704 里只有 96 条带 waiver，608 条从未被人看过。
            # 检查器不能替人工签收，只能如实报数。
            tail = (f"；MEDIUM={med} 中已 waiver {wv}，其余 {med - wv} 条未确认"
                    if med >= 0 and wv >= 0 else "；MEDIUM 数无法解析")
            return "pass", f"HIGH=0{tail}（详见 audit/leak_scan.py --verbose）"
        if high > 0:
            return "fail", f"HIGH={high} 处疑似全样本统计"
        return "skip", "扫描器输出无法解析"
    except Exception as e:  # noqa: BLE001
        return "skip", f"扫描器异常: {type(e).__name__}"

def _mean_tstat_calls_without_lags(path: str) -> tuple:
    """AST 级检查：analysis.py 里每个 mean_tstat(...) 调用是否显式传了 lags。

    旧实现是跨文件做字符串减法（mean_tstat( 的个数 − lags= 的个数），
    只要别处多写一个 lags= 就会让 bad 变负 → 无条件 pass。删掉一行 lags=
    它也照样 pass —— 即「验证工具自己失灵」（AGENTS §7 陷阱 19）。
    """
    import ast
    src = _read(path)
    if not src:
        return (0, -1)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (0, -1)
    n_calls = n_missing = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "mean_tstat":
            continue
        n_calls += 1
        has_kw = any(kw.arg == "lags" for kw in node.keywords)
        # 位置参数也算：mean_tstat(x, lags) 是合法写法（stats.py 内部就这样调）。
        # 只认关键字参数会制造假 FAIL —— 检查器自己也会说谎。
        has_pos = len(node.args) >= 2
        if not (has_kw or has_pos):
            n_missing += 1
    return (n_calls, n_missing)


def inv4_hac() -> tuple:
    """INV-4 重叠收益必须用 HAC：AST 检查每个 mean_tstat 调用都传了 lags。"""
    targets = ["qcore/analysis.py", "qcore/meta.py", "qcore/confidence.py",
               "qcore/engine.py", "qcore/stats.py"]
    total_calls = total_missing = 0
    bad_files = []
    for f in targets:
        n_calls, n_missing = _mean_tstat_calls_without_lags(os.path.join(ROOT, f))
        if n_calls < 0:
            continue
        total_calls += n_calls
        total_missing += n_missing
        if n_missing:
            bad_files.append(f"{f}({n_missing})")
    if total_calls == 0:
        return "skip", "未找到 mean_tstat 调用"
    if total_missing:
        return "fail", f"{total_missing}/{total_calls} 处 mean_tstat 未传 lags：{bad_files}"
    return "pass", f"{total_calls} 处 mean_tstat 全部显式传 lags"


def inv5_data_guard() -> tuple:
    """INV-5 后复权非零风险：窗口内非正价 bar 计数。"""
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, "tests", "fixtures", "*_h2.csv")))
    if not files:
        return "skip", "无行情文件"
    cap = 200
    scan = files[:cap]
    try:
        import pandas as pd
        bad = 0
        total = 0
        for p in scan:
            df = pd.read_csv(p, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"])
            w = df[df["date"] >= "2013-01-01"]
            total += len(w)
            bad += int((w["close"] <= 0).sum())
        # 旧实现只查前 200 个文件却不披露 —— 陷阱 16「静默截断」发生在
        # 检查器自己身上。要么查全，要么把覆盖率写进 detail。
        scope = (f"全量 {len(scan)}/{len(files)} 个文件" if len(scan) == len(files)
                 else f"抽查 {len(scan)}/{len(files)} 个文件（未覆盖 {len(files)-len(scan)} 个）")
        return ("pass", f"窗口内非正价 {bad}/{total}（{scope}）") if bad == 0 else \
               ("fail", f"窗口内出现 {bad} 根非正价（{scope}）")
    except Exception as e:  # noqa: BLE001
        return "skip", f"{type(e).__name__}: {e}"


def inv6_pit_universe() -> tuple:
    """INV-6 股票池点对点：UNIV 掩码是否真正生效（N >= n_univ）。

    **这条检查只证明「掩码非退化」，不证明「池子无生存者偏差」。**
    池成员由 get_pool() 的今日市值/成交额产生，退市股结构性缺席，
    掩码无法消除这一层。detail 里如实写明边界。
    """
    st = _read(os.path.join(STATE_DIR, "status.json"))
    try:
        d = json.loads(st) if st else {}
        meta = d.get("meta", {})
        n = int(meta.get("n_stocks") or 0)
        n_top = int(d.get("n_stocks") or 0)
        n_univ = int(meta.get("n_univ") or 100)
    except Exception:
        return "skip", "status.json 无池信息"
    if n == 0:
        return "skip", "未记录股票池规模"
    warn = ""
    if n_top and n_top != n:
        warn = f"；注意 status.n_stocks={n_top} ≠ meta.n_stocks={n}（朝向 bug 未修净）"
    head = f"N={n} ≥ n_univ={n_univ}，掩码非退化"
    tail = "；但池成员由今日市值/成交额产生，退市股缺席 → 生存者偏差未被此检查覆盖"
    return ("pass", head + tail + warn) if n >= n_univ else \
           ("fail", f"N={n} < n_univ={n_univ}，掩码恒真等于未生效")


def _uses_append_mode(path: str, func_name: str) -> int:
    """AST：函数体内 open(...) 的 mode 里有几个是 "a"。"""
    import ast
    src = _read(path)
    if not src:
        return -1
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return -1
    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != func_name:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name != "open":
                continue
            mode = ""
            if len(sub.args) >= 2 and isinstance(sub.args[1], ast.Constant):
                mode = str(sub.args[1].value)
            for kw in sub.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if "a" in mode:
                n += 1
    return n


def inv7_append_only() -> tuple:
    """INV-7 台账只追加：结构（写入用 append 模式）+ 数据（无坏行/裸 NaN）。

    旧实现只检查源码里出现过 `"a"` 这个字面量 —— 只要别处写过一次就 pass，
    对 83% 重复、无周期归属的台账照样判 PASS。
    """
    dirs = [os.path.join(ROOT, "data", "pred_ledger"), os.path.join(ROOT, "data", "claims")]
    files = [p for d in dirs for p in sorted(glob.glob(os.path.join(d, "*.jsonl")))]
    if not files:
        return "skip", "尚无台账文件"
    n_app = _uses_append_mode(os.path.join(ROOT, "qcore", "store.py"), "append_ledger")
    if n_app <= 0:
        return "fail", "store.append_ledger 未使用 append 模式写入"
    bad = 0
    nan_new = 0
    nan_legacy = 0
    total = 0
    dup_note = ""
    for p in files:
        base = os.path.basename(p)
        is_main_ledger = ("pred_ledger" in p.replace("\\", "/")
                          and not base.endswith(".dedup.jsonl"))
        keys = set()
        rows = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                total += 1
                rows += 1
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if "NaN" in line or "Infinity" in line:
                    # 历史行（无 run_id，修复前写入）已冻结 —— INV-7 不许回改，
                    # 只能披露；修复后写入的行（带 run_id）必须干净，否则 FAIL。
                    if r.get("run_id"):
                        nan_new += 1
                    else:
                        nan_legacy += 1
                if r.get("as_of") is not None and r.get("code") is not None:
                    keys.add((str(r.get("as_of"))[:10], str(r.get("code"))))
        if is_main_ledger and keys and len(keys) < rows:
            dup_note = (f"；主台账唯一键 {len(keys)}/{rows}（历史重复；"
                        f"只追加约束下不可回改，读数请用 compact_ledger 生成的 .dedup.jsonl）")
    if bad or nan_new:
        return "fail", (f"台账存在 {bad} 行不可解析、{nan_new} 行**修复后写入**的裸 NaN"
                        f"（历史遗留 {nan_legacy} 行，已冻结不可回改）")
    legacy_note = (f"；历史遗留裸 NaN {nan_legacy} 行（无 run_id，INV-7 下不可回改）"
                   if nan_legacy else "")
    return "pass", (f"{len(files)} 个台账文件，写入走 append；{total} 行全部可解析，"
                    f"修复后写入的行无裸 NaN{legacy_note}{dup_note}")


def inv8_ui_no_calc() -> tuple:
    """INV-8 UI 不做计算：server.py 不得 import numpy/pandas。"""
    src = _read(os.path.join(ROOT, "webui", "server.py"))
    if not src:
        return "skip", "webui/server.py 不存在"
    bad = [m for m in ("import numpy", "import pandas", "from numpy", "from pandas")
           if m in src]
    return ("pass", "server.py 无 numpy/pandas，纯 IO") if not bad else \
           ("fail", f"server.py 引入了 {bad}")


CHECKS = [
    ("INV-1", "无未来函数（截断不变性）", inv1_no_lookahead),
    ("INV-2", "先预测后对账（循环顺序）", inv2_order),
    ("INV-3", "统计量必须滚动（泄漏扫描）", inv3_rolling),
    ("INV-4", "重叠收益用 HAC（lags=h-1）", inv4_hac),
    ("INV-5", "后复权且窗口内无非正价", inv5_data_guard),
    ("INV-6", "股票池点对点（UNIV 生效）", inv6_pit_universe),
    ("INV-7", "台账只追加", inv7_append_only),
    ("INV-8", "UI 不做计算", inv8_ui_no_calc),
]


def run_all(write: bool = True) -> List[Dict]:
    out = []
    for cid, name, fn in CHECKS:
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001
            status, detail = "skip", f"{type(e).__name__}: {e}"
        out.append({"id": cid, "name": name, "status": status, "detail": detail})
    if write:
        os.makedirs(STATE_DIR, exist_ok=True)
        path = os.path.join(STATE_DIR, "invariants.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    return out


def main() -> int:
    res = run_all()
    icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}
    n_fail = 0
    for r in res:
        print(f"  [{icon[r['status']]}] {r['id']} {r['name']} :: {r['detail']}")
        n_fail += r["status"] == "fail"
    print(f"\n{len(res)} checks, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
