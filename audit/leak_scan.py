#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit/leak_scan.py -- 未来函数 / 数据泄漏 静态扫描器（独立红队审计，SPEC §0.0.2 / §10.3）

设计原则
--------
1. **纯静态**：只读源码，不 import qcore、不跑回测。可在任何模块未就绪时独立运行。
2. **逻辑语句级**：先把跨行链式调用拼成一条"逻辑语句"，再判断统计量是否挂在
   rolling / expanding / ewm / groupby 上。逐行正则会漏掉
   `(close - close.rolling(60).mean()) / close.rolling(60).std()` 这种换行写法。
3. **分级**：HIGH 必炸（exit 1）；MEDIUM 需人工确认（exit 0，除非 --strict）。
4. **可豁免**：行尾写 `# leak-scan: ok <理由>` 即跳过该行，理由会打印出来，
   避免审计变成"用 noqa 淹没问题"。

退出码
------
0  无 HIGH（可能含 MEDIUM / INFO）
1  发现 HIGH 级泄漏嫌疑
2  扫描器自身出错（文件读不到等）

用法
----
    python audit/leak_scan.py                # 扫 qcore/*.py
    python audit/leak_scan.py --path qcore   # 指定目录
    python audit/leak_scan.py --strict       # MEDIUM 也算失败
    python audit/leak_scan.py --verbose      # 打印所有命中（含 INFO）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# ------------------------------------------------------------------ 常量

# 允许"向过去看"的窗口构造器：出现这些词，说明统计量是滚动/扩张口径，安全
SAFE_WINDOW = re.compile(
    r"\b(rolling|expanding|ewm|cumsum|cummax|cummin|cumprod|groupby|resample|"
    r"shift\s*\(\s*\d+\s*\)|pct_change|diff|autocorr)\s*\("
)

# 全样本统计量（若接收者没有 SAFE_WINDOW -> 疑似全样本口径 = 隐性未来函数）
FULLSAMPLE_STAT = re.compile(
    r"\.(mean|std|median|var|quantile|min|max|sum|skew|kurt|mad|sem|nunique)\s*\("
)

# numpy 版本的全样本统计量
NP_FULLSAMPLE = re.compile(
    r"\bnp\.(nan)?(mean|std|median|percentile|quantile|min|max|ptp|var)\s*\("
)

# ---- HIGH：结构性未来函数，几乎不存在合法用法 ----
HIGH_PATTERNS = [
    (re.compile(r"\.shift\s*\(\s*-\s*\d"), "shift(-n) 向后取未来行（经典 lookahead）"),
    (re.compile(r"\.shift\s*\(\s*-\s*[A-Za-z_]\w*"), "shift(-var) 向后取未来行"),
    (re.compile(r"\.bfill\s*\("), "bfill 用未来值回填（时间序列上=偷看）"),
    (re.compile(r"method\s*=\s*['\"]bfill['\"]"), "fillna(method='bfill') 未来值回填"),
    (re.compile(r"method\s*=\s*['\"]backfill['\"]"), "fillna(method='backfill') 未来值回填"),
    (re.compile(r"limit_direction\s*=\s*['\"]both['\"]"), "interpolate(limit_direction='both') 向未来插值"),
    (re.compile(r"limit_direction\s*=\s*['\"]backward['\"]"), "interpolate(limit_direction='backward') 向未来插值"),
    (re.compile(r"reindex\s*\([^)]*method\s*=\s*['\"]bfill['\"]"), "reindex(method='bfill') 未来值回填"),
    (re.compile(r"center\s*=\s*True"), "rolling(center=True) 窗口含未来行"),
    (re.compile(r"merge_asof\s*\([^)]*direction\s*=\s*['\"]forward['\"]"), "merge_asof(direction='forward') 向未来对齐"),
    (re.compile(r"merge_asof\s*\([^)]*direction\s*=\s*['\"]nearest['\"]"), "merge_asof(direction='nearest') 可能取未来行"),
    (re.compile(r"\.iloc\s*\[[^:\]]*\+\s*(?:\d|[A-Za-z_]\w*)"), ".iloc[i+k] 取未来行（k>0）"),
    (re.compile(r"\.iat\s*\[[^:\]]*\+\s*(?:\d|[A-Za-z_]\w*)"), ".iat[i+k] 取未来行"),
    (re.compile(r"\bshift\s*\(\s*-"), "shift(-n) 未来行"),
]

# ---- MEDIUM：可能是合法横截面统计，也可能是全样本口径 ----
MEDIUM_PATTERNS = [
    (re.compile(r"\.rank\s*\(\s*pct\s*=\s*True"), ".rank(pct=True) 若为全样本排名则泄漏（按日横截面才安全）"),
    (re.compile(r"\.transform\s*\("), "groupby().transform 若聚合窗含未来行则泄漏"),
    (re.compile(r"\.agg\s*\("), ".agg() 聚合，需确认窗口"),
    (re.compile(r"\.apply\s*\("), ".apply() 自定义函数，需确认是否用了全序列"),
    (re.compile(r"\bnp\.sign\s*\([^)]*\)\s*\.cumsum"), "sign().cumsum() 全样本累计"),
]

# 行尾豁免标记
WAIVER = re.compile(r"#\s*leak-scan\s*:\s*ok\b[ \t]*(.*)$", re.IGNORECASE)

# 评估/报表类文件：全样本 mean/std 是"对结果做汇总"，属正常统计，不是特征泄漏。
# 这些文件里的 MEDIUM 降级为 INFO（仍统计，--verbose 可见），
# 但 HIGH 级结构性模式（shift(-1)/bfill/center=True）在任何文件里都照报。
EVAL_FILES = {"metrics.py", "report.py", "backtest.py", "stats.py", "store.py"}


def _blank_triple(text: str) -> str:
    """
    把三引号字符串（docstring）的内容替换成空格，保留换行与总长度。

    必要性：signals.py 的 docstring 里写着"禁止使用 shift(-n)、rolling(center=True)"，
    不屏蔽就会把**注释文字**当成真实代码报 HIGH —— 假阳性会让审计失去可信度。
    只屏蔽三引号（docstring 惯例），保留单/双引号字符串，
    因为 `fillna(method='bfill')` 这类模式必须看见引号里的内容。
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text.startswith('"""', i) or text.startswith("'''", i):
            q = text[i:i + 3]
            j = text.find(q, i + 3)
            if j < 0:
                j = n - 3
            for k in range(i + 3, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 3
        else:
            i += 1
    return "".join(out)

# 注释剥离（保留字符串内容，避免误伤；行号不变）
def _strip_comment(line: str) -> str:
    """去掉 # 之后的注释，但跳过字符串内的 #。"""
    out = []
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            out.append(ch)
            if ch == "\\":
                if i + 1 < len(line):
                    out.append(line[i + 1])
                    i += 2
                    continue
            elif ch == quote:
                quote = None
        else:
            if ch in "\"'":
                quote = ch
                out.append(ch)
            elif ch == "#":
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def _iter_logical_statements(src_lines):
    """
    把物理行拼成逻辑语句（括号平衡 + 反斜杠续行）。
    产出 (joined_text, [(start_idx, end_idx), ...]) —— 每行在 joined_text 中的偏移区间，
    用于把 joined 里的命中位置映射回物理行号。
    """
    stmts = []
    buf = []
    spans = []
    depth = 0
    for ln, raw in enumerate(src_lines):
        code = _strip_comment(raw.rstrip("\n"))
        if not buf and not code.strip():
            continue
        start_off = sum(len(b) + 1 for b in buf)
        buf.append(code)
        spans.append((ln, start_off, start_off + len(code)))
        depth += code.count("(") - code.count(")")
        depth += code.count("[") - code.count("]")
        depth += code.count("{") - code.count("}")
        cont = code.rstrip().endswith("\\")
        if depth <= 0 and not cont:
            stmts.append(("\n".join(buf), spans))
            buf, spans, depth = [], [], 0
    if buf:
        stmts.append(("\n".join(buf), spans))
    return stmts


def _line_of(spans, offset):
    """joined 偏移 -> 物理行号（0-based）。"""
    lo = spans[0][0]
    for ln, a, b in spans:
        if a <= offset <= b:
            return ln
        lo = ln
    return lo


def _receiver_start(s, dot_pos):
    """
    从 s[dot_pos]=='.' 向前回溯接收者表达式起点（括号平衡）。
    返回起始下标，使 s[start:dot_pos] 为接收者文本。
    """
    j = dot_pos - 1
    depth = 0
    while j >= 0:
        ch = s[j]
        if ch in ")]}":
            depth += 1
        elif ch in "([{":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and ch in ",;:=+*/%<>!&|^~":
            break
        elif depth == 0 and ch == "\n":
            break
        j -= 1
    return j + 1


WAIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waivers.json")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FILE_WAIVERS = None          # relpath -> reason


def _file_waivers() -> dict:
    """读 audit/waivers.json（文件级豁免）。

    行内 `# leak-scan: ok <理由>` 只能豁免具体那一行；实测 MEDIUM 有 700+ 条、
    散在 40 个文件，逐行加注释不现实。所以另有一条**可复核的机械规则**：
    不在打分/建模链路闭包内的文件，其全样本统计只影响事后指标，按文件豁免。
    规则与生成脚本见 audit/gen_waivers.py（可重复执行、可 --check）。
    """
    global _FILE_WAIVERS
    if _FILE_WAIVERS is not None:
        return _FILE_WAIVERS
    _FILE_WAIVERS = {}
    try:
        with open(WAIVER_PATH, encoding="utf-8") as f:
            d = json.load(f)
        for w in (d.get("waivers") or []):
            p = str(w.get("path") or "").replace("\\", "/")
            if p:
                _FILE_WAIVERS[p] = str(w.get("reason") or "file waiver")
    except Exception:
        pass
    return _FILE_WAIVERS


class Finding:
    __slots__ = ("path", "line", "sev", "reason", "code", "waived", "waiver")

    def __init__(self, path, line, sev, reason, code, waived=False, waiver=""):
        self.path = path
        self.line = line
        self.sev = sev
        self.reason = reason
        self.code = code
        self.waived = waived
        self.waiver = waiver

    def __str__(self):
        loc = f"{self.path}:{self.line}"
        code = self.code.strip()
        if len(code) > 110:
            code = code[:107] + "..."
        tag = " [WAIVED]" if self.waived else ""
        s = f"{loc}: [{self.sev}]{tag} {self.reason}\n      | {code}"
        if self.waived and self.waiver:
            s += f"\n      | waiver: {self.waiver.strip()}"
        return s


def scan_file(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()
    raw_lines = raw_text.splitlines(keepends=True)
    # 屏蔽 docstring 内容后再做模式匹配（保留行号与偏移）
    masked_lines = _blank_triple(raw_text).splitlines(keepends=True)

    # 整文件豁免（如文件头写 leak-scan: file-ok）
    head = "".join(raw_lines[:8])
    if re.search(r"leak-scan\s*:\s*file-ok", head, re.IGNORECASE):
        return []

    is_eval = os.path.basename(path) in EVAL_FILES
    findings = []
    for joined, spans in _iter_logical_statements(masked_lines):
        # 该语句内每行的豁免
        waiver_by_line = {}
        for ln, a, b in spans:
            m = WAIVER.search(raw_lines[ln])
            if m:
                waiver_by_line[ln] = m.group(1)

        def add(sev, reason, pos, line=None):
            ln = _line_of(spans, pos) if line is None else line
            waived = ln in waiver_by_line
            waiver_txt = waiver_by_line.get(ln, "")
            if not waived:
                rel = os.path.relpath(path, _ROOT).replace("\\", "/")
                fr = _file_waivers().get(rel)
                if fr:
                    waived, waiver_txt = True, f"[file] {fr}"
            findings.append(
                Finding(path, ln + 1, sev, reason,
                        raw_lines[ln].rstrip(), waived, waiver_txt)
            )

        # ---------- HIGH ----------
        for pat, reason in HIGH_PATTERNS:
            for m in pat.finditer(joined):
                add("HIGH", reason, m.start())

        # ---------- MEDIUM: 全样本统计量 ----------
        for m in FULLSAMPLE_STAT.finditer(joined):
            dot = m.start()
            rstart = _receiver_start(joined, dot)
            receiver = joined[rstart:dot]
            if SAFE_WINDOW.search(receiver):
                continue
            # 纯标量/字面量接收者（如 x.mean() 且 x 是 list）无法静态判定，仍报 MEDIUM
            name = m.group(1)
            # 噪声过滤：覆盖率统计 / 打印 / attrs 元数据，全样本口径也无害
            noisy = ("attrs[" in joined or joined.lstrip().startswith("print(")
                     or any(t in receiver for t in
                            ("notna(", "isna(", "isfinite(", "isnan(", "len(")))
            sev = "INFO" if (is_eval or noisy) else "MEDIUM"
            add(sev,
                f".{name}() 未挂在 rolling/expanding 链上 -> 疑似全样本口径"
                f"（评估/报表类文件里通常是正常汇总；特征类文件里需人工确认）",
                dot)

        for m in NP_FULLSAMPLE.finditer(joined):
            rstart = _receiver_start(joined, m.start())
            receiver = joined[rstart:m.start()]
            if SAFE_WINDOW.search(receiver):
                continue
            add("INFO" if is_eval else "MEDIUM",
                f"{m.group(0)} 数组级统计 -> 若作用于全样本序列则泄漏（需人工确认）",
                m.start())

        # ---------- MEDIUM: 其他可疑 ----------
        for pat, reason in MEDIUM_PATTERNS:
            for m in pat.finditer(joined):
                add("INFO" if is_eval else "MEDIUM", reason, m.start())

    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description="静态扫描未来函数/数据泄漏")
    ap.add_argument("--path", default="qcore", help="要扫描的目录或文件（默认 qcore）")
    ap.add_argument("--strict", action="store_true", help="MEDIUM 也视为失败")
    ap.add_argument("--verbose", action="store_true", help="打印全部命中")
    ap.add_argument("--quiet-waived", action="store_true", help="不打印已豁免项")
    args = ap.parse_args(argv)

    target = args.path
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.py")))
    else:
        files = sorted(glob.glob(target))
    files = [f for f in files if os.path.isfile(f) and not f.endswith("__init__.py")]

    if not files:
        print(f"[leak_scan] no python files under {target!r}")
        return 2

    all_findings = []
    for f in files:
        try:
            all_findings.extend(scan_file(f))
        except Exception as e:  # noqa: BLE001
            print(f"[leak_scan] ERROR scanning {f}: {e!r}")
            return 2

    print("=" * 78)
    print("audit/leak_scan.py  --  static look-ahead / data-leak scan")
    print("=" * 78)
    print(f"scanned {len(files)} file(s) under {target!r}")
    for f in files:
        print(f"  - {f}")
    print("-" * 78)

    active = [x for x in all_findings if not x.waived]
    waived = [x for x in all_findings if x.waived]
    waived_file = [x for x in waived if str(x.waiver).startswith("[file]")]
    waived_line = [x for x in waived if not str(x.waiver).startswith("[file]")]
    highs = [x for x in active if x.sev == "HIGH"]
    meds = [x for x in active if x.sev == "MEDIUM"]
    infos = [x for x in active if x.sev == "INFO"]

    show = active if args.verbose else [x for x in active if x.sev in ("HIGH", "MEDIUM")]
    if show:
        for x in sorted(show, key=lambda z: (z.path, z.line)):
            print(x)
            print()
    if not args.quiet_waived and waived:
        print(f"-- waived by '# leak-scan: ok' ({len(waived)}) --")
        for x in sorted(waived, key=lambda z: (z.path, z.line)):
            print(f"  {x.path}:{x.line}: {x.reason}")
            if x.waiver:
                print(f"      waiver: {x.waiver.strip()}")
        print()

    print("-" * 78)
    print(f"RESULT: HIGH={len(highs)}  MEDIUM={len(meds)}  INFO={len(infos)}  "
          f"WAIVED={len(waived)}（行内 {len(waived_line)} / 文件级 {len(waived_file)}）")
    if highs:
        print("VERDICT: FAIL -- structural look-ahead found. Build must not pass.")
        return 1
    if meds and args.strict:
        print("VERDICT: FAIL (--strict) -- MEDIUM findings require sign-off.")
        return 1
    if meds:
        print("VERDICT: PASS with warnings -- MEDIUM items need human confirmation")
        print("         (re-run with --verbose to list them).")
        return 0
    print("VERDICT: PASS -- no look-ahead pattern detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
