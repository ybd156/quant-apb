"""
负控测试：不变量检查器必须「能失败」。

AGENTS §7 陷阱 19：验证工具自己失灵 —— 冒烟测试对报错页面报「全过」。
这里对每条检查先构造**已知违规样本**，断言它确实返回 fail / 命中计数 > 0。
如果注入违规后仍然 PASS，说明检查器在说谎。

跑法：python -m tests.test_invariants_negative
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _tmpdir(prefix):
    base = os.path.join(ROOT, "tmp")
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f"{prefix}{os.getpid()}_{_SEQ[0]}")
    _SEQ[0] += 1
    os.makedirs(d, exist_ok=True)
    return d

from qcore import invariants as IV  # noqa: E402

PASS = 0
FAIL = 0
_SEQ = [0]


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _fake_read(mapping):
    """把 IV._read 换成按路径返回伪造源码。"""
    orig = IV._read

    def _r(p):
        key = os.path.basename(str(p))
        return mapping.get(key, orig(p))

    IV._read = _r
    return orig


def test_inv2_catches_maturity_bug():
    print("\n[1] INV-2 必须抓到 s = t（到期判据被删）")
    good = ("# ① 先预测\n    ...\n# ②③ 对账\n    s = t - 1 - h\n")
    bad = ("# ① 先预测\n    ...\n# ②③ 对账\n    s = t\n")
    orig = _fake_read({"engine.py": bad})
    try:
        st, detail = IV.inv2_order()
        check("注入 s = t -> FAIL", st == "fail", f"{st} :: {detail}")
    finally:
        IV._read = orig
    orig = _fake_read({"engine.py": good})
    try:
        st, detail = IV.inv2_order()
        check("正确判据 -> PASS", st == "pass", f"{st} :: {detail}")
    finally:
        IV._read = orig


def test_inv4_ast_counts_missing_lags():
    print("\n[2] INV-4 必须数出未传 lags 的 mean_tstat 调用")
    tmp = _tmpdir("inv4_")
    p = os.path.join(tmp, "analysis.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write("import x\n"
                "def a():\n"
                "    return x.mean_tstat(v, lags=4)\n"
                "def b():\n"
                "    return x.mean_tstat(v)\n")
    n_calls, n_missing = IV._mean_tstat_calls_without_lags(p)
    check("识别 2 处调用", n_calls == 2, str(n_calls))
    check("识别 1 处缺 lags", n_missing == 1, str(n_missing))

    # 旧实现（字符串计数相减）在这种样本上会给出 bad<=0 的假 pass：
    src_bad = ("x.mean_tstat(a)\n"
               "x.mean_tstat(b)\n"
               "other(lags=1)\nother(lags=2)\nother(lags=3)\n")
    old_bad = src_bad.count("mean_tstat(") - src_bad.count("lags=")
    check("旧字符串法会误判为 pass（记录 bug 指纹）", old_bad <= 0, f"old_bad={old_bad}")


def test_inv7_append_mode_ast():
    print("\n[3] INV-7 必须区分 append 与覆盖写")
    tmp = _tmpdir("inv7_")
    p_ok = os.path.join(tmp, "ok.py")
    with open(p_ok, "w", encoding="utf-8") as f:
        f.write("def append_ledger(rows):\n"
                "    with open('a.jsonl', 'a', encoding='utf-8') as fh:\n"
                "        fh.write('x')\n")
    p_bad = os.path.join(tmp, "bad.py")
    with open(p_bad, "w", encoding="utf-8") as f:
        f.write("def append_ledger(rows):\n"
                "    with open('a.jsonl', 'w', encoding='utf-8') as fh:\n"
                "        fh.write('x')\n")
    check("append 模式识别为 1", IV._uses_append_mode(p_ok, "append_ledger") == 1)
    check("覆盖模式识别为 0", IV._uses_append_mode(p_bad, "append_ledger") == 0)


def test_inv3_discloses_medium():
    print("\n[4] INV-3 不得把 MEDIUM 未确认写成「已人工确认」")
    st, detail = IV.inv3_rolling()
    if st == "skip":
        print(f"  [SKIP] {detail}")
        return
    check("detail 不含「已人工确认」", "已人工确认" not in detail, detail)
    check("detail 披露未确认条数", ("未确认" in detail) or ("waiver" in detail), detail)


def test_inv5_discloses_scope():
    print("\n[5] INV-5 必须披露抽查覆盖率")
    st, detail = IV.inv5_data_guard()
    if st == "skip":
        print(f"  [SKIP] {detail}")
        return
    check("detail 含覆盖率", ("抽查" in detail) or ("全量" in detail), detail)


def main() -> int:
    test_inv2_catches_maturity_bug()
    test_inv4_ast_counts_missing_lags()
    test_inv7_append_mode_ast()
    test_inv3_discloses_medium()
    test_inv5_discloses_scope()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
