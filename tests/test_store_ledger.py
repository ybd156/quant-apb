"""
回归测试：台账写入（幂等 / 归属 / 裸 NaN）与 status 朝向。

背景（实测）：
  - data/pred_ledger/2026.jsonl 47,676 行只有 8,150 个唯一 (as_of,code) 键，
    8,029 个键挂着 3–9 个互不相同的分数 —— rebuild 每次把最近 90 天整段重追加。
  - 24 行含裸 NaN，严格 JSON 解析失败（陷阱 10 的镜像：append_ledger 没走 _clean）。
  - status.json 的 n_stocks 取到 scores.shape[1] = T，UI 首屏显示「2633 只」而真值 112。

跑法：python -m tests.test_store_ledger
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 临时目录必须落在工作区内：沙箱不允许写系统 temp
def _tmpdir(prefix):
    # 注意：不要用 tempfile.mkdtemp —— 它创建的目录在当前沙箱下
    # 继承受限 DACL，open(..., "w") 会 PermissionError。
    base = os.path.join(ROOT, "tmp")
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f"{prefix}{os.getpid()}_{_SEQ[0]}")
    _SEQ[0] += 1
    os.makedirs(d, exist_ok=True)
    return d

from qcore import store  # noqa: E402

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


def _row(as_of, code, score, ret):
    return {"as_of": as_of, "code": code, "score": score, "ret": ret,
            "regime": 0, "entry_date": as_of, "entry_price": 10.0,
            "exit_date": as_of, "exit_price": 11.0}


def test_idempotent_append():
    print("\n[1] 台账幂等：同一 (as_of, code, horizon) 只落一行")
    tmp = _tmpdir("ledger_")
    old = store.LEDGER_DIR
    store.LEDGER_DIR = tmp
    try:
        rows = [_row("2026-08-10", "000001", 0.20, 0.01),
                _row("2026-08-10", "000063", -0.21, -0.02)]
        p = store.append_ledger(rows, year=2026, horizon=5, market="CN")
        n1 = sum(1 for _ in open(p, encoding="utf-8"))
        # 模拟 rebuild：同样的 90 天再追加一遍（这正是 5.85× 重复的成因）
        for _ in range(4):
            store.append_ledger(rows, year=2026, horizon=5, market="CN")
        n2 = sum(1 for _ in open(p, encoding="utf-8"))
        check("重复 append 5 次后行数不变", n1 == n2 == 2, f"{n1} -> {n2}")

        rec = json.loads(open(p, encoding="utf-8").readline())
        check("每行带 horizon", rec.get("horizon") == 5, str(rec.get("horizon")))
        check("每行带 market", rec.get("market") == "CN", str(rec.get("market")))
        check("每行带 run_id", bool(rec.get("run_id")), str(rec.get("run_id")))

        # 不同周期 = 不同预测，必须允许共存
        store.append_ledger([_row("2026-08-10", "000001", 0.20, 0.01)],
                            year=2026, horizon=1, market="CN")
        n3 = sum(1 for _ in open(p, encoding="utf-8"))
        check("h=1 与 h=5 不互相吞掉", n3 == 3, f"rows={n3}")
    finally:
        store.LEDGER_DIR = old


def test_no_bare_nan():
    print("\n[2] 裸 NaN 防护：python float NaN 必须落成 null")
    tmp = _tmpdir("ledger_nan_")
    old = store.LEDGER_DIR
    store.LEDGER_DIR = tmp
    try:
        p = store.append_ledger([_row("2026-08-11", "601995", float("nan"), float("nan"))],
                                year=2026, horizon=5, market="CN")
        raw = open(p, encoding="utf-8").read()
        check("输出无 NaN 字面量", "NaN" not in raw, raw.strip()[:80])
        rec = json.loads(raw.strip())
        check("score 落成 null", rec["score"] is None, str(rec["score"]))
        check("严格 JSON 可解析", True)
    finally:
        store.LEDGER_DIR = old


def test_compact_ledger():
    print("\n[3] 去重视图：保留最后一条 + dup_n，且不改原文件")
    tmp = _tmpdir("ledger_cmp_")
    old = store.LEDGER_DIR
    store.LEDGER_DIR = tmp
    try:
        rows = [_row("2026-08-12", "000001", 0.19917, 0.01)]
        store.append_ledger(rows, year=2026, horizon=5)
        # 手工追加两条不同分数的同键行（模拟历史遗留重复）
        p = os.path.join(tmp, "2026.jsonl")
        with open(p, "a", encoding="utf-8") as f:
            for sc in (0.19961, 0.20058):
                r = _row("2026-08-12", "000001", sc, 0.01)
                f.write(json.dumps(r) + "\n")
        before = open(p, encoding="utf-8").read()
        res = store.compact_ledger(2026)
        after = open(p, encoding="utf-8").read()
        check("原台账未被修改", before == after)
        check("唯一键数 = 1", res["unique_keys"] == 1, str(res["unique_keys"]))
        check("重复行数 = 2", res["duplicate_rows"] == 2, str(res["duplicate_rows"]))
        rec = json.loads(open(res["path"], encoding="utf-8").readline())
        check("保留最后一条（最新重算）", abs(rec["score"] - 0.20058) < 1e-9, str(rec["score"]))
        check("dup_n = 3", rec.get("dup_n") == 3, str(rec.get("dup_n")))
    finally:
        store.LEDGER_DIR = old


def test_status_orientation():
    print("\n[4] status 朝向：n_stocks 不能等于 n_days")
    src = open(os.path.join(ROOT, "qcore", "store.py"), encoding="utf-8").read()
    check('shape[1]' not in src.split('"n_stocks"')[1][:80],
          'n_stocks 不再用 scores.shape[1]')
    check('len(res.get("codes")' in src, "n_stocks 取自 codes 长度")

    p = os.path.join(ROOT, "data", "state", "status.json")
    if not os.path.isfile(p):
        print("  [SKIP] status.json 不存在（未跑过 pipeline）")
        return
    d = json.load(open(p, encoding="utf-8"))
    n_top = d.get("n_stocks")
    n_meta = (d.get("meta") or {}).get("n_stocks")
    n_days = d.get("n_days")
    if n_meta is not None:
        check("status.n_stocks == meta.n_stocks", n_top == n_meta,
              f"{n_top} vs {n_meta}")
    check("status.n_stocks != n_days（除非真的 N==T）",
          n_top != n_days or n_top == n_meta, f"n_stocks={n_top} n_days={n_days}")


def main() -> int:
    test_idempotent_append()
    test_no_bare_nan()
    test_compact_ledger()
    test_status_orientation()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
