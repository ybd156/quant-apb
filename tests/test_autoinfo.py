#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_autoinfo.py —— 自主情报模块的契约测试（不联网）

守住四件事：
  1. **查谁**：配额轮转（持仓/推荐/自选都要有），且不重复
  2. **只追加 + 幂等**：同一批情报跑两次，第二次必须 new=0、行数不变
  3. **缺口驱动**：没查到情报的 subject 进 gaps（下一轮优先）
  4. **到期验证**：方向 × 收益 > 0 才是命中；未到期不结算、不伪造

用法：python -m tests.test_autoinfo
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import autoinfo as A  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _sandbox():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "tmp", "autoinfo_test_%d" % os.getpid())
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    A.STATE = os.path.join(d, "state", "autoinfo.json")
    A.LEDGER = os.path.join(d, "autoinfo", "ledger.jsonl")
    A.VERIFIED = os.path.join(d, "autoinfo", "verified.jsonl")
    A.STATE_DIR = os.path.join(d, "state")
    return d


def main() -> int:
    print("=" * 70)
    print("autoinfo —— 自主情报契约")
    print("=" * 70)

    tmp = _sandbox()
    try:
        # ---------- 1. 查谁 ----------
        real_subjects = A.subjects
        A.subjects = lambda limit=8: [
            {"code": "600519", "name": "贵州茅台", "why": "持仓"},
            {"code": "000001", "name": "平安银行", "why": "推荐"},
            {"code": "000002", "name": "万科A", "why": "自选"},
        ][:limit]

        news = {
            "600519": [{"code": "600519", "as_of": "2026-11-10", "title": "茅台提价",
                        "url": "u1", "media": "eastmoney", "in_title": True, "source": "info"}],
            "000001": [],   # 缺口
            "000002": [{"code": "000002", "as_of": "2026-11-09", "title": "万科融资",
                        "url": "u2", "media": "sina", "in_title": False, "source": "info"}],
        }
        A.fetch_news = lambda code, name="", days=7: list(news.get(code) or [])
        A._market_items = lambda: []

        r1 = A.run(max_subjects=8, verbose=False)
        check("选到 subject 并产出情报", r1.get("ok") and r1.get("n_items", 0) >= 2,
              f"n_items={r1.get('n_items')}")
        check("缺口被记录（没查到的进 gaps）", "000001" in (r1.get("gaps") or []),
              f"gaps={r1.get('gaps')}")
        check("只登记带 code+url 的情报", r1.get("ledger", {}).get("new") == 2,
              f"ledger={r1.get('ledger')}")

        # ---------- 2. 幂等 + 只追加 ----------
        with open(A.LEDGER, encoding="utf-8") as f:
            before = [json.loads(x) for x in f if x.strip()]
        r2 = A.run(max_subjects=8, verbose=False)
        with open(A.LEDGER, encoding="utf-8") as f:
            after = [json.loads(x) for x in f if x.strip()]
        check("重跑不新增行（幂等）", len(after) == len(before) and r2["ledger"]["new"] == 0,
              f"{len(before)} -> {len(after)}")
        check("历史行内容未被改写", after[:len(before)] == before, "逐行相等")

        # ---------- 3. 打分 ----------
        hi = A.score({"in_title": True, "as_of": "2026-11-11", "source": "info"}, {})
        lo = A.score({"in_title": False, "as_of": "2026-11-11", "source": "info"}, {})
        live = A.score({"in_title": True, "as_of": "2026-11-11", "source": "live"}, {})
        check("标题命中 > 仅正文", hi > lo, f"{hi} > {lo}")
        check("快讯降权 < 个股新闻", live < hi, f"{live} < {hi}")

        # ---------- 4. 到期验证 ----------
        # 用"足够老"的日期，否则 verify 会判未到期（d1 = as_of + 1.5h+2 天）
        def fake_close(code, day):
            if code == "600519":
                return 100.0 if day <= "2026-10-02" else 110.0
            return 100.0 if day <= "2026-10-02" else 95.0

        A._close_on = fake_close
        rows = [{"as_of": "2026-10-01", "code": "600519", "url": "u1", "direction": 1, "h": 5},
                {"as_of": "2026-09-30", "code": "000002", "url": "u2", "direction": -1, "h": 5}]
        A.LEDGER = os.path.join(tmp, "autoinfo", "ledger2.jsonl")
        A.VERIFIED = os.path.join(tmp, "autoinfo", "verified2.jsonl")
        with open(A.LEDGER, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        v = A.verify(verbose=False)
        check("到期情报被结算", v.get("new") == 2, f"new={v.get('new')}")
        ver = [json.loads(x) for x in open(A.VERIFIED, encoding="utf-8") if x.strip()]
        by = {r["code"]: r for r in ver}
        check("看多且涨 → 命中", by.get("600519", {}).get("hit") is True,
              f"ret={by.get('600519', {}).get('ret')}")
        check("看空且跌 → 命中", by.get("000002", {}).get("hit") is True,
              f"ret={by.get('000002', {}).get('ret')}")
        v2 = A.verify(verbose=False)
        check("已验证的不重复结算", v2.get("new") == 0 and v2.get("already") == 2,
              f"new={v2.get('new')} already={v2.get('already')}")

        # 未到期：把 h 调到很大
        A.VERIFIED = os.path.join(tmp, "autoinfo", "verified3.jsonl")
        with open(A.LEDGER, "w", encoding="utf-8") as f:
            f.write(json.dumps({"as_of": "2026-11-11", "code": "600519", "url": "u9",
                                "direction": 1, "h": 100}) + "\n")
        v3 = A.verify(h=100, verbose=False)
        check("未到期不结算、不伪造", v3.get("new") == 0 and v3.get("not_matured") == 1,
              f"new={v3.get('new')} skip={v3.get('not_matured')}")

        # ---------- 5. 读数 ----------
        A.LEDGER = os.path.join(tmp, "autoinfo", "ledger2.jsonl")
        s = A.summary()
        check("summary 报样本不足而不是给结论", s.get("insufficient") is True,
              f"verified={s.get('verified')}")

        A.subjects = real_subjects
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
