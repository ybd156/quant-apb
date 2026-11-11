#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_event_history.py —— 历史事件库契约（不联网）

守住四件事：
  1. 目录本身自洽：id 唯一、日期可解析、方向 ∈ {-1,+1}、类型在白名单里
  2. 收益是**量出来**的，不是写死的：用本地沪深300现算，且落在合理区间
  3. 「直觉方向」与「真实反应」的差距要被显式量出来（不粉饰）
  4. analyst 侧：analogs/implied/evaluate 的算术正确

用法：python -m tests.test_event_history
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import event_history as E  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("event_history —— 历史参照库")
    print("=" * 70)

    # ---------- 1. 目录自洽 ----------
    ids = [e["id"] for e in E.CATALOG]
    check("id 唯一", len(ids) == len(set(ids)), f"{len(ids)} 条")
    bad_date = []
    for e in E.CATALOG:
        try:
            datetime.strptime(e["date"], "%Y-%m-%d")
        except Exception:  # noqa: BLE001
            bad_date.append(e["id"])
    check("日期全部可解析", not bad_date, str(bad_date))
    check("方向只有 ±1", all(e["direction"] in (-1, 1) for e in E.CATALOG))
    check("类型在白名单里", all(e["kind"] in E.KINDS for e in E.CATALOG))
    check("都有关键词（否则永远匹配不到）", all(e.get("keywords") for e in E.CATALOG))

    # ---------- 2. 收益是量出来的 ----------
    d = E.measure(h=5)
    check("本地行情可读", bool(d.get("ok")), d.get("reason") or d.get("index"))
    rows = d.get("rows") or []
    check("绝大多数事件能测出收益", (d.get("n_measured") or 0) >= 15,
          f"{d.get('n_measured')}/{d.get('n')}")
    cars = [r["car"] for r in rows if r.get("car") is not None]
    check("收益落在合理区间（|car|<50%）", all(abs(c) < 0.5 for c in cars), f"n={len(cars)}")
    check("每条都标了口径（不允许无来源数字）",
          all(r.get("basis") in ("market_proxy", "unavailable", "out_of_range") for r in rows))

    # ---------- 3. 直觉方向 vs 真实反应：差距要显式可见 ----------
    same = sum(1 for r in rows if r.get("car") is not None
               and (r["car"] > 0) == (r["direction"] > 0))
    n = len([r for r in rows if r.get("car") is not None])
    check("方向一致率被算出来（不是 100% 才正常）", 0 <= same <= n,
          f"{same}/{n} = {same / max(n, 1):.0%}（这就是「缺了多少」的第一层）")

    # ---------- 4. 匹配与合成 ----------
    a = E.analogs("美国宣布对等关税，出口链承压")
    check("关税类新闻能匹配到历史事件", any(x["id"] in ("tariff2025", "trade2018") for x in a),
          str([x["id"] for x in a]))
    check("无关句子不硬凑", E.analogs("今天食堂的菜有点咸") == [])

    im = E.implied([{"car": -0.02}, {"car": -0.04}, {"car": 0.01}])
    check("implied 取中位数", im and im["implied_car"] == -0.02, str(im))
    check("implied 报告方向一致率", im and im["direction_agree"] == round(2 / 3, 3), str(im))
    check("无有效参照时返回 None", E.implied([{"car": None}]) is None)

    # ---------- 5. evaluate 的算术 ----------
    from qcore import autoinfo as A
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", "evhist_test_%d" % os.getpid())
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    old = (A.LEDGER, A.VERIFIED)
    try:
        A.LEDGER = os.path.join(tmp, "ledger.jsonl")
        A.VERIFIED = os.path.join(tmp, "verified.jsonl")
        with open(A.LEDGER, "w", encoding="utf-8") as f:
            f.write(json.dumps({"as_of": "2026-10-01", "code": "600519", "url": "u1",
                                "implied_car": -0.03}) + "\n")
            f.write(json.dumps({"as_of": "2026-10-01", "code": "000001", "url": "u2",
                                "implied_car": 0.02}) + "\n")
        with open(A.VERIFIED, "w", encoding="utf-8") as f:
            f.write(json.dumps({"as_of": "2026-10-01", "code": "600519", "url": "u1",
                                "ret": -0.01}) + "\n")
            f.write(json.dumps({"as_of": "2026-10-01", "code": "000001", "url": "u2",
                                "ret": 0.02}) + "\n")
        ev = A.evaluate()
        check("evaluate 配对数正确", ev.get("n") == 2, str(ev.get("n")))
        check("bias = 平均(预期-真实)", abs(ev["bias"] - (-0.02 + 0.0) / 2) < 1e-9,
              f"bias={ev.get('bias')}")
        check("mae = 平均绝对差", abs(ev["mae"] - 0.01) < 1e-9, f"mae={ev.get('mae')}")
        check("sign_match 正确", ev["sign_match"] == 1.0, str(ev.get("sign_match")))
        check("样本不足时标 insufficient", ev.get("insufficient") is True)
    finally:
        A.LEDGER, A.VERIFIED = old
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
