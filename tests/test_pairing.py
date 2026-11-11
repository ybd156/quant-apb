#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_pairing.py —— 授权链接/授权码（训练机一步配置）

守住七件事：
  1. 码是 8 位、去掉了易混字符（0/O/1/I/L）；
  2. 链接三种形态都能解析回 (nas, code)：`#pair=` / `?pair=` / `apb://`；
  3. 裸码（大小写、空格、缺横线）也能归一；
  4. 兑换成功 → 签发令牌 + 码立即作废（**二次兑换必须被拒**）；
  5. 过期码被拒；
  6. 乱码被拒，且**连续失败会被限速**；
  7. 兑换拿到的令牌能通过 `tokens.verify()`（不是野字符串）。

用法：python -m tests.test_pairing
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import pairing as PR  # noqa: E402
from qcore import tokens as TK  # noqa: E402

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
    print("pairing —— 授权链接 + 授权码")
    print("=" * 70)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", f"pair_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    old_pair, old_tok = PR.PAIR_FILE, TK.PATH
    PR.PAIR_FILE = os.path.join(tmp, "pair_codes.json")
    TK.PATH = os.path.join(tmp, "tokens.json")
    try:
        r = PR.create(nas="https://nas.example.com", ttl_min=30, label="测试")
        code = r["code"]
        check("码是 8 位且带横线", len(code.replace("-", "")) == 8 and "-" in code, code)
        check("不含易混字符",
              not (set(code.replace("-", "")) &
                   {"0", "O", "1", "I", "L"}), code)
        check("给了三种链接", all(r["links"].get(k) for k in ("fragment", "query", "apb")),
              str(r["links"]))

        # 2) 链接解析
        for label, text in (
            ("#pair=", r["links"]["fragment"]),
            ("?pair=", r["links"]["query"]),
            ("apb://", r["links"]["apb"]),
        ):
            p = PR.parse_link(text)
            check(f"能解析 {label}", p.get("ok") and p["code"] == code
                  and p["nas"].rstrip("/") == "https://nas.example.com", str(p)[:100])

        # 3) 裸码归一
        bare = code.replace("-", "")
        for label, text in (("原样", code), ("小写", code.lower()),
                            ("无横线", bare), ("带空格", f" {code[:4]} {code[4:]} ")):
            p = PR.parse_link(text)
            check(f"裸码归一：{label}", p.get("ok") and p["code"] == code, str(p)[:80])
        check("乱码被拒", PR.parse_link("abc")["ok"] is False)
        check("空输入被拒", PR.parse_link("")["ok"] is False)

        # 4) 兑换 → 作废
        got = PR.redeem(code, ip="10.0.0.9")
        check("兑换成功", got.get("ok") is True, str(got)[:100])
        check("拿到了令牌", str(got.get("token") or "").startswith("apb_"), str(got.get("token"))[:14] + "…")
        ok, meta = TK.verify(str(got.get("token") or ""))
        check("令牌可被 tokens.verify 认", ok is True, str(meta)[:80])
        again = PR.redeem(code)
        check("二次兑换被拒（一次性）", again.get("ok") is False
              and again.get("reason") == "already_used", str(again.get("reason")))

        # 5) 过期
        r2 = PR.create(nas="https://nas.example.com", ttl_min=1)
        d = json.load(open(PR.PAIR_FILE, encoding="utf-8"))
        for c in d["codes"]:
            if c["code"] == r2["code"]:
                c["expires_ts"] = time.time() - 5
        json.dump(d, open(PR.PAIR_FILE, "w", encoding="utf-8"), ensure_ascii=False)
        ex = PR.redeem(r2["code"])
        check("过期码被拒", ex.get("ok") is False and ex.get("reason") == "expired",
              str(ex.get("reason")))

        # 6) 限速
        PR._FAILS.clear()
        codes = [PR.redeem("ZZZZ-ZZZZ", ip="1.2.3.4") for _ in range(PR.MAX_FAILS_PER_MIN + 3)]
        limited = [x for x in codes if x.get("reason") == "rate_limited"]
        check(f"连续失败会被限速（>{PR.MAX_FAILS_PER_MIN} 次/分钟）", len(limited) >= 1,
              f"限速 {len(limited)} 次")
        ok_other = PR.redeem("ZZZZ-ZZZZ", ip="9.9.9.9")
        check("别的 IP 不受影响", ok_other.get("reason") != "rate_limited",
              str(ok_other.get("reason")))

        # 7) 列表
        lst = PR.list_codes(include_used=True)
        check("列表能给出状态", lst["ok"] and all(
            r["state"] in ("active", "used", "expired") for r in lst["rows"]),
            str([r["state"] for r in lst["rows"]])[:80])
    finally:
        PR.PAIR_FILE, TK.PATH = old_pair, old_tok
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
