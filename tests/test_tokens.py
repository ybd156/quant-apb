"""
访问令牌回归测试：多令牌、哈希存储、单独吊销、scope。

背景（2026-11-10）：原来只有一个 `APB_TOKEN` 环境变量 ——
部署到别处不好改、多台训练机无法单独吊销、泄露只能全部换掉、
训练机没配 token 就 `login_required`（模型训好了传不回来）。

本测试钉住：
  1. 明文只在创建时返回一次，库里只存哈希
  2. 校验通过 / 吊销后立即失效 / 删除
  3. scope 生效（sync 令牌不能做 write）
  4. 列表绝不泄露明文与哈希

跑法：python -m tests.test_tokens
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore import tokens as TK  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    # 用独立文件，别碰真实 tokens.json
    tmp = os.path.join(ROOT, "tmp", "test_tokens")
    os.makedirs(tmp, exist_ok=True)
    TK.PATH = os.path.join(tmp, "tokens.json")
    TK.STATE = tmp
    if os.path.isfile(TK.PATH):
        os.remove(TK.PATH)

    print("[1] 创建")
    a = TK.create(name="训练机-A", note="家里那台")
    b = TK.create(name="云主机", scopes=["sync"])
    check("返回明文", bool(a.get("token")), (a.get("token") or "")[:14] + "…")
    check("明文带前缀", str(a["token"]).startswith(TK.TOKEN_PREFIX))
    check("两个令牌不同", a["token"] != b["token"])
    check("返回里有 warning", "只显示这一次" in str(a.get("warning")))

    print("\n[2] 存储不含明文")
    raw = open(TK.PATH, encoding="utf-8").read()
    check("文件里没有明文", a["token"] not in raw)
    check("文件里没有第二个明文", b["token"] not in raw)
    d = json.loads(raw)
    check("存的是哈希", all("hash" in t and len(t["hash"]) == 64
                          for t in d["tokens"]), f"{len(d['tokens'])} 条")

    print("\n[3] 校验")
    ok, rec = TK.verify(a["token"], scope="write")
    check("有效令牌通过", ok, rec.get("name") if rec else "")
    ok2, _ = TK.verify("apb_not_a_real_token")
    check("乱写的令牌不通过", ok2 is False)
    ok3, _ = TK.verify("")
    check("空令牌不通过", ok3 is False)

    print("\n[4] scope")
    ok4, rec4 = TK.verify(b["token"], scope="write")
    check("sync 令牌不能 write", ok4 is False and rec4 is not None,
          rec4.get("name") if rec4 else "")
    ok5, _ = TK.verify(b["token"], scope="sync")
    check("sync 令牌可以 sync", ok5 is True)

    print("\n[5] 列表不泄露")
    lst = TK.list_tokens()
    txt = json.dumps(lst, ensure_ascii=False)
    check("列表不含明文", a["token"] not in txt)
    check("列表不含哈希", "hash" not in txt)
    check("列表有两条", len(lst) == 2, str(len(lst)))
    check("有前缀便于识别", all(t.get("prefix") for t in lst))

    print("\n[6] 吊销后立即失效")
    check("吊销成功", TK.revoke(a["id"]) is True)
    ok6, _ = TK.verify(a["token"], scope="write")
    check("吊销后不通过", ok6 is False)
    ok7, _ = TK.verify(b["token"], scope="sync")
    check("另一个令牌不受影响", ok7 is True)
    check("重复吊销返回 False", TK.revoke(a["id"]) is False)

    print("\n[7] 删除")
    check("删除成功", TK.delete(a["id"]) is True)
    check("删完只剩一条", len(TK.list_tokens()) == 1)
    check("删除不存在的返回 False", TK.delete("tk_nope") is False)

    print("\n[8] 负控：校验器真的会拒绝")
    check("把哈希当令牌用不通过", TK.verify(d["tokens"][0]["hash"])[0] is False)

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
