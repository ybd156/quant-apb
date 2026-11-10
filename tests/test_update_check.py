"""
回归测试：更新检查必须能翻页（否则「最新版本」永远停在第一页的最大值）。

实测事故（2026-11-10）：GHCR 的 `/v2/<repo>/tags/list` 默认只返回 **100 个** tag
（按字典序）。版本数超过 100 后，0.28.x 落到第 2 页 —— `_tags_from` 不分页，
于是 `check()` 算出的 latest 停在 `0.27.0`，NAS 显示「已是最新，无需更新」，
训练机也就永远不更新。表现是「我明明发了新版，NAS 检测不到」。

本测试用假的 `_get_json` 模拟两页 tag，断言：
  1. 分页会把两页都取到（总数 = 两页之和）
  2. 取到的最大值是第 2 页里的新版本（而不是第 1 页的最大值）
  3. 单页不足 100 时立刻停，不会死循环

跑法：python -m tests.test_update_check
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore import update_check as UC  # noqa: E402

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


def _fake_pages(pages, calls=None):
    """构造假 _get_json：按 `last=` 参数返回对应页。"""
    def _get_json(url, headers=None, **kw):
        if calls is not None:
            calls.append(url)
        if "/token?" in url:
            return {"token": "fake"}
        # 取出 last 参数，决定返回第几页
        idx = 0
        if "last=" in url:
            last = url.split("last=")[1].split("&")[0]
            last = urllib_unquote(last)
            idx = pages.index(next(p for p in pages if p and p[-1] == last)) + 1
        return {"tags": list(pages[idx]) if idx < len(pages) else []}
    return _get_json


def urllib_unquote(s: str) -> str:
    import urllib.parse
    return urllib.parse.unquote(s)


def main() -> int:
    # ---- 1) 两页：第 1 页全是旧版本，第 2 页才有新版本（正是线上情形）
    # 第 1 页全是旧版本（都 < 0.27），且正好 100 个 —— 模拟 GHCR 的真实返回
    page1 = [f"0.{i}.{j}" for i in range(1, 11) for j in range(5)]
    page1 += ["v" + t for t in page1]                                                  # 共 100 个
    page2 = ["0.27.8", "v0.27.8", "0.28.0", "v0.28.0", "0.28.2", "v0.28.2"]          # 6 个
    pages = [page1, page2]
    calls = []
    orig = UC._get_json
    UC._get_json = _fake_pages(pages, calls)
    try:
        ok, tags, reason = UC._tags_from("ghcr.io")
        check("分页成功", ok, reason or "ok")
        check("两页都取到", len(tags) == len(page1) + len(page2),
              f"{len(tags)} == {len(page1) + len(page2)}")
        sem = [t for t in tags if UC._ver_key(t)]
        best = max(sem, key=UC._ver_key)
        check("最大值来自第 2 页（0.28.2）",
              UC._ver_key(best) == UC._ver_key("0.28.2"), best)
        check("确实发生了翻页（>=2 次 tags 请求）",
              sum(1 for u in calls if "tags/list" in u) >= 2,
              str(sum(1 for u in calls if "tags/list" in u)))
    finally:
        UC._get_json = orig

    # ---- 2) 单页不足 100：必须立刻停（不死循环）
    calls2 = []
    UC._get_json = _fake_pages([["0.28.0", "v0.28.0"]], calls2)
    try:
        ok, tags, _ = UC._tags_from("ghcr.io")
        check("单页即停", ok and len(tags) == 2, f"n={len(tags)}")
        check("只请求了 1 次 tags", sum(1 for u in calls2 if "tags/list" in u) == 1)
    finally:
        UC._get_json = orig

    # ---- 3) 无 token → 明确报 no_token，而不是静默空列表
    UC._get_json = lambda url, headers=None, **kw: ({}, {"_x": 1})[0]
    try:
        ok, tags, reason = UC._tags_from("ghcr.io")
        check("无 token 时 ok=False 且原因明确", (not ok) and reason == "no_token", reason)
    finally:
        UC._get_json = orig

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
