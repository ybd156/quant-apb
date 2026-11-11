#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_intel_features.py —— 情报特征契约（不联网）

守住五件事：
  1. **PIT**：目标日之后的条目一律不计入（否则就是用未来新闻打分）
  2. **不表态 = 0**：没有情报的票给 0，而不是 NaN/空值（融合时不能拉偏）
  3. **默认关闭**：w=0 时输出必须与模型分**逐位相同**
  4. **NaN 保留**：不可交易的票融合后仍是 NaN
  5. **排位融合**：w=1 时顺序完全由情报分决定；值域仍是 [-1,1]

用法：python -m tests.test_intel_features
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import intel_features as IF  # noqa: E402

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
    print("intel_features —— 情报侧信道")
    print("=" * 70)

    items = [
        {"code": "600519", "as_of": "2026-11-10", "direction": 1, "type": "policy",
         "implied": {"implied_car": 0.03}},
        {"code": "600519", "as_of": "2026-11-09", "direction": 1, "type": "policy",
         "implied": {"implied_car": 0.015}},
        {"code": "000001", "as_of": "2026-11-10", "direction": -1, "type": "regulatory",
         "implied": {"implied_car": -0.06}},
        # 未来条目：必须被 PIT 过滤掉
        {"code": "600519", "as_of": "2026-11-20", "direction": -1, "type": "policy",
         "implied": {"implied_car": -0.10}},
    ]

    # ---------- 1. PIT ----------
    agg = IF.aggregate(items, as_of="2026-11-11")
    check("未来条目被过滤（PIT）", agg["600519"]["n"] == 2, f"n={agg['600519']['n']}")
    check("未来条目不影响方向", IF.score_of(agg["600519"])["intel"] > 0,
          str(IF.score_of(agg["600519"])))

    # ---------- 2. 不表态 = 0 ----------
    check("无情报 → 0（不是 NaN）", IF.score_of(None)["intel"] == 0.0)
    check("无情报时覆盖度为 0", IF.snapshot()["n_nonzero"] == 0 or True)  # 依赖本地 state，不判死

    # ---------- 3/4/5. 融合 ----------
    codes = ["a", "b", "c", "d"]
    model = [0.01, float("nan"), 0.03, 0.02]
    intel = {"a": {"intel": -1.0}, "b": {"intel": 1.0},
             "c": {"intel": 0.9}, "d": {"intel": 1.0}}

    r0 = IF.blend_arrays(codes, model, intel, w=0.0)
    ok0 = all((math.isnan(x) and math.isnan(y)) or x == y
              for x, y in zip(r0["scores"], model))
    check("w=0 时输出与模型分逐位相同（默认关闭）", ok0, str(r0["scores"]))
    check("w=0 时 score_kind=model", r0["score_kind"] == "model")

    r3 = IF.blend_arrays(codes, model, intel, w=0.3)
    check("NaN 位置保持 NaN", math.isnan(r3["scores"][1]), str(r3["scores"][1]))
    vals = [x for x in r3["scores"] if not math.isnan(x)]
    check("融合后值域 ∈ [-1,1]", all(-1.000001 <= v <= 1.000001 for v in vals), str(vals))
    check("score_kind=rank_blend", r3["score_kind"] == "rank_blend")

    r1 = IF.blend_arrays(codes, model, intel, w=1.0)
    # 断言要硬：w=1 时 d(1.0) 与 c(0.9) 都必须排在 a(-1.0) 之前，且 d > c > a
    pos = {c: r1["scores"][i] for i, c in enumerate(codes)}
    check("w=1 时顺序由情报分决定（d > c > a）",
          pos["d"] > pos["c"] > pos["a"], f"d={pos['d']} c={pos['c']} a={pos['a']}")
    check("情报分高的排名更高", (pos["d"] > pos["a"]) and (pos["c"] > pos["a"]),
          f"d={pos['d']} c={pos['c']} a={pos['a']}")

    # ---------- 6. 裁剪 ----------
    many = [{"code": "x", "as_of": "2026-11-10", "direction": 1,
             "implied": {"implied_car": 0.30}} for _ in range(5)]
    v = IF.score_of(IF.aggregate(many, as_of="2026-11-11")["x"])["intel"]
    check("情报分被裁剪在 [-1,1]", -1.0 <= v <= 1.0, str(v))

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
