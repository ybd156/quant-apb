#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_roll_window.py —— 滚动和必须"与序列长度无关"（评审 R48 的地基）

背景：`_roll_sum` 原用前缀和差分 `cs[i]-cs[i-w]`。两个被减数随序列变长而变大，
舍入误差随之变化 → 同一个窗口在长序列与短序列上算出**不同**结果（实测末 800 根里
751 格差 1e-8 量级）。信号在阈值边界上会翻转 DIR，于是"增量 = 批量"做不到逐位相等。

本测试钉住：改用滑窗求和后，**同一窗口在任意长度序列上逐位相同**。

用法：python -m tests.test_roll_window
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore.signals import _roll_sum  # noqa: E402

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
    print("rolling window —— 截断逐位相等")
    print("=" * 70)

    rng = np.random.default_rng(11)
    x = np.cumprod(1 + rng.normal(0, 0.02, size=4000)) * 100

    for w in (5, 20, 60, 250):
        full = _roll_sum(x, w)
        for cut in (600, 1200):
            tail = _roll_sum(x[-cut:], w)
            d = np.abs(full[-cut:] - tail)
            n_diff = int(np.nansum((d > 0) & np.isfinite(d)))
            check(f"w={w} 末 {cut} 根逐位相等", n_diff == 0,
                  f"最大差 {np.nanmax(d):.2e}，不同格 {n_diff}")

    # 数值正确性：与朴素循环对照（容差 1e-9）
    w = 20
    got = _roll_sum(x, w)
    want = np.array([x[i - w + 1:i + 1].sum() if i >= w - 1 else np.nan
                     for i in range(len(x))])
    d = np.abs(got - want)
    check("与朴素求和在 1e-9 内一致", np.nanmax(d) < 1e-9, f"max {np.nanmax(d):.2e}")

    # 短序列 / 非法窗口
    check("n < w 返回全 NaN", np.all(np.isnan(_roll_sum(x[:3], 20))))
    check("w <= 0 返回全 NaN", np.all(np.isnan(_roll_sum(x[:10], 0))))

    # 常数序列：和必须精确 = w*const（前缀和在这里最容易掉精度）
    c = np.full(5000, 3.7)
    s = _roll_sum(c, 37)
    check("常数序列精确", np.allclose(s[36:], 37 * 3.7, rtol=0, atol=1e-12),
          f"max err {np.max(np.abs(s[36:] - 37 * 3.7)):.2e}")

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
