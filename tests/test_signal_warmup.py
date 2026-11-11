#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_signal_warmup.py —— 预热截断必须逐位相等（评审 R48 第二步）

`compute_signals(df, tail=k, warmup=W)` 只算最后 W+k 根。**W 用错就违反 INV-1**，
所以这里钉住两件事：

1. W=**400** 时，最后 k 行与全量计算**逐位相等**（DIR 整数、STR 浮点都相等）；
2. 不给 warmup（默认）时，行为与原来完全一致（回归保护）。

400 这个数字来自 `python -m tools.signal_warmup_probe` 的实测：
W=250 时仍有 4 个信号不同（atr_chan_20_1p5 / ma_slope_250_20 / macd_hist_12_26_9 /
rsi_21_ob65，EMA 递归 + 250 日窗口所致），W=400 时 106/106 全等。

用法：python -m tests.test_signal_warmup
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore.data import load_kline  # noqa: E402
from qcore.signals import compute_signals  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _cmp(full, part, tail):
    d0, s0 = full
    d1, s1 = part
    n_dir = 0
    max_str = 0.0
    bad = []
    for sid in d0.columns:
        a = d0[sid].to_numpy()[-tail:]
        b = d1[sid].to_numpy()[-tail:]
        nd = int(np.nansum(np.nan_to_num(a, nan=9) != np.nan_to_num(b, nan=9)))
        sa = s0[sid].to_numpy()[-tail:]
        sb = s1[sid].to_numpy()[-tail:]
        sd = float(np.nanmax(np.abs(np.nan_to_num(sa, nan=9) - np.nan_to_num(sb, nan=9))))
        if nd or sd > 0:
            bad.append(sid)
        n_dir += nd
        max_str = max(max_str, sd)
    return n_dir, max_str, bad


def main() -> int:
    print("=" * 70)
    print("signal warmup —— W=400 逐位相等")
    print("=" * 70)
    tail = 30
    codes = ("600519", "000001", "300750")
    for c in codes:
        try:
            df = load_kline(c)
        except Exception as e:  # noqa: BLE001
            check(f"{c} 行情可读", False, f"{type(e).__name__}"); continue
        if df is None or len(df) < 500:
            check(f"{c} 历史足够", False, f"bars={0 if df is None else len(df)}"); continue
        full = compute_signals(df)
        part = compute_signals(df, tail=tail, warmup=400)
        n_dir, max_str, bad = _cmp(full, part, tail)
        check(f"{c} W=400 末 {tail} 根逐位相等", n_dir == 0 and max_str == 0.0,
              f"DIR 差 {n_dir} 格，STR 最大差 {max_str:.1e}"
              + (f"，例：{', '.join(bad[:4])}" if bad else ""))
        # 默认路径必须与"不传 warmup"完全一致
        again = compute_signals(df, tail=tail)
        n2, s2, _ = _cmp(full, again, tail)
        check(f"{c} 不传 warmup 时行为不变", n2 == 0 and s2 == 0.0, f"{n2} 格 / {s2:.1e}")

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
