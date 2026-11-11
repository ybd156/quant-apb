#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_weighter_idempotent.py —— 学习器的"样本级幂等"（评审 R45）

背景：`live --loop` 每小时调一次 `step_once()`，而它的游标是"已封存日"
（数据到 11-10、h=5 → 11-04），所以同一个 as_of 再跑一次时
`run_walkforward` 会**把同一批到期样本再喂一遍**：`logw += η·y` 累积 24 次。

本测试用合成面板直接钉住两件事：
  1. 同一个面板 + 同一个权重器再跑一遍，`n_g`（每个信号的更新次数）**必须不变**
  2. 新建的权重器仍然会学（幂等不是"不学"）

用法：python -m tests.test_weighter_idempotent
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore.dataset import Panel3D  # noqa: E402
from qcore.engine import RunConfig, run_walkforward  # noqa: E402
from qcore.learner import AdaptiveWeighter  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _panel(T=60, N=4, S=3, h=1, seed=7):
    rng = np.random.default_rng(seed)
    dates = np.array([np.datetime64("2026-01-01") + np.timedelta64(i, "D") for i in range(T)])
    px = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(T, N)), axis=0)
    DIR = rng.integers(-1, 2, size=(T, N, S)).astype(np.int8)
    STR = np.full((T, N, S), 200, dtype=np.uint8)
    RET = {}
    for hh in (h,):
        arr = np.full((T, N), np.nan, dtype=np.float32)
        if T > 1 + hh:
            arr[: T - 1 - hh] = px[1 + hh:] / px[1: T - hh] - 1.0
        RET[hh] = arr
    return Panel3D(dates=dates, codes=[f"C{i}" for i in range(N)],
                   sids=[f"s{i}" for i in range(S)], DIR=DIR, STR=STR,
                   OPEN=px.astype(np.float32), CLOSE=px.astype(np.float32),
                   RET=RET,
                   SIGMA=np.full((T, N), 0.01, dtype=np.float32),
                   REGIME=np.zeros(T, dtype=np.int8),
                   VALID=np.ones((T, N), dtype=bool),
                   UNIV=np.ones((T, N), dtype=bool))


def main() -> int:
    print("=" * 70)
    print("weighter 样本级幂等（R45）")
    print("=" * 70)

    p3d = _panel()
    w = AdaptiveWeighter(3, n_regimes=1, sids=p3d.sids)
    cfg = RunConfig(horizon=1, warmup=0, record_ledger=False)

    run_walkforward(p3d, cfg, weighter=w)
    n1 = w.n_g.copy()
    lt1 = w.learned_through
    check("第一次跑确实学了", float(n1.sum()) > 0,
          f"Σn_g={n1.sum():.0f} learned_through={lt1}")
    check("游标被推进到最后可对账的样本日", bool(lt1), str(lt1))

    # 同一个面板、同一个权重器再跑一遍（模拟 loop 的第二次迭代）
    run_walkforward(p3d, cfg, weighter=w)
    n2 = w.n_g.copy()
    check("再跑一遍 n_g **不变**（样本级幂等）", np.array_equal(n1, n2),
          f"{n1.tolist()} vs {n2.tolist()}")
    check("游标不回退", w.learned_through == lt1, f"{w.learned_through} vs {lt1}")

    # 再跑第三遍，仍然不变
    run_walkforward(p3d, cfg, weighter=w)
    check("第三次跑也不变", np.array_equal(n2, w.n_g.copy()))

    # 全新权重器：应该重新学（幂等 ≠ 不学）
    w2 = AdaptiveWeighter(3, n_regimes=1, sids=p3d.sids)
    run_walkforward(p3d, cfg, weighter=w2)
    check("新权重器会正常学习（幂等不是关闭学习）", float(w2.n_g.sum()) > 0,
          f"Σn_g={w2.n_g.sum():.0f}")

    # 状态可以从 npz 往返（游标必须持久化，否则重启后又会重学）
    import tempfile
    from qcore import store
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", f"wtest_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    try:
        pth = store.save_weighter(w, path=os.path.join(tmp, "w.npz"), horizon=1)
        w3 = store.load_weighter(path=pth, horizon=1, strict=False)
        check("游标随 npz 往返", getattr(w3, "learned_through", None) == lt1,
              f"{getattr(w3, 'learned_through', None)} vs {lt1}")
        run_walkforward(p3d, cfg, weighter=w3)
        check("重启后读回游标 → 不重学", np.array_equal(n1, w3.n_g.copy()))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
