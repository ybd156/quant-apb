"""
回归测试：Deflated Sharpe 的单位。

Bug 指纹（实测）：`analysis.py:109` 与 `stats.py:262` 把「每期收益均值」
（≈4e-4）当成「每期 Sharpe」（mean/std）传给 PSR 公式，而 sr0_daily 是
Sharpe 单位（≈4.7e-2），两者差 ~100 倍 → DSR 恒被压在
Φ(-E[max SR]) ≈ 0.0078 的地板上，h=1/5/20 三个不同策略全给 0.008。

本测试锁定两件事：
  1. 传「每期 Sharpe」时，N=1 的 DSR 必须等于 Φ(sr·√(n-1))（Bailey & LdP）。
  2. 传「收益均值」时 DSR 会退化到地板值 —— 记录 bug 指纹，防止回潮。

跑法：python -m tests.test_dsr_units
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore import stats as S  # noqa: E402

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


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_identity_single_trial():
    print("\n[1] N=1 时 DSR 必须等于 Φ(sr·√(n-1))")
    n, sr = 1000, 0.42
    d = S.deflated_sharpe(sr, n, 1)
    expect = phi(sr * math.sqrt(n - 1))
    check("DSR 与解析式一致（误差<1e-9）",
          abs(d["dsr"] - expect) < 1e-9,
          f"dsr={d['dsr']:.10f} expect={expect:.10f}")


def test_monotone_in_sharpe():
    print("\n[2] DSR 必须随 Sharpe 单调上升（旧实现几乎是常数）")
    n, N = 2610, 73
    vals = [S.deflated_sharpe(sr, n, N)["dsr"] for sr in (0.01, 0.05, 0.10, 0.20)]
    check("严格递增", all(b > a for a, b in zip(vals, vals[1:])),
          " ".join(f"{v:.4f}" for v in vals))
    spread = vals[-1] - vals[0]
    check("极差 > 0.3（有判别力）", spread > 0.3, f"spread={spread:.4f}")


def test_bug_fingerprint():
    print("\n[3] 记录 bug 指纹：传收益均值会退化到地板值")
    n, N = 2610, 73
    # 年化 1.5 的 Sharpe → 每期 ≈ 0.0945，高于 N=73 的门槛 sr0≈0.0473
    sr_hi = 1.5
    d_ok = S.deflated_sharpe(sr_hi / math.sqrt(252), n, N)
    d_bug = S.deflated_sharpe(0.000419, n, N)   # 旧实现传入的「每期收益均值」
    check("传 Sharpe 时可判（dsr > 0.5）", d_ok["dsr"] > 0.5, f"{d_ok['dsr']:.4f}")

    # 反过来：年化 0.42 的 Sharpe 在 N=73 下**过不了**门槛 ——
    # 这不是策略的问题，是「11 年样本 × 73 次试验」的信息量上限（提案 S-2）。
    d_mid = S.deflated_sharpe(0.42 / math.sqrt(252), n, N)
    check("年化 0.42 Sharpe 在 N=73 下不显著（门槛 sr0≈0.0473/期）",
          d_mid["dsr"] < 0.5, f"{d_mid['dsr']:.4f}")
    check("传收益均值时被压到 <0.05（= 旧 0.008 的来源）",
          d_bug["dsr"] < 0.05, f"{d_bug['dsr']:.4f}")
    check("两者差异 > 10 倍", d_ok["dsr"] > 10 * d_bug["dsr"],
          f"{d_ok['dsr']:.4f} vs {d_bug['dsr']:.4f}")


def test_floor_is_ntrials_driven():
    print("\n[4] 地板值由 n_trials 决定（说明它与策略无关）")
    n = 2610
    a = S.deflated_sharpe(1e-6, n, 73)["dsr"]
    b = S.deflated_sharpe(1e-6, n, 73)["dsr"]
    c = S.deflated_sharpe(1e-6, n, 219)["dsr"]
    check("同 N 同 n 复现一致", abs(a - b) < 1e-12, f"{a:.6f}")
    check("N 越大地板越低", c < a, f"N=73 -> {a:.6f}, N=219 -> {c:.6f}")


def test_summarize_edge_uses_sharpe():
    print("\n[5] summarize_edge 的 DSR 必须基于 per-period Sharpe")
    rng = np.random.default_rng(7)
    ic = rng.normal(0.02, 0.06, 1200)     # per-period SR ≈ 0.333
    out = S.summarize_edge(ic, np.arange(1200), horizon=1, n_trials=1)
    sr_pp = out["sharpe_per_period"]
    expect = phi(sr_pp * math.sqrt(1200 - 1))
    check("输出带 sharpe_per_period", np.isfinite(sr_pp), f"{sr_pp:.4f}")
    check("DSR 与 sharpe_per_period 自洽",
          abs(out["deflated_sharpe"]["dsr"] - expect) < 1e-9,
          f"dsr={out['deflated_sharpe']['dsr']:.10f}")


def main() -> int:
    test_identity_single_trial()
    test_monotone_in_sharpe()
    test_bug_fingerprint()
    test_floor_is_ntrials_driven()
    test_summarize_edge_uses_sharpe()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
