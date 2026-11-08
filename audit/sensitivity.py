#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit/sensitivity.py -- 参数敏感性 + "孤峰"过拟合检测（SPEC §0.1.4 / §10.4）

要证明的事
----------
SPEC 说"参数不手调，任何魔法数字必须做敏感性分析"。
本文件对 6 个关键参数做单参数扫描（其余参数固定在 CONFIG 默认），
看指标随参数变化是**平缓**还是**断崖/孤峰**。

孤峰（isolated peak）定义
-------------------------
某个**内部**参数点的指标明显高于左右邻居，而左右邻居彼此接近：
      gain = (m_best - mean(m_left, m_right)) / max(|mean(m_left,m_right)|, eps)
      gain > 0.30  且  |m_left - m_right| < 0.5*|m_best - mean(邻居)|
=> 判定"孤峰"：该参数值像是被样本内挑出来的，换到相邻值就崩，过拟合风险高。

为什么要看"相邻点是否接近"
--------------------------
只看"最高点"会把单调趋势误判为孤峰。真正的孤峰是"尖"，两侧都塌，
且两侧高度相近 —— 说明那一点的特殊性不是来自参数趋势，而是来自噪声。

诚实声明
--------
若所有参数点的指标都 ≈ 0（本系统当前就接近这个情况），
那么"没有孤峰"只说明**各参数一样没用**，不代表稳健。
本文件会显式打印这个判断，避免把"均匀无效"包装成"参数稳健"。

用法
----
    python audit/sensitivity.py                # 全部 6 个参数
    python audit/sensitivity.py --top 30 --start 2015-01-01
    python audit/sensitivity.py --params tau,eta
    python audit/sensitivity.py --synthetic    # 不依赖 signals.py 的自检
退出码：0 无孤峰 / 1 发现孤峰（过拟合风险）/ 2 未就绪
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 单参数扫描表（±50% 及 SPEC 规定的档位）
SWEEPS = {
    "horizon":     [1, 5, 20],
    "decay_rho":   [0.99, 0.995, 0.999],
    "kappa":       [4.0, 8.0, 16.0],
    "eta":         [0.01, 0.02, 0.04],
    "tau":         [1.0, 1.5, 2.5],
    "eps_explore": [0.01, 0.03, 0.1],
}

PEAK_GAIN = 0.30      # 孤峰判据：高于邻居均值 30%
FLAT_TOL = 0.02       # 指标全部 |IC| < 此值 -> "均匀无效"，不是"稳健"


# ------------------------------------------------------------------ 指标
def rank_ic(scores: np.ndarray, rets: np.ndarray, min_n: int = 10):
    """(T,N) x (T,N) -> 逐日横截面 Spearman 的均值/标准差/天数。"""
    T = scores.shape[0]
    out = []
    for t in range(T):
        s = np.asarray(scores[t], dtype=np.float64)
        r = np.asarray(rets[t], dtype=np.float64)
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < min_n:
            continue
        rs = pd.Series(s[m]).rank().to_numpy()
        rr = pd.Series(r[m]).rank().to_numpy()
        if rs.std() < 1e-12 or rr.std() < 1e-12:
            continue
        out.append(float(np.corrcoef(rs, rr)[0, 1]))
    if not out:
        return float("nan"), float("nan"), 0
    a = np.asarray(out)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if a.size > 1 else 0.0), int(a.size)


def hit_rate(scores: np.ndarray, rets: np.ndarray):
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rets, dtype=np.float64)
    m = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > 1e-6) & (np.abs(r) > 1e-9)
    if m.sum() == 0:
        return float("nan")
    return float((np.sign(s[m]) == np.sign(r[m])).mean())


def _scores_TN(res, T, N):
    sc = np.asarray(res["scores"])
    if sc.shape == (T, N):
        return sc
    if sc.shape == (N, T):
        return sc.T
    raise ValueError(f"bad scores shape {sc.shape} for T={T} N={N}")


# ------------------------------------------------------------------ 单次评估
def make_cfg(base_cfg, pname, value):
    """
    构造一个只改 pname 的 RunConfig 副本。

    关键：`horizon` 在 RunConfig 上，而 decay_rho/kappa/eta/tau/eps_explore
    在 WeightConfig 上。若用 setattr 往 RunConfig 上塞 WeightConfig 的字段，
    不会报错但**完全不生效** —— 敏感性分析会得到"所有值 IC 完全相同"的假稳健。
    这里显式分流，并在改不到时报错，绝不静默降级。
    """
    if hasattr(base_cfg, pname) and pname != "weight_cfg":
        return dataclasses.replace(base_cfg, **{pname: value})
    if hasattr(base_cfg.weight_cfg, pname):
        wc = dataclasses.replace(base_cfg.weight_cfg, **{pname: value})
        return dataclasses.replace(base_cfg, weight_cfg=wc)
    raise AttributeError(f"parameter {pname!r} exists on neither RunConfig nor WeightConfig")


def eval_cfg(p3d, cfg, warmup=80):
    from qcore.engine import run_walkforward

    T, N = len(p3d.dates), len(p3d.codes)
    h = cfg.horizon
    if h not in p3d.RET:
        return float("nan"), float("nan"), float("nan"), 0
    RET = np.asarray(p3d.RET[h], dtype=np.float64)
    res = run_walkforward(p3d, cfg)
    sc = _scores_TN(res, T, N).astype(np.float64)
    ic, ic_sd, nd = rank_ic(sc, RET)
    hr = hit_rate(sc, RET)
    return ic, ic_sd, hr, nd


# ------------------------------------------------------------------ 孤峰判定
def detect_peak(values, metrics, param):
    """
    返回 (is_peak, gain, note)。只看内部点；两端点不参与孤峰判定。
    """
    m = np.asarray(metrics, dtype=np.float64)
    ok = np.isfinite(m)
    if ok.sum() < 3:
        return False, float("nan"), "insufficient finite points"
    # 用绝对值（IC 可能为负，"更强"指 |IC| 更大）
    mag = np.abs(m)
    best = int(np.nanargmax(np.where(ok, mag, -np.inf)))
    if best == 0 or best == len(m) - 1:
        return False, float("nan"), "best at boundary (monotone), not an isolated peak"
    left, right = m[best - 1], m[best + 1]
    if not (np.isfinite(left) and np.isfinite(right)):
        return False, float("nan"), "neighbor not finite"
    neigh = 0.5 * (left + right)
    denom = max(abs(neigh), 1e-6)
    gain = (abs(m[best]) - abs(neigh)) / denom
    # 两侧是否彼此接近（真孤峰的签名）
    close_sides = abs(left - right) < 0.5 * max(abs(m[best] - neigh), 1e-9)
    is_peak = (gain > PEAK_GAIN) and close_sides
    note = (f"best={values[best]} |IC|={abs(m[best]):.4f} vs neighbors "
            f"{left:+.4f}/{right:+.4f} (gain {gain:+.0%}"
            f"{', sides close' if close_sides else ', sides differ'})")
    return is_peak, gain, note


# ------------------------------------------------------------------ 合成自检
def _synthetic_panel(T=600, N=12, S=8, seed=7, horizons=(1, 5, 20)):
    from qcore.dataset import Panel3D

    rng = np.random.default_rng(seed)
    rets = 0.0004 + rng.normal(0, 0.017, (T, N))
    close = 12.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.003, (T, N)))
    DIR = rng.integers(-1, 2, size=(T, N, S)).astype(np.int8)
    STR = rng.integers(0, 256, size=(T, N, S)).astype(np.uint8)
    RET = {}
    for h in horizons:
        arr = np.full((T, N), np.nan, dtype=np.float32)
        if T > 1 + h:
            arr[:T - 1 - h] = open_[1 + h:] / open_[1:T - h] - 1.0
        RET[h] = arr
    pct = np.vstack([np.full((1, N), np.nan), close[1:] / close[:-1] - 1.0])
    SIGMA = pd.DataFrame(pct).rolling(20, min_periods=20).std().to_numpy(np.float32)
    return Panel3D(
        dates=np.arange(T).astype("datetime64[D]"),
        codes=[f"S{n:03d}" for n in range(N)],
        sids=[f"syn_{i}" for i in range(S)],
        DIR=DIR, STR=STR,
        OPEN=open_.astype(np.float32), CLOSE=close.astype(np.float32),
        RET=RET, SIGMA=SIGMA,
        REGIME=np.zeros(T, dtype=np.int8),
        VALID=np.ones((T, N), dtype=bool),
        index_close=np.nanmean(close, axis=1).astype(np.float32),
        str_dtype="uint8", meta={"synthetic": True},
    )


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="参数敏感性 + 孤峰过拟合检测")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--params", default=",".join(SWEEPS.keys()))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)

    from qcore.engine import RunConfig

    print("=" * 78)
    print("audit/sensitivity.py -- parameter sensitivity & isolated-peak detection")
    print("=" * 78)

    try:
        if args.synthetic:
            p3d = _synthetic_panel()
            print("mode: SYNTHETIC (self-check only)")
        else:
            from qcore.data import get_universe
            from qcore.dataset import build_dataset
            uni = get_universe(top_n=args.top)
            codes = uni["code"].astype(str).tolist()[:args.top]
            print(f"building dataset: top={args.top} start={args.start} "
                  f"extra_horizons=[1,5,20] ...")
            p3d = build_dataset(codes, start=args.start, horizon=args.horizon,
                                extra_horizons=[1, 5, 20], verbose=False)
            print(f"mode: REAL  T={len(p3d.dates)} N={len(p3d.codes)} "
                  f"S={len(p3d.sids)} {str(p3d.dates[0])[:10]}..{str(p3d.dates[-1])[:10]}")
    except Exception as e:  # noqa: BLE001
        print(f"NOT READY: {type(e).__name__}: {e}")
        return 2

    base_cfg = RunConfig(horizon=args.horizon, warmup=80)
    base_ic, base_sd, base_hr, base_nd = eval_cfg(p3d, base_cfg)
    print("-" * 78)
    print(f"  BASELINE: mean IC={base_ic:+.4f} (sd {base_sd:.4f}, days {base_nd})  "
          f"hit={base_hr:.4f}")
    print()

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    peaks = []
    all_metrics = []

    for pname in params:
        if pname not in SWEEPS:
            print(f"  [skip] unknown param {pname!r}")
            continue
        vals = SWEEPS[pname]
        ics, hrs, sds = [], [], []
        for v in vals:
            cfg = make_cfg(base_cfg, pname, v)
            ic, ic_sd, hr, nd = eval_cfg(p3d, cfg)
            ics.append(ic)
            sds.append(ic_sd)
            hrs.append(hr)
            all_metrics.append(ic)
        is_peak, gain, note = detect_peak(vals, ics, pname)
        if is_peak:
            peaks.append((pname, note))
        print(f"  {pname:12s} " + "  ".join(
            f"{v}:IC={ic:+.4f},hit={hr:.3f}" for v, ic, hr in zip(vals, ics, hrs)))
        flag = " <== ISOLATED PEAK" if is_peak else ""
        # 所有值指标完全相同 -> 参数根本没接上（典型接线 bug），不是"稳健"
        fin = [x for x in ics if np.isfinite(x)]
        if len(fin) >= 2 and (max(fin) - min(fin)) < 1e-12:
            print(f"  {'':12s} -> !! WARNING: metric identical for every value of "
                  f"{pname!r}; this parameter is NOT wired into the run.")
            flag += " <== PARAM NOT EFFECTIVE"
        print(f"  {'':12s} -> {note}{flag}")
        print()

    print("-" * 78)
    finite = [m for m in all_metrics if np.isfinite(m)]
    flat = bool(finite) and max(abs(m) for m in finite) < FLAT_TOL

    if peaks:
        print(f"VERDICT: HIGH RISK -- {len(peaks)} isolated peak(s) detected:")
        for name, note in peaks:
            print(f"    {name}: {note}")
        print("  An isolated peak means the metric collapses at neighbouring values:")
        print("  that parameter was likely fitted to this sample (overfitting).")
        return 1

    if flat:
        print("VERDICT: PASS (no isolated peak) BUT UNINFORMATIVE.")
        print(f"  All |IC| < {FLAT_TOL}: every parameter setting is equally useless.")
        print("  'No peak' here means 'uniformly dead', NOT 'robust'.")
        print("  Sensitivity analysis only certifies robustness when the metric is alive.")
        return 0

    print("VERDICT: PASS -- no isolated peak; metric varies smoothly across each sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
