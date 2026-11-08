#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit/placebo.py -- 安慰剂检验（SPEC §10.5）

要证明的事
----------
"评估/引擎这条链路本身有没有泄漏"。
做法：喂给系统**与未来收益无关**的信号，重跑全流程。若 OOS 指标仍然显著为正，
说明收益不是信号带来的，而是评估代码把未来信息漏进来了。

为什么不能用"逐 sid 固定翻符号"当安慰剂（重要）
------------------------------------------------
若把某个信号的方向**全局**翻过来，学习器会自适应地学出**负权重**，于是 IC 依然为正。
这不是泄漏，是学习器正常工作。用这种 placebo 会得到"安慰剂也有效 → 判定泄漏"的
**假阳性**。所以本文件的安慰剂必须切断"信号 ↔ 未来收益"的统计关联，且不给学习器
留下可利用的稳定结构：

  random_dir  (主)  把 DIR 换成 i.i.d. 随机符号，但**保持每个 sid 的经验边际**
                    （P(+1)/P(0)/P(-1) 与真实数据一致）。关联被彻底切断，
                    学习器无从适应 —— 这是真正的零假设。
  perm_time   (次)  每个 sid 沿时间轴整体循环平移一个随机 offset(>= h+2)。
                    保留发声频率与自相关结构，只错开"信号 -> 收益"的对齐。
  sign_flip_day(次) 逐日随机翻符号。最暴力，同时摧毁持续性。

判据
----
|mean Rank IC| < 0.03 且 命中率落在 0.5±0.02  ->  安慰剂归零，评估链路干净。
否则 -> 高危：评估代码泄漏。

诚实声明（本文件会打印）
------------------------
安慰剂"归零"只在**真实基线本身有效**时才有信息量。若真实 IC 也是 ~0，
那"安慰剂也 ~0"什么都不能证明（可能整套信号根本没 alpha）。所以本文件同时报告
真实基线 IC，并在 |baseline IC| < 0.02 时打 INCONCLUSIVE 标记。

用法
----
    python audit/placebo.py                 # 真实数据，3 种安慰剂 x 3 次重复
    python audit/placebo.py --reps 5
    python audit/placebo.py --top 25        # 用更小的股票池（快）
    python audit/placebo.py --synthetic     # 不依赖 signals.py 的合成自检
退出码：0 通过 / 1 高危(未归零) / 2 未就绪
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

IC_TOL = 0.03          # |mean IC| 容忍上限
HIT_TOL = 0.02         # 命中率偏离 0.5 的容忍
BASELINE_MIN = 0.02    # 真实基线 IC 低于此值 -> 安慰剂结论无信息量


# ------------------------------------------------------------------ 指标
def rank_ic(scores: np.ndarray, rets: np.ndarray, min_n: int = 10):
    """
    scores/rets: (T,N)。逐日横截面 Spearman。返回 (ic_array, n_days)。
    无 scipy：用 pandas 秩 + numpy 相关。
    """
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
    return np.asarray(out, dtype=np.float64)


def hit_rate(scores: np.ndarray, rets: np.ndarray, min_abs: float = 1e-6):
    """sign(score)==sign(ret) 的比例（只统计 |score|>min_abs 且收益有限的格）。"""
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rets, dtype=np.float64)
    m = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > min_abs) & (np.abs(r) > 1e-9)
    if m.sum() == 0:
        return float("nan"), 0
    return float((np.sign(s[m]) == np.sign(r[m])).mean()), int(m.sum())


# ------------------------------------------------------------------ 安慰剂构造
def _empirical_marginal(DIR: np.ndarray):
    """每个 sid 的 (P(+1), P(0), P(-1))。"""
    pos = (DIR > 0).mean(axis=(0, 1))
    neg = (DIR < 0).mean(axis=(0, 1))
    zer = 1.0 - pos - neg
    return pos, zer, neg


def make_placebo_panel(p3d, kind: str, seed: int, horizon: int):
    """返回一个新的 Panel3D，DIR 被替换成与未来收益无关的版本。"""
    rng = np.random.default_rng(seed)
    DIR = np.asarray(p3d.DIR)
    T, N, S = DIR.shape

    if kind == "random_dir":
        pos, zer, neg = _empirical_marginal(DIR)
        u = rng.random((T, N, S))
        new = np.zeros((T, N, S), dtype=np.int8)
        p_pos = pos[None, None, :]
        p_zer = (pos + zer)[None, None, :]
        new[u < p_pos] = 1
        new[(u >= p_pos) & (u < p_zer)] = 0
        new[u >= p_zer] = -1
        # 只在原本可能发声的位置放信号？不：安慰剂就是要完全随机，故保留全图
    elif kind == "perm_time":
        min_off = horizon + 2
        new = np.empty_like(DIR)
        for s in range(S):
            off = int(rng.integers(min_off, max(min_off + 1, T - 1)))
            new[:, :, s] = np.roll(DIR[:, :, s], off, axis=0)
    elif kind == "sign_flip_day":
        sgn = rng.choice(np.array([-1, 1], dtype=np.int8), size=(T, N, S))
        new = (DIR * sgn).astype(np.int8)
    else:
        raise ValueError(f"unknown placebo kind {kind!r}")

    return dataclasses.replace(p3d, DIR=new)


# ------------------------------------------------------------------ 合成自检
def _synthetic_panel(T=600, N=12, S=8, seed=7):
    """不依赖 signals.py 的最小 panel（用于验证本脚本自身能跑通）。"""
    from qcore.dataset import Panel3D

    rng = np.random.default_rng(seed)
    rets = 0.0004 + rng.normal(0, 0.017, (T, N))
    close = 12.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.003, (T, N)))
    DIR = rng.integers(-1, 2, size=(T, N, S)).astype(np.int8)
    STR = rng.integers(0, 256, size=(T, N, S)).astype(np.uint8)
    h, lag = 5, 1
    arr = np.full((T, N), np.nan, dtype=np.float32)
    if T > lag + h:
        arr[:T - lag - h] = open_[lag + h:] / open_[lag:T - h] - 1.0
    pct = np.vstack([np.full((1, N), np.nan), close[1:] / close[:-1] - 1.0])
    SIGMA = pd.DataFrame(pct).rolling(20, min_periods=20).std().to_numpy(np.float32)
    return Panel3D(
        dates=np.arange(T).astype("datetime64[D]"),
        codes=[f"S{n:03d}" for n in range(N)],
        sids=[f"syn_{i}" for i in range(S)],
        DIR=DIR, STR=STR,
        OPEN=open_.astype(np.float32), CLOSE=close.astype(np.float32),
        RET={h: arr}, SIGMA=SIGMA,
        REGIME=np.zeros(T, dtype=np.int8),
        VALID=np.ones((T, N), dtype=bool),
        index_close=np.nanmean(close, axis=1).astype(np.float32),
        str_dtype="uint8", meta={"synthetic": True},
    )


# ------------------------------------------------------------------ 主流程
def _scores_TN(res, T, N):
    sc = np.asarray(res["scores"])
    if sc.shape == (T, N):
        return sc
    if sc.shape == (N, T):
        return sc.T
    raise ValueError(f"bad scores shape {sc.shape} for T={T} N={N}")


def run(p3d, cfg, reps, kinds, verbose=True):
    from qcore.engine import run_walkforward

    T, N = len(p3d.dates), len(p3d.codes)
    h = cfg.horizon
    RET = np.asarray(p3d.RET[h], dtype=np.float64)          # (T,N)

    # ---- 真实基线
    res0 = run_walkforward(p3d, cfg)
    sc0 = _scores_TN(res0, T, N).astype(np.float64)
    ic0 = rank_ic(sc0, RET)
    hr0, n0 = hit_rate(sc0, RET)
    base_ic = float(np.nanmean(ic0)) if ic0.size else float("nan")
    base_hr = hr0
    # 基线 IC 是否显著不为 0：用自身标准误算 t 值，别拿固定阈值糊弄
    if ic0.size > 1:
        se = float(np.nanstd(ic0, ddof=1)) / np.sqrt(ic0.size)
        base_t = base_ic / se if se > 0 else 0.0
    else:
        base_t = 0.0

    if verbose:
        print(f"  baseline (real signals): mean IC={base_ic:+.4f} "
              f"(std {np.nanstd(ic0):.4f}, days {ic0.size}, t={base_t:+.2f})  "
              f"hit={base_hr:.4f} (n={n0})")

    rows = []
    worst_ic = 0.0
    worst_hr_dev = 0.0
    for kind in kinds:
        ics = []
        hrs = []
        for rep in range(reps):
            pp = make_placebo_panel(p3d, kind, seed=1000 + rep, horizon=h)
            res = run_walkforward(pp, cfg)
            sc = _scores_TN(res, T, N).astype(np.float64)
            ic = rank_ic(sc, RET)
            hr, _ = hit_rate(sc, RET)
            if ic.size:
                ics.append(float(np.nanmean(ic)))
            if np.isfinite(hr):
                hrs.append(hr)
        m_ic = float(np.mean(ics)) if ics else float("nan")
        s_ic = float(np.std(ics)) if len(ics) > 1 else 0.0
        m_hr = float(np.mean(hrs)) if hrs else float("nan")
        worst_ic = max(worst_ic, abs(m_ic) if np.isfinite(m_ic) else 0.0)
        worst_hr_dev = max(worst_hr_dev, abs(m_hr - 0.5) if np.isfinite(m_hr) else 0.0)
        rows.append((kind, m_ic, s_ic, m_hr, reps))
        if verbose:
            print(f"  placebo[{kind:14s}] mean IC={m_ic:+.4f} (sd {s_ic:.4f})  "
                  f"hit={m_hr:.4f}  reps={reps}")

    # 只有 random_dir / sign_flip_day 是**干净的零假设**：
    #   perm_time 保留横截面倾斜，会与"个股长期强弱"共振（=生存者偏差），
    #   不能用它判定"评估代码泄漏"，只能作为诊断。
    leak_kinds = {"random_dir", "sign_flip_day"}
    leak_rows = [r for r in rows if r[0] in leak_kinds]
    worst_ic_leak = max((abs(r[1]) for r in leak_rows), default=0.0)
    worst_hr_leak = max((abs(r[3] - 0.5) for r in leak_rows), default=0.0)
    clean = (worst_ic_leak < IC_TOL) and (worst_hr_leak < HIT_TOL)
    # 有信息量 = 基线 IC 既够大、又统计上显著（|t|>=2），否则"安慰剂归零"不说明问题
    informative = (np.isfinite(base_ic) and abs(base_ic) >= BASELINE_MIN
                   and abs(base_t) >= 2.0)
    return {
        "baseline_ic": base_ic, "baseline_hit": base_hr, "baseline_t": base_t,
        "rows": rows, "clean": clean, "informative": informative,
        "worst_ic": worst_ic, "worst_hit_dev": worst_hr_dev,
        "worst_ic_leak": worst_ic_leak, "worst_hr_leak": worst_hr_leak,
        "n_days_ic": int(ic0.size), "T": T, "N": N,
    }


def diagnose_cross_sectional(p3d, cfg, reps=2, verbose=True):
    """
    判定 perm_time 安慰剂的正 IC 从哪来。

    做法：把收益按股票**去均值**（RET[t,n] -= mean_t RET[:,n]），再算 IC。
      - 若去均值后 IC 归零 -> 正 IC 来自"个股长期强弱"（横截面倾斜）
        = 生存者偏差，**不是**评估泄漏。
      - 若去均值后 IC 仍在 -> 说明存在真正的时序对齐泄漏，高危。
    """
    from qcore.engine import run_walkforward

    T, N = len(p3d.dates), len(p3d.codes)
    h = cfg.horizon
    RET = np.asarray(p3d.RET[h], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(RET, axis=0)                 # 每只股票全期均值
    RET_dm = RET - mu[None, :]                       # 横截面去均值

    raw, dm = [], []
    for rep in range(reps):
        pp = make_placebo_panel(p3d, "perm_time", seed=1000 + rep, horizon=h)
        res = run_walkforward(pp, cfg)
        sc = _scores_TN(res, T, N).astype(np.float64)
        i1 = rank_ic(sc, RET)
        i2 = rank_ic(sc, RET_dm)
        if i1.size:
            raw.append(float(np.nanmean(i1)))
        if i2.size:
            dm.append(float(np.nanmean(i2)))
    m_raw = float(np.mean(raw)) if raw else float("nan")
    m_dm = float(np.mean(dm)) if dm else float("nan")
    sd_raw = float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0
    t_raw = (m_raw / (sd_raw / np.sqrt(len(raw)))) if sd_raw > 0 and len(raw) > 1 else 0.0
    if verbose:
        print(f"  DIAG perm_time: IC vs raw returns = {m_raw:+.4f} "
              f"(sd {sd_raw:.4f} over {len(raw)} reps, t={t_raw:+.2f})")
        print(f"                  IC vs cross-sectionally demeaned returns = {m_dm:+.4f}")
        if not np.isfinite(m_raw):
            print("       => no data.")
        elif abs(m_raw) <= IC_TOL:
            print("       => within tolerance; nothing to explain.")
        elif abs(t_raw) < 2.0:
            print("       => NOT statistically significant across reps (|t|<2).")
            print("          The apparent IC is placebo noise, not a stable effect.")
            print("          Increase --reps before drawing any conclusion.")
        elif np.isfinite(m_dm) and abs(m_dm) < IC_TOL:
            print("       => significant, and it vanishes after demeaning returns by stock:")
            print("          driven by PERSISTENT STOCK-LEVEL OUTPERFORMANCE")
            print("          (survivorship bias), NOT by evaluation leakage.")
        else:
            print("       => significant AND survives demeaning: genuine timing")
            print("          leakage in the evaluation path. HIGH RISK.")
    return m_raw, m_dm


def main(argv=None):
    ap = argparse.ArgumentParser(description="安慰剂检验：随机信号必须归零")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true", help="不依赖 signals.py 的自检")
    ap.add_argument("--kinds", default="random_dir,perm_time,sign_flip_day")
    args = ap.parse_args(argv)

    from qcore.engine import RunConfig

    print("=" * 78)
    print("audit/placebo.py -- placebo test (random signals must collapse to ~0)")
    print("=" * 78)

    cfg = RunConfig(horizon=args.horizon, warmup=80)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    try:
        if args.synthetic:
            p3d = _synthetic_panel()
            print("mode: SYNTHETIC (self-check only, not evidence about the real system)")
        else:
            from qcore.data import get_universe
            from qcore.dataset import build_dataset
            uni = get_universe(top_n=args.top)
            codes = uni["code"].astype(str).tolist()[:args.top]
            print(f"building dataset: top={args.top} start={args.start} h={args.horizon} ...")
            p3d = build_dataset(codes, start=args.start, horizon=args.horizon, verbose=False)
            print(f"mode: REAL  T={len(p3d.dates)} N={len(p3d.codes)} S={len(p3d.sids)} "
                  f"{str(p3d.dates[0])[:10]}..{str(p3d.dates[-1])[:10]}")
    except Exception as e:  # noqa: BLE001
        print(f"NOT READY: {type(e).__name__}: {e}")
        return 2

    print("-" * 78)
    out = run(p3d, cfg, args.reps, kinds)
    print("-" * 78)

    print(f"  baseline IC = {out['baseline_ic']:+.4f}   "
          f"baseline hit = {out['baseline_hit']:.4f}   IC days = {out['n_days_ic']}")
    print(f"  LEAK-TEST kinds worst |IC| = {out['worst_ic_leak']:.4f}  (tol {IC_TOL})")
    print(f"  LEAK-TEST kinds worst |hit-0.5| = {out['worst_hr_leak']:.4f}  (tol {HIT_TOL})")
    print(f"  (all kinds incl. diagnostics) worst |IC| = {out['worst_ic']:.4f}")
    print()
    if any(r[0] == "perm_time" for r in out["rows"]):
        diagnose_cross_sectional(p3d, cfg, reps=min(2, args.reps))
        print()

    if not out["clean"]:
        print("VERDICT: HIGH RISK -- placebo did NOT collapse to zero.")
        print("         Random signals should not predict returns. If they do, the")
        print("         evaluation/engine path is leaking future information.")
        return 1

    if not out["informative"]:
        print("VERDICT: PASS but INCONCLUSIVE.")
        print(f"         Placebo collapsed (good), BUT baseline |IC|={abs(out['baseline_ic']):.4f} "
              f"< {BASELINE_MIN} or |t|={abs(out['baseline_t']):.2f} < 2.0.")
        print("         'Placebo ~0' proves nothing when the real signal is also ~0.")
        print("         -> this test cannot distinguish 'no leak' from 'no alpha'.")
        return 0

    print("VERDICT: PASS -- placebo collapsed to ~0 while baseline is non-trivial.")
    print("         Evaluation path shows no sign of leaking future returns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
