#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_no_lookahead.py -- 截断不变性检验（Truncation Invariance）
SPEC §0.0.5 / §10.1 的硬检验：**第一铁律的证伪器**

原理
----
站在 T 日，只保留 date <= T 的行情，重跑全流程。
若 T 日输出（scores / weights / REGIME）与全量运行逐元素不同 -> 系统在 T 日用了未来信息。

关键设计（为什么不用 qcore.dataset.truncate）
----------------------------------------------
`dataset.truncate()` 切的是**已经算好的** RET/SIGMA 数组。RET[t] 在原 panel 里是用
OPEN[t+lag+h] 算的，本身含 T 之后的价格；切片只删行、不重算，于是泄漏值被原样复制过去。
后果：若引擎误写成 `RET[t]`（而非 `RET[t-1-h]`），用 truncate() 做的检验**永远发现不了**。
所以本文件一律走 **"截断原始行情 -> 重新 build -> 重跑"**，让 RET/SIGMA/DIR 全部
从截断后的价格重算。这才是 SPEC 说的"把数据集截断到 T 之后重跑全流程"。

三层检验
--------
L1 合成数据（今天就能跑，不依赖 signals.py）
   用可控的合成价格矩阵 + 一个"信号构造函数"重建 mini-pipeline，验证 engine + learner
   的逐日重放是否因果。这是泄漏最容易发生的地方（引擎循环/权重更新）。
L2 负对照（证明检验本身有效）
   把信号构造函数换成**故意偷看未来**的版本（等价 shift(-1)）。若 L1 的检验抓不出来，
   说明检验是摆设。必须 FAIL 才算"检验有效"。
L3 真实数据（signals.py 就绪后自动启用）
   打补丁把 qcore.dataset 的 build_panel/load_index 换成截断版，重跑 build_dataset。

用法
----
    python tests/test_no_lookahead.py          # 全部
    python tests/test_no_lookahead.py --l1     # 只跑合成
    python tests/test_no_lookahead.py -v       # 打印每项数值
退出码：0 全通过 / 1 有 FAIL / 2 全部 SKIP（环境未就绪）
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ATOL = 1e-12
# 注：engine 内部用 float32 存 scores/weights，float32 的相对精度约 1.2e-7。
# atol=1e-12 对 float32 而言等于"要求逐位相同"——正是我们要的严格度：
# 任何真实的重算差异都会远大于 1e-12；而同一份数据、同一顺序的运算必然逐位相同。

# ------------------------------------------------------------------ 可选依赖
HAVE_ENGINE = False
IMPORT_ERR = ""
try:
    from qcore.dataset import Panel3D  # noqa: F401
    from qcore.engine import RunConfig, run_walkforward

    HAVE_ENGINE = True
except Exception as e:  # noqa: BLE001
    IMPORT_ERR = f"{type(e).__name__}: {e}"

HAVE_SIGNALS = os.path.exists(os.path.join(ROOT, "qcore", "signals.py"))


# ================================================================== 工具
def _dir_of(shape, T, N, S):
    """判定 DIR/STR 的轴顺序，返回 'TNS' 或 'NTS'。"""
    if shape == (T, N, S):
        return "TNS"
    if shape == (N, T, S):
        return "NTS"
    raise ValueError(f"cannot infer axis order from shape {shape} vs (T,N,S)=({T},{N},{S})")


def scores_at(res, T, N, k):
    """取第 k 个交易日（0-based, T 轴）的横截面分数 (N,)。自动识别 (T,N) / (N,T)。"""
    sc = np.asarray(res["scores"])
    if sc.shape == (T, N):
        return sc[k, :]
    if sc.shape == (N, T):
        return sc[:, k]
    raise ValueError(f"scores shape {sc.shape} matches neither (T,N)=({T},{N}) nor (N,T)=({N},{T})")


def _maxdiff(a, b):
    """逐元素最大绝对差；NaN 位置不一致也算违规（返回 inf）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf"), "shape"
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return float("inf"), "nan-mask"
    if not fa.any():
        return 0.0, "all-nan"
    d = np.abs(a[fa] - b[fb])
    return float(d.max()) if d.size else 0.0, "ok"


# ================================================================== L1 合成 pipeline
def _synth_market(T, N, seed=7):
    """造一段有牛/熊/震荡结构的合成行情（正价格，对数随机游走）。"""
    rng = np.random.default_rng(seed)
    seg = T // 3
    drift = np.concatenate([
        np.full(seg, +0.0014),                       # 牛
        np.full(seg, -0.0016),                       # 熊
        np.full(T - 2 * seg, +0.00005),              # 震荡
    ])
    rets = drift[:, None] + rng.normal(0.0, 0.017, size=(T, N))
    close = 12.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0.0, 0.003, size=(T, N)))
    return open_.astype(np.float64), close.astype(np.float64)


def _sig_momentum(CLOSE, lag):
    """因果信号：lag 日动量方向。"""
    T, N = CLOSE.shape
    d = np.zeros((T, N), dtype=np.float64)
    s = np.zeros((T, N), dtype=np.float64)
    if T > lag:
        m = CLOSE[lag:] / CLOSE[:-lag] - 1.0
        d[lag:] = np.sign(m)
        s[lag:] = np.clip(np.abs(m) * 25.0, 0.0, 1.0)
    return d, s


def _sig_peek_next_close(CLOSE):
    """★ 故意泄漏 ★ 等价于 shift(-1)：用 t+1 的收盘价决定 t 的方向。

    这是最经典的未来函数。截断到 T 后，t=T-1 处没有 t+1 -> 方向归 0，
    因此**若检验有效，截断后 T 日分数必然变化**。
    """
    T, N = CLOSE.shape
    d = np.zeros((T, N), dtype=np.float64)
    s = np.zeros((T, N), dtype=np.float64)
    if T > 1:
        m = CLOSE[1:] / CLOSE[:-1] - 1.0      # m[i] 对应 t=i，用了 CLOSE[i+1]
        d[:-1] = np.sign(m)
        s[:-1] = np.clip(np.abs(m) * 25.0, 0.0, 1.0)
    return d, s


def _build_mini_panel(OPEN, CLOSE, test_builder, S_extra=5):
    """
    用给定的价格矩阵重建一个 Panel3D（相当于 mini 版 build_dataset）。
    **所有派生量都从传入的 OPEN/CLOSE 现算**，所以截断输入 = 真正的点对点重建。
    """
    T, N = CLOSE.shape

    # --- 信号矩阵：第 0 列是"被测信号"，其余是因果动量（保证截断一致）
    sids = ["test_sig"] + [f"mom_{k}" for k in (1, 2, 3, 5, 8, 13, 21)[:S_extra]]
    S = len(sids)
    DIR = np.zeros((T, N, S), dtype=np.float64)
    STR = np.zeros((T, N, S), dtype=np.float64)

    d0, s0 = test_builder(CLOSE)
    DIR[:, :, 0] = d0
    STR[:, :, 0] = s0
    for j, k in enumerate((1, 2, 3, 5, 8, 13, 21)[:S_extra], start=1):
        dj, sj = _sig_momentum(CLOSE, k)
        DIR[:, :, j] = dj
        STR[:, :, j] = sj

    # --- 收益矩阵 r_h(t) = OPEN[t+lag+h]/OPEN[t+lag]-1（越界 NaN）
    lag, h = 1, 5
    RET = {}
    arr = np.full((T, N), np.nan, dtype=np.float32)
    if T > lag + h:
        arr[:T - lag - h] = OPEN[lag + h:] / OPEN[lag:T - h] - 1.0
    RET[h] = arr

    # --- 条件波动率（因果 rolling 20）
    pct = np.vstack([np.full((1, N), np.nan), CLOSE[1:] / CLOSE[:-1] - 1.0])
    SIGMA = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(19, T):
        SIGMA[t] = np.nanstd(pct[t - 19:t + 1], axis=0, ddof=1)

    # --- regime（因果：60 日均线方向 + 20 日波动分位）
    ew = np.nanmean(CLOSE, axis=1)
    ma = np.full(T, np.nan)
    for t in range(59, T):
        ma[t] = ew[t - 59:t + 1].mean()
    up = (ew > ma)
    vol = SIGMA.mean(axis=1)
    REGIME = np.zeros(T, dtype=np.int8)
    for t in range(T):
        hi = 0
        if t >= 20 and np.isfinite(vol[t]):
            past = vol[max(0, t - 249):t + 1]
            past = past[np.isfinite(past)]
            hi = int(len(past) > 5 and vol[t] > np.median(past))
        REGIME[t] = int(2 * int(bool(up[t]) and np.isfinite(ma[t])) + hi)

    VALID = np.isfinite(OPEN) & np.isfinite(CLOSE) & (CLOSE > 0)

    return Panel3D(
        dates=np.arange(T).astype("datetime64[D]"),
        codes=[f"S{n:03d}" for n in range(N)],
        sids=sids,
        DIR=DIR.astype(np.int8),
        STR=(np.clip(STR, 0, 1) * 255).astype(np.uint8),
        OPEN=OPEN.astype(np.float32),
        CLOSE=CLOSE.astype(np.float32),
        RET=RET,
        SIGMA=SIGMA,
        REGIME=REGIME,
        VALID=VALID,
        index_close=ew.astype(np.float32),
        str_dtype="uint8",
        meta={"synthetic": True},
    )


def _pick_three_dates(CLOSE, warmup, T, horizon=5):
    """在牛/熊/震荡三段各挑一个日期（确定性，不靠随机种子）。"""
    ew = np.nanmean(CLOSE, axis=1)
    lo = max(warmup + 30, 70)
    hi = T - 2 - horizon          # 留出收益窗口
    if hi <= lo:
        raise RuntimeError(f"series too short: lo={lo} hi={hi}")
    win = 60
    fwd = np.full(T, np.nan)
    for t in range(lo, hi + 1):
        if t - win >= 0:
            fwd[t] = ew[t] / ew[t - win] - 1.0
    idx = [t for t in range(lo, hi + 1) if np.isfinite(fwd[t])]
    arr = np.array(idx)
    bull = int(arr[np.argmax(fwd[arr])])
    bear = int(arr[np.argmin(fwd[arr])])
    chop = int(arr[np.argmin(np.abs(fwd[arr]))])
    # 去重（极端情况下可能撞车）
    out = []
    for x in (bull, bear, chop):
        if x not in out:
            out.append(x)
    while len(out) < 3:
        cand = int(arr[len(arr) // 2 + len(out)])
        if cand not in out:
            out.append(cand)
        else:
            break
    return out, {"bull": bull, "bear": bear, "chop": chop}


def _compare_runs(res_full, res_tr, T_full, N_full, k, N_tr=None, idx_f=None, idx_t=None):
    """
    比较全量运行的第 k 日 vs 截断运行的最后一日。
    idx_f/idx_t: 按**代码**对齐的列索引（截断会因 min_bars 少掉若干股票，
                 绝不能按列号硬比 —— 那会把 A 股票的分跟 B 股票的分比）。
    """
    out = {}
    if N_tr is None:
        N_tr = N_full
    a = scores_at(res_full, T_full, N_full, k)
    b = scores_at(res_tr, k + 1, N_tr, k)
    if idx_f is not None:
        a = a[idx_f]
        b = b[idx_t]
    out["scores"] = _maxdiff(a, b)
    # 2) weights_hist（S 维一致，无需对齐）
    wh_f = np.asarray(res_full["weights_hist"])
    wh_t = np.asarray(res_tr["weights_hist"])
    out["weights_hist"] = _maxdiff(wh_f[k], wh_t[-1])
    # 3) REGIME
    out["REGIME"] = _maxdiff(np.asarray(res_full["REGIME"])[k], np.asarray(res_tr["REGIME"])[-1])
    # 4) 附加：前一日也要一致（抓"只错一天"的边界 bug）
    if k >= 1:
        a1 = scores_at(res_full, T_full, N_full, k - 1)
        b1 = scores_at(res_tr, k + 1, N_tr, k - 1)
        if idx_f is not None:
            a1 = a1[idx_f]
            b1 = b1[idx_t]
        out["scores[t-1]"] = _maxdiff(a1, b1)
    return out


def run_l1_synthetic(verbose=False):
    """L1：合成数据截断不变性（诚实构造函数 -> 必须全 0）"""
    T, N = 700, 12
    OPEN, CLOSE = _synth_market(T, N, seed=7)
    cfg = RunConfig(horizon=5, warmup=80)
    dates, tags = _pick_three_dates(CLOSE, cfg.warmup, T, cfg.horizon)

    p_full = _build_mini_panel(OPEN, CLOSE, _sig_momentum_wrap)
    res_full = run_walkforward(p_full, cfg)

    rows = []
    ok = True
    for k in dates:
        p_tr = _build_mini_panel(OPEN[:k + 1], CLOSE[:k + 1], _sig_momentum_wrap)
        res_tr = run_walkforward(p_tr, cfg)
        diffs = _compare_runs(res_full, res_tr, T, N, k)
        worst = max(v[0] for v in diffs.values())
        good = worst <= ATOL
        ok &= good
        rows.append((k, worst, good, diffs))

    if verbose:
        print("  L1 synthetic truncation (honest builder):")
        for k, worst, good, diffs in rows:
            print(f"    T={k:4d} worst={worst:.3e} {'OK' if good else 'LEAK'}  " +
                  " ".join(f"{n}={v[0]:.1e}" for n, v in diffs.items()))
    return ok, rows


def _sig_momentum_wrap(CLOSE):
    """L1 的被测信号：纯因果动量。"""
    return _sig_momentum(CLOSE, 10)


def run_l2_negative_control(verbose=False):
    """
    L2：负对照 —— 把被测信号换成 shift(-1) 偷看未来。
    检验**必须**报出违规（worst > ATOL），否则检验无效。
    返回 (detected, rows)
    """
    T, N = 700, 12
    OPEN, CLOSE = _synth_market(T, N, seed=7)
    cfg = RunConfig(horizon=5, warmup=80)
    dates, tags = _pick_three_dates(CLOSE, cfg.warmup, T, cfg.horizon)

    p_full = _build_mini_panel(OPEN, CLOSE, _sig_peek_next_close)
    res_full = run_walkforward(p_full, cfg)

    rows = []
    detected_any = False
    for k in dates:
        p_tr = _build_mini_panel(OPEN[:k + 1], CLOSE[:k + 1], _sig_peek_next_close)
        res_tr = run_walkforward(p_tr, cfg)
        diffs = _compare_runs(res_full, res_tr, T, N, k)
        worst = max(v[0] for v in diffs.values())
        hit = worst > ATOL
        detected_any |= hit
        rows.append((k, worst, hit, diffs))

    if verbose:
        print("  L2 negative control (shift(-1) leak injected) -- MUST be detected:")
        for k, worst, hit, diffs in rows:
            print(f"    T={k:4d} worst={worst:.3e} {'DETECTED' if hit else 'MISSED!!'}  " +
                  " ".join(f"{n}={v[0]:.1e}" for n, v in diffs.items()))
    return detected_any, rows


# ================================================================== L3 真实数据
def _truncated_build(codes, start, horizon, as_of, verbose=False):
    """
    截断原始行情后重建数据集：把 qcore.dataset 里的 build_panel / load_index
    换成"只保留 date <= as_of"的版本，然后调用 build_dataset。
    这样 RET/SIGMA/DIR/REGIME 全部从截断后的数据重算 —— 真正的点对点重建。
    """
    import qcore.dataset as DS

    orig_bp = DS.build_panel
    orig_li = DS.load_index

    def bp_trunc(*a, **kw):
        panel = orig_bp(*a, **kw)
        out = {}
        for c, df in panel.items():
            d = df[df["date"] <= as_of]
            if len(d) > 0:
                out[c] = d.reset_index(drop=True)
        return out

    def li_trunc(*a, **kw):
        ix = orig_li(*a, **kw)
        return ix[ix["date"] <= as_of].reset_index(drop=True)

    DS.build_panel = bp_trunc
    DS.load_index = li_trunc
    try:
        return DS.build_dataset(codes, start=start, horizon=horizon, verbose=verbose)
    finally:
        DS.build_panel = orig_bp
        DS.load_index = orig_li


def run_l3_real(verbose=False):
    """L3：真实数据截断不变性。signals.py 未就绪 -> SKIP。"""
    if not HAVE_SIGNALS:
        return None, "SKIP: qcore/signals.py not found"
    try:
        import pandas as pd
        from qcore.data import get_universe
        from qcore.dataset import build_dataset

        uni = get_universe(top_n=40)
        codes = uni["code"].astype(str).tolist()[:40]
        start = "2015-01-01"
        h = 5
        p3d = build_dataset(codes, start=start, horizon=h, verbose=False)
        cfg = RunConfig(horizon=h, warmup=80)

        T = len(p3d.dates)
        N = len(p3d.codes)
        if verbose:
            print(f"  L3 full dataset: T={T} N={N} S={len(p3d.sids)} "
                  f"{str(p3d.dates[0])[:10]}..{str(p3d.dates[-1])[:10]}")

        # 候选日期：要求当日股票覆盖度足够，否则截断后股票太少、检验变弱
        cov = np.asarray(p3d.VALID, dtype=np.float64).sum(axis=1) / max(N, 1)
        lo = max(cfg.warmup + 30, 70)
        hi = T - 2 - h
        if hi <= lo:
            return None, f"SKIP: series too short (T={T}, need > {lo + h + 2})"
        ix = np.asarray(p3d.index_close, dtype=np.float64)
        fwd = np.full(T, np.nan)
        for t in range(lo, hi + 1):
            if t - 60 >= 0 and np.isfinite(ix[t - 60]) and ix[t - 60] > 0:
                fwd[t] = ix[t] / ix[t - 60] - 1.0
        thr = 0.6 * float(np.nanmax(cov))
        cand = [t for t in range(lo, hi + 1) if np.isfinite(fwd[t]) and cov[t] >= thr]
        if len(cand) < 3:
            return None, f"SKIP: only {len(cand)} candidate dates with coverage>={thr:.0%}"
        arr = np.array(cand)
        picks = [int(arr[np.argmax(fwd[arr])]), int(arr[np.argmin(fwd[arr])]),
                 int(arr[np.argmin(np.abs(fwd[arr]))])]
        for t in arr:                      # 不足 3 个时补
            if len(set(picks)) >= 3:
                break
            if int(t) not in picks:
                picks.append(int(t))
        picks = sorted(set(picks))[:3]

        res_full = run_walkforward(p3d, cfg)
        full_set = set(p3d.codes)

        rows = []
        ok = True
        for k in picks:
            as_of = pd.Timestamp(p3d.dates[k])
            p_tr = _truncated_build(codes, start, h, as_of)
            res_tr = run_walkforward(p_tr, cfg)
            # 按代码对齐：截断后股票池会因 min_bars 收缩，必须按 code 匹配，
            # 不能按列号硬比（否则等于拿 A 股的分去比 B 股的分）
            common = [c for c in p_tr.codes if c in full_set]
            idx_f = [p3d.codes.index(c) for c in common]
            idx_t = [p_tr.codes.index(c) for c in common]
            if len(common) < 3:
                rows.append((str(as_of.date()), float("inf"), False,
                             {"align": (float("inf"), f"only {len(common)} common codes")}))
                ok = False
                continue
            diffs = _compare_runs(res_full, res_tr, T, N, k,
                                  N_tr=len(p_tr.codes), idx_f=idx_f, idx_t=idx_t)
            worst = max(v[0] for v in diffs.values())
            good = worst <= ATOL
            ok &= good
            rows.append((str(as_of.date()), worst, good, diffs))
            if verbose:
                print(f"    T={as_of.date()} cov={cov[k]:.2f} "
                      f"common={len(common)}/{N} worst={worst:.3e} "
                      f"{'OK' if good else 'LEAK'}")
        if verbose:
            print("  L3 real-data truncation summary:")
            for d, worst, good, diffs in rows:
                print(f"    T={d} worst={worst:.3e} {'OK' if good else 'LEAK'}  " +
                      " ".join(f"{n}={v[0]:.1e}" for n, v in diffs.items()))
        return ok, rows
    except Exception as e:  # noqa: BLE001
        return None, f"ERROR: {type(e).__name__}: {e}"


# ================================================================== L4 数据口径守卫
def run_l4_data_sanity(verbose=False):
    """
    后复权口径守卫（工程日志 docs/03_limitations.md 的可执行版本）。
    前复权(fqt=1)是减法复权，会把高分红股的历史价格压到 0 甚至负数，
    从而制造 |涨跌幅| > 21% 的假 bar。A 股涨跌停 20%（ST 5%），>21% 必为数据错误。

    例外（合法的大涨跌幅，不算污染）：
      - 上市首日：主板首日无涨跌幅限制（实际常见 +44%），科创板/创业板首日无限制
      - 科创板(688)/创业板(300,301)：**上市后前 5 个交易日无涨跌幅限制**
      - 股改 / 重大资产重组 **复牌首日**：无涨跌幅限制（2005-2007 常见，如 000858 +25.6%）
    所以每只股票跳过最初 5 根 bar 再判断，避免把 IPO 行情误判为数据缺陷。

    阈值依据（实测，不是拍脑袋）：
      - 前复权 bug 的签名是**量级性**错误：601088 出现过 +17400%、-91%、-68%，
        以及 close 被压到 0.01（甚至 -9.55）。所以 nonpos_px / close<0.2 是硬失败线。
      - 21%~100% 且**前面有停牌缺口**的，是股改/借壳/重组复牌（如 000858 +25.6%，
        000776 停牌 1212 天后 +394%）。属"合法但不是经济收益"的结构性跳变 -> WARN。
      - 21%~100% 且**前面没有停牌缺口**的，在连续交易日里不可能发生 -> FAIL（数据错误）。
    所以本检查按"是否能用停牌缺口解释"来分流，而不是一刀切。
    """
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
    if not files:      # CI 上没有缓存 -> 用仓库内置夹具
        files = sorted(glob.glob(os.path.join(ROOT, "tests", "fixtures", "*_h2.csv")))
    if not files:
        return None, "SKIP: no *_h2.csv (hfq) cache yet"
    import pandas as pd

    IPO_EXEMPT = 5          # 上市后前 5 个交易日不设涨跌幅限制（科创板/创业板）
    WARN = 21.0             # >21% 已超出 A 股任何常规涨跌停
    GAP_DAYS = 15           # 停牌缺口 >15 自然日 -> 复牌首日无涨跌停限制
    LIMIT_START = "1996-12-16"   # A 股涨跌停制度实施日；此前完全无涨跌幅限制
    SAMPLE_START = "2013-01-01"  # 本项目回测窗口起点（build_dataset 默认 start）
    bad_ret = 0             # 结构性跳变（有停牌解释）
    unexplained = 0         # 回测窗口内、无解释的异常涨跌幅 -> 硬失败
    unexplained_pre = 0     # 窗口外、无解释 -> 仅告警
    bad_px = 0
    bad_px_pre = 0
    near_zero = 0
    near_zero_pre = 0
    total = 0
    worst = []
    lim = pd.Timestamp(LIMIT_START)
    smp = pd.Timestamp(SAMPLE_START)
    for f in files:
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:  # noqa: BLE001
            continue
        d = d[(d["close"] > 0) & (d["volume"] > 0)].reset_index(drop=True)
        if not len(d):
            continue
        total += len(d)
        # 价格有效性：按回测窗口分流。
        #   窗口内（>=2013）出现非正/近零价格 = 硬缺陷，直接 FAIL；
        #   窗口外（1994-2012）只告警 —— 实测 354 根非正 bar 全部来自 5 只股票
        #   的 1994-2006 段（601607/000408/000425/600674/600673），
        #   说明**后复权也会越零**（缩股/重组类），并非只有前复权会。
        #   build_panel 有 close>0 过滤，这些 bar 不会进入回测，但必须在报告里留痕。
        neg_px = (d[["open", "high", "low", "close"]] <= 0).any(axis=1)
        in_w = d["date"] >= smp
        bad_px += int((neg_px & in_w).sum())
        bad_px_pre += int((neg_px & ~in_w).sum())
        nz = (d["close"] > 0) & (d["close"] < 0.2)
        near_zero += int((nz & in_w).sum())
        near_zero_pre += int((nz & ~in_w).sum())
        # 涨跌幅有效性：跳过上市初期无涨跌停窗口
        gap = d["date"].diff().dt.days
        dd = d.iloc[IPO_EXEMPT:]
        gg = gap.iloc[IPO_EXEMPT:]
        if not len(dd):
            continue
        pc = dd["pct_chg"].abs()
        # 1996-12-16 之前没有涨跌停制度，任何幅度都合法 -> 不参与判定
        in_limit_era = dd["date"] >= lim
        big = (pc > WARN) & in_limit_era
        if not big.any():
            continue
        structural = big & (gg > GAP_DAYS)
        bad_ret += int(structural.sum())
        unexp = big & ~(gg > GAP_DAYS)
        in_win = unexp & (dd["date"] >= smp)
        pre_win = unexp & ~(dd["date"] >= smp)
        unexplained += int(in_win.sum())
        unexplained_pre += int(pre_win.sum())
        worst.append((os.path.basename(f), int(big.sum()), int(unexp.sum()), float(pc.max())))
        for idx in dd.index[in_win][:3]:
            print(f"      !! UNEXPLAINED-IN-WINDOW {os.path.basename(f)} "
                  f"{str(d['date'][idx])[:10]} pct={d['pct_chg'][idx]:.1f}% (no suspension gap)")

    ratio = (bad_ret / total) if total else 0.0
    ok = (unexplained == 0 and bad_px == 0 and near_zero == 0)
    if verbose:
        print(f"  L4 hfq data sanity: files={len(files)} bars={total}")
        print(f"      hard defects: nonpos_px={bad_px} close<0.2={near_zero} "
              f"unexplained_in_window={unexplained}")
        print(f"      pre-window WARN only (before {SAMPLE_START}): "
              f"unexplained_ret={unexplained_pre} nonpos_px={bad_px_pre} "
              f"close<0.2={near_zero_pre}")
        print(f"      structural jumps (suspension/resumption, ratio {ratio:.1e}): {bad_ret}")
        for name, n, nu, mx in sorted(worst, key=lambda x: -x[3])[:5]:
            print(f"      WARN {name}: {n} bar(s) |pct|>21%, max={mx:.1f}%, "
                  f"unexplained={nu}")
        print(f"      NOTE: structural jumps are NOT economic returns -- they must be")
        print(f"            masked out of RET[h] (see docs/03_limitations.md).")
    return ok, {"files": len(files), "bars": total, "bad_ret": bad_ret,
                "unexplained": unexplained, "unexplained_pre": unexplained_pre,
                "bad_px": bad_px, "near_zero": near_zero, "ratio": ratio,
                "bad_px_pre": bad_px_pre, "near_zero_pre": near_zero_pre}


# ================================================================== 主程序
def main(argv=None):
    ap = argparse.ArgumentParser(description="截断不变性检验（第一铁律证伪器）")
    ap.add_argument("--l1", action="store_true", help="只跑合成数据检验")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("tests/test_no_lookahead.py -- Truncation Invariance (SPEC 0.0.5)")
    print("=" * 78)
    if not HAVE_ENGINE:
        print(f"qcore.engine/dataset unavailable -> cannot run.\n  {IMPORT_ERR}")
        return 2
    print(f"engine: OK   signals.py: {'OK' if HAVE_SIGNALS else 'MISSING (L3 will SKIP)'}")
    print("-" * 78)

    results = []

    ok1, rows1 = run_l1_synthetic(args.verbose)
    results.append(("L1 synthetic truncation (honest)", ok1,
                    f"{len(rows1)} dates, worst={max(r[1] for r in rows1):.2e}"))

    ok2, rows2 = run_l2_negative_control(args.verbose)
    results.append(("L2 negative control (shift(-1) MUST be caught)", ok2,
                    f"{len(rows2)} dates, max={max(r[1] for r in rows2):.2e}"))

    if args.l1:
        ok3, msg3 = None, "SKIP (--l1)"
    else:
        ok3, msg3 = run_l3_real(args.verbose)
    results.append(("L3 real-data truncation", ok3,
                    msg3 if isinstance(msg3, str) else
                    f"{len(msg3)} dates, worst={max(r[1] for r in msg3):.2e}"))

    ok4, msg4 = run_l4_data_sanity(args.verbose)
    results.append(("L4 hfq data sanity (no qfq-style fake returns)", ok4,
                    msg4 if isinstance(msg4, str) else
                    f"files={msg4['files']} bars={msg4['bars']} "
                    f"nonpos_px={msg4['bad_px']} close<0.2={msg4['near_zero']} "
                    f"unexplained_ret={msg4['unexplained']} "
                    f"structural_jumps={msg4['bad_ret']}"))

    print()
    n_fail = 0
    n_skip = 0
    for name, ok, detail in results:
        if ok is None:
            status = "SKIP"
            n_skip += 1
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            n_fail += 1
        print(f"  [{status:4s}] {name}")
        print(f"         {detail}")

    print("-" * 78)
    if n_fail:
        print(f"VERDICT: FAIL ({n_fail} check(s)) -- look-ahead or data defect present.")
        return 1
    if n_skip == len(results):
        print("VERDICT: SKIP -- environment not ready.")
        return 2
    print(f"VERDICT: PASS ({len(results) - n_skip} passed, {n_skip} skipped).")
    return 0


# ---- pytest 兼容（有 pytest 时会收集；没有也能 python 直接跑）----
def test_l1_truncation_invariance():
    if not HAVE_ENGINE:
        import pytest
        pytest.skip(IMPORT_ERR)
    ok, rows = run_l1_synthetic()
    assert ok, [(k, w) for k, w, g, _ in rows if not g]


def test_l2_negative_control_has_power():
    if not HAVE_ENGINE:
        import pytest
        pytest.skip(IMPORT_ERR)
    ok, rows = run_l2_negative_control()
    assert ok, "truncation test FAILED to detect an injected shift(-1) leak -- test is useless"


def test_l4_data_sanity():
    ok, info = run_l4_data_sanity()
    if ok is None:
        import pytest
        pytest.skip(str(info))
    assert ok, f"data sanity violated: {info}"


if __name__ == "__main__":
    sys.exit(main())
