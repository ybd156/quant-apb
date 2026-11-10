"""tests/test_signals.py —— 信号库验收（可直接 `python -m tests.test_signals` 运行）

覆盖：
  1. 真实数据跑通：取值域 / 无 NaN / 列数 / dtype
  2. 截断一致性（无未来函数的硬检验）：截断到 t 重算，第 t 行必须与全量逐元素相同
  3. 增量模式 tail=k 一致性（k=1/50/300）
  4. 每族至少 1 个非零触发（防死信号）
  5. 耗时（3000 根 < 40ms）与峰值内存（tracemalloc 打印）
  6. 健壮性：空表 / 缺列 / 全 NaN 列 / 常数价格 均不抛异常
  7. 批量吞吐（20 只外推 300 只）
"""
from __future__ import annotations

import glob
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

from qcore.signals import SignalSpec, compute_signals, signal_zoo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KLINE_DIR = os.path.join(ROOT, "data", "raw", "kline")
# 离线夹具：CI 上没有行情缓存时用它（真实数据的截断副本，随仓库提交）
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures")
ATOL = 1e-12

_PASS = 0
_FAIL = 0
_FAILS = []


def _ok(name: str, detail: str = "") -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))


def _bad(name: str, detail: str = "") -> None:
    global _FAIL
    _FAIL += 1
    _FAILS.append(f"{name} :: {detail}")
    print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


#: 依赖**外部数据**（资金流 / 大宗交易 / 龙虎榜）的信号族。
#: 这些族在没有喂数据时必须输出 0（预热），不算「死族」。
EXT_FAMILIES = ("flow", "block_trade", "billboard", "margin", "sentiment", "event",
                "holder", "lift", "forecast", "breadth")
EXT_PREFIX = ("flow", "bt", "bb", "mg", "snt", "evt", "hold", "lift", "fcst", "brd")


def check(name: str, fn) -> None:
    """跑一个断言块，异常即记失败（不中断其余用例）。"""
    try:
        detail = fn()
        _ok(name, "" if detail is None else str(detail))
    except AssertionError as e:
        _bad(name, str(e) or "assertion failed")
    except Exception as e:      # noqa: BLE001
        _bad(name, f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# 数据加载（只读本地缓存，绝不联网）
# ----------------------------------------------------------------------
def find_kline(code: str = "600519") -> str:
    """优先本地缓存，其次仓库内置夹具（保证 CI 可离线跑）。"""
    for base in (KLINE_DIR, FIXTURE_DIR):
        for pat in (f"{code}_h2.csv", f"{code}.csv"):
            p = os.path.join(base, pat)
            if os.path.exists(p):
                return p
        hits = sorted(glob.glob(os.path.join(base, f"{code}*.csv")))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no kline for {code} in {KLINE_DIR} or {FIXTURE_DIR}")


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def list_codes(limit: int = 20):
    paths = sorted(glob.glob(os.path.join(KLINE_DIR, "*_h2.csv")))[:limit]
    if not paths:
        paths = sorted(glob.glob(os.path.join(KLINE_DIR, "*.csv")))[:limit]
    return paths


# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("tests/test_signals.py :: qcore/signals.py acceptance")
    print("=" * 78)

    zoo = signal_zoo()
    sids = [s.sid for s in zoo]
    path = find_kline("600519")
    USING_FIXTURE = ("fixtures" in path.replace("\\", "/"))
    df = load_df(path)
    print(f"data  : {os.path.basename(path)}  bars={len(df)}  "
          f"{df['date'].iloc[0].date()} .. {df['date'].iloc[-1].date()}")
    print(f"zoo   : {len(zoo)} specs / {len(set(s.sid for s in zoo))} unique sids / "
          f"{len(set(s.family for s in zoo))} families")

    # ---------------- 1. 契约与取值域 ----------------
    print("\n[1] contract on real data")
    DIR, STR = compute_signals(df)

    def t_shape():
        assert isinstance(DIR, pd.DataFrame) and isinstance(STR, pd.DataFrame)
        assert DIR.shape == STR.shape == (len(df), len(zoo)), (DIR.shape, STR.shape)
        assert list(DIR.columns) == sids, "column order must equal signal_zoo()"
        assert DIR.index.equals(pd.DatetimeIndex(df["date"])), "index must be date"
        return f"shape={DIR.shape}"
    check("DIR/STR shape & columns == signal_zoo()", t_shape)

    def t_domain():
        d = DIR.to_numpy()
        assert np.isfinite(d).all(), "DIR contains NaN/inf"
        uniq = np.unique(d)
        assert set(np.round(uniq, 6)).issubset({-1.0, 0.0, 1.0}), f"DIR domain {uniq[:9]}"
        s = STR.to_numpy()
        assert np.isfinite(s).all(), "STR contains NaN/inf"
        assert s.min() >= 0.0 and s.max() <= 1.0, (s.min(), s.max())
        return f"DIR in {uniq.tolist()}  STR in [{s.min():.3f},{s.max():.3f}]"
    check("DIR subset of {-1,0,1} & STR subset of [0,1] & no NaN", t_domain)

    def t_dtype():
        assert DIR.dtypes.eq(np.float32).all(), DIR.dtypes.unique()
        assert STR.dtypes.eq(np.float32).all(), STR.dtypes.unique()
        a = DIR.to_numpy(dtype=np.float32)
        b = STR.to_numpy(dtype=np.float32)
        assert a.dtype == b.dtype == np.float32
        return f"dtype={a.dtype}, to_numpy(float32) ok, nbytes={a.nbytes + b.nbytes}"
    check("dtype float32 & to_numpy(float32) friendly", t_dtype)

    def t_spec():
        assert all(isinstance(s, SignalSpec) for s in zoo)
        # 上限随外部数据族扩展而上调（73 手写 + 20 外部数据 = 93）
        assert 50 <= len(zoo) <= 120, len(zoo)
        assert all(len(s.sid) > 3 and s.note for s in zoo)
        fam = {}
        for s in zoo:
            fam[s.family] = fam.get(s.family, 0) + 1
        # 16-24 族：9 个**外部数据族**（资金流/大宗/龙虎榜/融资融券/散户情绪/
        # 政策事件/股东增减持/限售解禁/业绩预告）。没有数据时输出 0 是正确行为（预热），
        # 由 [4b] 专测覆盖「喂数据必须发声」。
        assert len(fam) == 25, f"expected 25 families, got {len(fam)}: {fam}"
        assert all(v >= 2 for v in fam.values()), fam
        return f"{len(fam)} families, min variants={min(fam.values())}"
    check("SignalSpec dataclass / 25 families / >=2 variants each", t_spec)

    # ---------------- 2. 截断一致性（无未来函数硬检验） ----------------
    print("\n[2] truncation invariance (no-lookahead hard test)")

    def t_trunc():
        rng = np.random.default_rng(20261108)
        n = len(df)
        ts = sorted(int(x) for x in rng.integers(600, n - 1, size=3))
        worst = 0.0
        for t in ts:
            part = df.iloc[: t + 1]
            d2, s2 = compute_signals(part)
            a_d = DIR.iloc[t].to_numpy(dtype=np.float64)
            b_d = d2.iloc[-1].to_numpy(dtype=np.float64)
            a_s = STR.iloc[t].to_numpy(dtype=np.float64)
            b_s = s2.iloc[-1].to_numpy(dtype=np.float64)
            dd = np.abs(a_d - b_d).max()
            ds = np.abs(a_s - b_s).max()
            worst = max(worst, dd, ds)
            assert dd <= ATOL and ds <= ATOL, f"t={t} maxdiff DIR={dd:.3e} STR={ds:.3e}"
            assert np.array_equal(a_d, b_d), f"t={t} DIR not bitwise equal"
        return f"t={ts} max|diff|={worst:.1e} (bitwise equal)"
    check("3 random cutoffs: row t identical to full run", t_trunc)

    def t_trunc_multi():
        paths = list_codes(6)
        rng = np.random.default_rng(7)
        worst = 0.0
        used = []
        for p in paths[:3]:
            d0 = load_df(p)
            if len(d0) < 400:
                continue
            D0, S0 = compute_signals(d0)
            t = int(rng.integers(300, len(d0) - 1))
            d1, s1 = compute_signals(d0.iloc[: t + 1])
            dd = np.abs(D0.iloc[t].to_numpy(np.float64) - d1.iloc[-1].to_numpy(np.float64)).max()
            ds = np.abs(S0.iloc[t].to_numpy(np.float64) - s1.iloc[-1].to_numpy(np.float64)).max()
            worst = max(worst, dd, ds)
            assert dd <= ATOL and ds <= ATOL, f"{os.path.basename(p)} t={t} diff={dd:.2e}/{ds:.2e}"
            used.append(os.path.basename(p))
        return f"{len(used)} stocks, max|diff|={worst:.1e}"
    check("truncation invariance across 3 other stocks", t_trunc_multi)

    # ---------------- 3. 增量模式 tail=k ----------------
    print("\n[3] incremental mode tail=k")

    def t_tail():
        full_d = DIR.iloc[-1].to_numpy(dtype=np.float64)
        full_s = STR.iloc[-1].to_numpy(dtype=np.float64)
        worst = 0.0
        for k in (1, 50, 300):
            td, ts_ = compute_signals(df, tail=k)
            assert td.shape == (k, len(zoo)), (k, td.shape)
            assert td.index.equals(pd.DatetimeIndex(df["date"].iloc[-k:])), f"k={k} index"
            a = td.iloc[-1].to_numpy(dtype=np.float64)
            b = ts_.iloc[-1].to_numpy(dtype=np.float64)
            dd = np.abs(a - full_d).max()
            ds = np.abs(b - full_s).max()
            worst = max(worst, dd, ds)
            assert dd <= ATOL and ds <= ATOL, f"k={k} maxdiff DIR={dd:.3e} STR={ds:.3e}"
        return f"k=1/50/300 last row identical, max|diff|={worst:.1e}"
    check("tail=k last row == full-mode last row", t_tail)

    def t_tail_index():
        td, ts_ = compute_signals(df, tail=300)
        assert td.index.equals(STR.index[-300:])
        assert td.shape == ts_.shape == (300, len(zoo))
        return "index aligned to last 300 dates"
    check("tail=300 index alignment", t_tail_index)

    # ---------------- 4. 每族至少 1 个非零触发 ----------------
    print("\n[4] per-family liveness (no dead signals)")

    def t_live():
        nz = (DIR != 0).sum()
        fam_rows = []
        dead_fam = []
        dead_sid = []
        for fam in sorted(set(s.family for s in zoo)):
            members = [s.sid for s in zoo if s.family == fam]
            tot = int(nz[members].sum())
            fam_rows.append((fam, len(members), tot))
            if tot <= 0:
                dead_fam.append(fam)
        for sid in sids:
            if int(nz[sid]) <= 0:
                dead_sid.append(sid)
        # 三个外部数据族：本用例不喂数据，它们输出 0 是**正确行为**（预热不发声）。
        # 由下面的 [4b] 专测覆盖。
        dead_fam = [f for f in dead_fam if f not in EXT_FAMILIES]
        assert not dead_fam, f"dead families: {dead_fam}"
        # flow 的 sid 同理：不喂资金流数据时输出 0 是正确行为，由 [4b] 专测覆盖
        dead_sid = [x for x in dead_sid
                    if not any(x.startswith(p + "_") for p in EXT_PREFIX)]
        assert not dead_sid, f"dead sids: {dead_sid[:8]}"
        # 非零行上 STR 必须真的 >0
        d = DIR.to_numpy()
        s = STR.to_numpy()
        assert (s[d != 0] > 0).all(), "some nonzero DIR rows have STR==0"
        for fam, cnt, tot in fam_rows:
            print(f"        {fam:<12} variants={cnt:<2} nonzero_dir_rows={tot}")
        return f"{len(fam_rows)} families all alive"
    check("every family (and every sid) fires at least once", t_live)

    # ---------------- 4b. flow 族：喂数据必须发声，不喂必须全 0 ----------------
    print(chr(10) + "[4b] flow family (needs external capital-flow data)")

    def t_ext():
        import pandas as pd
        from qcore.signals import compute_signals
        df = load_df(find_kline())
        n = len(df)
        cases = [
            ("flow", {"main_ratio": 3.0, "super_ratio": 2.0}, {1.0}),
            ("block_trade", {"bt_shares_ratio": 1.0, "bt_premium": 8.0}, {1.0}),
            ("billboard", {"bb_net_ratio": 12.0}, {1.0}),
            # 只喂 net 列：short 变体是反向解释（融券上升看空），
            # 同时喂三列会让方向混在一起，测不出「方向是否正确」。
            ("margin", {"mg_net_ratio": 5.0}, {1.0}),
            # 情绪族同时有反向(snt_contra)与顺势(snt_follow)两个变体，
            # 同一份输入必然同时出现 -1 和 +1 —— 这是设计，不是错误。
            ("sentiment", {"snt_bull_ratio": 0.85}, {-1.0, 1.0}),
            ("event", {"evt_direction": 1.0, "evt_level": 3.0}, {1.0}),
            ("holder", {"hold_net_ratio": 2.0}, {1.0}),
            ("lift", {"fc_lift_ratio": 15.0}, {-1.0}),      # 解禁压力看空
            ("forecast", {"eps_revision": 2.0}, {1.0}),
            # breadth 同时有 contra 与 follow 两个变体，双向是设计
            ("breadth", {"breadth_z": 2.5, "up_ratio": 0.9}, {-1.0, 1.0}),
        ]
        out = []
        for fam, cols, expect in cases:
            expect = set(expect) if not isinstance(expect, set) else expect
            specs = [s for s in signal_zoo() if s.family == fam]
            assert specs, f"no {fam} signals in zoo"
            # 不喂数据 -> 必须全 0（预热），且不得报错
            d0, _ = compute_signals(df, specs=specs)
            assert float(abs(d0.to_numpy()).sum()) == 0.0, f"{fam} fired without data"
            # 喂数据 -> 必须发声且方向正确
            f = pd.DataFrame({"date": df["date"]})
            for c, v in cols.items():
                f[c] = np.full(n, float(v))
            d1, _ = compute_signals(df, flow_df=f, specs=specs)
            nz = int((d1.to_numpy() != 0).sum())
            assert nz > 0, f"{fam} did not fire even with data"
            vals = d1.to_numpy()[d1.to_numpy() != 0]
            got = set(vals.tolist())
            assert got and got <= expect, f"{fam} direction wrong: {got} not in {expect}"
            out.append(f"{fam}:{nz}")
        return "fired -> " + " ".join(out)

    check("外部数据族：无数据=0，有数据=发声且方向正确", t_ext)

    # ---------------- 5. 性能与内存 ----------------
    print("\n[5] performance / memory")

    df3k = df.iloc[-3000:].reset_index(drop=True)

    def t_speed():
        compute_signals(df3k)                      # 预热（导入 / 指数缓存）
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            compute_signals(df3k)
            best = min(best, time.perf_counter() - t0)
        print(f"        full-history 3000 bars : {best * 1000:.1f} ms")
        assert best < 0.040, f"{best * 1000:.1f} ms > 40 ms budget"
        return f"{best * 1000:.1f} ms"
    check("single stock 3000 bars < 40 ms", t_speed)

    def t_speed_tail():
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            compute_signals(df3k, tail=300)
            best = min(best, time.perf_counter() - t0)
        print(f"        tail=300 (3000 bars)   : {best * 1000:.1f} ms")
        return f"{best * 1000:.1f} ms"
    check("tail=300 timing (reported, not gated)", t_speed_tail)

    def t_mem():
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        compute_signals(df3k)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        print(f"        peak python alloc      : {peak / 1024:.0f} KiB (informational)")
        assert peak < 200 * 1024 * 1024, f"peak {peak / 1024 / 1024:.1f} MiB"
        return f"{peak / 1024:.0f} KiB"
    check("peak memory (tracemalloc, informational)", t_mem)

    def t_batch():
        avail = (sorted(glob.glob(os.path.join(KLINE_DIR, "*_h2.csv")))
                 or sorted(glob.glob(os.path.join(FIXTURE_DIR, "*_h2.csv"))))
        if len(avail) < 5:
            return f"SKIP: only {len(avail)} kline files available (need >=5)"
        paths = list_codes(20)
        frames = [load_df(p) for p in paths]
        compute_signals(frames[0])                 # 预热
        t0 = time.perf_counter()
        for f in frames:
            compute_signals(f)
        el = time.perf_counter() - t0
        per = el / max(len(frames), 1)
        print(f"        batch {len(frames)} stocks    : {el * 1000:.0f} ms "
              f"({per * 1000:.1f} ms/stock) -> 300 stocks ~= {per * 300:.1f} s")
        assert per * 300 < 20.0, f"extrapolated 300 stocks {per * 300:.1f}s > 20s"
        return f"{el * 1000:.0f} ms for {len(frames)}"
    check("batch throughput (20 stocks, extrapolate 300)", t_batch)

    # ---------------- 6. 健壮性 ----------------
    print("\n[6] robustness (never raise)")

    def t_empty():
        d, s = compute_signals(df.iloc[:0].reset_index(drop=True))
        assert d.shape == (0, len(zoo)) and s.shape == (0, len(zoo))
        assert list(d.columns) == sids
        return "empty df -> empty (0,S)"
    check("empty DataFrame", t_empty)

    def t_nan_col():
        bad = df.copy()
        for c in ("close", "open", "high", "low", "volume", "amount", "turnover"):
            bad[c] = np.nan
        d, s = compute_signals(bad)
        assert np.isfinite(d.to_numpy()).all() and np.isfinite(s.to_numpy()).all()
        assert (d.to_numpy() == 0).all(), "all-NaN price columns must degrade to 0"
        return "all-NaN columns -> all zeros, no exception"
    check("all-NaN price columns", t_nan_col)

    def t_missing_col():
        lean = df[["date", "close", "high", "low"]].copy()
        d, s = compute_signals(lean)
        assert np.isfinite(d.to_numpy()).all() and np.isfinite(s.to_numpy()).all()
        assert d.shape == (len(df), len(zoo))
        return "missing volume/amount/turnover/open -> no exception"
    check("missing columns", t_missing_col)

    def t_constant():
        flat = df.copy()
        for c in ("open", "high", "low", "close"):
            flat[c] = 100.0
        d, s = compute_signals(flat)
        assert np.isfinite(d.to_numpy()).all() and np.isfinite(s.to_numpy()).all()
        # rs 族是"个股 vs 指数"的相对量，个股走平时指数仍会动，故允许非零
        own = [x.sid for x in zoo if x.family != "rs"]
        assert (d[own].to_numpy() == 0).all(), "constant price -> self-referential signals must be 0"
        return "constant price -> own-price signals zero, no divide-by-zero"
    check("constant price series", t_constant)

    def t_tiny():
        tiny = df.iloc[:30].reset_index(drop=True)
        d, s = compute_signals(tiny)
        assert d.shape == (30, len(zoo))
        assert np.isfinite(d.to_numpy()).all() and np.isfinite(s.to_numpy()).all()
        return "30 bars (below warmup) -> zeros, no exception"
    check("short history below warmup", t_tiny)

    def t_cfg():
        sub = signal_zoo({"families": ["rsi", "macd"]})
        assert len(sub) == 12 and all(s.family in ("rsi", "macd") for s in sub)
        d, s = compute_signals(df, specs=sub)
        assert list(d.columns) == [x.sid for x in sub]
        return f"custom zoo -> {len(sub)} columns"
    check("signal_zoo(cfg) subset & custom specs", t_cfg)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    if _FAILS:
        for f in _FAILS:
            print("  FAILED:", f)
        print("=" * 78)
        return 1
    print("ALL GREEN")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
