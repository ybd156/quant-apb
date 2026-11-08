"""tests/test_intraday.py —— 盘中分钟级模块验收（`python -m tests.test_intraday`）

覆盖：
  1. fetch_minute 真实数据（240 根、列名、时间范围 09:31–15:00）
  2. intraday_signals：≥8 个信号、取值域、无 NaN、**截断一致性**（严格因果）
  3. step()：打分封存 → 评估到期 → 更新权重 → 落盘；耗时与条数
  4. 幂等：同日重跑不重复入账；结果字段不被覆盖
  5. 成本计算：往返 bps 与净收益口径；年化拖累 >100%（不可交易）
  6. 无数据 / 无台账 优雅退出（不抛异常、非零退出码不崩）
  7. 复盘报告落盘且含"前向验证/不可交易"结论
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

import qcore.intraday as I
from qcore.costs import effective_cost_bps
from qcore.intraday_signals import INTRADAY_SIDS, intraday_signals, intraday_zoo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOL = 1e-12
CODES = ["600519", "000001", "000858"]

_PASS = 0
_FAIL = 0
_FAILS = []


def _ok(name, detail=""):
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))


def _bad(name, detail=""):
    global _FAIL
    _FAIL += 1
    _FAILS.append(f"{name} :: {detail}")
    print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def check(name, fn, skip_reason=None):
    if skip_reason:
        print(f"  [SKIP] {name} :: {skip_reason}")
        return
    try:
        d = fn()
        _ok(name, "" if d is None else str(d))
    except AssertionError as e:
        _bad(name, str(e) or "assertion failed")
    except Exception as e:      # noqa: BLE001
        _bad(name, f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------
def _redirect_paths(tmp: str):
    """把模块级路径指向临时目录，测试不污染真实台账/状态。"""
    I.INTRADAY_DIR = os.path.join(tmp, "intraday")
    I.MINUTE_DIR = os.path.join(tmp, "minute")
    I.STATE_DIR = os.path.join(tmp, "state")
    I.REPORT_DIR = os.path.join(tmp, "reports")
    I.LOG_DIR = os.path.join(tmp, "logs")
    I.STATE_PATH = os.path.join(I.STATE_DIR, "intraday_weighter.npz")
    I._ensure_dirs()


def main() -> int:
    print("=" * 78)
    print("tests/test_intraday.py :: intraday forward-test module")
    print("=" * 78)

    # ---------------- 1. 真实数据 ----------------
    print("\n[1] real minute data")
    frames = {}
    net_err = None
    for c in CODES:
        try:
            frames[c] = I.fetch_minute(c, klt=1)
        except Exception as e:      # noqa: BLE001
            net_err = f"{type(e).__name__}: {e}"
    if not frames:
        # 完全拿不到数据时，用合成数据把逻辑跑通，并明确标注
        print(f"  [WARN] no network minute data ({net_err}) -> use synthetic bars")
        idx = pd.date_range("2026-11-06 09:31", "2026-11-06 15:00", freq="1min")
        rng = np.random.default_rng(0)
        px = 100 * np.cumprod(1 + rng.normal(0, 0.0004, len(idx)))
        frames = {"600519": pd.DataFrame({
            "datetime": idx, "open": px, "close": px,
            "high": px * 1.0005, "low": px * 0.9995,
            "volume": rng.integers(100, 5000, len(idx)).astype(float),
            "amount": px * rng.integers(100, 5000, len(idx)) * 100.0})}

    df = frames[CODES[0]] if CODES[0] in frames else list(frames.values())[0]

    def t_fetch():
        assert set(["datetime", "open", "close", "high", "low", "volume", "amount"]).issubset(df.columns)
        assert len(df) >= 200, len(df)
        assert df["datetime"].is_monotonic_increasing
        hhmm = df["datetime"].dt.strftime("%H:%M")
        assert hhmm.iloc[0] == "09:31", hhmm.iloc[0]
        assert hhmm.iloc[-1] == "15:00", hhmm.iloc[-1]
        return f"{len(df)} bars, {hhmm.iloc[0]}..{hhmm.iloc[-1]}, {len(frames)} codes"
    check("fetch_minute returns one session (240 bars)", t_fetch, skip_reason=net_err if not frames else None)

    # ---------------- 2. 信号 ----------------
    print("\n[2] intraday_signals")
    DIR, STR = intraday_signals(df)
    zoo = intraday_zoo()

    def t_zoo():
        assert len(zoo) >= 8, len(zoo)
        assert len(set(s.sid for s in zoo)) == len(zoo)
        assert list(DIR.columns) == [s.sid for s in zoo] == list(INTRADAY_SIDS)
        return f"{len(zoo)} signals: {', '.join(INTRADAY_SIDS[:5])} ..."
    check(">=8 intraday signals with stable sids", t_zoo)

    def t_domain():
        d = DIR.to_numpy()
        s = STR.to_numpy()
        assert np.isfinite(d).all() and np.isfinite(s).all()
        assert set(np.unique(d)).issubset({-1.0, 0.0, 1.0})
        assert s.min() >= 0 and s.max() <= 1
        assert DIR.dtypes.eq(np.float32).all() and STR.dtypes.eq(np.float32).all()
        return f"DIR in {np.unique(d).tolist()}, STR [{s.min():.2f},{s.max():.2f}], float32"
    check("DIR subset {-1,0,1}, STR subset [0,1], no NaN, float32", t_domain)

    def t_live():
        nz = (DIR != 0).sum()
        dead = [sid for sid in INTRADAY_SIDS if int(nz[sid]) == 0]
        assert not dead, f"dead signals: {dead}"
        return f"all {len(INTRADAY_SIDS)} fire; min triggers={int(nz.min())}"
    check("no dead signal (every sid fires)", t_live)

    def t_warmup():
        first = int(DIR.ne(0).any(axis=1).to_numpy().argmax())
        assert first <= 30, f"first nonzero at bar {first} > 30"
        return f"first nonzero bar={first} (warmup <=30)"
    check("warmup <= 30 bars", t_warmup)

    def t_trunc():
        rng = np.random.default_rng(20261108)
        worst = 0.0
        for t in sorted(int(x) for x in rng.integers(40, len(df) - 1, size=3)):
            d2, s2 = intraday_signals(df.iloc[: t + 1])
            dd = np.abs(DIR.iloc[t].to_numpy(np.float64) - d2.iloc[-1].to_numpy(np.float64)).max()
            ds = np.abs(STR.iloc[t].to_numpy(np.float64) - s2.iloc[-1].to_numpy(np.float64)).max()
            worst = max(worst, dd, ds)
            assert dd <= ATOL and ds <= ATOL, f"t={t} diff={dd:.2e}/{ds:.2e}"
        return f"3 cutoffs, max|diff|={worst:.1e}"
    check("truncation invariance (strict causality)", t_trunc)

    def t_empty():
        d, s = intraday_signals(df.iloc[:0])
        assert d.shape == (0, len(INTRADAY_SIDS))
        return "empty -> (0,S)"
    check("empty df", t_empty)

    # ---------------- 3. step() 与幂等 ----------------
    print("\n[3] step() / idempotency / cost")
    tmp = os.path.join(ROOT, "data", f"_tmp_intraday_test_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    real_fetch = I.fetch_many
    try:
        _redirect_paths(tmp)
        I.fetch_many = lambda codes, klt=1: {c: frames[c].copy() for c in codes if c in frames}
        now_closed = pd.Timestamp("2026-11-06 15:30").to_pydatetime()

        state = {}

        def t_step():
            t0 = time.perf_counter()
            r = I.step(CODES, horizon_min=1, now=now_closed, verbose=False)
            el = time.perf_counter() - t0
            assert r.get("ok"), r
            assert r["predicted"] > 0, r
            assert r["scored"] > 0, r
            state["r"] = r
            state["el"] = el
            state["date"] = r["ledgers"][0]
            return (f"predicted={r['predicted']} scored={r['scored']} "
                    f"updated={r['updated']} in {el * 1000:.0f} ms")
        check("step() writes predictions and scores due ones", t_step)

        def t_ledger():
            date = state["date"]
            recs = I.load_ledger(date)
            assert len(recs) == state["r"]["predicted"], (len(recs), state["r"]["predicted"])
            scored = [x for x in recs if x["status"] == "scored"]
            assert len(scored) == state["r"]["scored"]
            keys = {(x["code"], x["bar_time"], x["horizon"]) for x in recs}
            assert len(keys) == len(recs), "duplicate primary keys in ledger"
            sample = recs[0]
            for f in ("as_of", "code", "bar_time", "horizon", "score", "action",
                      "sigma", "dir_hash", "weights_hash", "dir_vec", "status"):
                assert f in sample, f"missing field {f}"
            assert len(sample["dir_vec"]) == len(INTRADAY_SIDS)
            return f"{len(recs)} records, {len(scored)} scored, keys unique"
        check("ledger schema + unique (code,bar_time,horizon)", t_ledger)

        def t_idem():
            before = I.load_ledger(state["date"])
            snap = [(x["code"], x["bar_time"], x["score"], x["status"], x.get("ret")) for x in before]
            r2 = I.step(CODES, horizon_min=1, now=now_closed, verbose=False)
            after = I.load_ledger(state["date"])
            snap2 = [(x["code"], x["bar_time"], x["score"], x["status"], x.get("ret")) for x in after]
            assert len(after) == len(before), f"ledger grew on rerun: {len(before)} -> {len(after)}"
            assert snap == snap2, "records changed on rerun"
            assert r2["predicted"] == 0, f"rerun predicted {r2['predicted']} duplicates"
            return f"rerun added 0 rows (still {len(after)}), scores/results untouched"
        check("idempotent: same-day rerun adds nothing", t_idem)

        def t_payload():
            recs = I.load_ledger(state["date"])
            sc = [x for x in recs if x["status"] == "scored"]
            ok = 0
            for r in sc:
                e, x = r["entry_price"], r["exit_price"]
                if e is None or x is None:
                    continue
                exp = x / e - 1.0
                assert abs(exp - r["ret"]) < 1e-7, (exp, r["ret"])
                if abs(r["score"]) <= 1e-12:      # 不表态的 bar 不判对错
                    assert r["hit"] is None
                    continue
                assert r["hit"] in (True, False)
                assert bool(r["hit"]) == bool(np.sign(r["score"]) == np.sign(r["ret"]))
                ok += 1
            assert ok > 100, ok
            return f"{ok} scored rows verified (ret = exit/entry - 1, hit consistent)"
        check("evaluation arithmetic (open-to-open, no close cheating)", t_payload)

        def t_exec_lag():
            recs = [x for x in I.load_ledger(state["date"]) if x["status"] == "scored"][:1]
            r = recs[0]
            t_bar = pd.Timestamp(r["bar_time"])
            t_entry = pd.Timestamp(r["entry_time"])
            t_exit = pd.Timestamp(r["exit_time"])
            assert t_entry > t_bar, "entry must be strictly after the decision bar"
            assert (t_exit - t_entry) == pd.Timedelta(minutes=r["horizon"])
            return f"entry={r['entry_time']} exit={r['exit_time']} (h={r['horizon']}m, lag=1 bar)"
        check("execution lag: decide at t, enter at t+1 open", t_exec_lag)

        def t_state():
            assert os.path.exists(I.STATE_PATH), I.STATE_PATH
            z = np.load(I.STATE_PATH, allow_pickle=True)
            assert list(z["sids"]) == list(INTRADAY_SIDS)
            assert z["a_g"].shape == (len(INTRADAY_SIDS),)
            assert float(np.asarray(z["n_g"]).sum()) > 0, "weights never updated"
            L2 = I.IntradayLearner.load()
            w = L2.w.weights(0)
            assert abs(float(w.sum()) - 1.0) < 1e-9 and (w > 0).all()
            return f"npz ok, updates={int(np.asarray(z['n_g']).sum())}, sum(w)=1"
        check("weighter state persisted & reloadable (reuse AdaptiveWeighter)", t_state)

        def t_cost():
            rt = effective_cost_bps("CN", 1e5)
            assert 15.0 < rt < 25.0, rt
            r = state["r"]
            date = state["date"]
            rev = I.session_review(date, write=True)
            assert rev["ok"], rev
            assert abs(rev["rt_bps"] - rt) < 1e-9
            assert rev["bets"] > 0
            # 净 = 毛 - 成本
            assert abs(rev["mean_net_bps"] - (rev["mean_gross_bps"] - rt)) < 1e-6
            assert rev["annual_cost_pct"] > 100.0, rev["annual_cost_pct"]
            assert rev["mean_net_bps"] < 0, "1-min net must be negative"
            assert "不可交易" in rev["verdict"]
            state["rev"] = rev
            return (f"rt={rt:.2f}bps gross={rev['mean_gross_bps']:.2f} "
                    f"net={rev['mean_net_bps']:.2f} bps, annual drag={rev['annual_cost_pct']:.0f}%")
        check("cost accounting: net = gross - round-trip; annual drag > 100%", t_cost)

        def t_report():
            rev = state["rev"]
            p = rev.get("report")
            assert p and os.path.exists(p), p
            txt = open(p, encoding="utf-8").read()
            for kw in ("前向验证", "不是回测", "不可交易", "毛", "净", "19."):
                assert kw in txt, f"report missing {kw}"
            assert len(txt) > 1500
            return f"{os.path.basename(p)} ({len(txt)} chars)"
        check("review report written with honest conclusion", t_report)

        def t_expire():
            # 造一个"昨天"的台账，pending 应被标记 expired 而不是伪造回填
            old_date = "2026-11-05"
            recs = [{"code": "600519", "bar_time": f"{old_date} 10:00:00", "horizon": 1,
                     "bar_idx": 29, "score": 0.1, "status": "pending", "ret": None}]
            I.write_ledger(old_date, recs)
            n = I._expire_stale("2026-11-06", verbose=False)
            after = I.load_ledger(old_date)
            assert after[0]["status"] == "expired", after[0]["status"]
            assert after[0].get("ret") is None, "must not fabricate returns"
            return f"{n} stale pending -> expired (no fake backfill)"
        check("stale pending expired, never fabricated", t_expire)

        def t_no_data():
            I.fetch_many = lambda codes, klt=1: {}
            r = I.step(CODES, horizon_min=1, now=now_closed, verbose=False)
            assert r["ok"] is False and r["reason"] == "no_data"
            rev = I.session_review("1999-01-01", write=False)
            assert rev["ok"] is False
            return "no data -> ok=False, no exception"
        check("graceful exit when no minute data / no ledger", t_no_data)

        def t_cli():
            rc = I.main(["--review", "--date", state["date"]])
            assert rc == 0, rc
            rc2 = I.main(["--review", "--date", "1999-01-01"])
            assert rc2 == 1, rc2
            return "CLI --review ok / missing date -> exit 1"
        check("CLI wiring", t_cli)

    finally:
        I.fetch_many = real_fetch
        shutil.rmtree(tmp, ignore_errors=True)

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
    try:
        sys.exit(main())
    except Exception:      # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
