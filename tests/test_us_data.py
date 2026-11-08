"""tests/test_us_data.py —— 美股/港股数据层验收（`python -m tests.test_us_data`）

覆盖：
  1. ticker 解析：AAPL/MSFT/NVDA/TSLA/AMD 五个全对（105.xxx）
  2. 美股日线：AAPL/MSFT/NVDA 各 >500 根；无负价、无 |pct_chg|>50%；日期升序唯一
  3. 港股日线：00700.HK / 09988.HK 各 >500 根；同样断言；'0700.HK' 与 '00700' 等价
  4. 基准：SPY / HSI 各 >1000 根
  5. 列名契约：与 qcore/data.py（A股缓存）**逐列一致**
  6. 缓存命中：第二次调用 0 次网络请求且 <50ms
  7. 失败路径：解析不到的 ticker / 无数据的 secid 抛清晰异常（不静默返回空表）
  8. 美股池：列名/去重/市值覆盖率
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

import qcore.us_data as ud
from qcore.data import KLINE_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_KLINE_DIR = os.path.join(ROOT, "data", "raw", "kline")
EXPECT = ["date", "open", "close", "high", "low", "volume", "amount",
          "amplitude", "pct_chg", "change", "turnover", "code"]

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


def _check(name: str, cond: bool, detail: str = "") -> bool:
    (_ok if cond else _bad)(name, detail)
    return bool(cond)


def _quality(name: str, df: pd.DataFrame, min_bars: int = 500) -> None:
    """通用行情体检：根数 / 负价 / 极端涨跌 / 日期单调 / 列契约。"""
    n = len(df)
    if not _check(f"{name} bars > {min_bars}", n > min_bars, f"bars={n}"):
        return
    _check(f"{name} columns contract", list(df.columns) == EXPECT,
           f"{list(df.columns) if list(df.columns) != EXPECT else 'ok'}")
    _check(f"{name} no negative price",
           bool((df[["open", "high", "low", "close"]] > 0).all().all()),
           f"min_close={df['close'].min():.4g}")
    mx = float(df["pct_chg"].abs().max())
    _check(f"{name} no |pct_chg|>50", mx <= 50.0, f"max={mx:.2f}%")
    _check(f"{name} date strictly ascending", bool(df["date"].is_monotonic_increasing))
    _check(f"{name} no duplicate dates", int(df["date"].duplicated().sum()) == 0)
    _check(f"{name} no NaN in OHLC", int(df[["open", "high", "low", "close"]].isna().sum().sum()) == 0)
    _check(f"{name} pct_chg consistent with close",
           bool(np.allclose(df["pct_chg"].to_numpy()[1:],
                            (df["close"].to_numpy()[1:] / df["close"].to_numpy()[:-1] - 1) * 100,
                            atol=1e-6, rtol=0)),
           "recomputed from close")
    _check(f"{name} dropped_jumps is small", len(df.attrs.get("dropped_jumps", [])) <= 5,
           f"dropped={len(df.attrs.get('dropped_jumps', []))}")


# ---------------------------------------------------------------- 1
def test_resolve():
    print("\n[1] resolve_ticker")
    exp = {"AAPL": "105.AAPL", "MSFT": "105.MSFT", "NVDA": "105.NVDA",
           "TSLA": "105.TSLA", "AMD": "105.AMD"}
    got = {}
    for t, want in exp.items():
        try:
            got[t] = ud.resolve_ticker(t)
            _check(f"resolve {t}", got[t] == want, f"got={got[t]} want={want}")
        except Exception as e:  # noqa: BLE001
            _bad(f"resolve {t}", f"{type(e).__name__}: {e}")
    _check("resolve SPY -> AMEX 107", ud.resolve_ticker("SPY") == "107.SPY",
           ud.resolve_ticker("SPY"))
    # 港股两种写法都要能吃
    _check("resolve 0700.HK -> 116.00700", ud.resolve_ticker("0700.HK") == "116.00700")
    _check("resolve 00700 -> 116.00700", ud.resolve_ticker("00700") == "116.00700")
    # 磁盘缓存文件存在
    _check("us_map.json written", os.path.exists(ud.US_MAP_PATH), ud.US_MAP_PATH)


# ---------------------------------------------------------------- 2
def test_us_kline():
    print("\n[2] US daily bars")
    for t in ("AAPL", "MSFT", "NVDA"):
        try:
            df = ud.load_us_kline(t)
            _quality(t, df, min_bars=500)
            print(f"       {t}: bars={len(df)} {df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()} "
                  f"last_close={df['close'].iloc[-1]:.4g}")
        except Exception as e:  # noqa: BLE001
            _bad(f"{t} load", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 3
def test_hk_kline():
    print("\n[3] HK daily bars")
    for c in ("00700.HK", "09988.HK"):
        try:
            df = ud.load_hk_kline(c)
            _quality(c, df, min_bars=500)
            print(f"       {c}: bars={len(df)} {df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()} "
                  f"last_close={df['close'].iloc[-1]:.4g}")
        except Exception as e:  # noqa: BLE001
            _bad(f"{c} load", f"{type(e).__name__}: {e}")
    # 代码写法等价性
    try:
        a = ud.load_hk_kline("00700")
        b = ud.load_hk_kline("0700.HK")
        _check("HK code forms equivalent", len(a) == len(b) and
               float(a["close"].iloc[-1]) == float(b["close"].iloc[-1]),
               f"{len(a)} vs {len(b)}")
    except Exception as e:  # noqa: BLE001
        _bad("HK code forms equivalent", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 4
def test_benchmarks():
    print("\n[4] benchmarks")
    for name, fn in (("SPY", ud.load_us_benchmark), ("HSI", ud.load_hk_benchmark)):
        try:
            df = fn()
            n = len(df)
            _check(f"{name} bars > 1000", n > 1000, f"bars={n}")
            _check(f"{name} no negative price", bool((df["close"] > 0).all()),
                   f"min={df['close'].min():.4g}")
            _check(f"{name} date ascending", bool(df["date"].is_monotonic_increasing))
            print(f"       {name}: bars={n} {df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()} "
                  f"last={df['close'].iloc[-1]:.4g}")
        except Exception as e:  # noqa: BLE001
            _bad(f"{name} benchmark", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 5
def test_column_contract():
    print("\n[5] column contract vs qcore/data.py (A-share cache)")
    files = sorted(glob.glob(os.path.join(A_KLINE_DIR, "*_h2.csv")))
    if not files:
        _bad("A-share reference file", "no data/raw/kline/*_h2.csv found")
        return
    ref = pd.read_csv(files[0], nrows=2)
    _check("A-share reference columns == KLINE_COLS+code",
           list(ref.columns) == EXPECT, f"{files[0]} -> {list(ref.columns)}")
    us = ud.load_us_kline("AAPL")
    hk = ud.load_hk_kline("00700")
    _check("US columns identical to A-share", list(us.columns) == list(ref.columns))
    _check("HK columns identical to A-share", list(hk.columns) == list(ref.columns))
    _check("dtypes match (date datetime64, rest numeric)",
           str(us["date"].dtype).startswith("datetime64")
           and all(np.issubdtype(us[c].dtype, np.number) for c in EXPECT[1:-1]),
           str({c: str(us[c].dtype) for c in EXPECT}))
    _check("KLINE_COLS sanity", list(KLINE_COLS) == EXPECT[:-1])


# ---------------------------------------------------------------- 6
def test_cache_hit():
    print("\n[6] cache hit -> no network, <50ms")
    ud.load_us_kline("AAPL")  # warm
    path = os.path.join(ud.US_KLINE_DIR, "AAPL_h2.csv")
    _check("cache file exists", os.path.exists(path), path)

    calls = {"n": 0}
    orig = ud._get_json

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("network hit on cache path")

    ud._get_json = _boom
    try:
        t0 = time.perf_counter()
        df = ud.load_us_kline("AAPL")
        dt = (time.perf_counter() - t0) * 1000.0
    finally:
        ud._get_json = orig
    _check("cache hit -> 0 network calls", calls["n"] == 0, f"calls={calls['n']}")
    _check("cache hit < 50ms", dt < 50.0, f"{dt:.1f}ms")
    _check("cache hit returns full frame", len(df) > 500, f"bars={len(df)}")

    # 港股同理
    ud.load_hk_kline("00700")
    calls["n"] = 0
    ud._get_json = _boom
    try:
        t0 = time.perf_counter()
        hk = ud.load_hk_kline("00700")
        dth = (time.perf_counter() - t0) * 1000.0
    finally:
        ud._get_json = orig
    _check("HK cache hit -> 0 network calls", calls["n"] == 0, f"calls={calls['n']}")
    _check("HK cache hit < 50ms", dth < 50.0, f"{dth:.1f}ms")
    _check("HK cache hit returns full frame", len(hk) > 500, f"bars={len(hk)}")


# ---------------------------------------------------------------- 7
def test_failures():
    print("\n[7] failure paths are loud, not silent")
    try:
        ud.resolve_ticker("ZZZZNOTATICKER")
        _bad("resolve bogus raises", "returned without raising")
    except Exception as e:  # noqa: BLE001
        _check("resolve bogus raises", isinstance(e, RuntimeError),
               f"{type(e).__name__}: {str(e)[:70]}")
    try:
        ud.fetch_hk_kline("99999", repair=True)  # 不存在的港股代码
        _bad("bogus HK code raises", "returned without raising")
    except Exception as e:  # noqa: BLE001
        _check("bogus HK code raises", isinstance(e, RuntimeError),
               f"{type(e).__name__}: {str(e)[:70]}")
    try:
        ud._norm_hk_code("ABCDEF")
        _bad("bad HK code rejected", "accepted non-numeric")
    except ValueError as e:
        _check("bad HK code rejected", True, str(e)[:50])
    # 批量函数单只失败不中断
    ok = ud.download_us(["AAPL", "ZZZZNOTATICKER", "MSFT"], verbose=False)
    _check("batch survives single failure", ok == 2, f"ok={ok}/3")


# ---------------------------------------------------------------- 8
def test_pool():
    print("\n[8] US pool")
    try:
        p = ud.get_us_pool(80)
        _check("pool columns", list(p.columns) == ["code", "name", "mktcap", "amount", "sector"],
               str(list(p.columns)))
        _check("pool non-empty", len(p) > 0, f"n={len(p)}")
        _check("pool codes unique", int(p["code"].duplicated().sum()) == 0)
        cov = float(p["mktcap"].notna().mean())
        _check("pool mktcap coverage > 0", cov > 0, f"{cov:.1%}")
        _check("pool mktcap positive where present",
               bool((p["mktcap"].dropna() > 0).all()))
        print(f"       pool: n={len(p)} mktcap_cov={cov:.1%} "
              f"amount_cov={p['amount'].notna().mean():.1%} source={p.attrs.get('source')}")
        # 池子里的代码要能解析（抽样 5 只）
        for t in p["code"].head(5):
            try:
                _check(f"pool code resolvable {t}", ud.resolve_ticker(t).endswith(t),
                       ud.resolve_ticker(t))
            except Exception as e:  # noqa: BLE001
                _bad(f"pool code resolvable {t}", f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        _bad("get_us_pool", f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 72)
    print("tests/test_us_data.py -- US/HK data layer acceptance")
    print("=" * 72)
    t0 = time.perf_counter()
    test_resolve()
    test_us_kline()
    test_hk_kline()
    test_benchmarks()
    test_column_contract()
    test_cache_hit()
    test_failures()
    test_pool()
    el = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"PASS={_PASS}  FAIL={_FAIL}  elapsed={el:.1f}s")
    if _FAILS:
        print("failures:")
        for f in _FAILS:
            print("  -", f)
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
