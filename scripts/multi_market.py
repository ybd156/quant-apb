"""
多市场对照 —— 同一套 73 个信号，跑 A 股 / 美股 / 港股，比 IC。

## 要回答的问题

A 股上 IC = −0.0185（t=−2.48），到底是：
  (a) 这套信号本身没用（普适性差），还是
  (b) A 股特有的结构（涨跌停、T+1、散户占比高）把它压反了？

判断方法：**同一套信号换市场跑**。
  · 如果三个市场都显著为负 → 信号集本身有问题
  · 如果只有 A 股为负 → 是市场结构问题，换市场就能救
  · 如果都不显著 → 信号集没有信息量，与市场无关

## 口径

每个市场用自己的基准指数、自己的交易日历、自己的成本模型。
不混用日历（INV：不同市场的停牌/涨跌停规则不同）。
"""
from __future__ import annotations

import sys
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, ".")

from qcore.dataset import build_dataset
from qcore.engine import run_walkforward, RunConfig
from qcore import analysis as A, stats as ST


def _us_pool(n: int):
    from qcore.us_data import get_pool_us, load_us_kline
    p = get_pool_us(top_n=n)
    return p, (lambda c: load_us_kline(c)), False


def _hk_pool(n: int):
    from qcore.us_data import load_hk_kline
    from qcore.data import get_pool
    # 港股池：优先用项目已有的港股清单，没有就跳过
    try:
        from qcore.us_data import get_pool_hk  # type: ignore
        p = get_pool_hk(top_n=n)
        return p, (lambda c: load_hk_kline(c)), False
    except Exception:
        return None, None, None


def _cn_pool(n: int):
    from qcore.data import get_pool, load_kline
    p = get_pool(top_n=n)
    return p, (lambda c: load_kline(c)), True


MARKETS = {"CN": _cn_pool, "US": _us_pool, "HK": _hk_pool}


def run_one(market: str, n: int = 60, start: str = "2016-01-01",
            horizon: int = 5, verbose: bool = True) -> Optional[Dict]:
    fn = MARKETS.get(market)
    if fn is None:
        return None
    try:
        pool, loader, pad = fn(n)
    except Exception as e:  # noqa: BLE001
        return {"market": market, "ok": False, "error": f"{type(e).__name__}: {e}"[:100]}
    if pool is None or len(pool) == 0:
        return {"market": market, "ok": False, "error": "no pool"}

    codes = [str(c) for c in pool["code"].tolist()]
    try:
        p3d = build_dataset(codes, start=start, horizon=horizon,
                            loader=loader, pad=pad, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"market": market, "ok": False, "error": f"dataset: {e}"[:100]}

    T, N, S = p3d.DIR.shape
    try:
        res = run_walkforward(p3d, RunConfig(horizon=horizon, warmup=120,
                                             record_ledger=False))
    except Exception as e:  # noqa: BLE001
        return {"market": market, "ok": False, "error": f"engine: {e}"[:100]}

    sc = res["scores_TN"]
    ret = p3d.RET[horizon]
    ic = A.daily_ic_series(sc.T, ret.T)   # daily_ic_series 约定 (N,T)
    ic = ic[np.isfinite(ic)]
    if ic.size < 60:
        return {"market": market, "ok": False, "error": f"IC 样本不足 {ic.size}"}
    st = ST.mean_tstat(ic, lags=max(horizon - 1, 0))

    # 多空价差
    sp = []
    for t in range(T):
        x, y = sc[t], ret[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < max(10, int(0.2 * N)):   # 阈值随池子大小自适应，之前写死 40 会让小池子永远为空
            continue
        lo, hi = np.percentile(x[m], 20), np.percentile(x[m], 80)
        sp.append(np.nanmean(y[m][x[m] >= hi]) - np.nanmean(y[m][x[m] <= lo]))
    sp = np.asarray(sp)
    sps = ST.mean_tstat(sp, lags=max(horizon - 1, 0)) if sp.size > 60 else {"mean": np.nan, "t": np.nan}

    return {"market": market, "ok": True, "T": int(T), "N": int(N), "S": int(S),
            "ic": float(st["mean"]), "t_ic": float(st["t"]),
            "spread": float(sps["mean"]), "t_spread": float(sps["t"]),
            "n_ic": int(ic.size)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="CN,US,HK")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args(argv)

    rows = []
    for m in [x.strip() for x in a.markets.split(",") if x.strip()]:
        print(f"\n=== {m} ===", flush=True)
        r = run_one(m, n=a.n, start=a.start, horizon=a.horizon)
        if r is None:
            print("  跳过（无该市场池）")
            continue
        if not r.get("ok"):
            print(f"  失败: {r.get('error')}")
            rows.append(r)
            continue
        print(f"  T={r['T']} N={r['N']} S={r['S']}")
        print(f"  IC={r['ic']:+.4f} (t={r['t_ic']:+.2f}, n={r['n_ic']})  "
              f"价差={r['spread']*100:+.4f}% (t={r['t_spread']:+.2f})", flush=True)
        rows.append(r)

    ok = [r for r in rows if r.get("ok")]
    print("\n" + "=" * 62)
    print(f"{'市场':<6}{'IC':>10}{'t':>8}{'价差%':>10}{'t':>8}{'N':>6}")
    for r in ok:
        print(f"{r['market']:<6}{r['ic']:>+10.4f}{r['t_ic']:>8.2f}"
              f"{r['spread']*100:>+10.4f}{r['t_spread']:>8.2f}{r['N']:>6}")
    if ok:
        neg = [r for r in ok if r["ic"] < 0 and abs(r["t_ic"]) >= 2]
        pos = [r for r in ok if r["ic"] > 0 and abs(r["t_ic"]) >= 2]
        print(f"\n显著为负: {len(neg)}/{len(ok)}   显著为正: {len(pos)}/{len(ok)}")
        if len(neg) == len(ok):
            print("→ 三个市场都显著为负：信号集本身有问题，换市场救不了")
        elif len(pos) >= 1 and len(neg) >= 1:
            print("→ 市场间分化：部分是市场结构问题")
        else:
            print("→ 都不显著：信号集没有信息量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
