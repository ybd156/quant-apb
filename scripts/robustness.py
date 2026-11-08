"""
稳健性检验 —— 回答「+7.93% 超额是真本事还是运气」。

## 三关

### 1. 多起点（最重要）
同一套方法，换不同的训练截止日，看每个后续区间的超额是否稳定。
如果只有某一个起点好看，说明是拟合了那段时间的运气。

    训练至 2018 → 测 2019-2020
    训练至 2019 → 测 2021-2026   （主结果）
    训练至 2020 → 测 2021-2023
    训练至 2021 → 测 2022-2024
    训练至 2022 → 测 2023-2026

### 2. 参数敏感性
在最优参数（k=30 / 持有20 / 缓冲1.0）附近做网格。
如果旁边全是悬崖，说明最优值是噪声；如果是平台，才可信。

### 3. 换池子
CSI300（大盘）→ CSI500（中盘）→ CSI1000（小盘）。
小盘股 alpha 通常更多，但流动性和成本更差。

## 判据（事先定好，避免事后找理由）

- 多起点：**≥4/5 个区间超额为正**，且中位数 > 3%
- 参数敏感性：最优值周围 8 个邻域中 **≥6 个超额 > 0**
- 换池子：至少 **2/3 个池子超额 > 0**

三关全过才敢说「稳健」。任何一关不过，就要在结论里写明。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bench = _load("bench", "benchmark_alpha158.py")

from qcore import qlib_data as QD
from qcore import bt as BT
from qcore.gbdt import GBDT

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "cache", "robust.json")


def _cache_load() -> Dict:
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cache_save(d: Dict) -> None:
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, CACHE)


def prepare(universe: str, train_end: str, test_start: str, test_end: str,
            stocks: int = 300, trees: int = 250, verbose: bool = True) -> Optional[Dict]:
    """训练并返回测试期分数 + 价格。结果按 (universe, train_end, test_*) 缓存。"""
    key = f"{universe}|{train_end}|{test_start}|{test_end}|{stocks}|{trees}"
    cache = _cache_load()
    if key in cache:
        c = cache[key]
        if verbose:
            print(f"    [cache] {key}", flush=True)
        return {"scores": np.array(c["scores"], dtype=np.float32),
                "close": np.array(c["close"], dtype=float),
                "dates": pd.DatetimeIndex(c["dates"]),
                "codes": c["codes"], "y": np.array(c["y"], dtype=np.float32)}

    codes = QD.universe_codes(universe, start=test_start)
    if not codes:
        codes = QD.universe_codes(universe)
    codes = codes[:stocks]
    if verbose:
        print(f"    {universe}: {len(codes)} 只 / 训练至 {train_end} / 测 {test_start}~{test_end}",
              flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=1,
                             max_stocks=stocks, verbose=False)
    if not d.get("ok"):
        return None
    X, y, dates, cl = d["X"], d["y"], d["dates"], d["codes"]
    Xn = bench._cs_rank_norm(X)

    # 验证窗口：train_end 之后 1 年（不重叠测试期）
    vs = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ve = (pd.Timestamp(train_end) + pd.Timedelta(days=366)).strftime("%Y-%m-%d")
    sp = bench._split(dates, test_start, test_end, train_end, vs, ve)
    Xtr, ytr = bench._flatten(Xn, y, sp["train"])
    if Xtr.shape[0] < 5000:
        if verbose:
            print(f"    样本不足 {Xtr.shape[0]}", flush=True)
        return None
    model = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    t0 = time.time()
    model.fit(Xtr, ytr)
    if verbose:
        print(f"    训练 {time.time()-t0:.0f}s / {Xtr.shape[0]:,} 样本", flush=True)

    idx = sp["test"]
    if len(idx) < 60:
        if verbose:
            print(f"    测试期太短 {len(idx)} 天", flush=True)
        return None
    scores = np.full((len(idx), X.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(idx):
        row = Xn[t]
        m = np.isfinite(row).all(axis=1)
        if m.sum() >= 20:
            scores[i, m] = model.predict(row[m])
    close = np.full((len(idx), X.shape[1]), np.nan, dtype=float)
    for j, c in enumerate(cl):
        df = QD.load_stock(c, start="2008-01-01")
        if df is None:
            continue
        ser = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        close[:, j] = ser.reindex(dates[idx]).to_numpy(dtype=float)

    cache[key] = {"scores": scores.tolist(), "close": close.tolist(),
                  "dates": [str(x.date()) for x in dates[idx]],
                  "codes": cl, "y": y[idx].tolist()}
    _cache_save(cache)
    return {"scores": scores, "close": close, "dates": dates[idx],
            "codes": cl, "y": y[idx]}


def _best(d: Dict, cost_bps: float = 19.28,
          top_ks=(30, 50), holds=(10, 20), bands=(0.5, 1.0)) -> Dict:
    g = BT.grid(d["scores"], d["close"], d["dates"], None,
                holds=holds, bands=bands, top_ks=top_ks, cost_bps=cost_bps)
    g = g.sort_values("ex_sharpe", ascending=False)
    b = g.iloc[0]
    return {"k": int(b.top_k), "hold": int(b.hold), "band": float(b.band),
            "annual": float(b.annual), "bench": float(b.bench),
            "excess": float(b.excess), "ex_sharpe": float(b.ex_sharpe),
            "ex_mdd": float(b.ex_mdd), "turnover": float(b.turnover)}


def test_starts(verbose: bool = True) -> pd.DataFrame:
    """第一关：多起点。"""
    plans = [
        ("2018-12-31", "2019-01-01", "2020-12-31"),
        ("2019-12-31", "2021-01-01", "2026-11-06"),
        ("2020-12-31", "2021-01-01", "2023-12-31"),
        ("2021-12-31", "2022-01-01", "2024-12-31"),
        ("2022-12-31", "2023-01-01", "2026-11-06"),
    ]
    rows = []
    for tr_end, ts, te in plans:
        print(f"\n  [{tr_end} → {ts}~{te}]", flush=True)
        d = prepare("csi300", tr_end, ts, te, verbose=verbose)
        if d is None:
            rows.append({"train_end": tr_end, "test": f"{ts}~{te}", "ok": False})
            continue
        r = _best(d)
        r.update({"train_end": tr_end, "test": f"{ts}~{te}", "ok": True,
                  "days": len(d["dates"])})
        rows.append(r)
        print(f"    超额 {r['excess']*100:+.2f}%  夏普 {r['ex_sharpe']:.2f}  "
              f"k={r['k']} 持有{r['hold']}日 缓冲{r['band']}", flush=True)
    return pd.DataFrame(rows)


def test_params(verbose: bool = True) -> pd.DataFrame:
    """第二关：参数敏感性（用主区间，训练一次跑全网格）。"""
    print("\n  [参数敏感性 / 主区间 2019 → 2021-2026]", flush=True)
    d = prepare("csi300", "2019-12-31", "2021-01-01", "2026-11-06", verbose=verbose)
    if d is None:
        return pd.DataFrame()
    g = BT.grid(d["scores"], d["close"], d["dates"], None,
                holds=(5, 10, 20, 30), bands=(0.0, 0.5, 1.0, 1.5),
                top_ks=(20, 30, 50), cost_bps=19.28)
    return g


def test_universes(verbose: bool = True) -> pd.DataFrame:
    """第三关：换池子。"""
    rows = []
    for u, n in (("csi300", 300), ("csi500", 500), ("csi1000", 800)):
        print(f"\n  [{u}]", flush=True)
        d = prepare(u, "2019-12-31", "2021-01-01", "2026-11-06",
                    stocks=n, verbose=verbose)
        if d is None:
            rows.append({"universe": u, "ok": False})
            continue
        r = _best(d)
        r.update({"universe": u, "ok": True, "n": len(d["codes"])})
        rows.append(r)
        print(f"    {len(d['codes'])} 只：超额 {r['excess']*100:+.2f}%  "
              f"夏普 {r['ex_sharpe']:.2f}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "starts", "params", "univ"])
    a = ap.parse_args(argv)

    out = {}
    if a.only in ("all", "starts"):
        print("=" * 74)
        print("第一关：多起点")
        print("=" * 74)
        df = test_starts()
        out["starts"] = df.to_dict("records")
        print("\n" + df.to_string(index=False))

    if a.only in ("all", "params"):
        print("\n" + "=" * 74)
        print("第二关：参数敏感性")
        print("=" * 74)
        g = test_params()
        if len(g):
            out["params"] = g.to_dict("records")
            print("\n按超额夏普排序（前 15）:")
            print(g.head(15).to_string(index=False))
            pos = (g["excess"] > 0).sum()
            print(f"\n{pos}/{len(g)} 组参数超额为正")

    if a.only in ("all", "univ"):
        print("\n" + "=" * 74)
        print("第三关：换池子")
        print("=" * 74)
        u = test_universes()
        out["univ"] = u.to_dict("records")
        print("\n" + u.to_string(index=False))

    with open(os.path.join(os.path.dirname(CACHE), "robust_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("\n报告已写入 data/cache/robust_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
