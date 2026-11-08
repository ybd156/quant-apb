# -*- coding: utf-8 -*-
"""A vs B 配对检验：同一天上两个模型的 IC 之差是否显著（HAC 口径）。"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qcore import stats as ST
from qcore.gbdt import GBDT
from qcore.neutral import industry_neutralize
from scripts.benchmark_alpha158 import _cs_rank_norm
from scripts.model_improve import (GBDT_KW, TEST, TRAIN, eval_days, flatten,
                                   idx_of, load_panel)

TREES = 250


def daily_ics(model, tdays):
    out = []
    for (t, Xday, yday) in tdays:
        m = np.isfinite(Xday).all(axis=1)
        if m.sum() < 20:
            out.append(np.nan)
            continue
        s = model.predict(Xday[m])
        r = yday[m]
        mm = np.isfinite(s) & np.isfinite(r)
        out.append(float(np.corrcoef(s[mm], r[mm])[0, 1]) if mm.sum() >= 20 else np.nan)
    return np.asarray(out)


def main() -> int:
    X, y, dates, codes, names, industry = load_panel()
    print("标准化 ...", flush=True)
    Xn = _cs_rank_norm(X)
    del X
    tr = idx_of(dates, TRAIN)
    te = idx_of(dates, TEST)
    Xtr, ytr, dtr, ctr = flatten(Xn, y, tr)
    tdays = eval_days(Xn, y, te)
    y_neu = industry_neutralize(y, industry).astype(np.float32)
    print(f"train={Xtr.shape[0]:,} test_days={len(tdays)}", flush=True)

    res = {}
    for tag, tgt in (("A", ytr), ("B", y_neu[dtr, ctr])):
        t0 = time.time()
        m = GBDT(n_estimators=TREES, random_state=42, **GBDT_KW).fit(Xtr, tgt)
        ics = daily_ics(m, tdays)
        res[tag] = ics
        print(f"  {tag}: IC={np.nanmean(ics):+.4f}  {time.time()-t0:.0f}s", flush=True)

    a, b = res["A"], res["B"]
    d = b - a
    st = ST.mean_tstat(d, lags=0)
    st_a = ST.mean_tstat(a, lags=0)
    st_b = ST.mean_tstat(b, lags=0)
    win = float(np.nanmean((d > 0).astype(float)))
    print()
    print("=" * 72)
    print("A vs B 配对检验（同一交易日上 IC_B - IC_A）")
    print("=" * 72)
    print(f"  A: IC={st_a['mean']:+.4f}  t={st_a['t']:+.2f}  p={st_a['p']:.2e}")
    print(f"  B: IC={st_b['mean']:+.4f}  t={st_b['t']:+.2f}  p={st_b['p']:.2e}")
    print(f"  Δ = B - A : mean={st['mean']:+.4f}  HAC t={st['t']:+.2f}  "
          f"p={st['p']:.2e}  95%CI=[{st['lo']:+.4f}, {st['hi']:+.4f}]")
    print(f"  日数={st['n']}  B 优于 A 的天数占比={win:.1%}")
    np.savez(os.path.join(ROOT, "data", "cache", "mi_ab_daily_ic.npz"),
             a=a, b=b, dates=dates.to_numpy()[te])
    print(f"  日频 IC 序列已存 data/cache/mi_ab_daily_ic.npz")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
