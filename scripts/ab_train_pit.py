"""
双轨训练复核（决策 1 的 C 方案）—— 同一份数据，快照 vs 点对点，各训一次，比 holdout IC。

## 为什么要有这个脚本

评审 R4 的 A/B 是**一次性手动跑的**（写在一段 heredoc 里），跑完数字留在报告里，
下次谁想复核都得重写。这个脚本把它固化：

    python -m scripts.ab_train_pit --horizon 5 --stocks 300 --trees 250

跑完写 `data/state/ab_train.json`，并打印对比表。**不改任何生产模型**，
只为了让人在"切不切默认"这件事上有一份可重跑的证据。

## 指标口径（与报告一致，必须一起读）

- IC = **逐日横截面 Spearman** 的均值；`se = std/sqrt(天数)`，`t = IC/se`；
- 不含成本、不含组合构建、未做多重检验校正；
- 训练/测试切分：`train_end` 之前训练，`test_start ~ test_end` 测试。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_ltb():
    spec = importlib.util.spec_from_file_location(
        "ltb", os.path.join(ROOT, "scripts", "lowturnover_bt.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ic(d) -> dict:
    sc, y = d["scores"], d["y"]
    ics = []
    for t in range(sc.shape[0]):
        s, r = sc[t], y[t]
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < 30:
            continue
        a = np.argsort(np.argsort(s[m]))
        b = np.argsort(np.argsort(r[m]))
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    ics = np.array(ics)
    ic = float(np.nanmean(ics)) if ics.size else float("nan")
    se = float(np.nanstd(ics) / np.sqrt(ics.size)) if ics.size else float("nan")
    return {"ic": round(ic, 5), "n_days": int(ics.size),
            "se": round(se, 5),
            "t": round(ic / se, 2) if se and np.isfinite(se) and se > 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="快照 vs 点对点 双轨训练复核")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--train-end", default="2019-12-31")
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--test-end", default="2026-11-06")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "state", "ab_train.json"))
    a = ap.parse_args(argv)

    ltb = _load_ltb()
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "config": {"horizon": a.horizon, "stocks": a.stocks, "trees": a.trees,
                      "train_end": a.train_end, "test_start": a.test_start,
                      "test_end": a.test_end},
           "runs": {}}

    for tag, pit in (("snapshot", False), ("pit", True)):
        t0 = time.time()
        d = ltb.prepare("csi300", stocks=a.stocks, train_end=a.train_end,
                        test_start=a.test_start, test_end=a.test_end,
                        horizon=a.horizon, trees=a.trees, verbose=False, pit=pit)
        r = _ic(d)
        r.update({"codes": len(d.get("codes") or []),
                  "seconds": round(time.time() - t0, 1)})
        out["runs"][tag] = r
        print(f"  {tag:<9} IC={r['ic']:+.5f}  t={r['t']:+.2f}  "
              f"n={r['n_days']}  codes={r['codes']}  {r['seconds']:.0f}s", flush=True)

    s, p = out["runs"].get("snapshot", {}), out["runs"].get("pit", {})
    if s.get("ic") and p.get("ic"):
        out["delta_ic"] = round(p["ic"] - s["ic"], 5)
        out["delta_pct"] = round((p["ic"] / s["ic"] - 1) * 100, 1) if s["ic"] else None
        print(f"\n  点对点 − 快照：ΔIC {out['delta_ic']:+.5f}（{out['delta_pct']:+.1f}%）")
    out["verdict"] = ("pit_better" if (p.get("ic") or 0) > (s.get("ic") or 0)
                      else "snapshot_better_or_equal")
    out["caveat"] = ("单次运行、单一种子；IC 为逐日横截面 Spearman 均值；"
                     "不含成本与组合构建，未做多重检验校正")

    try:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, a.out)
        print(f"  写入 {a.out}")
    except Exception as e:  # noqa: BLE001
        print(f"  写盘失败：{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
