"""
信号预热探测 —— 回答一个问题：**信号能只算最后 N 根吗？**

## 为什么现在可以问这个问题

`_roll_sum` 已从"前缀和差分"改成滑窗求和（评审 R48）：只依赖窗口内 w 个数，
与序列长度无关，实测截断差 **0 格**。于是"少算历史"第一次有了逐位相等的可能。

但"可能"不等于"成立"：只要还有任何一个信号用了 expanding 统计、或者窗口比
候选预热更长，尾巴就会变。本脚本逐信号、逐股地验证，给出**最小安全预热**。

## 判据（与 INV-1 同口径）

对每只票：
    全量  = compute_signals(df)
    截断  = compute_signals(df.tail(W + tail))
比较两者最后 tail 行的 DIR（必须逐位相等）与 STR（必须逐位相等，不是"接近"）。

## 用法

    python -m tools.signal_warmup_probe                 # 默认 10 只票、tail=40
    python -m tools.signal_warmup_probe --stocks 20 --tail 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore.data import get_pool, load_kline  # noqa: E402
from qcore.signals import compute_signals, signal_zoo  # noqa: E402

#: 候选预热长度（交易日）。250 覆盖年线类，520 覆盖两年类。
WARMUPS = (120, 180, 250, 320, 400, 520)


def _cmp(full, part, tail: int, sids: List[str]) -> Dict[str, Dict]:
    """比较最后 tail 行。返回 {sid: {dir_diff, str_diff}}。"""
    out = {}
    d0, s0 = full
    d1, s1 = part
    for sid in sids:
        if sid not in d0.columns or sid not in d1.columns:
            out[sid] = {"dir_diff": -1, "str_diff": 1.0, "missing": True}
            continue
        a = d0[sid].to_numpy()[-tail:]
        b = d1[sid].to_numpy()[-tail:]
        sa = s0[sid].to_numpy()[-tail:]
        sb = s1[sid].to_numpy()[-tail:]
        dd = int(np.nansum(np.nan_to_num(a, nan=9) != np.nan_to_num(b, nan=9)))
        sd = float(np.nanmax(np.abs(np.nan_to_num(sa, nan=9) - np.nan_to_num(sb, nan=9))))
        out[sid] = {"dir_diff": dd, "str_diff": sd}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="信号预热探测")
    ap.add_argument("--stocks", type=int, default=10)
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    sids = [s.sid for s in signal_zoo()]
    codes = get_pool(top_n=max(a.stocks, 20))["code"].tolist()[: a.stocks]
    print(f"探测：{len(codes)} 只票 × {len(WARMUPS)} 个预热候选 × {len(sids)} 个信号，"
          f"tail={a.tail}", flush=True)

    worst: Dict[int, Dict[str, Dict]] = {w: {} for w in WARMUPS}
    for i, code in enumerate(codes, 1):
        try:
            df = load_kline(code)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < max(WARMUPS) + a.tail + 20:
            continue
        full = compute_signals(df)
        for w in WARMUPS:
            part = compute_signals(df.tail(w + a.tail).reset_index(drop=True))
            r = _cmp(full, part, a.tail, sids)
            for sid, v in r.items():
                cur = worst[w].get(sid)
                if cur is None:
                    worst[w][sid] = dict(v)
                else:
                    cur["dir_diff"] = max(cur["dir_diff"], v["dir_diff"])
                    cur["str_diff"] = max(cur["str_diff"], v["str_diff"])
        if i % 5 == 0:
            print(f"  ... {i}/{len(codes)}", flush=True)

    report = []
    for w in WARMUPS:
        bad = [s for s, v in worst[w].items() if v["dir_diff"] != 0 or v["str_diff"] > 0]
        report.append({"warmup": w, "n_signals": len(worst[w]),
                       "n_mismatch": len(bad),
                       "mismatch": sorted(bad)[:12]})
        print(f"  warmup={w:<4} 不一致信号 {len(bad):>3}/{len(worst[w])}"
              + (f"  例：{', '.join(sorted(bad)[:6])}" if bad else "  ✅ 全部逐位相等"),
              flush=True)

    ok = [r for r in report if r["n_mismatch"] == 0]
    best = min(ok, key=lambda r: r["warmup"]) if ok else None
    print(f"\n最小安全预热：{best['warmup'] if best else '无（本候选集内做不到逐位相等）'}")
    if a.json:
        print(json.dumps({"tail": a.tail, "report": report, "best": best},
                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
