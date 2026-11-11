"""
tick 批量容量探测（评审 R46）—— 一次能要多少只？会不会被限流？

## 背景

`qcore.tick.snapshot(codes)` 把**所有代码塞进一个 URL**（`qt.gtimg.cn/q=sh600519,sz000001,...`），
没有分块。当前盘中轮询只覆盖自选股（53 只），要扩到全池（300 只）之前必须先知道：

1. 单次请求最多能吃多少只（URL 长度 / 服务端上限）；
2. 300 只时延迟多少、返回多少行、缺多少；
3. 降级链（腾讯 → 新浪 → 东财）在 300 只时是否仍可用。

## 判据

- `rows/n` 覆盖率 ≥ 98% 视为可用；
- 单次延迟 < 3s 视为可接受（盘中 3s 一轮）；
- 任何 size 出现 `all_sources_failed` 就记下来 —— 那是硬上限。

## 用法

    python -m tools.tick_capacity_probe                 # 60/150/300 三档
    python -m tools.tick_capacity_probe --sizes 100,300,600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore import tick as T  # noqa: E402
from qcore.data import get_pool  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="tick 批量容量探测")
    ap.add_argument("--sizes", default="60,150,300")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args(argv)

    sizes = [int(x) for x in str(a.sizes).split(",") if x.strip()]
    pool = get_pool(top_n=max(sizes) + 50)
    codes = pool["code"].tolist()

    out = []
    for n in sizes:
        sub = codes[:n]
        url_len = len("https://qt.gtimg.cn/q=" + ",".join(T._norm(c) for c in sub))
        best = None
        for k in range(a.repeat):
            t0 = time.time()
            r = T.snapshot(sub)
            dt = time.time() - t0
            rec = {"n": n, "attempt": k + 1, "ok": bool(r.get("ok")),
                   "source": r.get("source"), "rows": r.get("n"),
                   "coverage": round((r.get("n") or 0) / n, 4) if r.get("ok") else 0.0,
                   "missing": len(r.get("missing") or []),
                   "elapsed_s": round(dt, 2), "url_chars": url_len,
                   "stale": r.get("stale"), "reason": r.get("reason")}
            out.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            if best is None or (rec["coverage"], -rec["elapsed_s"]) > (best["coverage"], -best["elapsed_s"]):
                best = rec
            time.sleep(0.5)

    print("\n=== 汇总 ===")
    for n in sizes:
        rs = [r for r in out if r["n"] == n and r["ok"]]
        if not rs:
            print(f"{n:>4} 只：全部失败")
            continue
        cov = min(r["coverage"] for r in rs)
        lat = max(r["elapsed_s"] for r in rs)
        src = sorted({r["source"] for r in rs})
        print(f"{n:>4} 只：最低覆盖 {cov*100:.1f}% · 最长 {lat:.2f}s · 源 {src} "
              f"· URL {rs[0]['url_chars']} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
