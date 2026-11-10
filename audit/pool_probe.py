"""
生存者偏差探针 —— 回答「点对点股票池到底有没有生效」。

## 为什么需要它

INV-6 的自检（`qcore/invariants.py::inv6_pit_universe`）只证明
「UNIV 掩码非退化（N >= n_univ）」，**不证明池成员随时间变化**。
而真正的生存者偏差来自池的**构造**：`get_pool()` 用「今日市值 top-N」取候选池，
退市股结构性缺席。掩码再怎么点对点，也只能在幸存者池内排名。

本探针给三个可观测指标：

1. **池快照**：`data/raw/meta/pool.csv` 的行数、市值下限、ST/退市名称计数。
   行数恰好等于 `max_pages * 100` 且市值下限很高 → 池是「今日大市值快照」。
2. **UNIV 周转率**：从真实面板算 UNIV 掩码在首尾交易日的 Jaccard 相似度。
   Jaccard == 1.0 说明池成员恒定 = 掩码没在换人 = 点对点没生效。
3. **退市股覆盖**：本地无退市名单，**明确标 not_measured**，不猜。

跑法：
    python -m audit.pool_probe
    python -m audit.pool_probe --codes 30 --start 2020-01-01 --json

退出码：0 = 通过；1 = 发现「池静态」（点对点未生效）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _pool_snapshot() -> dict:
    p = os.path.join(ROOT, "data", "raw", "meta", "pool.csv")
    if not os.path.isfile(p):
        return {"ok": False, "reason": "no_pool_csv", "path": p}
    import pandas as pd
    d = pd.read_csv(p, dtype={"code": str})
    out = {"ok": True, "path": p, "n_rows": int(len(d)),
           "n_cols": int(d.shape[1])}
    if "mktcap" in d.columns:
        mc = pd.to_numeric(d["mktcap"], errors="coerce") / 1e8
        out["mktcap_yi"] = {"min": round(float(mc.min()), 1),
                            "median": round(float(mc.median()), 1),
                            "max": round(float(mc.max()), 1)}
    if "name" in d.columns:
        nm = d["name"].astype(str)
        out["n_st_or_delisted"] = int(nm.str.contains("ST|退", na=False).sum())
    if "code" in d.columns:
        pre = d["code"].astype(str).str[0]
        out["prefix_counts"] = {k: int(v) for k, v in pre.value_counts().items()}
    out["page_cap_hint"] = (out["n_rows"] % 100 == 0)
    return out


def _univ_turnover(n_codes: int, start: str) -> dict:
    """UNIV 掩码的首尾 Jaccard + 相邻窗口周转。"""
    from qcore.dataset import build_dataset
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
    if len(files) < 5:
        return {"ok": False, "reason": "no_local_kline",
                "hint": "先跑一次 python -m qcore.pipeline 生成 data/raw/kline"}
    codes = [os.path.basename(f).split("_")[0] for f in files[:n_codes]]
    p3d = build_dataset(codes, start=start, horizon=5, verbose=False)
    import numpy as np
    U = p3d.UNIV                      # (T,N) bool
    dates = np.asarray(p3d.dates)
    T, N = U.shape

    def _set(t):
        return {p3d.codes[i] for i in np.flatnonzero(U[t])}

    # 跳过预热期：amt60 的 min_periods=20，前 20+ 天 rk 全 NaN → UNIV 全 False，
    # 直接拿第 0 天比会把「预热」误读成「池在换人」（Jaccard=0 的假信号）。
    warm = min(60, max(0, T - 2))
    first, last = _set(warm), _set(T - 1)
    mid = _set((warm + T - 1) // 2)
    j_first_last = (len(first & last) / len(first | last)) if (first | last) else None
    j_first_mid = (len(first & mid) / len(first | mid)) if (first | mid) else None
    # 逐日变动：平均每天有多少只进出
    churn = []
    for t in range(1, T):
        a, b = _set(t - 1), _set(t)
        if a or b:
            churn.append(len(a ^ b))
    # 掩码到底有没有「排除」东西？这才是「点对点是否生效」的直接判据。
    # N <= n_univ 时 rk 全 ≤ n_univ，UNIV 恒等于 isfinite(rk)，等于没排除。
    # 注意：不能用 VALID|UNIV 反推 traded —— VALID ⊆ UNIV，所以并集恒等于 UNIV，
    # 排除率会永远算出 0.00%（本探针的第一版就踩了这个坑）。
    # traded 必须从价格矩阵重算。
    traded = np.isfinite(p3d.OPEN) & np.isfinite(p3d.CLOSE)
    n_traded = int(traded.sum())
    n_univ = int((traded & U).sum())
    excl = (1.0 - n_univ / n_traded) if n_traded else None
    return {"ok": True, "codes": codes, "T": int(T), "N": int(N),
            "n_traded_cells": n_traded, "n_univ_cells": n_univ,
            "exclusion_rate": None if excl is None else round(excl, 4),
            "date_from": str(dates[0])[:10], "date_to": str(dates[-1])[:10],
            "univ_size_first": len(first), "univ_size_last": len(last),
            "jaccard_first_last": None if j_first_last is None else round(j_first_last, 4),
            "jaccard_first_mid": None if j_first_mid is None else round(j_first_mid, 4),
            "avg_daily_churn": round(sum(churn) / len(churn), 2) if churn else None,
            "univ_full_days": int((U.sum(axis=1) == N).sum()),
            "mask_degenerate": bool((U[warm:].sum(axis=1) == N).mean() > 0.95),
            "note": (f"用 {len(codes)} 只本地票的小面板测算；"
                     f"UNIV 规模受 n_univ 与 N 的关系影响，"
                     f"N 接近 n_univ 时掩码近似恒真（见 INV-6 警告）")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生存者偏差探针")
    ap.add_argument("--codes", type=int, default=30, help="本地票数（默认 30）")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    snap = _pool_snapshot()
    turn = _univ_turnover(a.codes, a.start)

    findings = []
    verdict = "unknown"

    if snap.get("ok"):
        if snap.get("page_cap_hint") and snap["n_rows"] >= 500:
            findings.append(
                f"pool.csv 恰好 {snap['n_rows']} 行（100 的整数倍）→ 池是"
                f"「东财 clist 分页上限」截出来的今日市值快照，不是全市场")
        mc = snap.get("mktcap_yi") or {}
        if mc.get("min") is not None and mc["min"] > 100:
            findings.append(
                f"池内最小市值 {mc['min']} 亿 → 小市值与退市股结构性缺席")
        if snap.get("n_st_or_delisted", 0) <= 2:
            findings.append(
                f"池内含 ST/退市名称 {snap.get('n_st_or_delisted', 0)} 只 → "
                f"退市股基本不进池（生存者偏差的直接证据）")

    if turn.get("ok"):
        j = turn.get("jaccard_first_last")
        er = turn.get("exclusion_rate")
        if er is not None and er < 0.02:
            findings.append(
                f"掩码排除率仅 {er:.2%}（{turn['n_traded_cells']} 个可交易股票-日里"
                f"只挡掉 {turn['n_traded_cells'] - turn['n_univ_cells']} 个）→ "
                f"N={turn['N']} 接近/低于 n_univ，点对点选池在本面板**实质无效**")
            verdict = "mask_noop"
        elif turn.get("mask_degenerate"):
            # N <= n_univ 时掩码恒真 —— 这是 INV-6 自己写明的失效情形。
            # 它不是「代码 bug」，而是「这个池太小，点对点无从生效」。
            findings.append(
                f"N={turn['N']} ≤ n_univ → 掩码在 {turn['univ_full_days']}/{turn['T']} 天"
                f"选中全部股票（退化），点对点选池在本面板**无从生效**")
            verdict = "mask_degenerate"
        elif j is not None and j >= 0.999:
            findings.append(
                f"UNIV 首尾 Jaccard = {j} 且未选中全部 → 池成员恒定，点对点选池**未生效**")
            verdict = "pool_static"
        elif j is not None:
            findings.append(
                f"UNIV 首尾 Jaccard = {j}，平均每日进出 {turn['avg_daily_churn']} 只 → 掩码在换人")
            verdict = "ok"
    else:
        findings.append(f"UNIV 周转未测：{turn.get('reason')}")

    findings.append("退市股覆盖：not_measured（本地无退市名单，需外部数据源）")

    res = {"ok": verdict not in ("pool_static", "mask_noop"), "verdict": verdict,
           "pool_snapshot": snap, "univ_turnover": turn,
           "findings": findings,
           "limitation": ("本探针只能证明「掩码是否在换人」与「池是不是今日快照」；"
                          "无法量化生存者偏差的收益幅度 —— 那需要退市股全历史行情。")}
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print("=" * 72)
        print("生存者偏差探针")
        print("=" * 72)
        if snap.get("ok"):
            print(f"池快照: {snap['n_rows']} 行 · 市值 {snap.get('mktcap_yi')} 亿 · "
                  f"ST/退 {snap.get('n_st_or_delisted')} 只")
        if turn.get("ok"):
            print(f"UNIV  : {turn['T']} 天 × {turn['N']} 只 · "
                  f"{turn['date_from']}~{turn['date_to']} · "
                  f"首尾 Jaccard {turn['jaccard_first_last']} · "
                  f"日均进出 {turn['avg_daily_churn']} 只")
        print()
        for f in findings:
            print("  · " + f)
        print(f"\nVERDICT: {verdict}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
