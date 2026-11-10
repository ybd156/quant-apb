"""
价格口径探针 —— 东财 fqt=2 到底是乘法复权还是加法复权，以及它会不会污染 Rank IC。

## 背景

AGENTS INV-5 断言「fqt=2 是后复权（乘法复权）」。Data Engineer 的实测给出反例：
601088 的 `h2 − raw` 在连续 10 个交易日**恒等于 20.93**，2560 天里只变 13 次
（= 除息日）。这是**加法复权**的签名：`p_adj(t) = p_raw(t) + 累计分红(t)`。

后果：`r_adj = Δp / (p_raw + C)`，分母被分红抬高 → 收益率被按个股压缩，
压缩因子 = p_raw / (p_raw + C)，随个股与时间变化。

## 这个探针要回答的关键问题

Rank IC 是**横截面秩相关**。所以真正决定「IC 有没有被污染」的不是压缩多少，
而是**压缩因子在不同股票之间是否一致**：

  · 若跨股近似一致 → 只等比缩放，秩不变 → **IC 基本不受影响**，受影响的是幅度
  · 若跨股差异大 → 会**重排**股票 → IC 被系统性偏置

因此核心指标是「同一交易日、同一 5 日窗口内，两种口径下前向收益的**横截面 Spearman**」。

## 跑法

    python -m audit.basis_probe --codes 40
    python -m audit.basis_probe --codes 60 --json

首次运行会为每只票拉一次 fqt=0（约 0.6–1.6s/只），写入 `data/raw/kline/{code}_h0.csv`；
之后再跑就是本地读取。

## 边界（不许越界声称）

本探针**不重算 IC**。它只回答「压缩因子是否跨股一致」与「秩是否被重排」。
真正的 IC 前后对比需要全池 fqt=0 + 正确处理除息（需要分红数据），属独立立项。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

H = 5          # 前向窗口（与基线 h=5 对齐）
EPS = 1e-6


def _pair(code: str):
    """取同一只票的 fqt=2 与 fqt=0，按日期对齐。"""
    from qcore import data as D
    a = D.load_kline(code, adjust=2)
    b = D.load_kline(code, adjust=0)
    a = a[["date", "open", "close"]].rename(columns={"open": "o2", "close": "c2"})
    b = b[["date", "open", "close"]].rename(columns={"open": "o0", "close": "c0"})
    m = a.merge(b, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return m


def _one(m: pd.DataFrame) -> dict:
    """单只票：偏移量签名 + 压缩因子 + 除息日。"""
    off = (m["c2"] - m["c0"]).to_numpy(dtype=np.float64)
    d_off = np.diff(off)
    exdiv = np.abs(d_off) > 1e-4                 # 偏移量变化 = 除息（或送转）
    r2 = m["c2"].pct_change().to_numpy(dtype=np.float64)
    r0 = m["c0"].pct_change().to_numpy(dtype=np.float64)
    ok = (~np.r_[False, exdiv]) & np.isfinite(r2) & np.isfinite(r0) & (np.abs(r0) > EPS)
    ratio = r2[ok] / r0[ok]
    # 偏移量恒定的最大连续段长度（加法复权的签名：段内 offset 不变）
    seg, best = 0, 0
    for flag in (~np.r_[True, exdiv]):
        seg = seg + 1 if flag else 0
        best = max(best, seg)
    off_max = float(np.abs(off).max()) if len(off) else 0.0
    flat_share = float(1.0 - exdiv.mean()) if len(exdiv) else 0.0
    # 分类（互斥）：
    #   none          offset 恒为 0            -> 该股从未除权/分红
    #   additive      offset 恒定但非 0（>=90% 交易日不变）-> 加法复权
    #   multiplicative 其余                   -> 乘法复权（offset = raw·(k-1) 随价格每日变）
    if off_max < 1e-6:
        kind = "none"
    elif flat_share >= 0.90:
        kind = "additive"
    else:
        kind = "multiplicative"
    return {
        "kind": kind,
        "offset_max": off_max,
        "n_days": int(len(m)),
        "n_exdiv": int(exdiv.sum()),
        "n_ratio": int(ok.sum()),
        "compress_median": float(np.median(ratio)) if ok.sum() else None,
        "offset_flat_share": float(1.0 - exdiv.mean()) if len(exdiv) else None,
        "longest_flat_run": int(best),
        "offset_first": float(off[0]) if len(off) else None,
        "offset_last": float(off[-1]) if len(off) else None,
    }


def _xs_spearman(pairs: dict, h: int) -> dict:
    """每个交易日，两种口径下 h 日前向收益的横截面 Spearman。"""
    wide = {}
    for code, m in pairs.items():
        o2 = m["o2"].to_numpy(dtype=np.float64)
        o0 = m["o0"].to_numpy(dtype=np.float64)
        d = pd.DatetimeIndex(m["date"])
        n = len(m)
        if n < h + 3:
            continue
        # r_h(t) = O[t+1+h]/O[t+1] - 1（与 dataset.RET 的口径一致）
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.full(n, np.nan)
            r0 = np.full(n, np.nan)
            hi = n - 1 - h
            r2[:hi] = o2[1 + h:1 + h + hi] / o2[1:1 + hi] - 1.0
            r0[:hi] = o0[1 + h:1 + h + hi] / o0[1:1 + hi] - 1.0
        for i, dt in enumerate(d):
            if not (np.isfinite(r2[i]) and np.isfinite(r0[i])):
                continue
            slot = wide.setdefault(dt, {})
            slot[code] = (r2[i], r0[i])
    rhos = []
    ns = []
    for dt, slot in wide.items():
        if len(slot) < 10:                        # 截面太薄不做秩相关
            continue
        a = np.array([v[0] for v in slot.values()])
        b = np.array([v[1] for v in slot.values()])
        if np.std(a) < EPS or np.std(b) < EPS:
            continue
        # Spearman = 秩上的 Pearson。pandas 的 method="spearman" 会去 import scipy，
        # 而本项目禁止 scipy —— 自己算，两行。
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
        ns.append(len(slot))
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    return {"n_days": int(rhos.size),
            "median_n_names": int(np.median(ns)) if ns else 0,
            "spearman_mean": float(rhos.mean()) if rhos.size else None,
            "spearman_median": float(np.median(rhos)) if rhos.size else None,
            "spearman_p05": float(np.percentile(rhos, 5)) if rhos.size else None,
            "frac_below_0_99": float((rhos < 0.99).mean()) if rhos.size else None}


def main(argv=None) -> int:
    try:                       # Windows 控制台是 GBK，'−' 等字符会 UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="价格口径探针（fqt=2 是否加法复权 / 秩是否被重排）")
    ap.add_argument("--codes", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
    if len(files) < 10:
        print("[SKIP] 本地行情不足（data/raw/kline/*_h2.csv < 10）")
        return 0
    codes = [os.path.basename(f).split("_")[0] for f in files[:a.codes]]

    pairs, per = {}, {}
    failed = []
    for c in codes:
        try:
            m = _pair(c)
        except Exception as e:  # noqa: BLE001
            failed.append({"code": c, "error": f"{type(e).__name__}: {e}"[:120]})
            continue
        if len(m) < 300:
            continue
        pairs[c] = m
        per[c] = _one(m)
    if not pairs:
        print("[SKIP] 无可用样本")
        return 0

    comp = np.array([v["compress_median"] for v in per.values()
                     if v["compress_median"] is not None])
    flat = np.array([v["offset_flat_share"] for v in per.values()
                     if v["offset_flat_share"] is not None])
    xs = _xs_spearman(pairs, H)
    kinds = {}
    for v in per.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    additive_codes = sorted(c for c, v in per.items() if v["kind"] == "additive")

    findings = []
    if comp.size:
        findings.append(
            f"压缩因子（r_h2/r_fqt0，非除息日）：中位 {np.median(comp):.4f} · "
            f"p05 {np.percentile(comp,5):.4f} · p95 {np.percentile(comp,95):.4f} · "
            f"{int((comp<0.9).sum())}/{comp.size} 只 <0.90")
    if flat.size:
        findings.append(
            f"偏移量 (h2−raw) 恒定的交易日占比（中位）{np.median(flat)*100:.1f}% —— "
            f"**这个数不能当加法复权证据**：从未分红/除权的股票 offset 恒为 0，"
            f"会把中位数拉高。判定必须看下面的逐股分类。")
    if xs.get("spearman_median") is not None:
        findings.append(
            f"两种口径下 {H} 日前向收益的横截面 Spearman：中位 "
            f"{xs['spearman_median']:.4f}（p05 {xs['spearman_p05']:.4f}，"
            f"{xs['n_days']} 天，中位截面 {xs['median_n_names']} 只）")
        findings.append(
            f"秩被重排的交易日占比（Spearman<0.99）："
            f"{xs['frac_below_0_99']*100:.1f}%")

    smed = xs.get("spearman_median")
    fbad = xs.get("frac_below_0_99")
    if smed is None:
        verdict = "insufficient"
    elif smed >= 0.995 and (fbad or 0) < 0.20:
        verdict = "rank_preserved"
    elif smed >= 0.98:
        # 中位很好但分布有厚尾：47% 的交易日秩相关 <0.99 就不能说「秩不变」
        verdict = "rank_mildly_reordered"
    else:
        verdict = "rank_materially_reordered"

    res = {
        "ok": True, "verdict": verdict,
        "n_codes": len(pairs), "failed": failed,
        "adjust_kind_counts": kinds,
        "additive_codes": additive_codes[:50],
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "compression": {
            "median": float(np.median(comp)) if comp.size else None,
            "p05": float(np.percentile(comp, 5)) if comp.size else None,
            "p95": float(np.percentile(comp, 95)) if comp.size else None,
            "n_below_0_90": int((comp < 0.90).sum()) if comp.size else 0,
            "n": int(comp.size),
        },
        "offset_flat_share_median": float(np.median(flat)) if flat.size else None,
        "xs_spearman": xs,
        "findings": findings,
        "interpretation": {
            "rank_preserved": "压缩因子跨股近似一致 → 秩不变 → **Rank IC 基本不受污染**，"
                              "受影响的只是收益幅度（多空年化的经济解读要打折）",
            "rank_mildly_reordered": "存在轻度重排 → Rank IC 可能有小幅偏置，需全量 A/B 确认",
            "rank_materially_reordered": "重排显著 → **Rank IC 被系统性偏置**，"
                                         "现有 IC 数字必须重算后才能引用",
            "insufficient": "样本不足，不下结论",
        }.get(verdict, ""),
        "limitation": ("本探针不重算 IC；它只证明「压缩是否存在」与「秩是否被重排」。"
                       "全量 A/B 需全池 fqt=0 + 分红数据以正确处理除息日。"),
    }
    # 落盘，供 qcore.claims 的 return_scale_median 口径自动结算
    try:
        from qcore import store
        store.write_json("basis", res)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 写 data/state/basis.json 失败: {e}")

    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print("=" * 74)
        print(f"价格口径探针 · {len(pairs)} 只 · h={H}")
        print("=" * 74)
        if kinds:
            print("  · 复权方式分类: " + " / ".join(
                f"{k}={v}" for k, v in sorted(kinds.items())))
        for f in findings:
            print("  · " + f)
        print(f"\nVERDICT: {verdict}")
        print("  含义: " + res["interpretation"])
        print("  边界: " + res["limitation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
