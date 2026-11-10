"""
复权口径对 Rank IC 的影响 —— 用同一份分数，在两个价格口径下各算一遍 IC。

## 为什么需要它

`audit/basis_probe.py` 已经证明：东财 fqt=2 对绝大多数股票是乘法复权
（压缩因子中位 1.0000），但两种口径下 5 日前向收益的横截面 Spearman
中位 0.9908、47% 交易日 <0.99 —— **秩有轻度重排**。

秩被重排，就意味着 Rank IC 会变。变多少？这个脚本量化它。

## 三个口径

| 代号 | 含义 |
|---|---|
| `h2`   | 现行：东财 fqt=2（后复权） |
| `raw`  | 东财 fqt=0（不复权）。**除息日会有虚假下跌**，是下界参考，不是真值 |
| `adj`  | 真值代理：非除息窗口用 raw，含除息事件的窗口用 h2（避开虚假跌幅） |

`h2` vs `adj` 的差 = **纯压缩效应**（这才是「复权口径」本身的影响）；
`h2` vs `raw` 的差 = 压缩 + 除息缺口的合计。

## 分数从哪来

不重算信号逻辑：直接用 `engine.run_walkforward` 产出的同一份 `scores`，
只把 `RET` 换成另一个口径 —— 这样两次 IC 的**唯一差异**就是价格口径。

## 跑法

    python -m audit.basis_ic --codes 60
    python -m audit.basis_ic --codes 60 --start 2016-01-01 --json

## 边界

- 面板规模受 `--codes` 限制（默认 60）。截面越薄，IC 噪声越大；
  这里只做**同一分数、两种口径的相对比较**，不当绝对基线引用。
- `adj` 是代理，不是精确真值：它假设「除息窗口内 h2 的收益率等于真实收益率」，
  这在乘法复权下成立，在少数加法复权股票上不成立（实测 2/36）。
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关（不依赖 scipy）。"""
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _daily_ic(scores: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """(N,T) scores × (T,N) ret → 每日横截面 Rank IC 序列。"""
    N, T = scores.shape
    out = np.full(T, np.nan)
    for t in range(T):
        s = scores[:, t]
        r = ret[t]
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() >= 8:
            out[t] = _spearman(s[m], r[m])
    return out


def _open0(codes, dates, lag=1, h=5):
    """用 fqt=0 的 OPEN 重建 (T,N) 前向收益矩阵；缺数据处 NaN。"""
    from qcore import data as D
    cols = {}
    for c in codes:
        try:
            df = D.load_kline(c, adjust=0)
        except Exception:
            continue
        s = pd.Series(df["open"].to_numpy(dtype=np.float64),
                      index=pd.DatetimeIndex(df["date"]))
        cols[c] = s
    if not cols:
        return None
    O = pd.DataFrame(cols).reindex(pd.DatetimeIndex(dates)).to_numpy(dtype=np.float64)
    T, N = O.shape
    r = np.full((T, N), np.nan)
    hi = T - 1 - h - lag
    with np.errstate(invalid="ignore", divide="ignore"):
        r[:max(hi, 0)] = (O[lag + h:lag + h + hi] / O[lag:lag + hi] - 1.0)
    return r


def _classify(codes):
    """逐股判定复权方式。返回 {code: 'multiplicative'|'additive'|'none'}。

    为什么不再检测「除息日」：乘法复权下 r_h2 恒等于真实总收益（数学恒等）——
      p_adj = p_raw·k(t)，r_true = p_adj[t+1]/p_adj[t] − 1 = r_h2
    所以对乘法复权的股票，"真值口径"就是 h2 本身，不需要修。真正需要修的是
    **加法复权**那少数股票（r_h2 = Δp/(p_raw+C)，被分红压缩）。
    检测除息日在数值上极不稳（CSV 只留 2 位小数，比值抖动 ≈1e-3），
    第一版因此把 93.8% 的单元格误标成除息 —— 直接放弃这条路，改用分类。
    """
    from qcore import data as D
    out = {}
    for c in codes:
        try:
            a = D.load_kline(c, adjust=2)[["date", "close"]]
            b = D.load_kline(c, adjust=0)[["date", "close"]]
        except Exception:
            continue
        m = a.merge(b, on="date", how="inner").sort_values("date")
        if len(m) < 60:
            continue
        off = (m["close_x"] - m["close_y"]).to_numpy(dtype=np.float64)
        d_off = np.abs(np.diff(off))
        off_max = float(np.abs(off).max()) if off.size else 0.0
        flat = float((d_off < 1e-4).mean()) if d_off.size else 0.0
        if off_max < 1e-6:
            out[c] = "none"
        elif flat >= 0.90:
            out[c] = "additive"
        else:
            out[c] = "multiplicative"
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="复权口径对 Rank IC 的影响")
    ap.add_argument("--codes", type=int, default=60)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from qcore.dataset import build_dataset
    from qcore.engine import run_walkforward, RunConfig
    from qcore import stats as S

    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
    if len(files) < 20:
        print("[SKIP] 本地行情不足（data/raw/kline/*_h2.csv < 20）")
        return 0
    codes = [os.path.basename(f).split("_")[0] for f in files[:a.codes]]

    print(f"[1/4] build_dataset  {len(codes)} 只 …", flush=True)
    p3d = build_dataset(codes, start=a.start, horizon=a.horizon, verbose=False)
    print(f"      T={p3d.shape[0]} N={p3d.shape[1]} S={p3d.shape[2]}", flush=True)

    print("[2/4] walkforward（只取 scores，不重算信号逻辑）…", flush=True)
    res = run_walkforward(p3d, RunConfig(horizon=a.horizon, warmup=120,
                                        record_ledger=False))
    scores = np.asarray(res["scores"], dtype=np.float64)        # (N,T)
    print(f"      scores {scores.shape}", flush=True)

    print("[3/4] 重建 fqt=0 的前向收益 + 逐股复权方式 …", flush=True)
    r2 = np.asarray(p3d.RET[a.horizon], dtype=np.float64)       # (T,N)
    r0 = _open0(list(p3d.codes), p3d.dates, lag=1, h=a.horizon)
    if r0 is None:
        print("[SKIP] fqt=0 数据不可用")
        return 0
    kinds = _classify(list(p3d.codes))
    add_codes = sorted(c for c, k in kinds.items() if k == "additive")
    kc = {}
    for k in kinds.values():
        kc[k] = kc.get(k, 0) + 1
    print("      复权方式: " + " / ".join(f"{k}={v}" for k, v in sorted(kc.items()))
          + (f"；加法复权: {add_codes[:8]}" if add_codes else ""), flush=True)

    ok = np.isfinite(r2) & np.isfinite(r0)
    r2m = np.where(ok, r2, np.nan)
    r0m = np.where(ok, r0, np.nan)
    # 只**修正**加法复权那几只票的收益（换成 raw 口径 = 它们各自的真实收益，
    # 除息日除外），其余股票原样保留 —— 截面成分不变，
    # 于是 ΔIC 只反映「收益口径」而不是「换了股票池」。
    add_mask = np.zeros(p3d.shape[1], dtype=bool)
    pos = {c: i for i, c in enumerate(p3d.codes)}
    for c in add_codes:
        if c in pos:
            add_mask[pos[c]] = True
    r2k = np.where(ok, np.where(add_mask[None, :], r0, r2), np.nan)

    print("[4/4] 三个对照各算一遍 Rank IC …", flush=True)
    out = {}
    for name, mat in (("h2", r2m), ("h2_fix_additive", r2k), ("raw", r0m)):
        ic = _daily_ic(scores, mat)
        ic_ok = ic[np.isfinite(ic)]
        st = S.mean_tstat(ic_ok, lags=max(a.horizon - 1, 0)) if ic_ok.size > 5 else {}
        out[name] = {"ic_mean": float(ic_ok.mean()) if ic_ok.size else None,
                     "hac_t": float(st.get("t")) if st else None,
                     "hac_se": float(st.get("se")) if st else None,
                     "n_days": int(ic_ok.size),
                     "_series": ic}

    base = out["h2"]
    for name in ("h2_fix_additive", "raw"):
        b, o = base.get("ic_mean"), out[name].get("ic_mean")
        if b is not None and o is not None:
            d = float(o - b)
            out[name]["delta_ic"] = d
            out[name]["delta_pct"] = float(d / abs(b) * 100) if abs(b) > 1e-12 else None
        else:
            out[name]["delta_ic"] = None
            out[name]["delta_pct"] = None
        sa, sb = base["_series"], out[name]["_series"]
        m = np.isfinite(sa) & np.isfinite(sb)
        out[name]["ic_series_corr"] = _spearman(sa[m], sb[m]) if m.sum() > 5 else None

    rels = [abs(v) for v in (out["h2_fix_additive"].get("delta_pct"),
                             out["raw"].get("delta_pct")) if v is not None]
    rel = max(rels) if rels else None
    if rel is None:
        verdict = "insufficient"
    elif rel < 10:
        verdict = "basis_negligible"
    elif rel < 35:
        verdict = "basis_material"
    else:
        verdict = "basis_dominant"

    res_out = {
        "ok": True, "verdict": verdict,
        "n_codes": int(p3d.shape[1]), "T": int(p3d.shape[0]),
        "horizon": a.horizon, "start": a.start,
        "adjust_kind_counts": kc,
        "additive_codes": add_codes[:20],
        "by_basis": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in out.items()},
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "interpretation": {
            "basis_negligible": "复权口径（修正加法复权股票收益 / 改用 fqt=0 原始价）"
                                "对 Rank IC 的影响 <10% 相对值 → "
                                "现有 IC 的**方向与量级都可引用**",
            "basis_material": "口径使 IC 变动 10–35% → 方向可引用、幅度不可直接引用，"
                              "对外应同时给两个口径",
            "basis_dominant": "口径使 IC 变动 >35% → 现有 IC 数字**不可引用**，"
                              "必须改用修正口径重算基线",
            "insufficient": "样本不足，不下结论",
        }.get(verdict, ""),
        "limitation": (f"面板仅 {int(p3d.shape[1])} 只（基线 283 只），只做同分数两口径的"
                       f"相对比较，不当绝对基线。乘法复权下 r_h2 数学上恒等于真实总收益，"
                       f"所以真正的风险只来自少数加法复权股票与 fqt=0 的除息缺口。"),
    }
    try:
        from qcore import store
        store.write_json("basis_ic", res_out)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 写 basis_ic.json 失败: {e}")

    if a.json:
        print(json.dumps(res_out, ensure_ascii=False, indent=1))
    else:
        print("=" * 76)
        print(f"复权口径对 Rank IC 的影响 · {p3d.shape[1]} 只 × {p3d.shape[0]} 天 · h={a.horizon}")
        print("=" * 76)
        for name, lab in (("h2", "现行 fqt=2（全部股票）"),
                          ("h2_fix_additive", "修正加法复权股票收益"),
                          ("raw", "fqt=0 原始价（含除息缺口）")):
            r = out[name]
            d = r.get("delta_ic")
            print(f"  {name:15s} {lab:24s} IC {r['ic_mean']:+.5f}  t {r['hac_t']:+.2f}  "
                  f"n={r['n_days']}"
                  + (f"  ΔIC {d:+.5f} ({r['delta_pct']:+.1f}%)" if d is not None else ""))
        print()
        print("  IC 序列相关: 修正加法复权 "
              f"{out['h2_fix_additive'].get('ic_series_corr')} · "
              f"raw {out['raw'].get('ic_series_corr')}")
        print()
        print(f"VERDICT: {verdict}")
        print("  含义: " + res_out["interpretation"])
        print("  边界: " + res_out["limitation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
