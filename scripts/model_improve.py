#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型层四项改进的实测对比 —— A/B/C/D/E/F 六组，同一口径，同一评估函数。

## 评估口径（**固定，不因实验而变**）

| 项目 | 取值 |
|---|---|
| 股票池 | CSI300 点对点成分（`universe_codes('csi300', start='2017-06-01')`，300 只） |
| 特征 | Alpha158（`qcore.alpha158`），逐日横截面 rank 正态化（Qlib CSRankNorm 等价） |
| 标签 | `close[T+1+h]/close[T+1]-1`，h=1 |
| 训练 | 2008-01-01 .. 2019-12-31（早停/集成权重用的验证集 = 2019 全年，在训练窗口内） |
| 测试 | 2021-01-01 .. 2026-11-06（2020 整年丢弃，防止验证/测试贴边） |
| 模型 | GBDT(n_estimators=250, depth=4, lr=0.1, min_samples_leaf=200, subsample=.8, colsample=.8, l2=1, max_bins=32) |
| 指标 | 逐日横截面 Pearson IC → `stats.mean_tstat(lags=0)`；RankIC = 逐日 Spearman 均值 |

**所有实验共用同一批样本行**：训练/测试的行掩码只由「158 个特征是否全部有限」决定，
在实验之间不变。所以 A→F 的差异只来自模型/标签/权重，不来自样本选择。

## 六组实验

    A 基线        原始 158 因子 + GBDT
    B 行业中性化   标签换成「行业内去均值」后的残差（评估仍用原始标签）
    C 特征筛选    158 -> 60（select_by_stability，只在训练窗口内做）
    D 时间衰减    样本按半衰期 250 天重采样（GBDT 不支持 sample_weight，见 qcore.sample_weight）
    E 集成        BlendModel（GBDT×3 + 岭回归 + 等权 rank，按验证集 IC 加权）
    F 全组合      B+C+D+E

## 用法

    python scripts/model_improve.py                 # 跑 A-F 全部
    python scripts/model_improve.py --only A,B      # 只跑指定实验
    python scripts/model_improve.py --trees 20 --only A --smoke   # 冒烟测试

结果同时写入 `data/cache/mi_results.json`，已跑过的实验默认复用（--force 重跑）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qcore.ensemble_model import BlendModel                      # noqa: E402
from qcore.feature_select import select_by_stability, stability_scores  # noqa: E402
from qcore.gbdt import GBDT                                      # noqa: E402
from qcore.neutral import industry_neutralize, group_sizes       # noqa: E402
from qcore.sample_weight import resample_by_weight, time_decay_weights  # noqa: E402
from scripts.benchmark_alpha158 import _cs_rank_norm, _ic        # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")
PANEL = os.path.join(CACHE, "mi_panel_v1.npz")
IND_CSV = os.path.join(CACHE, "mi_industry_v1.csv")
RESULTS = os.path.join(CACHE, "mi_results.json")

TRAIN = ("2008-01-01", "2019-12-31")
VALID = ("2019-01-01", "2019-12-31")
TEST = ("2021-01-01", "2026-12-31")

EXPS = [
    ("A", "基线（158 因子 + GBDT）"),
    ("B", "行业中性化（标签 -> 行业残差）"),
    ("C", "特征筛选（158 -> 60，按 IC 稳定性）"),
    ("D", "时间衰减（半衰期 250 天，重采样近似）"),
    ("E", "多模型集成（BlendModel）"),
    ("F", "全组合（B+C+D+E）"),
]
EXP_DESC = dict(EXPS)

GBDT_KW = dict(max_depth=4, learning_rate=0.1, min_samples_leaf=200,
               subsample=0.8, colsample=0.8, l2=1.0, max_bins=32)


# ---------------------------------------------------------------------- #
# 数据
# ---------------------------------------------------------------------- #
def load_panel():
    if not os.path.exists(PANEL):
        raise SystemExit(f"面板缓存不存在：{PANEL}\n先运行 python tmp/build_panel.py")
    z = np.load(PANEL, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    dates = pd.to_datetime(z["dates"])
    codes = [str(c) for c in z["codes"]]
    names = [str(n) for n in z["names"]]
    ind_df = pd.read_csv(IND_CSV, dtype={"code": str})
    imap = dict(zip(ind_df["code"].str.zfill(6), ind_df["industry"].astype(str)))
    industry = np.array([imap.get(c[2:] if c[:2] in ("sh", "sz", "bj") else c, "NA")
                         for c in codes], dtype=object)
    return X, y, dates, codes, names, industry


def idx_of(dates: pd.DatetimeIndex, span) -> np.ndarray:
    a, b = span
    return np.where((dates >= a) & (dates <= b))[0]


def flatten(X: np.ndarray, y: np.ndarray, days: np.ndarray):
    """把 (T,N,K)/(T,N) 的指定日期展平成样本，返回 (Xf, yf, day_pos, col_pos)。

    col_pos 是每个样本的股票列号，用于把 (T,N) 的标签/行业残差按样本对齐取出来。
    """
    xs, ys, ds, cs = [], [], [], []
    for t in days:
        m = np.isfinite(y[t]) & np.isfinite(X[t]).all(axis=1)
        if m.sum() < 20:
            continue
        xs.append(X[t][m])
        ys.append(y[t][m])
        ds.append(np.full(int(m.sum()), int(t), dtype=np.int64))
        cs.append(np.flatnonzero(m).astype(np.int64))
    if not xs:
        K = X.shape[2]
        e = np.empty(0)
        return np.empty((0, K), np.float32), e, e.astype(np.int64), e.astype(np.int64)
    return (np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.float64),
            np.concatenate(ds), np.concatenate(cs))


def eval_days(Xn, y, days):
    """测试集逐日数据：[(t, X_day, y_day), ...]（y 一律是**原始标签**）。"""
    out = []
    for t in days:
        out.append((int(t), Xn[t], y[t]))
    return out


# ---------------------------------------------------------------------- #
# 评估（口径固定：调用 benchmark_alpha158._ic）
# ---------------------------------------------------------------------- #
def evaluate(model, tdays, sel=None, dates=None):
    N = tdays[0][1].shape[0]
    score = np.full((len(tdays), N), np.nan, dtype=np.float32)
    ytest = np.full((len(tdays), N), np.nan, dtype=np.float32)
    for i, (t, Xday, yday) in enumerate(tdays):
        row = Xday if sel is None else Xday[:, sel]
        m = np.isfinite(row).all(axis=1)
        if m.sum() >= 20:
            score[i, m] = model.predict(row[m])
        ytest[i] = yday
    res = _ic(score, ytest)          # 官方口径：benchmark_alpha158._ic
    # 附加：分年度 IC（只用来看"改进是不是只来自某一年"，不参与判定）
    if dates is not None:
        yrs, ics = [], []
        for i in range(len(tdays)):
            s, r = score[i], ytest[i]
            m = np.isfinite(s) & np.isfinite(r)
            if m.sum() < 20:
                continue
            v = float(np.corrcoef(s[m], r[m])[0, 1])
            if np.isfinite(v):
                yrs.append(int(pd.Timestamp(dates[tdays[i][0]]).year))
                ics.append(v)
        yrs = np.array(yrs)
        ics = np.array(ics)
        res["year_ic"] = {str(y): round(float(ics[yrs == y].mean()), 4)
                          for y in sorted(set(yrs.tolist()))}
        res["year_n"] = {str(y): int((yrs == y).sum()) for y in sorted(set(yrs.tolist()))}
    return res


# ---------------------------------------------------------------------- #
# 单个实验
# ---------------------------------------------------------------------- #
def run_experiment(tag, ctx, trees=250, top_n=60, half_life=250.0,
                   seed=42, n_folds=5, verbose=True):
    Xtr, ytr_raw, dtr = ctx["Xtr"], ctx["ytr"], ctx["dtr"]
    Xva, yva_raw, dva = ctx["Xva"], ctx["yva"], ctx["dva"]
    tdays = ctx["tdays"]
    sel = None
    t0 = time.time()
    info = {}

    use_neutral = tag in ("B", "F")
    use_select = tag in ("C", "F")
    use_decay = tag in ("D", "F")
    use_blend = tag in ("E", "F")

    # ---- 标签：行业残差 ----
    ytr, yva = ytr_raw, yva_raw
    if use_neutral:
        ytr = ctx["ytr_neu"]
        yva = ctx["yva_neu"]

    # ---- 特征：稳定性筛选（只用训练窗口，绝不含测试期）----
    if use_select:
        sel = ctx["sel"]
        info["n_features"] = len(sel)

    Xtr_f = Xtr if sel is None else Xtr[:, sel]
    Xva_f = Xva if sel is None else Xva[:, sel]

    # ---- 样本：时间衰减重采样 ----
    if use_decay:
        w = time_decay_weights(ctx["dates"][dtr], half_life_days=half_life)
        Xtr_f, ytr_f, ridx = resample_by_weight(Xtr_f, ytr, w,
                                                random_state=seed)
        wva = time_decay_weights(ctx["dates"][dva], half_life_days=half_life)
        Xva_f, yva_f, _ = resample_by_weight(Xva_f, yva, wva, random_state=seed + 1)
        eff = float(np.power(0.5, 4.0))     # 4 年前样本权重占比示意
        info["decay_eff_weight_4y"] = eff
        if verbose:
            print(f"    [D] 重采样后训练样本 {Xtr_f.shape[0]:,} 行"
                  f"（原始 {Xtr.shape[0]:,}），有效权重 4 年前 = {eff:.3f}", flush=True)
    else:
        ytr_f, yva_f = ytr, yva

    # ---- 模型 ----
    if use_blend:
        model = BlendModel(n_gbdt=3, n_estimators=trees,
                           ridge_alpha=1.0, verbose=verbose, **GBDT_KW)
        model.fit(Xtr_f, ytr_f, eval_set=(Xva_f, yva_f), eval_days=dva)
        info["weights"] = model.weights
        info["component_ic"] = {k: round(v, 5) for k, v in model.ic_detail.items()}
    else:
        model = GBDT(n_estimators=trees, random_state=seed, **GBDT_KW)
        model.fit(Xtr_f, ytr_f)
        info["n_trees"] = int(getattr(model, "n_trees_", 0))

    # ---- 评估（对**原始标签**打分，口径不随实验变化）----
    res = evaluate(model, tdays, sel=sel, dates=ctx["dates"])
    res["sec"] = round(time.time() - t0, 1)
    res["n_features"] = int(sel.size if sel is not None else Xtr.shape[1])
    res["train_rows"] = int(Xtr_f.shape[0])
    res.update(info)
    return res


# ---------------------------------------------------------------------- #
def fmt_line(tag, res, base_ic=None):
    d = ""
    if base_ic is not None and np.isfinite(base_ic):
        d = f"  ΔIC={res['ic'] - base_ic:+.4f}"
    return (f"{tag} {EXP_DESC[tag]:<26s} IC={res['ic']:+.4f}  t={res['t']:+6.2f}  "
            f"RankIC={res['rank_ic']:+.4f}  ICIR={res['icir']:+.3f}{d}  "
            f"[{res['n_features']}特征 {res['train_rows']:,}行 {res['sec']}s]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="模型层四项改进实测对比")
    ap.add_argument("--only", default="A,B,C,D,E,F", help="只跑指定实验，逗号分隔")
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--half-life", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="忽略已有结果缓存")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试：只保留 2025 年测试集，用于快速验证流程")
    a = ap.parse_args(argv)

    only = [s.strip().upper() for s in a.only.split(",") if s.strip()]
    bad = [s for s in only if s not in EXP_DESC]
    if bad:
        print(f"未知实验代号: {bad}；可选 {list(EXP_DESC)}")
        return 2

    t_start = time.time()
    print("=" * 100)
    print("模型层四项改进 —— A/B/C/D/E/F 实测对比")
    print("=" * 100)

    X, y, dates, codes, names, industry = load_panel()
    T, N, K = X.shape
    print(f"面板 X={X.shape}  {dates[0].date()} .. {dates[-1].date()}  股票 {N} 只")

    sz = group_sizes(industry)
    print(f"行业映射：{len(set(industry))} 个行业，单只行业 {int((sz == 1).sum())} 个，"
          f"未知(NA/-) {int(((industry == 'NA') | (industry == '-')).sum())} 只")

    print("横截面 rank 正态化 ...", flush=True)
    t0 = time.time()
    Xn = _cs_rank_norm(X)
    del X
    print(f"  完成 {time.time() - t0:.0f}s", flush=True)

    tr_idx = idx_of(dates, TRAIN)
    va_idx = idx_of(dates, VALID)
    te_idx = idx_of(dates, TEST)
    if a.smoke:
        te_idx = te_idx[dates[te_idx] >= "2025-01-01"]
    print(f"训练 {dates[tr_idx[0]].date()}..{dates[tr_idx[-1]].date()} ({len(tr_idx)} 天) | "
          f"验证 {dates[va_idx[0]].date()}..{dates[va_idx[-1]].date()} ({len(va_idx)} 天) | "
          f"测试 {dates[te_idx[0]].date()}..{dates[te_idx[-1]].date()} ({len(te_idx)} 天)")

    print("展平样本 ...", flush=True)
    t0 = time.time()
    Xtr, ytr, dtr, ctr = flatten(Xn, y, tr_idx)
    Xva, yva, dva, cva = flatten(Xn, y, va_idx)
    tdays = eval_days(Xn, y, te_idx)
    print(f"  train={Xtr.shape[0]:,}  valid={Xva.shape[0]:,}  test_days={len(tdays)}  "
          f"{time.time() - t0:.0f}s")

    # 行业中性化标签（整个面板一次算完，逐日横截面）
    y_neu = industry_neutralize(y, industry).astype(np.float32)
    ytr_neu = y_neu[dtr, ctr]
    yva_neu = y_neu[dva, cva]
    print(f"行业中性化标签：train 残差 std={float(np.nanstd(ytr_neu)):.5f} "
          f"(原始 {float(np.nanstd(ytr)):.5f})", flush=True)

    ctx = dict(Xtr=Xtr, ytr=ytr, dtr=dtr, Xva=Xva, yva=yva, dva=dva,
               tdays=tdays, dates=dates.to_numpy(),
               ytr_neu=ytr_neu, yva_neu=yva_neu)

    # 特征筛选（只跑一次，C 和 F 共用）
    if any(t in ("C", "F") for t in only):
        print(f"特征稳定性筛选（{a.folds} 折，训练窗口内）...", flush=True)
        t0 = time.time()
        st = stability_scores(Xn[tr_idx], y[tr_idx], names, n_folds=a.folds)
        sel = select_by_stability(Xn[tr_idx], y[tr_idx], names,
                                  n_folds=a.folds, top_n=a.top_n)
        ctx["sel"] = np.array(sel, dtype=np.int64)
        ctx["sel_names"] = [names[i] for i in sel]
        mean_ic = st["mean_ic"]
        cons = st["consistency"]
        print(f"  {time.time() - t0:.0f}s  选出 {len(sel)} 个特征"
              f"（|IC| 均值 {np.abs(mean_ic[sel]).mean():.4f}，"
              f"符号一致率 {cons[sel].mean():.2f}）")
        print(f"  Top-10: {', '.join(ctx['sel_names'][:10])}")
        dropped = [names[i] for i in np.argsort(-np.abs(mean_ic))
                   if i not in set(sel)][:8]
        print(f"  被丢弃的高|IC|特征样例: {', '.join(dropped)}")

    # ---- 逐组运行 ----
    results = {}
    if os.path.exists(RESULTS) and not a.force:
        try:
            results = json.load(open(RESULTS, encoding="utf-8"))
        except Exception:
            results = {}
    base_ic = results.get("A", {}).get("ic")

    print("-" * 100)
    for tag in only:
        key = f"{tag}|{a.trees}|{a.top_n}|{a.half_life}|{a.smoke}"
        if key in results and not a.force:
            print(fmt_line(tag, results[key], base_ic) + "   [缓存]")
            if tag == "A":
                base_ic = results[key]["ic"]
            continue
        print(f">>> 实验 {tag}: {EXP_DESC[tag]}", flush=True)
        res = run_experiment(tag, ctx, trees=a.trees, top_n=a.top_n,
                             half_life=a.half_life, seed=a.seed, n_folds=a.folds)
        results[key] = res
        json.dump(results, open(RESULTS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        if tag == "A":
            base_ic = res["ic"]
        print(fmt_line(tag, res, base_ic), flush=True)

    # ---- 汇总表 ----
    print()
    print("=" * 100)
    print(f"汇总（测试集 {dates[te_idx[0]].date()} .. {dates[te_idx[-1]].date()}，"
          f"h=1，{a.trees} 棵树）")
    print("=" * 100)
    rows = []
    for tag, desc in EXPS:
        key = f"{tag}|{a.trees}|{a.top_n}|{a.half_life}|{a.smoke}"
        r = results.get(key)
        if r is None:
            continue
        rows.append((tag, desc, r))
    if rows:
        b_ic = next((r["ic"] for t, _, r in rows if t == "A"), None)
        print(f"{'组':<3s}{'配置':<30s}{'IC':>9s}{'t':>8s}{'RankIC':>9s}"
              f"{'ICIR':>8s}{'ΔIC':>9s}{'ΔIC%':>8s}")
        print("-" * 100)
        for tag, desc, r in rows:
            d = (r["ic"] - b_ic) if (b_ic is not None) else float("nan")
            dp = (d / abs(b_ic) * 100.0) if (b_ic not in (None, 0)) else float("nan")
            print(f"{tag:<3s}{desc:<30s}{r['ic']:>+9.4f}{r['t']:>+8.2f}"
                  f"{r['rank_ic']:>+9.4f}{r['icir']:>+8.3f}{d:>+9.4f}{dp:>+8.1f}%")
        print("-" * 100)

        # 分年度 IC：判断"改进是否只来自某一年"
        years = sorted({y for _, _, r in rows for y in r.get("year_ic", {})})
        if years:
            print("\n分年度 IC（各年逐日横截面 IC 的均值，仅用于稳健性观察）")
            print(f"{'组':<3s}" + "".join(f"{y:>9s}" for y in years))
            print("-" * 100)
            for tag, desc, r in rows:
                yi = r.get("year_ic", {})
                print(f"{tag:<3s}" + "".join(
                    f"{(yi[y] if y in yi else float('nan')):>+9.4f}" for y in years))
            print("-" * 100)

        # 结论
        if b_ic is not None:
            better = [(t, r["ic"] - b_ic) for t, _, r in rows if t != "A"]
            ok = [t for t, d in better if d > 0.0005]
            flat = [t for t, d in better if abs(d) <= 0.0005]
            bad = [t for t, d in better if d < -0.0005]
            print("结论：")
            print(f"  有效（ΔIC > +0.0005）: {', '.join(ok) if ok else '无'}")
            print(f"  持平（|ΔIC| ≤ 0.0005）: {', '.join(flat) if flat else '无'}")
            print(f"  变差（ΔIC < -0.0005）: {', '.join(bad) if bad else '无'}")

    print(f"\n总耗时 {time.time() - t_start:.0f}s  结果缓存 -> {RESULTS}")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
