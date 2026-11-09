"""
自蒸馏 —— 让 N100 也能自己训练自己。

## 问题（实测数据）

本机（16 核 / 94 GFLOPS）训练一个周期：
    特征构建   22s
    GBDT 训练  576s   ← 瓶颈
    合计      598s

N100 约 1/5 算力 → **每个周期 50 分钟**，6 个周期 5 小时。
不是不能跑，是没法每天跑、没法多周期跑。

## 蒸馏能解决什么，不能解决什么

**能**：把大模型（teacher，250 树 × 158 特征）压成小模型（student，80 树 × 60 特征），
训练快 6-7 倍，推理也快。student 每天可以重训。

**不能**：蒸馏**不会创造新信息**。student 的上限就是 teacher。
如果 teacher 的 IC 是 0.026，student 最好也是 0.026，通常更低一点。

所以蒸馏的定位是「**用少量精度换可部署性**」，不是"提升效果"。
任何声称"蒸馏提升 IC"的说法都要警惕——那通常是把 teacher 的过拟合当成了信号。

## 关键：软标签 vs 硬标签

    student_soft  用 teacher 的连续输出当标签（回归）
    student_hard  用原始收益当标签（回归）

理论上 soft 更好：teacher 的输出已经**降噪**过（250 棵树的平均），
student 学的是"老师怎么看"，而不是"市场怎么抖"。

但这是理论，必须实测。本模块的 `compare()` 就是干这个的。

## 用法

```python
from qcore.distill import distill, compare
r = compare()          # 三方对照：teacher / student_soft / student_hard
d = distill()          # 蒸馏并保存 student
```

## N100 友好配置

`profile_for(cpu_count)` 按核数自动选配置：
  16 核 → 250 树 / 158 特征 / 全量（本机，598s）
   8 核 → 150 树 / 100 特征（约 200s）
   4 核（N100）→ 80 树 / 60 特征（约 90s，蒸馏后）
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
OUT = os.path.join(STATE, "distill.json")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def profile_for(cpu_count: int) -> Dict:
    """
    按 CPU 核数选训练配置。这是**保守**的估计，实际以 benchmark 为准。
    """
    n = max(int(cpu_count), 1)
    if n >= 16:
        return {"name": "full", "trees": 250, "n_features": 158,
                "note": "全量，约 600s"}
    if n >= 8:
        return {"name": "medium", "trees": 150, "n_features": 100,
                "note": "约 200s"}
    if n >= 4:
        return {"name": "n100", "trees": 80, "n_features": 60,
                "note": "N100 档，蒸馏后约 90s"}
    return {"name": "minimal", "trees": 50, "n_features": 40,
            "note": "极简，约 60s"}


def _load_data(universe: str = "csi300", stocks: int = 300,
               horizon: int = 1, verbose: bool = False) -> Optional[Dict]:
    """加载特征与标签（复用 benchmark_alpha158 的构建逻辑）。"""
    import importlib.util
    here = os.path.join(ROOT, "scripts")
    spec = importlib.util.spec_from_file_location(
        "bench", os.path.join(here, "benchmark_alpha158.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    from qcore import qlib_data as QD
    train_end = (datetime.now().replace() - __import__("datetime").timedelta(days=30)
                 ).strftime("%Y-%m-%d")
    codes = QD.universe_codes(universe, start=train_end)
    if verbose:
        print(f"  载入 {len(codes)} 只 / horizon={horizon}", flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=horizon,
                             max_stocks=stocks, verbose=verbose)
    if not d.get("ok"):
        return None
    d["_bench"] = bench
    return d


def _split(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
           valid_days: int = 250) -> Tuple:
    """
    时间序列切分：最后 valid_days 个交易日做验证，绝不随机打乱。
    """
    ud = np.unique(dates)
    if len(ud) <= valid_days + 50:
        cut = ud[int(len(ud) * 0.8)]
    else:
        cut = ud[-valid_days]
    tr = dates < cut
    va = dates >= cut
    return X[tr], y[tr], X[va], y[va]


def _rank(a: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # 并列取平均秩
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    """逐日横截面 Rank IC 的均值。"""
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 20:
            continue
        p, t = pred[m], y[m]
        if np.std(p) <= 0 or np.std(t) <= 0:
            continue
        v = _spearman(p, t)
        if np.isfinite(v):
            ics.append(v)
    return float(np.nanmean(ics)) if ics else float("nan")


def compare(universe: str = "csi300", stocks: int = 300, horizon: int = 1,
            student_trees: int = 80, student_features: int = 60,
            verbose: bool = True) -> Dict:
    """
    三方对照：teacher（全量）/ student_soft（学 teacher）/ student_hard（学原始标签）。
    同一个验证集上比 Rank IC 和训练耗时。
    """
    from qcore.gbdt import GBDT
    from qcore.feature_select import select_by_stability

    t0 = time.time()
    d = _load_data(universe, stocks, horizon, verbose)
    if d is None:
        return {"ok": False, "reason": "no_data"}
    bench = d["_bench"]
    X, y, dates = d["X"], d["y"], d["dates"]
    t_load = time.time() - t0

    if verbose:
        print(f"  数据 {X.shape}，载入 {t_load:.0f}s", flush=True)
    Xn = bench._cs_rank_norm(X)
    Xtr, ytr, Xva, yva = _split(Xn, y, dates)
    dtr, dva = dates[dates < np.unique(dates)[-250]], dates[dates >= np.unique(dates)[-250]]
    if verbose:
        print(f"  训练 {Xtr.shape[0]} 行 / 验证 {Xva.shape[0]} 行", flush=True)

    res: Dict[str, Dict] = {}

    # ---- teacher：全量 ----
    t0 = time.time()
    teacher = GBDT(n_estimators=250, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    teacher.fit(Xtr, ytr)
    t_teacher = time.time() - t0
    p_teacher = teacher.predict(Xva)
    res["teacher"] = {"trees": 250, "features": X.shape[2],
                      "train_s": round(t_teacher, 1),
                      "ic": round(_ic(p_teacher, yva, dva), 5)}
    if verbose:
        print(f"  teacher: {t_teacher:.0f}s IC={res['teacher']['ic']:+.5f}", flush=True)

    # ---- 选特征 ----
    t0 = time.time()
    _names = [f'f{i}' for i in range(Xtr.shape[2])]
    idx = np.asarray(select_by_stability(Xtr, ytr, _names,
                                         top_n=student_features))[:student_features]
    t_sel = time.time() - t0
    if verbose:
        print(f"  选特征 {len(idx)}/{X.shape[2]}，{t_sel:.0f}s", flush=True)

    # ---- student_soft：学 teacher 的输出 ----
    t0 = time.time()
    teacher_train_pred = teacher.predict(Xtr)
    s_soft = GBDT(n_estimators=student_trees, max_depth=4, learning_rate=0.12,
                   min_samples_leaf=300, subsample=0.8, colsample=0.9,
                   l2=1.0, max_bins=32, random_state=42)
    s_soft.fit(Xtr[:, :, idx], teacher_train_pred)
    t_soft = time.time() - t0
    res["student_soft"] = {
        "trees": student_trees, "features": len(idx),
        "train_s": round(t_soft, 1),
        "ic": round(_ic(s_soft.predict(Xva[:, :, idx]), yva, dva), 5),
        "speedup_vs_teacher": round(t_teacher / max(t_soft, 1e-9), 1),
    }

    # ---- student_hard：学原始收益 ----
    t0 = time.time()
    s_hard = GBDT(n_estimators=student_trees, max_depth=4, learning_rate=0.12,
                   min_samples_leaf=300, subsample=0.8, colsample=0.9,
                   l2=1.0, max_bins=32, random_state=42)
    s_hard.fit(Xtr[:, :, idx], ytr)
    t_hard = time.time() - t0
    res["student_hard"] = {
        "trees": student_trees, "features": len(idx),
        "train_s": round(t_hard, 1),
        "ic": round(_ic(s_hard.predict(Xva[:, :, idx]), yva, dva), 5),
        "speedup_vs_teacher": round(t_teacher / max(t_hard, 1e-9), 1),
    }

    t_ic = res["teacher"]["ic"]
    for k in ("student_soft", "student_hard"):
        v = res[k]
        v["ic_retention"] = (round(v["ic"] / t_ic, 3)
                             if t_ic and np.isfinite(t_ic) and t_ic != 0 else None)

    verdict = []
    soft, hard = res["student_soft"], res["student_hard"]
    if soft["ic"] > hard["ic"]:
        verdict.append(f"软标签优于硬标签（{soft['ic']:+.5f} vs {hard['ic']:+.5f}）")
    elif hard["ic"] > soft["ic"]:
        verdict.append(f"硬标签反而更好（{hard['ic']:+.5f} vs {soft['ic']:+.5f}）"
                       "—— 蒸馏在本例无增益，直接用硬标签")
    else:
        verdict.append("两者持平")
    if t_ic and np.isfinite(t_ic):
        verdict.append(f"student 保留 teacher IC 的 {soft['ic_retention']:.0%}"
                       f"（{soft['train_s']}s vs {res['teacher']['train_s']}s）")

    out = {"ok": True, "at": _now(), "universe": universe, "horizon": horizon,
           "n_rows": int(X.shape[0]), "n_features": int(X.shape[2]),
           "load_s": round(t_load, 1), "results": res,
           "verdict": "；".join(verdict),
           "note": "蒸馏不创造信息，student 上限是 teacher；"
                   "本表用同一时间序列验证集，不打乱"}
    os.makedirs(STATE, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, OUT)
    if verbose:
        print(f"\n  结论：{out['verdict']}", flush=True)
    return out


def distill(universe: str = "csi300", stocks: int = 300, horizon: int = 1,
            trees: Optional[int] = None, n_features: Optional[int] = None,
            verbose: bool = True) -> Dict:
    """
    训练 teacher → 蒸馏 student → 保存 student 作为该 horizon 的部署模型。
    trees/n_features 为 None 时按本机核数自动选。
    """
    from qcore.gbdt import GBDT
    from qcore.feature_select import select_by_stability
    from qcore import live_model as LM

    prof = profile_for(os.cpu_count() or 4)
    trees = int(trees or prof["trees"])
    n_features = int(n_features or prof["n_features"])
    if verbose:
        print(f"  配置: {prof['name']} ({prof['note']}) "
              f"trees={trees} features={n_features}", flush=True)

    d = _load_data(universe, stocks, horizon, verbose)
    if d is None:
        return {"ok": False, "reason": "no_data"}
    bench = d["_bench"]
    X, y, dates = d["X"], d["y"], d["dates"]
    Xn = bench._cs_rank_norm(X)

    t0 = time.time()
    teacher = GBDT(n_estimators=250, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    teacher.fit(Xn, y)
    t_teacher = time.time() - t0
    if verbose:
        print(f"  teacher 训练完成 {t_teacher:.0f}s", flush=True)

    t0 = time.time()
    _names = [f'f{i}' for i in range(Xn.shape[2])]
    idx = np.asarray(select_by_stability(Xn, y, _names, top_n=n_features))[:n_features]
    t_sel = time.time() - t0

    t0 = time.time()
    soft = teacher.predict(Xn)
    student = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.12,
                 min_samples_leaf=300, subsample=0.8, colsample=0.9,
                 l2=1.0, max_bins=32, random_state=42)
    student.fit(Xn[:, :, idx], soft)
    t_student = time.time() - t0

    meta = {
        "universe": universe, "n_stocks": len(d["codes"]),
        "n_factors": int(len(idx)), "n_trees": int(trees),
        "train_end": (datetime.now() - __import__("datetime").timedelta(days=30)
                      ).strftime("%Y-%m-%d"),
        "train_n": int(Xn.shape[0]), "horizon": int(horizon),
        "distilled": True, "teacher_trees": 250, "teacher_seconds": round(t_teacher, 1),
        "student_seconds": round(t_student, 1), "feature_idx": [int(i) for i in idx],
        "trained_at": _now(),
    }
    mp = LM.save_model(student, meta, horizon=horizon)
    out = {"ok": True, "model": mp, "trees": trees, "n_features": int(len(idx)),
           "teacher_s": round(t_teacher, 1), "student_s": round(t_student, 1),
           "total_s": round(t_teacher + t_sel + t_student, 1),
           "speedup": round((t_teacher + t_student) / max(t_student, 1e-9), 1),
           "profile": prof, "at": _now()}
    if verbose:
        print(f"  蒸馏完成：student {t_student:.0f}s（teacher {t_teacher:.0f}s），"
              f"保存到 {mp}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "--distill" in sys.argv:
        h = 1
        for a in sys.argv:
            if a.startswith("--horizon="):
                h = int(a.split("=", 1)[1])
        print(json.dumps(distill(horizon=h), ensure_ascii=False, indent=1))
    else:
        print("本机核数:", os.cpu_count(), "-> 建议配置:", json.dumps(
            profile_for(os.cpu_count() or 4), ensure_ascii=False))
        print()
        print(json.dumps(compare(), ensure_ascii=False, indent=1)[:1600])
