"""
实时演算 —— 在 NAS 上按需更新数据、训练、预测、盯市。

## N100 的现实约束

实测（本机 16 核）：
  · Alpha158 算 300 只：21 秒
  · 横截面标准化：17 秒
  · GBDT 训练 250 棵 / 615k 样本：**315 秒**

N100 单线程约为桌面 2~2.5× 慢 → 全量重训约 **10-13 分钟**。

所以**不能每次预测都重训**。分级调度：

| 频率 | 任务 | 耗时 |
|---|---|---|
| 交易时段每 3 秒 | 秒级快照（`qcore.tick`） | <1s |
| 交易时段每 60 秒 | 行情/宽度/情绪（`qcore.market_live`） | ~2s |
| 每日收盘后 15:10 | 增量数据 + 打分 + 盯市 | ~30s |
| 每周（可配） | 重训 GBDT | 10-13 分钟（夜间） |
| 每 6 小时 | 数据源探活（`qcore.health`） | ~5s |

## 模型持久化

训练好的 GBDT 用 pickle 存到 `data/model/gbdt.pkl`，附 `meta.json` 记训练时间、
样本数、IC。预测时直接加载，不重训。

## 增量加树（可选）

GBDT 支持 warm-start：保留旧树，在**最近 N 天**的新数据上再加 M 棵树。
这样每日成本从 10 分钟降到 30 秒，代价是对旧规律的适应变慢。
用 `warm_trees` 控制，默认关闭（宁可慢也要稳）。
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "data", "model")
STATE = os.path.join(ROOT, "data", "state")
MODEL_PATH = os.path.join(MODEL_DIR, "gbdt.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------------ 模型存取
def save_model(model, meta: Dict) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=4)
    meta = dict(meta)
    meta["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return MODEL_PATH


def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None, {}
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    meta = {}
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass
    return m, meta


def model_age_days() -> Optional[float]:
    _, meta = load_model()
    s = meta.get("saved_at")
    if not s:
        return None
    try:
        return (datetime.now().astimezone() - datetime.fromisoformat(s)).total_seconds() / 86400
    except Exception:
        return None


# ------------------------------------------------------------------ 训练
def train(universe: str = "csi300", stocks: int = 300, trees: int = 250,
          train_end: Optional[str] = None, verbose: bool = True) -> Dict:
    """训练并保存模型。train_end 默认为「今天往前 30 天」，留出验证缓冲。"""
    import importlib.util
    here = os.path.join(ROOT, "scripts")

    def _load(name, fn):
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, fn))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    bench = _load("bench", "benchmark_alpha158.py")
    if train_end is None:
        train_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    from qcore import qlib_data as QD
    from qcore.gbdt import GBDT

    codes = QD.universe_codes(universe, start=train_end)
    if verbose:
        print(f"  {universe}: {len(codes)} 只，训练截至 {train_end}", flush=True)
    d = bench.build_features(codes, start="2008-01-01", horizon=1,
                             max_stocks=stocks, verbose=verbose)
    if not d.get("ok"):
        return {"ok": False, "reason": "build_features failed"}
    X, y, dates = d["X"], d["y"], d["dates"]

    t0 = time.time()
    Xn = bench._cs_rank_norm(X)
    if verbose:
        print(f"  标准化 {time.time()-t0:.0f}s", flush=True)

    sp = bench._split(dates, "2021-01-01", "2099-12-31", train_end,
                      (datetime.strptime(train_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                      "2099-12-31")
    Xtr, ytr = bench._flatten(Xn, y, sp["train"])
    if Xtr.shape[0] < 5000:
        return {"ok": False, "reason": f"样本不足 {Xtr.shape[0]}"}
    t0 = time.time()
    model = GBDT(n_estimators=trees, max_depth=4, learning_rate=0.1,
                 min_samples_leaf=200, subsample=0.8, colsample=0.8,
                 l2=1.0, max_bins=32, random_state=42)
    model.fit(Xtr, ytr)
    dt = time.time() - t0

    meta = {"universe": universe, "n_stocks": len(d["codes"]),
            "n_factors": int(X.shape[2]), "n_trees": int(trees),
            "train_end": train_end, "train_n": int(Xtr.shape[0]),
            "train_seconds": round(dt, 1),
            "trained_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    save_model(model, meta)
    if verbose:
        print(f"  训练完成 {dt:.0f}s，已保存", flush=True)
    return {"ok": True, **meta}


# ------------------------------------------------------------------ 预测
def predict(top_k: int = 30, universe: str = "csi300", stocks: int = 300,
            verbose: bool = True) -> Dict:
    """用已保存的模型给最新一天打分，写 picks.json。"""
    import importlib.util
    here = os.path.join(ROOT, "scripts")
    spec = importlib.util.spec_from_file_location("bench", os.path.join(here, "benchmark_alpha158.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    from qcore import qlib_data as QD
    from qcore import names as NAMES

    model, meta = load_model()
    if model is None:
        return {"ok": False, "reason": "model_not_trained"}

    codes = QD.universe_codes(universe, start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"))
    if not codes:
        codes = QD.universe_codes(universe)
    d = bench.build_features(codes, start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"),
                             horizon=1, max_stocks=stocks, verbose=False)
    if not d.get("ok"):
        return {"ok": False, "reason": "features_failed"}
    X, dates, cl = d["X"], d["dates"], d["codes"]
    Xn = bench._cs_rank_norm(X)

    last = len(dates) - 1
    row = Xn[last]
    ok = np.isfinite(row).all(axis=1)
    if ok.sum() < 20:
        return {"ok": False, "reason": "insufficient_valid_features"}
    score = np.full(len(cl), np.nan)
    score[ok] = model.predict(row[ok])

    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf))[:top_k]
    nm = {}
    for k, v in (NAMES.load() or {}).items():
        nm[str(k)] = v.get("name") if isinstance(v, dict) else str(v)

    rows = []
    for rank, j in enumerate(order, 1):
        code = cl[j]
        bare = code[2:] if code[:2] in ("sh", "sz", "bj") else code
        rows.append({"rank": rank, "code": bare, "qlib_code": code,
                     "name": nm.get(bare) or nm.get(bare.zfill(6)) or "",
                     "score": round(float(score[j]), 6),
                     "weight": round(1.0 / len(order), 4)})

    payload = {"as_of": str(dates[last].date()), "rows": rows, "n": len(rows),
               "top_k": top_k, "model_age_days": round(model_age_days() or 0, 1),
               "note": "实时打分（模型未重训时用最近一次保存的模型）"}
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, "picks.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(STATE, "picks.json"))
    if verbose:
        print(f"  预测完成 {payload['as_of']}，{len(rows)} 只", flush=True)
    return {"ok": True, **payload}


# ------------------------------------------------------------------ 调度
def should_retrain(max_age_days: float = 7.0) -> bool:
    a = model_age_days()
    return a is None or a >= max_age_days


def run_once(retrain_if_due: bool = True, max_age_days: float = 7.0,
             verbose: bool = True) -> Dict:
    """跑一轮：按需重训 → 预测 → 写健康度。"""
    from qcore import health
    out = {"at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if retrain_if_due and should_retrain(max_age_days):
        if verbose:
            print("[live] 模型过期，开始重训 ...", flush=True)
        out["train"] = train(verbose=verbose)
    else:
        out["train"] = {"ok": True, "skipped": True,
                        "age_days": round(model_age_days() or 0, 1)}
    out["predict"] = predict(verbose=verbose)
    try:
        out["health"] = health.check_all().get("summary")
    except Exception as e:  # noqa: BLE001
        out["health"] = {"error": str(e)[:80]}
    return out


def loop(interval_s: int = 3600, max_age_days: float = 7.0,
         verbose: bool = True) -> None:
    """常驻：每小时检查一次，收盘后跑一次完整演算。"""
    last_run_day = None
    while True:
        try:
            now = datetime.now()
            # 15:10 之后跑当天第一次
            if now.weekday() < 5 and (now.hour, now.minute) >= (15, 10):
                if last_run_day != now.date():
                    if verbose:
                        print(f"[live] {now:%H:%M} 收盘演算 ...", flush=True)
                    r = run_once(max_age_days=max_age_days, verbose=verbose)
                    last_run_day = now.date()
                    if verbose:
                        print(f"[live] done: {json.dumps(r, ensure_ascii=False)[:200]}", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[live] ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        print(json.dumps(train(), ensure_ascii=False, indent=1))
    elif "--predict" in sys.argv:
        print(json.dumps({k: v for k, v in predict().items() if k != "rows"},
                         ensure_ascii=False, indent=1))
    elif "--loop" in sys.argv:
        loop()
    else:
        print(json.dumps(run_once(), ensure_ascii=False, indent=1)[:1200])
