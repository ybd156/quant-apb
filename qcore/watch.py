"""
自选跟踪 —— 用户添加标的即开始实时预测与跟踪演算。

## 工作流

    用户添加 600519
      → 立刻：拉数据 → 算 Alpha158 → 用已训练模型打分 → 记录进场价
      → 每日：更新最新价、分数、自添加以来的收益
      → 累积：命中率（预测涨且真涨 / 预测跌且真跌）

## 为什么需要"参考横截面"

GBDT 吃的是**横截面 rank 标准化后的特征**。只给 3 只股票做 rank 没有意义
（rank 只在组内比较）。所以打分时把「CSI300 全池 ∪ 自选」放在一起做横截面，
再取自选那几只的结果。

基准池的特征矩阵缓存到 `data/model/base_features.npz`，
有效期默认 1 天——每天只需重算一次，之后加多少只都是秒级。

## 学习

每次更新都记录 `pred` 与 `actual`，累积成命中率。
这是唯一诚实的"模型在实盘上准不准"的度量，比回测更可信。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
MODEL_DIR = os.path.join(ROOT, "data", "model")
WATCH_PATH = os.path.join(STATE, "watchlist.json")
BASE_PATH = os.path.join(MODEL_DIR, "base_features.npz")
BASE_META = os.path.join(MODEL_DIR, "base_meta.json")

os.makedirs(STATE, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _bare(code: str) -> str:
    c = str(code).strip().lower()
    return c[2:] if c[:2] in ("sh", "sz", "bj") else c.zfill(6)


def _qlib(code: str) -> str:
    c = str(code).strip().lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c
    c6 = c.zfill(6)
    if c6[0] in ("6", "9"):
        return "sh" + c6
    if c6[:2] in ("43", "83", "87", "92"):
        return "bj" + c6
    return "sz" + c6


# ------------------------------------------------------------------ 基准特征缓存
def _base_fresh(max_age_hours: float = 24.0) -> bool:
    if not os.path.isfile(BASE_PATH) or not os.path.isfile(BASE_META):
        return False
    try:
        with open(BASE_META, encoding="utf-8") as f:
            m = json.load(f)
        t = datetime.fromisoformat(m["built_at"])
        return (datetime.now().astimezone() - t).total_seconds() < max_age_hours * 3600
    except Exception:
        return False


def _bench():
    import importlib.util
    p = os.path.join(ROOT, "scripts", "benchmark_alpha158.py")
    spec = importlib.util.spec_from_file_location("bench", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_base(universe: str = "csi300", stocks: int = 300,
               force: bool = False, verbose: bool = True) -> Dict:
    """构建/刷新 CSI300 基准特征矩阵（每日一次）。"""
    if not force and _base_fresh():
        with open(BASE_META, encoding="utf-8") as f:
            return {"ok": True, "cached": True, **json.load(f)}
    from qcore import qlib_data as QD
    bench = _bench()
    codes = QD.universe_codes(universe, start=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"))
    if not codes:
        codes = QD.universe_codes(universe)
    d = bench.build_features(codes, start="2008-01-01", horizon=1,
                             max_stocks=stocks, verbose=verbose)
    if not d.get("ok"):
        return {"ok": False, "reason": "features_failed"}
    np.savez_compressed(
        BASE_PATH,
        X=d["X"][-1:].astype(np.float32),          # 只存最后一天，省内存
        codes=np.array(d["codes"]),
        dates=np.array([str(d["dates"][-1].date())]),
    )
    meta = {"built_at": _now(), "universe": universe, "n": len(d["codes"]),
            "date": str(d["dates"][-1].date())}
    with open(BASE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"  基准特征已构建：{meta['n']} 只 @ {meta['date']}", flush=True)
    return {"ok": True, "cached": False, **meta}


# ------------------------------------------------------------------ 打分
def score(codes: Iterable[str], verbose: bool = True) -> Dict:
    """
    给任意股票打分。做法：把「基准池 ∪ 目标」放一起算横截面，再取目标。
    返回 {code: {"score":..., "rank_pct":...}}
    """
    from qcore import qlib_data as QD
    from qcore import live_model as LM

    model, meta = LM.load_model()
    if model is None:
        return {"ok": False, "reason": "model_not_trained"}

    targets = [_qlib(c) for c in codes]
    base = build_base(verbose=verbose)
    if not base.get("ok"):
        return {"ok": False, "reason": base.get("reason", "base_failed")}

    # 基准池最后一天的特征
    z = np.load(BASE_PATH, allow_pickle=True)
    Xbase = z["X"]                        # (1, N, K)
    cbase = [str(c) for c in z["codes"]]
    base_date = str(z["dates"][0])

    bench = _bench()
    # 目标股的特征（多取一年预热）
    extra = [c for c in targets if c not in set(cbase)]
    Xs, cs = [Xbase[0]], list(cbase)
    if extra:
        d = bench.build_features(extra, start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"),
                                 horizon=1, max_stocks=len(extra), verbose=False)
        if d.get("ok"):
            Xs.append(d["X"][-1])
            cs += list(d["codes"])
    X = np.concatenate(Xs, axis=0)[None, ...]      # (1, N', K)

    Xn = bench._cs_rank_norm(X)
    row = Xn[0]
    ok = np.isfinite(row).all(axis=1)
    sc = np.full(len(cs), np.nan)
    if ok.sum() >= 20:
        sc[ok] = model.predict(row[ok])

    pct = pd.Series(sc).rank(pct=True).to_numpy()
    out = {}
    for i, c in enumerate(cs):
        if c in set(targets):
            out[_bare(c)] = {"score": (round(float(sc[i]), 6) if np.isfinite(sc[i]) else None),
                             "rank_pct": (round(float(pct[i]), 4) if np.isfinite(pct[i]) else None)}
    return {"ok": True, "as_of": base_date, "n_pool": len(cs), "scores": out}


# ------------------------------------------------------------------ 自选管理
def load() -> Dict:
    try:
        with open(WATCH_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"items": [], "updated_at": None}


def _save(d: Dict) -> None:
    d["updated_at"] = _now()
    tmp = WATCH_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, WATCH_PATH)


def _name_map() -> Dict[str, str]:
    from qcore import names as NAMES
    out = {}
    for k, v in (NAMES.load() or {}).items():
        out[_bare(k)] = v.get("name") if isinstance(v, dict) else str(v)
    return out


def _last_price(code: str) -> Optional[float]:
    try:
        from qcore import qlib_data as QD
        df = QD.load_stock(_qlib(code), start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        if df is None or len(df) == 0:
            return None
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def add(codes: Iterable[str], verbose: bool = True) -> Dict:
    """添加标的：立刻打分 + 记录进场价。"""
    cur = load()
    items = cur.get("items") or []
    have = {it["code"] for it in items}
    new = [_bare(c) for c in codes if _bare(c) not in have]
    if not new:
        return {"ok": True, "added": 0, "items": items}

    sc = score(new, verbose=verbose)
    nm = _name_map()
    added = []
    for c in new:
        px = _last_price(c)
        s = (sc.get("scores") or {}).get(c) or {}
        it = {"code": c, "name": nm.get(c, ""), "added_at": _now(),
              "added_price": px, "last_price": px,
              "entry_score": s.get("score"), "entry_rank_pct": s.get("rank_pct"),
              "score": s.get("score"), "rank_pct": s.get("rank_pct"),
              "pred": ("up" if (s.get("rank_pct") or 0) >= 0.5 else "down")
              if s.get("rank_pct") is not None else None,
              "ret_since_add": 0.0 if px else None,
              "hit": None, "updated_at": _now()}
        items.append(it)
        added.append(it)
    cur["items"] = items
    _save(cur)
    return {"ok": True, "added": len(added), "items": added,
            "as_of": sc.get("as_of"), "n_pool": sc.get("n_pool")}


def remove(codes: Iterable[str]) -> Dict:
    cur = load()
    drop = {_bare(c) for c in codes}
    before = len(cur.get("items") or [])
    cur["items"] = [it for it in (cur.get("items") or []) if it["code"] not in drop]
    _save(cur)
    return {"ok": True, "removed": before - len(cur["items"])}


def update(verbose: bool = True) -> Dict:
    """刷新：最新价、最新分数、自添加收益、命中与否。"""
    cur = load()
    items = cur.get("items") or []
    if not items:
        return {"ok": True, "n": 0, "items": []}
    codes = [it["code"] for it in items]
    sc = score(codes, verbose=verbose)
    scores = sc.get("scores") or {}
    for it in items:
        c = it["code"]
        px = _last_price(c)
        s = scores.get(c) or {}
        it["last_price"] = px
        if s.get("score") is not None:
            it["score"] = s["score"]
            it["rank_pct"] = s["rank_pct"]
            it["pred"] = "up" if (s["rank_pct"] or 0) >= 0.5 else "down"
        ap = it.get("added_price")
        if px and ap:
            it["ret_since_add"] = round(px / ap - 1.0, 6)
            # 命中：预测涨且收益>0，或预测跌且收益<0
            if it.get("pred"):
                it["hit"] = bool((it["pred"] == "up") == (it["ret_since_add"] > 0))
        it["updated_at"] = _now()
    cur["items"] = items
    _save(cur)
    return {"ok": True, "n": len(items), "as_of": sc.get("as_of"), "items": items}


def summary() -> Dict:
    cur = load()
    items = cur.get("items") or []
    if not items:
        return {"ok": True, "n": 0, "hit_rate": None, "avg_ret": None}
    rets = [it["ret_since_add"] for it in items if it.get("ret_since_add") is not None]
    hits = [it["hit"] for it in items if it.get("hit") is not None]
    return {"ok": True, "n": len(items),
            "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
            "n_evaluated": len(hits),
            "avg_ret": round(float(np.mean(rets)), 6) if rets else None,
            "updated_at": cur.get("updated_at")}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "add":
        print(json.dumps(add(sys.argv[2:]), ensure_ascii=False, indent=1)[:800])
    elif len(sys.argv) > 2 and sys.argv[1] == "remove":
        print(json.dumps(remove(sys.argv[2:]), ensure_ascii=False, indent=1))
    elif "--update" in sys.argv:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
        r = update()
        for it in r["items"][:10]:
            print(f"  {it['code']:<8}{it.get('name',''):<8} score={it.get('score')} "
                  f"ret={it.get('ret_since_add')} hit={it.get('hit')}")
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
