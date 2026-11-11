"""
情报特征 —— 把"自己找来的新闻 + 历史参照"变成模型能用的一列数。

## 为什么单独一层，而不是直接塞进 Alpha158

Alpha158 是 **158 维定宽**矩阵（`_validate` 会校验特征数，R36）。
往里加列 = 老模型全部作废 + 训练机必须重训 + 特征顺序错位风险。
而情报特征是**当日才有、不可回补**的（`pit_backfillable=False`），
天然不适合进"2008 年起回测"的那张表。

所以做成**侧信道**：模型分不变，情报分单独算，再按权重与模型分**排位融合**。
好处是：
  · 不破坏 158 维契约，不用重训也能立即生效（权重默认 0 = 关闭）；
  · 融合前后的分数都留档，事后能归因"这笔赚的是模型还是情报"；
  · 情报分本身也进台账，走同一套"到期验证"（autoinfo.verify/evaluate）。

## 分数怎么来（可解释、无魔法）

    情报分 = clip( 0.6·方向票 + 0.4·历史参照, -1, 1 )
      方向票   = Σ(direction_i × decay_i) / Σ|direction_i|      近期新闻的方向净额
      历史参照 = implied_car / scale                            参考同类事件的真实反应
    decay_i  = 0.5 ** (age_days / 3)                              3 天半衰期

两个来源都要求 `as_of <= 目标日`（PIT）；没有情报 → 该票情报分 0（**不是**空值，
0 表示"不表态"，融合时按 0 处理，不会把分数拉偏）。

## 融合（rank blend）

    z(x) = 2 × (rank(x) / n) − 1        ∈ [-1, 1]
    combined = (1−w)·z(model) + w·z(intel)

用排位而不是原值：GBDT 分数是"预测收益"（量级 0.003），情报分是 [-1,1]，
直接线性相加会被量级吃掉。排位融合只保留"谁排在谁前面"，
并且**输出也是排位分**（`score_kind="rank_blend"`），下游一律不得把它当收益估计。

## 用法

    python -m qcore.intel_features            # 看今天的情报分
    python -m qcore.intel_features --check    # 自检（含 PIT 与空数据分支）
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state", "autoinfo.json")
SCALE = 0.03          # 历史参照 CAR 的归一尺度（3% 视为"一整票"）
HALF_LIFE = 3.0


def _read(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _age(day: str, ref: Optional[str] = None) -> float:
    try:
        d = datetime.strptime(str(day)[:10], "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return 999.0
    r = datetime.strptime(ref, "%Y-%m-%d") if ref else datetime.now()
    return max(0.0, (r - d).days)


def aggregate(items: Iterable[Dict], as_of: Optional[str] = None) -> Dict[str, Dict]:
    """把情报条目按 code 聚合。**只吃 as_of <= 目标日** 的条目（PIT）。"""
    ref = as_of or datetime.now().strftime("%Y-%m-%d")
    out: Dict[str, Dict] = {}
    for it in items or []:
        code = str(it.get("code") or "")
        if not code:
            continue
        day = str(it.get("as_of") or "")[:10]
        if not day or day > ref:
            continue
        a = out.setdefault(code, {"vote": 0.0, "abs": 0.0, "n": 0,
                                  "cars": [], "types": [], "last": day})
        d = int(it.get("direction") or 0)
        dec = 0.5 ** (_age(day, ref) / HALF_LIFE)
        a["vote"] += d * dec
        a["abs"] += abs(d) * dec
        a["n"] += 1
        a["last"] = max(a["last"], day)
        imp = (it.get("implied") or {}).get("implied_car")
        if imp is not None:
            a["cars"].append(float(imp))
        if it.get("type"):
            a["types"].append(it["type"])
    return out


def score_of(agg: Optional[Dict]) -> Dict:
    """单票情报分：0.6 方向票 + 0.4 历史参照。无情报 → 0（不表态）。"""
    if not agg or not agg.get("n"):
        return {"intel": 0.0, "vote": 0.0, "analog": 0.0, "n": 0, "basis": "none"}
    vote = agg["vote"] / agg["abs"] if agg["abs"] > 1e-9 else 0.0
    cars = agg.get("cars") or []
    analog = (sum(cars) / len(cars)) / SCALE if cars else 0.0
    return {"intel": round(_clip(0.6 * vote + 0.4 * _clip(analog)), 4),
            "vote": round(_clip(vote), 4), "analog": round(_clip(analog), 4),
            "n": int(agg["n"]), "basis": "news+analog" if cars else "news"}


def snapshot(as_of: Optional[str] = None) -> Dict:
    d = _read(STATE, {})
    items = d.get("items") or []
    agg = aggregate(items, as_of=as_of)
    by_code = {c: score_of(a) for c, a in agg.items()}
    n_nonzero = sum(1 for v in by_code.values() if abs(v["intel"]) > 1e-9)
    return {"as_of": as_of or datetime.now().strftime("%Y-%m-%d"),
            "generated_at": d.get("generated_at"),
            "n_codes": len(by_code), "n_nonzero": n_nonzero,
            "by_code": by_code,
            "note": "情报分 ∈ [-1,1]，0=不表态；只吃 as_of <= 目标日的条目"}


def _ranks(vals: List[float]) -> List[float]:
    """把有限值映射到 [-1,1] 的排位分（并列取平均秩）。NaN 原样保留为 nan。"""
    idx = [i for i, v in enumerate(vals) if isinstance(v, float) and math.isfinite(v)]
    if not idx:
        return [float("nan")] * len(vals)
    order = sorted(idx, key=lambda i: vals[i])
    n = len(order)
    out = [float("nan")] * len(vals)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        p = 2.0 * ((i + j) / 2.0) / max(n - 1, 1) - 1.0     # [-1,1]
        for k in range(i, j + 1):
            out[order[k]] = p
        i = j + 1
    return out


def blend_arrays(codes: List[str], model_scores: List[float],
                 intel: Dict[str, Dict], w: float = 0.3) -> Dict:
    """排位融合。model_scores 可含 NaN（不可交易），NaN 位置输出仍为 NaN。"""
    w = max(0.0, min(1.0, float(w)))
    ms = [float(x) if x is not None else float("nan") for x in model_scores]
    isc = []
    for c in codes:
        v = (intel.get(str(c)) or {}).get("intel")
        isc.append(float(v) if v is not None else 0.0)
    zm = _ranks(ms)
    zi = _ranks(isc)
    out, used = [], 0
    for i, c in enumerate(codes):
        if not math.isfinite(ms[i]):
            out.append(float("nan"))
            continue
        if w <= 0:
            out.append(ms[i])
            continue
        z = (1 - w) * zm[i] + w * (zi[i] if math.isfinite(zi[i]) else 0.0)
        out.append(round(z, 6))
        used += 1
    return {"scores": out, "n_blended": used, "w": w,
            "score_kind": "rank_blend" if w > 0 else "model",
            "n_intel": sum(1 for c in codes if abs((intel.get(str(c)) or {}).get("intel") or 0) > 1e-9)}


def summary() -> Dict:
    s = snapshot()
    vals = [v["intel"] for v in s["by_code"].values()]
    return {"ok": True, "as_of": s["as_of"], "n_codes": s["n_codes"],
            "n_nonzero": s["n_nonzero"],
            "mean": round(sum(vals) / len(vals), 4) if vals else None,
            "max_abs": round(max((abs(v) for v in vals), default=0.0), 4),
            "note": "情报分覆盖度；n_nonzero=0 说明今天没有可用情报（正常：离线/未采集）"}


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        # 自检：PIT 过滤 / 空数据 / 融合不改变 NaN
        fake = [{"code": "600519", "as_of": "2026-11-10", "direction": 1, "implied": {"implied_car": 0.03}},
                {"code": "600519", "as_of": "2026-11-20", "direction": -1, "implied": {"implied_car": -0.03}}]
        a = aggregate(fake, as_of="2026-11-11")
        assert a["600519"]["n"] == 1, "PIT 过滤失败：未来条目被计入"
        assert abs(score_of(a["600519"])["intel"]) > 0.5, "情报分方向不对"
        assert score_of(None)["intel"] == 0.0, "空数据应为 0"
        r = blend_arrays(["a", "b", "c"], [0.1, float("nan"), 0.3], {"a": {"intel": 1.0}}, w=0.3)
        assert math.isnan(r["scores"][1]), "NaN 必须保留"
        assert r["score_kind"] == "rank_blend"
        print('{"ok": true, "checks": ["PIT", "score", "empty", "nan", "kind"]}')
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
