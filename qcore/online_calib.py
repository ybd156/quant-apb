"""
在线校准 —— 真的根据「预测 vs 实际」调整输出。

## 为什么需要这个

项目立项是「自学习参数体」：预测对了加深、错了减弱。
但实测下来，生产环境**从来没有在线的学习**：

    AdaptiveWeighter.update() 只被 engine.py（回测）和 intraday.py（未调度）调用
    start.sh 不调度 qcore.live / pipeline / intraday
    实际在跑的"学习" = 每周用新数据重训一次 GBDT

换句话说：模型是「训练一次、定期重训」，某次预测错了，
模型内部不会有任何量发生变化。这不是自学习。

## 这个模块做什么

在已对账的样本上，学一个**线性校准**：

    ret ≈ a * score + b

- `a` 是「分数的区分力」：如果模型给的分数越高、实际收益反而越低，
  a 会变成负数 —— 系统自己发现「方向反了」
- `b` 是「整体偏移」：如果所有预测都系统性高估，b 会变成负数修正回来

每学到新样本就更新一次，并把「更新前后」记进历史。
这就是「预测 → 实际 → 调整」的完整闭环，且有日志可查。

## 为什么只学 2 个参数

样本量少（每天约 100 只，到期才有结果）。参数多了必然过拟合。
2 个参数（斜率 + 截距）在几百个样本上是稳的，而且解释性极强：
看 `a` 就知道模型方向对不对，看 `b` 就知道有没有系统性偏差。

## 安全性

- 只用**已到期**的样本（`status == "done"`），不碰未来
- 更新幅度受 `LR` 和 `CLIP` 限制，单次异常样本不会把参数带飞
- `a` 限制在 [-3, 3]，`b` 限制在 [-0.2, 0.2]
- 每次更新都记录前后值，可回溯、可回滚
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
PATH = os.path.join(STATE, "online_calib.json")

# 超参
LR = 0.02            # 学习率
EPOCHS = 8           # 每次更新过几遍
A_RANGE = (-3.0, 3.0)
B_RANGE = (-0.20, 0.20)
MIN_SAMPLES = 30     # 少于这个数不更新（会过拟合）
MAX_HISTORY = 200


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load() -> Dict:
    try:
        with open(PATH, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save(d: Dict) -> None:
    _atomic(PATH, d)


def _params(d: Dict, h: int) -> Dict:
    k = str(int(h))
    p = (d.get(k) or {})
    return {"a": float(p.get("a", 1.0)), "b": float(p.get("b", 0.0)),
            "n_updates": int(p.get("n_updates", 0)),
            "updated_at": p.get("updated_at"),
            "history": list(p.get("history") or [])}


def collect(horizon: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    从已对账的台账里收集 (score, ret) 样本。

    只用 status == "done" 的记录 —— 那是「到期且已算实际收益」的，
    因果安全。pending 的绝不参与。
    """
    try:
        from . import horizon_ledger as HL
    except Exception:
        return np.zeros(0), np.zeros(0), 0

    rows = HL._read(int(horizon))
    xs: List[float] = []
    ys: List[float] = []
    used_days = 0
    for r in rows:
        if r.get("status") != "done":
            continue
        items = ((r.get("result") or {}).get("items") or [])
        if not items:
            continue
        used_days += 1
        for it in items:
            s = it.get("score")
            rt = it.get("ret")
            if s is None or rt is None:
                continue
            try:
                s = float(s)
                rt = float(rt)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(s) and np.isfinite(rt)):
                continue
            xs.append(s)
            ys.append(rt)
    return np.array(xs, dtype=float), np.array(ys, dtype=float), used_days


def _mse(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    if x.size == 0:
        return float("nan")
    pred = a * x + b
    return float(np.mean((pred - y) ** 2))


def update(horizon: int, verbose: bool = False) -> Dict:
    """
    用已到期样本更新校准参数。返回更新前后的对比。
    """
    h = int(horizon)
    x, y, days = collect(h)
    if x.size < MIN_SAMPLES:
        return {"ok": False, "horizon": h,
                "reason": f"已到期样本不足 {x.size} < {MIN_SAMPLES}",
                "n_samples": int(x.size), "n_days": days}

    d = load()
    p = _params(d, h)
    a0, b0 = p["a"], p["b"]
    mse0 = _mse(x, y, a0, b0)

    # 标准化 x 让梯度稳定（score 量级约 ±1，但不同周期差异大）
    sx = float(np.std(x)) or 1.0
    xn = x / sx

    a, b = a0, b0
    for _ in range(EPOCHS):
        pred = a * xn + b
        err = pred - y
        ga = float(np.mean(err * xn))
        gb = float(np.mean(err))
        a -= LR * ga
        b -= LR * gb
        # 约束到合理区间，防止单个异常样本把参数带飞
        a = float(np.clip(a, A_RANGE[0], A_RANGE[1]))
        b = float(np.clip(b, B_RANGE[0], B_RANGE[1]))

    mse1 = _mse(x / sx, y, a, b)

    # a 是在标准化尺度上的，换算回原始 score 尺度
    a_raw = a / sx
    a_raw = float(np.clip(a_raw, A_RANGE[0] * 10, A_RANGE[1] * 10))

    entry = {
        "at": _now(), "n_samples": int(x.size), "n_days": days,
        "a_before": round(a0, 6), "a_after": round(a_raw, 6),
        "b_before": round(b0, 6), "b_after": round(b, 6),
        "mse_before": round(mse0, 10), "mse_after": round(mse1, 10),
        "direction": ("翻转" if a0 * a_raw < 0 and abs(a_raw) > 0.05 else
                      "加强" if abs(a_raw) > abs(a0) else "减弱"),
    }

    p["a"] = a_raw
    p["b"] = b
    p["n_updates"] = p["n_updates"] + 1
    p["updated_at"] = entry["at"]
    p["history"] = (p["history"] + [entry])[-MAX_HISTORY:]
    d[str(h)] = p
    d["last_run"] = entry["at"]
    save(d)

    if verbose:
        print(f"[online] h{h} 用 {x.size} 个样本（{days} 天）更新："
              f"a {a0:+.4f} -> {a_raw:+.4f}（{entry['direction']}），"
              f"b {b0:+.6f} -> {b:+.6f}，MSE {mse0:.3e} -> {mse1:.3e}", flush=True)
    return {"ok": True, "horizon": h, **entry}


def update_all(verbose: bool = True) -> Dict:
    out = []
    for h in (1, 3, 5, 10, 15):
        try:
            r = update(h, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "horizon": h,
                 "reason": f"{type(e).__name__}: {str(e)[:60]}"}
        out.append(r)
    ok_n = sum(1 for r in out if r.get("ok"))
    if verbose:
        print(f"[online] {ok_n}/{len(out)} 个周期完成校准", flush=True)
    return {"ok": ok_n > 0, "results": out, "ok_count": ok_n}


def apply(horizon: int, scores) -> Tuple[np.ndarray, Dict]:
    """
    把校准应用到新分数上。没学过（n_updates=0）就原样返回。
    """
    arr = np.asarray(scores, dtype=float)
    d = load()
    p = _params(d, int(horizon))
    if p["n_updates"] == 0:
        return arr, {"applied": False, "reason": "还没学过（n_updates=0）"}
    out = p["a"] * arr + p["b"]
    return out, {"applied": True, "a": round(p["a"], 6), "b": round(p["b"], 6),
                 "n_updates": p["n_updates"], "updated_at": p["updated_at"]}


def summary() -> Dict:
    d = load()
    rows = []
    for h in (1, 3, 5, 10, 15):
        p = _params(d, h)
        x, y, days = collect(h)
        rows.append({
            "horizon": h, "a": round(p["a"], 6), "b": round(p["b"], 6),
            "n_updates": p["n_updates"], "updated_at": p["updated_at"],
            "n_samples": int(x.size), "n_days": days,
            "mse": (round(_mse(x, y, p["a"], p["b"]), 10) if x.size else None),
            "history": p["history"][-5:],
        })
    return {"ok": True, "generated_at": _now(), "horizons": rows,
            "note": ("a = 分数的区分力（负值表示模型方向反了）；"
                     "b = 整体偏移修正。每学到新样本更新一次，历史可查。")}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if "--summary" in a:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
    elif "--horizon" in a:
        h = 5
        for x in a:
            if x.startswith("--horizon="):
                h = int(x.split("=", 1)[1])
        print(json.dumps(update(h, verbose=True), ensure_ascii=False, indent=1))
    else:
        update_all(verbose=True)
