"""
模型健康度 —— 明确回答「这个模型还能不能用」。

## 为什么必须有这个

现在有告警（模型超龄、数据源挂），但没有**"模型已经废了"的硬判据**。
结果就是：IC 衰减了几个月，用户还在照着 picks 下单。

这比任何参数优化都重要——**知道什么时候该停手，比多赚 1% 更值钱**。

## 判据（事先定好，不事后找理由）

| 级别 | 条件 | 动作 |
|---|---|---|
| healthy  | 滚动 60 日 IC > 0 且 t > 1 | 正常使用 |
| watch    | 滚动 60 日 IC > 0 但 t ≤ 1 | 减半仓位 |
| degraded | 滚动 60 日 IC ≤ 0，或连续 10 日 IC 为负 | 停用，重训 |
| failed   | 滚动 60 日 IC < 0 且 t < −2 | **立即停用**，不再生成 picks |

## IC 从哪来

`qcore.live_model.predict` 每天打分后，用**已到期的**样本对账，把每日 IC
追加到 `data/state/ic_history.jsonl`。到期判据：T 日信号在 T+1+h 才可评价（INV-2）。

## 因果红线

评价 T 日信号必须等 T+1+h 的数据。**不能用当天数据评价当天信号**——
这个坑在 `session.postclose` 里已经踩过一次。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
HIST = os.path.join(STATE, "ic_history.jsonl")

# 判据阈值（集中在这里，改阈值不改逻辑）
WINDOW = 60            # 滚动窗口（交易日）
T_WATCH = 1.0          # t 低于此值进入 watch
T_FAIL = -2.0          # t 低于此值判 failed
CONSEC_NEG = 10        # 连续多少日 IC 为负进入 degraded


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_history() -> List[Dict]:
    out = []
    if not os.path.isfile(HIST):
        return out
    try:
        with open(HIST, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out


def record(ic: float, as_of: str, n: int = 0, source: str = "live") -> str:
    """追加一条 IC 记录。重复日期会覆盖（同一天只留一条）。"""
    hist = [h for h in load_history() if str(h.get("as_of")) != str(as_of)]
    hist.append({"as_of": str(as_of), "ic": float(ic), "n": int(n),
                 "source": source, "recorded_at": _now()})
    hist.sort(key=lambda h: str(h.get("as_of")))
    os.makedirs(STATE, exist_ok=True)
    tmp = HIST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for h in hist:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, HIST)
    return HIST


def _tstat(x: np.ndarray) -> float:
    if x.size < 5:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(x) / (sd / np.sqrt(x.size)))


def evaluate() -> Dict:
    """按判据给出健康度。"""
    hist = load_history()
    if len(hist) < 20:
        return {"ok": True, "level": "unknown", "n_days": len(hist),
                "note": f"IC 样本仅 {len(hist)} 天，不足 20 天，暂不判定",
                "usable": True, "position_scale": 1.0}

    ics = np.array([h["ic"] for h in hist if isinstance(h.get("ic"), (int, float))],
                   dtype=float)
    ics = ics[np.isfinite(ics)]
    if ics.size < 20:
        return {"ok": True, "level": "unknown", "n_days": int(ics.size),
                "note": "有效 IC 不足 20 天", "usable": True, "position_scale": 1.0}

    win = ics[-WINDOW:]
    mean_ic = float(np.mean(win))
    t = _tstat(win)

    # 连续为负天数（从末尾往前数）
    consec = 0
    for v in ics[::-1]:
        if v < 0:
            consec += 1
        else:
            break

    if mean_ic < 0 and np.isfinite(t) and t < T_FAIL:
        level, scale, usable = "failed", 0.0, False
        note = f"滚动 {len(win)} 日 IC {mean_ic:+.4f} 且 t={t:.2f} < {T_FAIL} —— 立即停用"
    elif mean_ic <= 0 or consec >= CONSEC_NEG:
        level, scale, usable = "degraded", 0.0, False
        note = (f"滚动 IC {mean_ic:+.4f}，连续 {consec} 日为负 —— 停用并重训"
                if consec >= CONSEC_NEG else f"滚动 IC {mean_ic:+.4f} ≤ 0 —— 停用并重训")
    elif not np.isfinite(t) or t <= T_WATCH:
        level, scale, usable = "watch", 0.5, True
        note = f"滚动 IC {mean_ic:+.4f} 但 t={t:.2f} ≤ {T_WATCH} —— 建议减半仓位"
    else:
        level, scale, usable = "healthy", 1.0, True
        note = f"滚动 IC {mean_ic:+.4f}，t={t:.2f} —— 正常使用"

    return {"ok": True, "level": level, "usable": usable,
            "position_scale": scale, "mean_ic": round(mean_ic, 6),
            "t": round(t, 3) if np.isfinite(t) else None,
            "consec_negative": consec, "n_days": int(ics.size),
            "window": int(len(win)), "note": note,
            "as_of": hist[-1].get("as_of") if hist else None}


def should_generate_picks() -> Dict:
    """是否还应该生成推荐。failed/degraded 时返回 False。"""
    h = evaluate()
    return {"ok": True, "generate": bool(h.get("usable", True)),
            "level": h.get("level"), "position_scale": h.get("position_scale", 1.0),
            "reason": h.get("note")}


def summary() -> Dict:
    h = evaluate()
    hist = load_history()
    return {**h, "history_days": len(hist),
            "first": hist[0].get("as_of") if hist else None,
            "last": hist[-1].get("as_of") if hist else None}


if __name__ == "__main__":
    import json
    print("当前:", json.dumps(summary(), ensure_ascii=False, indent=1))
    print()
    # 用合成数据演示四种状态
    import tempfile
    real = HIST
    try:
        for tag, seq in (
            ("healthy", list(np.full(60, 0.02) + np.random.default_rng(1).normal(0, 0.01, 60))),
            ("watch",   list(np.full(60, 0.005) + np.random.default_rng(2).normal(0, 0.03, 60))),
            ("degraded", list(np.full(60, -0.01) + np.random.default_rng(3).normal(0, 0.01, 60))),
            ("failed",  list(np.full(60, -0.03) + np.random.default_rng(4).normal(0, 0.01, 60))),
        ):
            globals()["HIST"] = os.path.join(tempfile.gettempdir(), f"ic_{tag}.jsonl")
            if os.path.exists(HIST):
                os.remove(HIST)
            for i, v in enumerate(seq):
                record(float(v), f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}")
            e = evaluate()
            print(f"  {tag:<9} -> level={e['level']:<9} scale={e['position_scale']} "
                  f"usable={e['usable']} | {e['note'][:52]}")
    finally:
        globals()["HIST"] = real
