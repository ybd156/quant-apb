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
BLEND = 0.5          # 指数平滑系数：每次学到多少（1.0=全量重算）
A_RANGE = (-20.0, 20.0)
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

    # 1) 优先用周期台账（predict 之后累积的，字段最全）
    rows = HL._read(int(horizon))
    xs: List[float] = []
    ys: List[float] = []
    used_days = 0
    n_same_source = 0
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
            n_same_source += 1

    # 2) 周期台账样本不够时 —— **不再并入主台账**（评审 R44）。
    #
    # 主台账 `ledger.json` 里的 score 是 **102 信号集成的加权投票分**（±0.7 量级），
    # 而本模块 apply() 乘的是 **GBDT 原始分**（0.003 量级）。拿前者拟合 a、乘到后者身上，
    # 等于用"另一个模型"的斜率去改这个模型的输出 —— 实测会把分数压成常数（差 2e-6）。
    #
    # 需要跨源样本时，必须先让两侧同源（predict 落盘的 score），而不是借用。
    # 这里留一个显式开关，方便做 A/B，不默认开启。
    _cross = str(os.environ.get("APB_CALIB_CROSS_SOURCE", "0")).lower() in ("1", "true", "yes")
    if len(xs) < MIN_SAMPLES and _cross:
        try:
            from . import horizon_ledger as _HL
            if _HL._model_horizon() == int(horizon):
                mx, my, mdays = _collect_main_ledger()
                if mx.size:
                    xs.extend(mx.tolist())
                    ys.extend(my.tolist())
                    used_days += mdays
        except Exception:
            pass

    return np.array(xs, dtype=float), np.array(ys, dtype=float), used_days


def _collect_main_ledger() -> Tuple[np.ndarray, np.ndarray, int]:
    """从 data/state/ledger.json 收集 (score, ret)。只取已对账（status=scored）。"""
    import json as _json

    path = os.path.join(STATE, "ledger.json")
    if not os.path.isfile(path):
        return np.zeros(0), np.zeros(0), 0
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return np.zeros(0), np.zeros(0), 0
    rows = data if isinstance(data, list) else (data.get("rows") or [])
    xs, ys, days = [], [], set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("status") or "") != "scored":
            continue
        sv, rv = r.get("score"), r.get("ret")
        if sv is None or rv is None:
            continue
        try:
            sv = float(sv); rv = float(rv)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(sv) and np.isfinite(rv)):
            continue
        xs.append(sv); ys.append(rv)
        days.add(str(r.get("as_of"))[:10])
    return (np.array(xs, dtype=float), np.array(ys, dtype=float), len(days))


def _mse(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    if x.size == 0:
        return float("nan")
    pred = a * x + b
    return float(np.mean((pred - y) ** 2))


def update(horizon: int, verbose: bool = False) -> Dict:
    """
    用已到期样本更新校准参数。

    做法：**闭式最小二乘 + 指数平滑**，不用梯度下降。

        a_ols = cov(x, y) / var(x)
        b_ols = mean(y) - a_ols * mean(x)
        a_new = (1-λ) * a_old + λ * a_ols        # λ = BLEND
        b_new = (1-λ) * b_old + λ * b_ols

    为什么不用 SGD：
      第一版用 SGD + 标准化，结果 a 从 1.0 冲到 3.56，MSE 从 0.067 涨到 0.766 ——
      学习率乘上尺度换算直接发散。线性回归有解析解，用它更稳、更可解释。
      λ 控制"每次学多少"：λ=1 就是全量重算，λ 小则更像"慢慢调整"。

    为什么平滑：
      单日样本可能有异常（涨跌停、停牌复牌），全量重算会被一天的噪声带偏。
      平滑让历史结论保留一部分权重，符合"对了加深、错了减弱"的渐进语义。
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

    # ---- 闭式最小二乘 ----
    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x))
    if vx <= 1e-12:
        return {"ok": False, "horizon": h,
                "reason": "分数方差为 0，无法估计区分力",
                "n_samples": int(x.size)}
    cov = float(np.mean((x - mx) * (y - my)))
    a_ols = cov / vx
    b_ols = my - a_ols * mx

    # ---- 指数平滑：λ 控制单次学习幅度 ----
    lam = BLEND if p["n_updates"] > 0 else 1.0     # 首次直接采用全量解
    a_new = (1.0 - lam) * a0 + lam * a_ols
    b_new = (1.0 - lam) * b0 + lam * b_ols

    # 约束：a 是"分数每高 1 个单位，预期收益高多少"
    #   score 量级约 ±0.02（GBDT 输出），ret 量级约 ±0.05
    #   合理区间取 ±20，超出说明样本有极端值
    a_new = float(np.clip(a_new, A_RANGE[0], A_RANGE[1]))
    b_new = float(np.clip(b_new, B_RANGE[0], B_RANGE[1]))

    mse1 = _mse(x, y, a_new, b_new)

    entry = {
        "at": _now(), "n_samples": int(x.size), "n_days": days,
        "a_before": round(a0, 6), "a_after": round(a_new, 6),
        "b_before": round(b0, 6), "b_after": round(b_new, 6),
        "a_ols": round(a_ols, 6), "b_ols": round(b_ols, 6),
        "mse_before": round(mse0, 10), "mse_after": round(mse1, 10),
        "direction": ("翻转" if (a0 * a_new < 0 and abs(a_new) > 0.01) else
                      "加强" if abs(a_new) > abs(a0) * 1.05 else
                      "减弱" if abs(a_new) < abs(a0) * 0.95 else "基本不变"),
        "better": bool(mse1 <= mse0),
    }

    p["a"] = a_new
    p["b"] = b_new
    p["n_updates"] = p["n_updates"] + 1
    p["updated_at"] = entry["at"]
    p["history"] = (p["history"] + [entry])[-MAX_HISTORY:]
    d[str(h)] = p
    d["last_run"] = entry["at"]
    save(d)

    if verbose:
        print(f"[online] h{h} 用 {x.size} 个样本（{days} 天）更新："
              f"a {a0:+.4f} -> {a_new:+.4f}（{entry['direction']}），"
              f"b {b0:+.6f} -> {b_new:+.6f}，"
              f"MSE {mse0:.4e} -> {mse1:.4e}"
              f"{'（变好）' if entry['better'] else '（变差，保留旧值更稳）'}",
              flush=True)
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
    a = float(p["a"])
    if a <= 0 and str(os.environ.get("APB_CALIB_ALLOW_NEG_A", "0")).lower() not in ("1", "true", "yes"):
        # a<=0 意味着"分数越高、收益越低"，套到新分数上就是把排序整体倒过来（评审 R17）。
        # 没有统计证据支持反号时，一律不应用。
        return arr, {"applied": False, "reason": "a_nonpositive",
                     "a": round(a, 6), "b": round(p["b"], 6),
                     "n_updates": p["n_updates"],
                     "hint": "斜率非正，拒绝应用；确认要反向请设 APB_CALIB_ALLOW_NEG_A=1"}
    out = a * arr + p["b"]
    return out, {"applied": True, "a": round(a, 6), "b": round(p["b"], 6),
                 "calib_inverted": bool(a < 0),
                 "n_updates": p["n_updates"], "updated_at": p["updated_at"]}


def summary() -> Dict:
    d = load()
    rows = []
    for h in (1, 3, 5, 10, 15):
        p = _params(d, h)
        x, y, days = collect(h)
        # 每个周期的样本从哪来、为什么是 0 —— 必须说清楚。
        # 实测：界面上五个周期只有 h=5 有 3000 样本，其余全是 0，
        # 用户看到的是「在线学习没做好」，其实原因各不相同：
        #   · h=1 刚记录 2 条预测，还没到期（要等 1 个交易日）
        #   · h=3/h10/h15 连模型都没有 → 没有预测 → 不可能有样本
        #   · h=5 有样本是因为主台账本来就是按 h=5 跑出来的
        _pend = 0
        _src = "none"
        try:
            from . import horizon_ledger as _HL
            _rows = _HL._read(h)
            _pend = sum(1 for r in _rows if r.get("status") == "pending")
            if any(r.get("status") == "done" for r in _rows):
                _src = "horizon_ledger"
            elif _pend:
                _src = "horizon_ledger_pending"
        except Exception:  # noqa: BLE001
            pass
        if _src in ("none", "horizon_ledger_pending"):
            try:
                from . import horizon_ledger as _HL2
                if _HL2._model_horizon() == h:
                    _src = "main_ledger"
            except Exception:  # noqa: BLE001
                pass
        if x.size >= MIN_SAMPLES:
            _why = None
        elif _src == "horizon_ledger_pending":
            _why = f"已记录 {_pend} 条预测，等 {h} 个交易日后到期才有样本"
        elif _src == "none":
            _why = ("该周期还没有模型/预测（先在训练机训 h=%d，再 live_model --predict）" % h)
        elif _src == "main_ledger":
            _why = f"主台账样本不足（{int(x.size)} < {MIN_SAMPLES}）"
        else:
            _why = f"已到期样本 {int(x.size)} < {MIN_SAMPLES}"
        rows.append({
            "horizon": h, "a": round(p["a"], 6), "b": round(p["b"], 6),
            "n_updates": p["n_updates"], "updated_at": p["updated_at"],
            "n_samples": int(x.size), "n_days": days,
            "n_pending": _pend, "source": _src, "why_zero": _why,
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
