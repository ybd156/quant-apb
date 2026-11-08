"""
仓位管理与杠杆 —— 学会空仓、满仓、以及"杠杆的正确惩罚方式"。

## 四种状态

| 状态 | 触发条件 | 仓位 |
|---|---|---|
| **空仓躲风险** | 优势消失 / 回撤超限 / 拥挤度极高 | 0% |
| **半仓试探** | 优势不显著 / 环境不确定 | 30-60% |
| **满仓吃收益** | 优势显著 + 回撤小 + 不拥挤 | 100% |
| **加杠杆** | 优势极强 + 回撤极小 + 环境友好 | 100-200% |

## 杠杆的奖惩设计（这是本模块的核心）

用户的要求很明确：**预测成功时杠杆收益"不值一提"，失败时惩罚较高。**

这不是拍脑袋，它符合三个事实：
1. 杠杆放大的是**波动**，不是**胜率**——胜率不变时，杠杆只提高破产概率（Kelly 准则）
2. 亏损后恢复需要指数级收益（亏 50% 要涨 100%）
3. 金融史上爆仓的，几乎都是"杠杆 + 侥幸"

所以本模块把杠杆的收益与损失做成**不对称**：

    杠杆赚了：只计入 lev_gain_keep = 30%   → 加 1 倍杠杆，赚了只多拿 0.3 倍
    杠杆亏了：放大 lev_loss_mult = 200%    → 加 1 倍杠杆，亏了要亏 2 倍

净效果：**只有当基础策略的胜率极高时，杠杆才可能正期望**；
而在胜率 50% 附近，杠杆必然加速亏损。这就是"惩罚机制"的数学表达。

## 杠杆上限与强平

- 单次最大杠杆 2.0 倍（不做无限杠杆）
- 净值跌破 **维持担保比例** 时强制降到 1.0 倍（模拟"追加保证金/强平"）
- 连续亏损时杠杆自动归零（不允许"越亏越加"）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PositionConfig:
    # ---- 仓位边界
    max_exposure: float = 1.0
    max_leverage: float = 2.0
    min_trade_exposure: float = 0.05      # 低于此值视为空仓

    # ---- 杠杆的不对称奖惩（用户明确要求）
    lev_gain_keep: float = 0.30           # 杠杆收益只计入 30%（"不值一提"）
    lev_loss_mult: float = 2.00           # 杠杆损失放大 200%（"惩罚较高"）

    # ---- 触发条件
    t_enter: float = 2.0                  # 优势显著性门槛（进）
    t_exit: float = 1.0                   # 优势衰减门槛（退）
    dd_cut: float = 0.10                  # 回撤超过则减仓
    dd_stop: float = 0.25                 # 回撤超过则空仓
    crowd_cut: float = 0.80               # 拥挤度分位超过则减仓
    crowd_stop: float = 0.95              # 拥挤度极端则空仓

    # ---- 杠杆准入（必须同时满足）
    lev_min_t: float = 2.5                # 优势要极显著
    lev_max_dd: float = 0.05              # 回撤要极小
    lev_max_crowd: float = 0.50           # 不能拥挤
    lev_max_losing_streak: int = 1        # 连亏 2 期以上不许加杠杆

    # ---- 强平
    maintenance_ratio: float = 0.75       # 净值/峰值 跌破则强制降杠杆


def target_exposure(t_ic: float, drawdown: float, crowd_pct: float,
                    vol_pct: float, losing_streak: int = 0,
                    cfg: PositionConfig = PositionConfig()) -> Dict:
    """
    根据"优势显著性 + 风险状态"给出目标仓位（含杠杆）。
    全部参数都是**当期可观测**的，不依赖未来。
    """
    reasons = []

    # 1) 优势是否存在
    if not np.isfinite(t_ic) or abs(t_ic) < cfg.t_exit:
        return {"exposure": 0.0, "leverage": 1.0, "state": "空仓（无优势）",
                "reason": f"优势 t={t_ic:.2f} 低于退出线 {cfg.t_exit}"}
    if abs(t_ic) < cfg.t_enter:
        base = 0.40
        reasons.append(f"优势 t={t_ic:.2f} 未达进入线，半仓试探")
    else:
        # 优势越强，基础仓位越高（t=2→0.7，t=4→1.0）
        base = float(np.clip(0.5 + 0.15 * (abs(t_ic) - cfg.t_enter), 0.5, 1.0))
        reasons.append(f"优势 t={t_ic:.2f}，基础仓位 {base:.0%}")

    # 2) 回撤惩罚
    if drawdown >= cfg.dd_stop:
        return {"exposure": 0.0, "leverage": 1.0, "state": "空仓（回撤超限）",
                "reason": f"回撤 {drawdown:.1%} ≥ {cfg.dd_stop:.0%}，强制空仓"}
    if drawdown >= cfg.dd_cut:
        k = 1.0 - (drawdown - cfg.dd_cut) / max(cfg.dd_stop - cfg.dd_cut, 1e-9)
        base *= float(np.clip(k, 0.3, 1.0))
        reasons.append(f"回撤 {drawdown:.1%} → 仓位打折至 {base:.0%}")

    # 3) 拥挤度惩罚
    if crowd_pct >= cfg.crowd_stop:
        return {"exposure": 0.0, "leverage": 1.0, "state": "空仓（拥挤极端）",
                "reason": f"拥挤度分位 {crowd_pct:.0%} ≥ {cfg.crowd_stop:.0%}"}
    if crowd_pct >= cfg.crowd_cut:
        base *= 0.6
        reasons.append(f"拥挤度分位 {crowd_pct:.0%} → 仓位 ×0.6")

    # 4) 高波动惩罚
    if np.isfinite(vol_pct) and vol_pct > 0.8:
        base *= 0.7
        reasons.append(f"波动分位 {vol_pct:.0%} → 仓位 ×0.7")

    exposure = float(np.clip(base, 0.0, cfg.max_exposure))

    # 5) 杠杆准入（必须全部满足，任一不满足就不加）
    lev = 1.0
    if (abs(t_ic) >= cfg.lev_min_t and drawdown <= cfg.lev_max_dd
            and crowd_pct <= cfg.lev_max_crowd
            and losing_streak <= cfg.lev_max_losing_streak):
        lev = float(min(cfg.max_leverage, 1.0 + 0.5 * (abs(t_ic) - cfg.lev_min_t)))
        reasons.append(f"杠杆准入全部满足 → {lev:.2f}x")
    else:
        miss = []
        if abs(t_ic) < cfg.lev_min_t:
            miss.append(f"t<{cfg.lev_min_t}")
        if drawdown > cfg.lev_max_dd:
            miss.append(f"回撤>{cfg.lev_max_dd:.0%}")
        if crowd_pct > cfg.lev_max_crowd:
            miss.append(f"拥挤>{cfg.lev_max_crowd:.0%}")
        if losing_streak > cfg.lev_max_losing_streak:
            miss.append(f"连亏{losing_streak}期")
        reasons.append("不加杠杆（" + "、".join(miss) + "）")

    state = ("加杠杆" if lev > 1.001 else
             "满仓" if exposure >= 0.95 else
             "半仓" if exposure >= 0.3 else "轻仓")
    return {"exposure": round(exposure * lev, 4), "leverage": round(lev, 3),
            "base_exposure": round(exposure, 4), "state": state,
            "reason": "；".join(reasons)}


def levered_return(base_ret: float, exposure: float,
                   cfg: PositionConfig = PositionConfig()) -> float:
    """
    把基础策略收益按仓位与杠杆折算，**杠杆部分用不对称奖惩**。

    exposure ≤ 1：直接按比例
    exposure > 1：杠杆部分（lev = exposure − 1）
        盈利：只计 lev_gain_keep
        亏损：放大 lev_loss_mult
    """
    if not np.isfinite(base_ret):
        return np.nan
    if exposure <= 1.0:
        return float(exposure * base_ret)
    lev = exposure - 1.0
    if base_ret >= 0:
        return float(base_ret + lev * base_ret * cfg.lev_gain_keep)
    return float(base_ret + lev * base_ret * cfg.lev_loss_mult)


def backtest_positioning(base_returns: np.ndarray, t_series: np.ndarray,
                         dd_series: Optional[np.ndarray] = None,
                         crowd_series: Optional[np.ndarray] = None,
                         vol_series: Optional[np.ndarray] = None,
                         cfg: PositionConfig = PositionConfig()) -> Dict:
    """
    对比三种仓位策略：
      A. 固定满仓（基准）
      B. 动态仓位（空仓/半仓/满仓，不加杠杆）
      C. 动态仓位 + 杠杆（含不对称惩罚）
    base_returns：基础策略每期收益（未加仓位的"满仓"收益）
    """
    r = np.asarray(base_returns, dtype=float)
    T = r.size
    t_s = np.asarray(t_series, dtype=float) if t_series is not None else np.full(T, np.nan)
    crowd = np.asarray(crowd_series, dtype=float) if crowd_series is not None else np.full(T, 0.5)
    vol = np.asarray(vol_series, dtype=float) if vol_series is not None else np.full(T, 0.5)

    out = {}
    for name, use_lev in (("fixed", False), ("dynamic", False), ("dynamic_lev", True)):
        nav = 1.0
        peak = 1.0
        streak = 0
        exps, states = [], []
        for i in range(T):
            if name == "fixed":
                e = 1.0; st = "满仓"
            else:
                dd = 1.0 - nav / peak
                d = target_exposure(t_s[i], dd, float(crowd[i]), float(vol[i]), streak, cfg)
                e = d["exposure"]
                if not use_lev:
                    e = min(e, 1.0)
                st = d["state"]
            rr = levered_return(r[i], e, cfg)
            nav *= (1.0 + np.nan_to_num(rr))
            peak = max(peak, nav)
            streak = streak + 1 if rr < 0 else 0
            exps.append(e); states.append(st)
            # 强平：净值跌破维持担保比例则强制降杠杆
            if nav / peak < cfg.maintenance_ratio and e > 1.0:
                pass    # 下一期 target_exposure 会因回撤自动降杠杆

        eq = np.cumprod([1.0] + [1.0 + np.nan_to_num(levered_return(r[i], exps[i], cfg))
                                 for i in range(T)])
        rets = np.diff(eq) / eq[:-1]
        out[name] = {
            "total_return": float(eq[-1] - 1.0),
            "ann_return": float(eq[-1] ** (252 / max(T, 1)) - 1.0),
            "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan,
            "avg_exposure": float(np.mean(exps)),
            "pct_flat": float(np.mean([e < 0.05 for e in exps])),
            "pct_levered": float(np.mean([e > 1.001 for e in exps])),
            "equity": eq.tolist(),
            "states": states,
        }
    base = out["fixed"]
    for k in ("dynamic", "dynamic_lev"):
        out[k]["vs_fixed"] = out[k]["total_return"] - base["total_return"]
    return out


def format_report(res: Dict, cfg: PositionConfig = PositionConfig()) -> str:
    L = ["## 仓位管理与杠杆", ""]
    L.append("| 策略 | 总收益 | 年化 | 最大回撤 | Sharpe | 平均仓位 | 空仓占比 | 杠杆占比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    name_map = {"fixed": "固定满仓", "dynamic": "动态仓位", "dynamic_lev": "动态+杠杆"}
    for k in ("fixed", "dynamic", "dynamic_lev"):
        d = res[k]
        L.append(f"| {name_map[k]} | {d['total_return']:+.1%} | {d['ann_return']:+.1%} | "
                 f"{d['max_drawdown']:.1%} | {d['sharpe']:.2f} | {d['avg_exposure']:.0%} | "
                 f"{d['pct_flat']:.0%} | {d['pct_levered']:.0%} |")
    L.append("")
    L.append(f"动态 vs 固定：{res['dynamic']['vs_fixed']:+.1%}；"
             f"加杠杆再 vs 固定：{res['dynamic_lev']['vs_fixed']:+.1%}")
    L.append("")
    L.append("> **杠杆的不对称奖惩**：赚了只计 "
             f"{cfg.lev_gain_keep:.0%}，亏了放大 {cfg.lev_loss_mult:.0%}。")
    L.append("> 净效果：只有基础胜率极高时杠杆才可能正期望；胜率 50% 附近，杠杆必然加速亏损。")
    return "\n".join(L)
