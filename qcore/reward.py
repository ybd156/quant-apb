"""
奖惩机制 —— 让"实践出真知"变成可执行的资源分配规则。

## 为什么单独的权重学习不够

现在的学习器只做一件事：**调整听谁的多**。但它没有"后果"：
- 一个信号连续亏 20 次，权重只降到 1/73，仍然占着资金
- 系统整体回撤 30%，仓位不变
- 我说错了 10 次，你看到的还是同一套结论

**奖惩机制要解决的是"后果"**：分数直接决定**给多少资源**，以及**什么时候停手**。

## 三层奖惩

| 层级 | 打分对象 | 奖惩后果 |
|---|---|---|
| **信号级** | 每个信号的近期表现 | 分数 → 权重上限/下限（赢家放大 2 倍，输家砍到 10%） |
| **系统级** | 策略整体净值 | 回撤/连亏 → **仓位乘数**（0~1），最坏情况强制空仓 |
| **判断级** | 我给出的结论 | 命中率 → **信任乘数**，低于 50% 就在界面上标"别信我" |

## 核心设计：罚得比奖得快

```
reward_lr = 0.10     # 做对了，慢慢加
penalty_lr = 0.30    # 做错了，立刻砍
```

这是**非对称奖惩**，理由有三：
1. 金融亏损是不可逆的（亏 50% 要涨 100% 才回本）
2. 行为金融的损失厌恶（Kahneman & Tversky 1979）说明人对亏损更敏感，
   系统也该如此
3. 防止"一次侥幸大赚"掩盖长期无能 —— 奖励慢，惩罚快，长期看只有真本事能活

## 关键原则

- **分数不是永久的**：用半衰期加权，60 天前的表现只算一半
- **有下限**：最差砍到 10%，不清零（保留"再给一次机会"，避免永久饿死）
- **有上限**：最好放大到 2 倍，不无限加（防止单信号绑架整个组合）
- **可解释**：每一次奖惩都写清"因为什么、扣了多少、什么时候能恢复"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RewardConfig:
    # ---- 非对称学习率（罚比奖快 3 倍）
    reward_lr: float = 0.10
    penalty_lr: float = 0.30
    half_life_days: int = 60

    # ---- 信号级边界
    max_boost: float = 2.0
    min_cut: float = 0.10

    # ---- 系统级（仓位乘数）
    dd_levels: Tuple[Tuple[float, float], ...] = (
        (0.10, 0.80), (0.15, 0.60), (0.20, 0.40), (0.30, 0.15), (0.40, 0.0))
    losing_streak_levels: Tuple[Tuple[int, float], ...] = (
        (5, 0.80), (8, 0.60), (12, 0.30), (20, 0.0))
    min_multiplier: float = 0.0
    recovery_periods: int = 20       # 恢复需要连续盈利期数

    # ---- 判断级（对我的信任）
    trust_levels: Tuple[Tuple[float, float], ...] = (
        (0.65, 1.0), (0.55, 0.85), (0.45, 0.60), (0.0, 0.35))


# ------------------------------------------------------------------ 信号级
def _decay_weights(n: int, half_life: int) -> np.ndarray:
    """半衰期加权：越近的样本权重越大。"""
    age = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (age / max(half_life, 1))


def signal_scorecard(payoffs: np.ndarray, sids: List[str],
                     cfg: RewardConfig = RewardConfig()) -> pd.DataFrame:
    """
    payoffs: (T, S) 每信号每期的标准化收益（NaN=未发声）
    返回每信号的记分卡：命中率 / 平均收益 / 样本数 / 分数 / 权重乘数 / 等级
    """
    T, S = payoffs.shape
    w = _decay_weights(T, cfg.half_life_days)
    rows = []
    for j in range(S):
        col = payoffs[:, j]
        m = np.isfinite(col)
        n = int(m.sum())
        if n < 10:
            rows.append({"sid": sids[j], "n": n, "hit": np.nan, "mean": np.nan,
                         "score": 0.0, "multiplier": 1.0, "grade": "N/A",
                         "reason": "样本不足，保持中性"})
            continue
        y = col[m]
        ww = w[m] / w[m].sum()
        hit = float((y > 0).mean())
        mean = float((y * ww).sum())
        # 分数 = 命中率偏离 0.5 的方向 + 加权平均收益
        score = (hit - 0.5) * 2.0 + mean
        # 非对称乘数：正分用 reward_lr，负分用 penalty_lr
        if score >= 0:
            mult = 1.0 + cfg.reward_lr * score * 10
        else:
            mult = 1.0 + cfg.penalty_lr * score * 10
        mult = float(np.clip(mult, cfg.min_cut, cfg.max_boost))
        grade = ("A" if mult >= 1.4 else "B" if mult >= 1.1 else
                 "C" if mult >= 0.8 else "D" if mult >= 0.4 else "E")
        rows.append({
            "sid": sids[j], "n": n, "hit": round(hit, 4), "mean": round(mean, 5),
            "score": round(score, 5), "multiplier": round(mult, 3), "grade": grade,
            "reason": (f"命中率 {hit:.1%}，加权均值 {mean:+.4f} → "
                       f"{'奖励' if score >= 0 else '惩罚'} ×{mult:.2f}"),
        })
    return pd.DataFrame(rows).sort_values("multiplier", ascending=False)


# ------------------------------------------------------------------ 系统级
def position_multiplier(equity: np.ndarray,
                        cfg: RewardConfig = RewardConfig()) -> Dict:
    """
    系统级仓位乘数：回撤 + 连续亏损 两条线取更严厉的那个。
    equity: 策略净值序列（等间隔，通常是非重叠调仓后的净值）。
    """
    e = np.asarray(equity, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if e.size < 5:
        return {"ok": False, "multiplier": 1.0, "reason": "样本不足，不施加限制"}

    peak = np.maximum.accumulate(e)
    dd = float(1.0 - e[-1] / peak[-1])          # 当前回撤
    max_dd = float((1.0 - e / peak).max())      # 历史最大回撤

    # 连续亏损期数
    r = np.diff(e) / e[:-1]
    streak = 0
    for x in r[::-1]:
        if x < 0:
            streak += 1
        else:
            break

    m_dd = 1.0
    for th, mult in cfg.dd_levels:
        if dd >= th:
            m_dd = mult
    m_streak = 1.0
    for th, mult in cfg.losing_streak_levels:
        if streak >= th:
            m_streak = mult

    mult = float(min(m_dd, m_streak))
    reasons = []
    if m_dd < 1.0:
        reasons.append(f"当前回撤 {dd:.1%} → 仓位 ×{m_dd:.2f}")
    if m_streak < 1.0:
        reasons.append(f"连续亏损 {streak} 期 → 仓位 ×{m_streak:.2f}")
    if not reasons:
        reasons.append(f"回撤 {dd:.1%}、连亏 {streak} 期，均未触发限制")

    return {"ok": True, "multiplier": round(mult, 3),
            "drawdown": round(dd, 4), "max_drawdown": round(max_dd, 4),
            "losing_streak": int(streak), "reason": "；".join(reasons),
            "action": ("**强制空仓**" if mult <= 0.001 else
                       "大幅降杠杆" if mult < 0.4 else
                       "适度降杠杆" if mult < 0.8 else "正常仓位")}


# ------------------------------------------------------------------ 判断级
def trust_multiplier(claims_score: Dict,
                     cfg: RewardConfig = RewardConfig()) -> Dict:
    """
    对我的判断的信任乘数：命中率低 → 降低信任，并在界面上明说。
    claims_score 来自 qcore.claims.score()。
    """
    hit = claims_score.get("hit_rate")
    n = claims_score.get("resolved", 0)
    if hit is None or n < 5:
        return {"ok": False, "multiplier": 0.6,
                "reason": f"已结算判断仅 {n} 条，样本不足 → 信任打 6 折（默认谨慎）",
                "label": "证据不足"}
    mult = 1.0
    for th, m in cfg.trust_levels:
        if hit >= th:
            mult = m
            break
    label = ("可信" if mult >= 0.9 else "需打折" if mult >= 0.6 else "不可信")
    return {"ok": True, "multiplier": round(mult, 2), "hit_rate": hit,
            "n_resolved": n, "label": label,
            "reason": f"我的判断命中率 {hit:.0%}（n={n}）→ 信任 ×{mult:.2f}"}


# ------------------------------------------------------------------ 总装
def allocate(base_weights: np.ndarray, scorecard: pd.DataFrame, sids: List[str],
             pos_mult: float = 1.0, trust_mult: float = 1.0) -> Dict:
    """
    把"基础权重 × 信号奖惩 × 系统仓位 × 判断信任"合成最终权重。
    返回最终权重与可解释的奖惩明细。
    """
    w = np.asarray(base_weights, dtype=float).copy()
    mult = np.ones(len(sids))
    idx = {s: i for i, s in enumerate(sids)}
    for r in scorecard.itertuples(index=False):
        if r.sid in idx:
            mult[idx[r.sid]] = float(r.multiplier)

    w_rewarded = w * mult
    if w_rewarded.sum() <= 0:
        w_rewarded = w
    w_rewarded = w_rewarded / w_rewarded.sum()

    # 系统级与判断级作用在**总仓位**上，不改相对权重
    exposure = float(np.clip(pos_mult * trust_mult, 0.0, 1.0))
    final = w_rewarded * exposure

    top_up = scorecard.nlargest(3, "multiplier")[["sid", "multiplier", "grade"]]
    top_dn = scorecard.nsmallest(3, "multiplier")[["sid", "multiplier", "grade"]]
    return {
        "final_weights": final,
        "relative_weights": w_rewarded,
        "exposure": round(exposure, 3),
        "pos_multiplier": round(float(pos_mult), 3),
        "trust_multiplier": round(float(trust_mult), 3),
        "rewarded_up": top_up.to_dict("records"),
        "rewarded_down": top_dn.to_dict("records"),
        "summary": (f"总仓位 {exposure:.0%}（系统×{pos_mult:.2f} × 信任×{trust_mult:.2f}）；"
                    f"最强信号 {top_up.iloc[0]['sid']}(×{top_up.iloc[0]['multiplier']:.2f})，"
                    f"最弱 {top_dn.iloc[0]['sid']}(×{top_dn.iloc[0]['multiplier']:.2f})"),
    }


def reward_report(scorecard: pd.DataFrame, pos: Dict, trust: Dict,
                  top: int = 10) -> str:
    """生成人类可读的奖惩报告（Markdown）。"""
    L = ["## 奖惩记分卡", ""]
    if pos.get("ok"):
        L.append(f"**系统仓位：{pos['action']}（×{pos['multiplier']}）**")
        L.append(f"> {pos['reason']}")
        L.append("")
    L.append(f"**对我的信任：{trust.get('label','?')}（×{trust.get('multiplier',1)}）**")
    L.append(f"> {trust.get('reason','')}")
    L.append("")
    L.append("### 受奖励（权重放大）")
    L.append("| 信号 | 等级 | 命中率 | 加权均值 | 乘数 | 原因 |")
    L.append("|---|---|---|---|---|---|")
    for r in scorecard.head(top).itertuples(index=False):
        L.append(f"| {r.sid} | {r.grade} | {r.hit:.1%} | {r.mean:+.4f} | ×{r.multiplier:.2f} | {r.reason} |")
    L.append("")
    L.append("### 受惩罚（权重削减）")
    L.append("| 信号 | 等级 | 命中率 | 加权均值 | 乘数 | 原因 |")
    L.append("|---|---|---|---|---|---|")
    for r in scorecard.tail(top).iloc[::-1].itertuples(index=False):
        L.append(f"| {r.sid} | {r.grade} | {r.hit:.1%} | {r.mean:+.4f} | ×{r.multiplier:.2f} | {r.reason} |")
    L.append("")
    L.append("> 罚得比奖得快（penalty_lr=0.30 vs reward_lr=0.10）：")
    L.append("> 亏损不可逆，一次侥幸大赚不该掩盖长期无能。")
    return "\n".join(L)
