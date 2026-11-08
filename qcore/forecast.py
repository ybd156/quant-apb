"""
前景与空间测算 —— 把"这个策略能走多远"算成可证伪的区间，而不是拍脑袋。

核心工具是 Grinold 基本法则：

        IR ≈ IC · √BR

IR  = 信息比率（年化超额 / 年化跟踪误差）
IC  = 预测值与实际收益的横截面相关系数
BR  = 广度（每年独立下注次数）

这个公式的价值不在于精确，而在于它**逼你把话说清楚**：
想让 IR 从 0.3 提到 1.0，要么 IC 翻倍，要么广度翻 11 倍。没有第三条路。

我们把"未来空间"拆成四项，每项都单独给区间和前提：

  ① IC 空间   —— 当前 IC 距离该类策略的经验上限还有多远
  ② 广度空间 —— 股票数 × 调仓频率，以及"有效独立下注数"（考虑相关性）
  ③ 成本空间 —— 换手 × 双边成本，吃掉多少 IR
  ④ 执行空间 —— 滑点、冲击成本、容量

**任何一项的乐观估计都必须写明"什么条件下才成立"**，否则就是画饼。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# 经验上限（来自公开文献与业界实践，作为"天花板"参照，不是承诺）
IC_CEILING = {
    "technical": 0.030,      # 纯技术指标横截面 IC 的经验上限量级
    "fundamental": 0.050,    # 基本面因子
    "alt_data": 0.060,       # 另类数据（另需验证）
    "composite": 0.070,      # 多源合成（极少见）
}


@dataclass
class ProjectionConfig:
    n_names: int = 100               # 平均持仓/可选标的数
    rebalance_days: int = 5          # 调仓间隔（交易日）
    cost_bps: float = 10.0           # 单边成本
    turnover_per_rebalance: float = 0.7
    annual_vol: float = 0.22         # 组合年化波动（用于把 IR 换成收益）
    family: str = "technical"
    n_signals: int = 73
    effective_signals: Optional[float] = None   # 有效独立信号数（相关性折扣后）
    avg_correlation: float = 0.30               # 持仓间平均相关（A股大票经验值 0.2-0.4）


def _ir(ic: float, breadth: float) -> float:
    return float(ic * math.sqrt(max(breadth, 1.0)))


def project(ic: float, ic_t: float, cfg: ProjectionConfig = ProjectionConfig()) -> Dict:
    """
    输入实测 IC 与 HAC t 值，输出前景与空间。
    ic_t 用于决定"这个 IC 本身可不可信"——不显著时所有推演都要打折。
    """
    periods_per_year = 252.0 / max(cfg.rebalance_days, 1)
    # 名义广度：持仓数 × 年调仓次数
    breadth_nominal = cfg.n_names * periods_per_year
    # 有效广度：横截面下注高度相关，必须扣相关性折扣
    #   BR_eff = periods × N / (1 + (N-1)·ρ)
    rho = float(np.clip(cfg.avg_correlation, 0.0, 0.95))
    n_indep = cfg.n_names / (1.0 + (cfg.n_names - 1) * rho)
    breadth = n_indep * periods_per_year
    cost_drag = (cfg.turnover_per_rebalance * cfg.cost_bps / 1e4
                 * 2.0 * periods_per_year)          # 双边

    cur_ir_gross = _ir(ic, breadth)
    cur_ir_net = cur_ir_gross - cost_drag / max(cfg.annual_vol, 1e-6)
    ceiling = IC_CEILING.get(cfg.family, 0.03)
    ic_gap = max(ceiling - abs(ic), 0.0)

    # 三档情景：每档都写明"需要什么才成立"
    def scen(ic_s: float, label: str, premise: str) -> Dict:
        ir_g = _ir(ic_s, breadth)
        ir_n = ir_g - cost_drag / max(cfg.annual_vol, 1e-6)
        return {"label": label, "ic": ic_s, "ir_gross": ir_g, "ir_net": ir_n,
                "ann_excess": ir_n * cfg.annual_vol, "premise": premise}

    if ic >= 0:
        scenarios = [
            scen(ic, "悲观：维持现状", "IC 不再提升，成本不变"),
            scen(ic * 1.6, "中性：IC 提升 60%", "补入基本面/资金面信号，且信号间相关性下降"),
            scen(ceiling, "乐观：逼近同类上限",
                 f"IC 达到 {cfg.family} 类策略经验上限 {ceiling:.3f}，且成本不恶化"),
        ]
    else:
        # IC 为负时，"维持现状"是亏钱的；中性/乐观必须先解决方向问题
        scenarios = [
            scen(ic, "悲观：维持现状（方向仍是反的）",
                 "不做任何改动 → 系统性反向，持续亏损"),
            scen(abs(ic) * 1.6, "中性：先修正方向，再提升 60%",
                 "必须先证明这个负 IC 是真实反转效应（独立样本复核），再反向后使用"),
            scen(ceiling, "乐观：修正方向并达到同类上限",
                 f"方向修正 + IC 达到 {ceiling:.3f}；跨度 {abs(ic) + ceiling:.3f}，"
                 "这需要换信息源，不是调参能达到的"),
        ]

    # 置信度：由 IC 的统计显著性决定，不显著就直说
    if not np.isfinite(ic_t):
        conf = "unknown"
    elif abs(ic_t) >= 3:
        conf = "medium"          # 即便显著，也不给 high —— 样本外永远未知
    elif abs(ic_t) >= 2:
        conf = "low"
    else:
        conf = "none"

    falsifiers = [
        {"claim": "IC 在当前信号集上可转正", "check": "滚动 250 日 HAC t > 2",
         "horizon": "6 个月"},
        {"claim": "净 IR > 0.3", "check": "扣双边成本后滚动 IR", "horizon": "12 个月"},
        {"claim": "信号有效独立数 > 45", "check": "条件相关 |ρ|>0.8 的信号对占比 < 15%",
         "horizon": "1 次重构后"},
        {"claim": "优势不是生存者偏差", "check": "相对『当日池等权』而非沪深300 的超额",
         "horizon": "立即"},
    ]

    return {
        "input": {"ic": ic, "ic_t": ic_t, "family": cfg.family},
        "current": {
            "breadth": breadth, "breadth_nominal": breadth_nominal,
            "independent_names": n_indep, "avg_correlation": rho,
            "periods_per_year": periods_per_year,
            "ir_gross": cur_ir_gross, "ir_net": cur_ir_net,
            "cost_drag_ir": cost_drag / max(cfg.annual_vol, 1e-6),
            "ann_excess": cur_ir_net * cfg.annual_vol,
        },
        "headroom": {
            "ic_ceiling": ceiling, "ic_gap": ic_gap,
            "ic_gap_pct": (ic_gap / ceiling) if ceiling else None,
            "breadth_now": breadth,
            "breadth_nominal": breadth_nominal,
            "breadth_if_double_names": (2 * cfg.n_names / (1.0 + (2 * cfg.n_names - 1) * rho))
                                       * periods_per_year,
            "ir_if_ic_ceiling": _ir(ceiling, breadth) - cost_drag / max(cfg.annual_vol, 1e-6),
            "cost_drag_ir": cost_drag / max(cfg.annual_vol, 1e-6),
            "note": ("有效广度已扣相关性折扣 BR_eff = 周期 × N/(1+(N−1)ρ)；"
                     "名义广度会算出高得离谱的 IR，那是自欺。"
                     "广度翻倍只带来 √2 倍 IR，IC 才是杠杆最大的一项。"),
        },
        "scenarios": scenarios,
        "confidence": conf,
        "falsifiers": falsifiers,
        "verdict": _verdict(ic, ic_t, cur_ir_net, conf),
    }


def _verdict(ic: float, t: float, ir_net: float, conf: str) -> str:
    if not np.isfinite(t) or conf == "unknown":
        return "样本不足，无法给出前景判断"
    if ic <= 0:
        return ("当前 IC ≤ 0：这套代码现在没有可用的选股能力，"
                "讨论『未来空间』没有意义，应先解决方向问题")
    if t < 2:
        return ("IC 为正但统计不显著：可能是运气。可以继续观察，"
                "但不能据此放大仓位")
    if ir_net <= 0:
        return "IC 显著但扣成本后为负：有预测力但没有可交易的净收益"
    return "IC 显著且扣成本后为正：具备进入模拟盘验证的资格（不是直接上实盘）"


def decompose_capacity(daily_amount: np.ndarray, participation: float = 0.05,
                       n_positions: int = 10) -> Dict:
    """
    容量估算：按"最多吃掉某只股票当日成交额的 x%"倒推可管理规模。
    这是很多策略被忽略的天花板 —— 小资金能赚，大资金赚不到。
    """
    a = np.asarray(daily_amount, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 20:
        return {"ok": False, "reason": "insufficient_data"}
    med = float(np.median(a))
    per_name = med * participation
    return {
        "ok": True,
        "median_daily_amount": med,
        "participation": participation,
        "capacity_per_name": per_name,
        "capacity_total": per_name * n_positions,
        "note": "按中位成交额估算；实际容量还要看冲击成本随规模的非线性上升。",
    }
