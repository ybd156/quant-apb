"""
交易成本模型 —— 把"回测赚钱、实盘亏钱"的那个缺口算清楚。

## 为什么必须分市场建模

三个市场的成本结构差一个数量级：

| 市场 | 显性成本（单边） | 关键差异 |
|---|---|---|
| A股 | 佣金 0.025% + 过户 0.001% + 规费 0.00541% + **印花税 0.05%（仅卖出）** | 印花税只在卖出收 |
| 美股 | 佣金 0 + SEC fee 0.00000278（仅卖出）+ TAF 每股 $0.000166（仅卖出） | 几乎为零，成本主要是价差 |
| 港股 | 佣金 0.03% + **印花税 0.1%（双边）** + 交易征费 0.0027% + 交易费 0.00565% + 交收费 0.002% + FRC 0.00015% | **双边印花税，往返 ~0.28%** |

港股往返成本约 28 bps，是 A股（约 11 bps）的 2.5 倍。
**同一个策略在 A股能活，搬到港股可能直接亏死** —— 这就是必须分市场算的原因。

## 费率来源与有效期

所有费率都带 `verified_on` 字段。费率会变（例如 A股印花税 2023-08-28 从 0.1% 减半到 0.05%），
**实盘前必须重新核对**，不能假定本文件永远正确。

## 滑点模型

    slippage_bps = base_spread_bps/2 + impact_coef · sqrt(participation)

`participation` = 本笔成交额 / 该股当日成交额。冲击成本随规模开方增长，
这是业界通用近似（Almgren-Chriss 简化形式）。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CostSpec:
    market: str
    name: str
    commission_bps: float = 0.0          # 券商佣金（单边）
    commission_min: float = 0.0          # 单笔最低佣金（本币）
    stamp_duty_bps: float = 0.0          # 印花税
    stamp_side: str = "none"             # sell | both | none
    transfer_bps: float = 0.0            # 过户费/交收费
    exchange_bps: float = 0.0            # 交易所交易费/经手费
    regulator_bps: float = 0.0           # 证管费/交易征费/FRC
    sec_fee_bps_sell: float = 0.0        # 美国 SEC fee（仅卖出）
    taf_per_share: float = 0.0           # 美国 FINRA TAF（仅卖出，每股）
    system_fee_per_order: float = 0.0    # 每笔系统使用费（本币）
    settlement_min: float = 0.0
    settlement_max: float = 1e18
    spread_bps: float = 5.0              # 买卖价差（全额），滑点取其一半
    impact_coef: float = 8.0             # 冲击成本系数（bps / sqrt(participation)）
    lot_size: int = 100
    verified_on: str = ""
    note: str = ""

    # ---- 单边费率合计（不含滑点与最低收费）
    def variable_bps(self, side: str) -> float:
        b = self.commission_bps + self.transfer_bps + self.exchange_bps + self.regulator_bps
        if self.stamp_side == "both" or (self.stamp_side == "sell" and side == "sell"):
            b += self.stamp_duty_bps
        if side == "sell":
            b += self.sec_fee_bps_sell
        return b


# 费率快照（2026-11 核对；实盘前请复核）
MARKETS: Dict[str, CostSpec] = {
    "CN": CostSpec(
        market="CN", name="A股（沪深）",
        commission_bps=2.5, commission_min=5.0,
        stamp_duty_bps=5.0, stamp_side="sell",      # 2023-08-28 起减半至 0.05%，仅卖出
        transfer_bps=0.1,                            # 2022-04-29 起 0.001%，双边
        exchange_bps=0.341,                          # 经手费 0.00341%，双边
        regulator_bps=0.2,                           # 证管费 0.002%，双边
        spread_bps=8.0, impact_coef=8.0, lot_size=100,
        verified_on="2026-11",
        note="印花税仅卖出单边；佣金通常已包含规费，此处分开列示便于审计",
    ),
    "US": CostSpec(
        market="US", name="美股",
        commission_bps=0.0, commission_min=0.0,      # 主流券商零佣金
        stamp_duty_bps=0.0, stamp_side="none",
        transfer_bps=0.0, exchange_bps=0.0, regulator_bps=0.0,
        sec_fee_bps_sell=0.278,                      # SEC Section 31 fee，仅卖出
        taf_per_share=0.000166,                      # FINRA TAF，仅卖出，上限 $8.30
        spread_bps=3.0, impact_coef=6.0, lot_size=1,
        verified_on="2026-11",
        note="零佣金时代，真实成本主要是价差与冲击；TAF 按股数计，需 shares 参数",
    ),
    "HK": CostSpec(
        market="HK", name="港股",
        commission_bps=3.0, commission_min=3.0,      # 券商约 0.03%，最低 HK$3
        stamp_duty_bps=10.0, stamp_side="both",      # 0.1%，双边
        transfer_bps=0.2,                            # 交收费 0.002%，最低2/最高100
        exchange_bps=0.565,                          # 联交所交易费 0.00565%，双边
        regulator_bps=0.27,                          # 交易征费 0.0027%，双边
        system_fee_per_order=0.5,                    # 交易系统使用费 HK$0.5/笔
        settlement_min=2.0, settlement_max=100.0,
        spread_bps=10.0, impact_coef=10.0, lot_size=100,
        verified_on="2026-11",
        note="双边印花税 0.1% 是主要成本；另有 FRC 征费 0.00015% 未单列（量级可忽略）",
    ),
}


def spec(market: str = "CN") -> CostSpec:
    m = (market or "CN").upper()
    if m not in MARKETS:
        raise KeyError(f"unknown market {market}; choose from {list(MARKETS)}")
    return MARKETS[m]


def impact_bps(participation: float, s: CostSpec) -> float:
    """冲击成本：随参与率开方增长。"""
    p = max(float(participation), 0.0)
    return s.impact_coef * math.sqrt(p) if p > 0 else 0.0


def one_way(notional: float, side: str, market: str = "CN",
            shares: Optional[float] = None, price: Optional[float] = None,
            participation: float = 0.0) -> Dict:
    """单边成本明细（金额 + bps）。"""
    s = spec(market)
    n = max(float(notional), 1e-9)
    side = side.lower()
    comp = {}

    comm = n * s.commission_bps / 1e4
    if s.commission_min:
        comm = max(comm, s.commission_min)
    comp["commission"] = comm

    comp["stamp_duty"] = (n * s.stamp_duty_bps / 1e4
                          if (s.stamp_side == "both" or
                              (s.stamp_side == "sell" and side == "sell")) else 0.0)

    transfer = n * s.transfer_bps / 1e4
    if s.settlement_min:
        transfer = max(transfer, s.settlement_min)
    if s.settlement_max < 1e17:
        transfer = min(transfer, s.settlement_max)
    comp["transfer"] = transfer

    comp["exchange"] = n * s.exchange_bps / 1e4
    comp["regulator"] = n * s.regulator_bps / 1e4

    sec = n * s.sec_fee_bps_sell / 1e4 if side == "sell" else 0.0
    comp["sec_fee"] = sec

    taf = 0.0
    if side == "sell" and s.taf_per_share and shares:
        taf = min(float(shares) * s.taf_per_share, 8.30)
    comp["taf"] = taf

    comp["system_fee"] = s.system_fee_per_order

    comp["slippage"] = n * (s.spread_bps / 2.0 + impact_bps(participation, s)) / 1e4

    total = sum(comp.values())
    return {"market": s.market, "side": side, "notional": n,
            "components": comp, "total": total, "total_bps": total / n * 1e4}


def round_trip_bps(market: str = "CN", notional: float = 1e5,
                   participation: float = 0.0, lot_value: Optional[float] = None) -> Dict:
    """
    一次完整往返（买入 + 卖出）的成本，单位 bps。
    notional 影响"最低佣金"的相对权重 —— 小资金成本率更高，这一点很多回测都忽略了。
    """
    s = spec(market)
    buy = one_way(notional, "buy", market, participation=participation)
    sell = one_way(notional, "sell", market, participation=participation)
    total = buy["total"] + sell["total"]
    return {
        "market": s.market, "name": s.name, "notional": notional,
        "buy_bps": buy["total_bps"], "sell_bps": sell["total_bps"],
        "round_trip_bps": total / notional * 1e4,
        "breakdown": {"buy": buy["components"], "sell": sell["components"]},
        "verified_on": s.verified_on,
    }


def effective_cost_bps(market: str = "CN", notional: float = 1e5,
                       participation: float = 0.0) -> float:
    """给回测用的单一往返成本（bps）。"""
    return float(round_trip_bps(market, notional, participation)["round_trip_bps"])


def cost_scenarios(markets: Optional[List[str]] = None,
                   notionals: Optional[List[float]] = None) -> List[Dict]:
    """规模 × 市场的成本矩阵 —— 用来说明"小资金成本率更高"和"港股更贵"。"""
    markets = markets or ["CN", "US", "HK"]
    notionals = notionals or [1e4, 1e5, 1e6, 1e7, 1e8]
    rows = []
    for m in markets:
        for n in notionals:
            r = round_trip_bps(m, n)
            rows.append({"market": r["market"], "name": r["name"],
                         "notional": n, "round_trip_bps": round(r["round_trip_bps"], 2),
                         "buy_bps": round(r["buy_bps"], 2), "sell_bps": round(r["sell_bps"], 2)})
    return rows


def annual_cost_drag(market: str, turnover_per_rebalance: float,
                     rebalance_days: int = 5, notional: float = 1e5) -> Dict:
    """年化成本拖累：换手 × 往返成本 × 年调仓次数。"""
    periods = 252.0 / max(rebalance_days, 1)
    rt = effective_cost_bps(market, notional)
    drag = turnover_per_rebalance * rt / 1e4 * periods
    return {"market": market, "round_trip_bps": rt, "periods_per_year": periods,
            "annual_cost_pct": drag * 100,
            "note": "按每次调仓换手 turnover 比例计；实际换手需从回测取"}


def net_of_costs(gross_ret: np.ndarray, turnover: np.ndarray, market: str = "CN",
                 notional: float = 1e5) -> np.ndarray:
    """把毛收益序列扣成净收益（逐期）。"""
    rt = effective_cost_bps(market, notional) / 1e4
    t = np.asarray(turnover, dtype=np.float64)
    return np.asarray(gross_ret, dtype=np.float64) - t * rt


def summary_table(markets: Optional[List[str]] = None) -> List[Dict]:
    """报告用的市场成本对比表（10 万本币口径）。"""
    out = []
    for m in (markets or ["CN", "US", "HK"]):
        s = spec(m)
        r = round_trip_bps(m, 1e5)
        out.append({
            "market": s.market, "name": s.name,
            "commission_bps": s.commission_bps,
            "stamp_duty_bps": s.stamp_duty_bps, "stamp_side": s.stamp_side,
            "other_bps": round(s.transfer_bps + s.exchange_bps + s.regulator_bps
                               + s.sec_fee_bps_sell, 4),
            "slippage_bps": s.spread_bps / 2.0,
            "round_trip_bps": round(r["round_trip_bps"], 2),
            "verified_on": s.verified_on,
        })
    return out


if __name__ == "__main__":
    for r in summary_table():
        print(r)
    print()
    for r in cost_scenarios():
        print(r)
