"""
多周期预测 —— T+0 / T+1 / T+3 / 1周 / 2周 / 15天，每个周期一套模型。

## 为什么每个周期要单独训练

不同持有期的**标签分布完全不同**：
- T+1：日噪声占主导，信噪比低，但换手最快
- T+15：趋势占主导，信噪比高，但资金占用久、回撤大

用同一个模型预测所有周期，等于假设"什么周期看什么因子"都一样，这显然不成立。
所以每个 horizon 训练独立的 GBDT，独立评估 IC，独立产出 picks。

## 标签定义（关键）

| horizon | 标签 | 执行假设 |
|---|---|---|
| h=0 | `close[t]/open[t] − 1` | **当日开盘买、收盘卖**（日内，仅 T+0 市场） |
| h≥1 | `close[t+1+h]/close[t+1] − 1` | T 日出信号，**T+1 收盘成交**，持有 h 天 |

h=0 只对美股有意义（A 股 T+1 制度不允许当日卖）。

## 成本随周期变化（这是最容易被忽略的）

持有期越短，**同样的往返成本摊到越少的收益上**：
- h=1：年换手 250 次 × 19.28bps ≈ **48%/年** 成本拖累
- h=5：约 9.6%/年
- h=20：约 2.4%/年

所以短周期模型必须有**显著更高的 IC** 才能覆盖成本。
`expected_net_annual()` 就是算这个的。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# horizon -> 展示名。0 = 日内（仅 T+0 市场）
HORIZONS: Dict[int, Dict] = {
    0: {"label": "T+0 日内", "note": "当日开盘买收盘卖，仅美股等 T+0 市场",
        "markets": ["US"]},
    1: {"label": "T+1 短线", "note": "次日卖出，换手最快、成本最重",
        "markets": ["CN", "US", "HK"]},
    3: {"label": "T+3 中短线", "note": "持 3 个交易日",
        "markets": ["CN", "US", "HK"]},
    5: {"label": "1 周 中线", "note": "持 5 个交易日",
        "markets": ["CN", "US", "HK"]},
    10: {"label": "2 周", "note": "持 10 个交易日",
         "markets": ["CN", "US", "HK"]},
    15: {"label": "15 天 中长线", "note": "持 15 个交易日",
         "markets": ["CN", "US", "HK"]},
}

DEFAULT_HORIZON = 1

# 各市场往返成本（bps），来自 qcore/costs.py 实测
ROUND_TRIP_BPS = {"CN": 19.28, "US": 3.28, "HK": 38.17}


def all_horizons() -> List[int]:
    return sorted(HORIZONS.keys())


def label_for(h: int) -> str:
    return (HORIZONS.get(h) or {}).get("label", f"h={h}")


def supports(h: int, market: str = "CN") -> bool:
    """该市场是否支持这个周期（A 股不支持 h=0）。"""
    m = str(market).upper()
    return m in ((HORIZONS.get(h) or {}).get("markets") or [])


def annual_cost_drag(h: int, market: str = "CN", turnover: float = 0.8) -> float:
    """
    年化成本拖累（小数）。turnover = 每次调仓换手比例。
    持 h 天 → 一年调仓 250/h 次。
    """
    bps = ROUND_TRIP_BPS.get(str(market).upper(), 19.28)
    n = 250.0 / max(int(h), 1)
    return n * turnover * bps / 1e4


def expected_net_annual(ic: float, h: int, market: str = "CN",
                        n_stocks: int = 300, rho: float = 0.30,
                        turnover: float = 0.8) -> Dict:
    """
    用 Grinold 基本法则估算扣成本后的年化超额。

        IR ≈ IC × √BR_eff
        BR_eff = (250/h) × N / (1 + (N-1)ρ)      ← 相关性调整后的有效广度
        年化超额 ≈ IR × 年化波动（取 20%）

    这是**粗估**，只用来判断"这个周期有没有可能覆盖成本"，不能当收益承诺。
    """
    h = max(int(h), 1)
    n = max(int(n_stocks), 2)
    br_eff = (250.0 / h) * n / (1.0 + (n - 1) * max(min(rho, 0.99), 0.0))
    ir = float(ic) * np.sqrt(max(br_eff, 1e-9))
    gross = ir * 0.20                       # 假设组合年化波动 20%
    cost = annual_cost_drag(h, market, turnover)
    net = gross - cost
    return {"h": h, "market": market, "ic": round(float(ic), 5),
            "br_eff": round(br_eff, 1), "ir": round(ir, 3),
            "gross_annual": round(gross, 4),
            "cost_annual": round(cost, 4),
            "net_annual": round(net, 4),
            "verdict": ("扣成本后仍为正" if net > 0 else "扣成本后为负，该周期不可交易")}


def table(ic_by_h: Dict[int, float], market: str = "CN") -> List[Dict]:
    """把各周期 IC 换算成扣成本后的年化，一张表看哪个周期值得做。"""
    rows = []
    for h in all_horizons():
        if h not in ic_by_h:
            continue
        if not supports(h, market):
            rows.append({"h": h, "label": label_for(h), "supported": False,
                         "note": f"{market} 不支持该周期"})
            continue
        r = expected_net_annual(ic_by_h[h], h, market)
        r["label"] = label_for(h)
        r["supported"] = True
        rows.append(r)
    return rows


if __name__ == "__main__":
    import json
    print("支持的周期:")
    for h, v in HORIZONS.items():
        print(f"  h={h:<3} {v['label']:<14} {v['note']}")

    print("\n=== 成本拖累随周期变化（CN，换手 80%）===")
    for h in (1, 3, 5, 10, 15):
        c = annual_cost_drag(h, "CN", 0.8)
        print(f"  h={h:<3} 年换手 {250/h:>6.1f} 次 → 成本拖累 {c*100:>6.2f}%")

    print("\n=== 假设各周期 IC 都是 0.03，扣成本后还剩多少 ===")
    print(f"{'h':<5}{'周期':<16}{'IR':>7}{'毛年化':>9}{'成本':>8}{'净年化':>9}  判定")
    for r in table({h: 0.03 for h in all_horizons()}, "CN"):
        if not r.get("supported"):
            print(f"{r['h']:<5}{r['label']:<16}{'—':>7}{'—':>9}{'—':>8}{'—':>9}  {r['note']}")
            continue
        print(f"{r['h']:<5}{r['label']:<16}{r['ir']:>7.3f}{r['gross_annual']*100:>8.1f}%"
              f"{r['cost_annual']*100:>7.1f}%{r['net_annual']*100:>8.1f}%  {r['verdict']}")
