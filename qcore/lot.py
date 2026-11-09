"""
最小交易单位与可成交性 —— 把「理论权重」变成「真能下的单」。

## 为什么这一层不可省

回测说「每只 5% 权重」，但实盘：

| 本金 | 每只 5% | 茅台（1453 元）一手 | 能买吗 |
|---|---|---|---|
| 10 万 | 5,000 元 | 145,300 元 | ❌ |
| 100 万 | 50,000 元 | 145,300 元 | ❌ |
| 200 万 | 100,000 元 | 145,300 元 | ❌ |

实测本项目当前 20 只推荐里，**10 万本金有 8 只买不起一手**。
如果不做这层，实盘会自动退化成「只买便宜股」——悄悄改变了策略。

## 三个硬约束

1. **最小交易单位**：A 股 100 股/手；港股每手不同（100/500/1000/2000）；美股 1 股
2. **涨跌停不可成交**：涨停买不进、跌停卖不出
3. **T+1**：当天买入当天不能卖（影响止损，见 `bt_rules`）

## 处理策略（不是简单砍掉）

买不起的票权重不能浪费，按比例**重新分配给买得起的**，
并如实报告「因为整手约束损失了多少仓位」和「实际持仓数」。

## 用法

```python
from qcore.lot import plan
r = plan(capital=100_000, weights={...}, prices={...}, market="CN")
# r["orders"]      真正能下的单 [{code, shares, amount, weight}]
# r["dropped"]     买不起被剔除的 [{code, price, need, have}]
# r["cash_drag"]   被迫闲置的现金比例
# r["effective_n"] 实际持仓数
```
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 每手股数。A 股统一 100；港股不统一，必须逐只查，这里给常见值兜底
LOT_CN = 100
LOT_US = 1
HK_DEFAULT_LOT = 500


@dataclass
class LotRule:
    market: str = "CN"
    lot: int = LOT_CN
    min_amount: float = 0.0      # 单笔最小金额（券商限制）
    allow_odd: bool = False      # 是否允许零股（A 股卖出可以，买入不行）


def _lot_for(code: str, market: str, hk_lots: Optional[Dict[str, int]] = None) -> int:
    if market == "US":
        return LOT_US
    if market == "HK":
        if hk_lots and code in hk_lots:
            return int(hk_lots[code])
        return HK_DEFAULT_LOT
    return LOT_CN


def plan(capital: float, weights: Dict[str, float], prices: Dict[str, float],
         market: str = "CN", hk_lots: Optional[Dict[str, int]] = None,
         max_iter: int = 5) -> Dict:
    """
    把目标权重转成实际可下的单。
    weights: {code: 目标权重}（应已归一化）
    prices : {code: 现价}（真实价，不是复权价）
    返回详细报告，含被剔除的票和现金拖累。
    """
    if capital <= 0 or not weights:
        return {"ok": False, "reason": "bad_input"}
    mkt = str(market).upper()

    w = {k: float(v) for k, v in weights.items() if float(v) > 0}
    px = {k: float(prices.get(k) or 0) for k in w}
    missing = [k for k in w if not px.get(k) or px[k] <= 0]

    orders: List[Dict] = []
    dropped: List[Dict] = []
    all_dropped: List[Dict] = []      # 累积所有轮被剔除的，用于报告
    pool = {k: v for k, v in w.items() if k not in missing}

    for _ in range(max_iter):
        if not pool:
            break
        tot = sum(pool.values())
        if tot <= 0:
            break
        norm = {k: v / tot for k, v in pool.items()}
        orders, dropped = [], []
        for code, wt in norm.items():
            lot = _lot_for(code, mkt, hk_lots)
            budget = capital * wt
            p = px[code]
            shares = int(budget // (p * lot)) * lot
            if shares < lot:
                dropped.append({"code": code, "price": p, "weight": round(wt, 4),
                                "budget": round(budget, 2),
                                "need_one_lot": round(p * lot, 2),
                                "lot": lot})
                continue
            amt = shares * p
            orders.append({"code": code, "price": p, "shares": shares,
                           "amount": round(amt, 2),
                           "weight_plan": round(wt, 4),
                           "weight_actual": round(amt / capital, 4),
                           "lot": lot})
        if not dropped:
            break
        all_dropped.extend(dropped)
        # 把被剔除的权重从池子里去掉，重新分配
        for d in dropped:
            pool.pop(d["code"], None)

    used = sum(o["amount"] for o in orders)
    cash_drag = max(0.0, 1.0 - used / capital) if capital else 1.0
    return {
        "ok": True, "market": mkt, "capital": round(capital, 2),
        "orders": sorted(orders, key=lambda x: -x["amount"]),
        "dropped": all_dropped,
        "n_target": len(w), "n_actual": len(orders),
        "used": round(used, 2), "cash": round(capital - used, 2),
        "cash_drag": round(cash_drag, 4),
        "missing_price": missing,
        "verdict": (
            "全部可建仓" if not all_dropped else
            f"{len(all_dropped)} 只因整手约束买不起"
            f"（最贵一手需 {max(d['need_one_lot'] for d in all_dropped):,.0f} 元），"
            f"实际持仓 {len(orders)}/{len(w)}，现金闲置 {cash_drag:.1%}"
        ),
    }


def feasible_capital(weights: Dict[str, float], prices: Dict[str, float],
                     market: str = "CN", hk_lots: Optional[Dict[str, int]] = None,
                     coverage: float = 0.95) -> Dict:
    """
    反解：要让至少 coverage 比例的票能买一手，需要多少本金。
    用于告诉用户「本金至少要多少」。
    """
    if not weights:
        return {"ok": False, "reason": "bad_input"}
    mkt = str(market).upper()
    tot = sum(weights.values()) or 1.0
    need = []
    for code, wt in weights.items():
        p = float(prices.get(code) or 0)
        if p <= 0:
            continue
        lot = _lot_for(code, mkt, hk_lots)
        # 该票要买一手，需要本金 >= p*lot / (wt/tot)
        need.append((p * lot) / max(wt / tot, 1e-9))
    if not need:
        return {"ok": False, "reason": "no_price"}
    need.sort()
    idx = max(0, min(len(need) - 1, int(len(need) * coverage) - 1))
    return {"ok": True, "market": mkt,
            "min_capital_for_all": round(need[-1], 0),
            f"min_capital_for_{int(coverage*100)}pct": round(need[idx], 0),
            "median": round(need[len(need) // 2], 0),
            "note": "要让全部票都能买一手所需本金；高于这个数就不会被整手约束剔除"}


if __name__ == "__main__":
    import json
    from qcore import tick

    # 用当前真实推荐 + 真实价实测
    try:
        picks = json.load(open("data/state/picks.json", encoding="utf-8"))
        codes = [r["code"] for r in picks["rows"]]
        snap = tick.snapshot(codes)
        px = {}
        for r in (snap.get("rows") or []):
            c = r.get("code", "")
            px[c[2:] if c[:2] in ("sh", "sz", "bj") else c] = r.get("price")
        w = {c: 1.0 / len(codes) for c in codes}
        for cap in (100_000, 300_000, 1_000_000, 3_000_000):
            r = plan(cap, w, px, "CN")
            print(f"本金 {cap:>9,}: 实际 {r['n_actual']}/{r['n_target']} 只，"
                  f"现金闲置 {r['cash_drag']:.1%} | {r['verdict'][:60]}")
        print()
        print("需要的本金:", json.dumps(feasible_capital(w, px, "CN"), ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        print("实测跳过:", e)

# ------------------------------------------------------------------ 小资金模式
def small_capital(capital: float, scores: Dict[str, float], prices: Dict[str, float],
                  market: str = "CN", hk_lots: Optional[Dict[str, int]] = None,
                  max_hold: int = 5) -> Dict:
    """
    小资金（2000/5000/10000 元）专用：不做等权，直接买**买得起的高分票**。

    为什么不能沿用等权逻辑：
      2000 元 / 20 只 = 100 元/只，连一手都买不起。
      小资金的正确做法是**集中持有 1-3 只买得起的高分票**，
      接受"无法分散"这个事实，而不是假装分散。

    返回可买清单（按分数排序，买得起才进）、能持几只、以及诚实的风险提示。
    """
    if capital <= 0 or not prices:
        return {"ok": False, "reason": "bad_input"}
    mkt = str(market).upper()

    cands = []
    for code, p in prices.items():
        try:
            p = float(p)
        except Exception:
            continue
        if p <= 0:
            continue
        lot = _lot_for(code, mkt, hk_lots)
        one_lot = p * lot
        if one_lot <= capital:
            cands.append({"code": code, "price": p, "lot": lot,
                          "one_lot_cost": round(one_lot, 2),
                          "score": float(scores.get(code) or 0.0)})

    cands.sort(key=lambda x: -x["score"])
    picked, used = [], 0.0
    for c in cands:
        if len(picked) >= max_hold:
            break
        if used + c["one_lot_cost"] <= capital:
            picked.append(c)
            used += c["one_lot_cost"]

    affordable = len(cands)
    cash = capital - used
    warn = []
    if affordable == 0:
        warn.append(f"本金 {capital:,.0f} 买不起任何一只的一手（最便宜一手 "
                    f"{min((float(p) * _lot_for(k, mkt, hk_lots) for k, p in prices.items() if float(p or 0) > 0), default=0):,.0f} 元）")
    elif len(picked) < 3:
        warn.append(f"只能持有 {len(picked)} 只 —— 无法分散，单票风险等于全部风险")
    if cash / capital > 0.3:
        warn.append(f"闲置现金 {cash/capital:.0%}（整手约束导致）")

    return {"ok": True, "market": mkt, "capital": round(capital, 2),
            "n_affordable": affordable, "n_hold": len(picked),
            "hold": picked, "used": round(used, 2), "cash": round(cash, 2),
            "cash_drag": round(cash / capital, 4) if capital else 0.0,
            "warnings": warn,
            "note": "小资金模式：集中持有买得起的高分票，不做等权分散"}


def scenarios(scores: Dict[str, float], prices: Dict[str, float],
              market: str = "CN", levels=(2000, 5000, 10000, 30000, 100000),
              hk_lots: Optional[Dict[str, int]] = None) -> Dict:
    """给一组本金档位，各自能买几只。"""
    out = []
    for cap in levels:
        r = small_capital(cap, scores, prices, market, hk_lots)
        out.append({"capital": cap, "n_affordable": r.get("n_affordable"),
                    "n_hold": r.get("n_hold"),
                    "cash_drag": r.get("cash_drag"),
                    "top": (r.get("hold") or [{}])[0].get("code") if r.get("hold") else None,
                    "warnings": r.get("warnings")})
    return {"ok": True, "market": market, "levels": out}
