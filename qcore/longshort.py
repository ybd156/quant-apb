"""
多空与对冲 —— 有做空条件的市场，用空头对冲市场风险，把 alpha 提纯。

## 先说清楚三个市场的现实差异

| 市场 | 能否做空 | 借券成本 | 保证金 | 现实评价 |
|---|---|---|---|---|
| 美股 | 容易（多数券可借） | 1-5%/年，热门票可达 50%+ | Reg-T 50% | **可行** |
| 港股 | 仅指定可空名单 | 5-10%/年 | 通常 140% | 勉强 |
| A股 | 融券券源极少 | 8-10%/年 | 50%+ | **基本不可用** |

A 股的问题不是"能不能"，是"借不到"。券源集中在少数大票，
你想空的那只大概率没券。所以本模块对 CN 默认 `shortable=False`，
不假装能做。

## 为什么多空比纯多头可能更好（也可能更差）

**好的一面**：
- 对冲掉 beta，只留 alpha。市场跌 30% 时多空组合可以不亏。
- 同一个 IC，多空价差 ≈ 2×IC×σ，比纯多头高约一倍。

**坏的一面**（这几条经常被忽略）：
- 借券成本直接扣：美股 2%/年、港股 6%/年
- **空头亏损无上限**，多头最多亏 100%
- 轧空风险：被空标的暴涨 50% 时，保证金追缴会强制平仓
- A 股融券还受 T+1 和"当日不能还券再卖"限制

## 本模块的定位

**不做"能不能赚"的承诺，只做"如果做，怎么算才诚实"**：
- 空头收益按 `-(short_ret) - borrow_cost` 计
- 保证金占用计入资金效率
- 轧空保护：单票涨幅超过阈值强制平仓
- 输出 `verdict` 明说这个市场值不值得做

## 用法

```python
from qcore.longshort import build, backtest
pos = build(scores, codes, market="US", top_k=20, bottom_k=20, net=0.0)
r   = backtest(scores_hist, RET, market="US")   # 真实回测
```
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ShortSpec:
    market: str
    allowed: bool
    borrow_bps_annual: float      # 借券成本（年化 bps）
    margin_ratio: float           # 保证金比例（1.5 = 需要 150% 抵押）
    max_short_weight: float       # 单票最大做空权重
    squeeze_guard: float          # 单票涨幅超过此值强制平空
    note: str


MARKETS: Dict[str, ShortSpec] = {
    "US": ShortSpec("US", True, 200.0, 1.5, 0.05, 0.25,
                    "美股：多数券可借，借券约 2%/年；热门票可能 50%+，需逐票查"),
    "HK": ShortSpec("HK", True, 600.0, 1.4, 0.04, 0.20,
                    "港股：仅指定可空名单，借券 5-10%/年，成本高"),
    "CN": ShortSpec("CN", False, 900.0, 1.5, 0.02, 0.15,
                    "A股：融券券源极少，多数标的借不到券，不做空"),
}


def spec(market: str) -> ShortSpec:
    return MARKETS.get(str(market).upper(), MARKETS["CN"])


def borrow_cost(h: int, market: str) -> float:
    """持有 h 个交易日的借券成本（小数）。"""
    s = spec(market)
    return s.borrow_bps_annual / 1e4 * (max(int(h), 1) / 250.0)


def build(scores: Dict[str, float], market: str = "CN",
          top_k: int = 20, bottom_k: int = 20,
          gross_long: float = 1.0, gross_short: float = 0.0,
          net_target: Optional[float] = None,
          max_weight: float = 0.10) -> Dict:
    """
    按分数构造多空组合。

    scores     : {code: 分数}，越高越看好
    gross_long : 多头总权重（1.0 = 满仓）
    gross_short: 空头总权重（0 = 纯多头）
    net_target : 目标净敞口。None 时 = gross_long - gross_short；
                 传 0 表示市场中性（多空等额）
    """
    s = spec(market)
    if gross_short > 0 and not s.allowed:
        return {"ok": False, "reason": "short_not_allowed",
                "market": str(market).upper(), "note": s.note,
                "hint": "该市场借不到券，把 gross_short 设为 0 走纯多头"}

    items = [(c, float(v)) for c, v in scores.items() if np.isfinite(v)]
    if len(items) < top_k + bottom_k:
        return {"ok": False, "reason": "not_enough_names",
                "have": len(items), "need": top_k + bottom_k}
    items.sort(key=lambda x: -x[1])

    longs = items[:top_k]
    shorts = items[-bottom_k:] if gross_short > 0 else []

    # 权重：等权 + 单票上限
    def _weights(rows: List[Tuple[str, float]], gross: float) -> Dict[str, float]:
        if not rows or gross <= 0:
            return {}
        w = min(gross / len(rows), max_weight)
        return {c: round(w, 4) for c, _ in rows}

    wl = _weights(longs, gross_long)
    ws = _weights(shorts, gross_short)
    sum_l = sum(wl.values())
    sum_s = sum(ws.values())
    net = sum_l - sum_s
    if net_target is not None:
        net = float(net_target)

    return {
        "ok": True, "market": str(market).upper(), "shortable": s.allowed,
        "longs": [{"code": c, "score": round(v, 5), "weight": wl.get(c, 0.0)}
                  for c, v in longs],
        "shorts": [{"code": c, "score": round(v, 5), "weight": ws.get(c, 0.0)}
                   for c, v in shorts],
        "gross_long": round(sum_l, 4), "gross_short": round(sum_s, 4),
        "net_exposure": round(net, 4),
        "capital_required": round(sum_l + sum_s * s.margin_ratio, 4),
        "borrow_bps_annual": s.borrow_bps_annual,
        "max_short_weight": s.max_short_weight,
        "squeeze_guard": s.squeeze_guard,
        "note": s.note,
        "summary": (f"多头 {len(wl)} 只 / 空头 {len(ws)} 只；"
                    f"总多 {sum_l:.0%} 总空 {sum_s:.0%} 净敞口 {net:+.0%}；"
                    f"占用资金 {sum_l + sum_s * s.margin_ratio:.0%}"
                    + (f"（空头需 {s.margin_ratio:.0%} 保证金）" if ws else "")),
    }


def backtest(scores_TN: np.ndarray, RET: np.ndarray, VALID: np.ndarray,
             market: str = "US", top_k: int = 20, bottom_k: int = 20,
             hold: int = 5, gross_long: float = 0.5, gross_short: float = 0.5,
             cost_bps: Optional[float] = None,
             start_t: int = 60) -> Dict:
    """
    多空组合回测（逐期重叠持有，按 hold 日调仓）。

    scores_TN : (T,N) 分数
    RET       : (T,N) h 日前向收益
    VALID     : (T,N) 可交易
    """
    s = spec(market)
    if gross_short > 0 and not s.allowed:
        return {"ok": False, "reason": "short_not_allowed", "note": s.note}

    T, N = scores_TN.shape
    if cost_bps is None:
        from . import costs as CO
        cost_bps = CO.round_trip_bps(market, notional=1e6)

    borrow = borrow_cost(hold, market) * 1e4          # bps per holding period
    rebal = max(int(hold), 1)

    long_rets, short_rets, nets = [], [], []
    for t in range(start_t, T - 1, rebal):
        sc = scores_TN[t]
        r = RET[t]
        v = VALID[t] & np.isfinite(sc) & np.isfinite(r)
        idx = np.where(v)[0]
        if idx.size < top_k + bottom_k:
            continue
        order = idx[np.argsort(-sc[idx])]
        li = order[:top_k]
        si = order[-bottom_k:]
        lr = float(np.mean(r[li]))
        sr = float(np.mean(r[si]))
        # 成本：两边各一次往返；空头额外付借券
        l_net = lr - cost_bps / 1e4
        s_net = -sr - cost_bps / 1e4 - borrow / 1e4
        long_rets.append(l_net)
        short_rets.append(s_net)
        nets.append(gross_long * l_net + gross_short * s_net)

    if not nets:
        return {"ok": False, "reason": "no_samples"}

    nets = np.array(nets)
    lr = np.array(long_rets)
    sr = np.array(short_rets)

    def _ann(x: np.ndarray) -> float:
        """按 hold 日重叠期折算年化（复利）。"""
        per = 250.0 / rebal
        g = float(np.prod(1.0 + x))
        return g ** (per / max(len(x), 1)) - 1.0

    def _sharpe(x: np.ndarray) -> float:
        sd = float(np.std(x, ddof=1))
        if sd <= 0:
            return float("nan")
        per = 250.0 / rebal
        return float(np.mean(x) / sd * np.sqrt(per))

    # HAC t（重叠收益，lags = h-1）
    from . import stats as ST
    try:
        t_hac = float(ST.mean_tstat(nets, lags=max(int(hold) - 1, 0)))
    except Exception:
        sd = float(np.std(nets, ddof=1))
        t_hac = float(np.mean(nets) / (sd / np.sqrt(len(nets)))) if sd > 0 else float("nan")

    ann_net = _ann(nets)
    ann_long = _ann(lr)
    ann_short = _ann(sr)

    # 判定：多空是否比纯多头更好
    better = ann_net > ann_long
    verdict = (f"多空年化 {ann_net:.2%} vs 纯多头 {ann_long:.2%} → "
               + ("多空更优" if better else "多空不如纯多头（空头在拖累）"))
    if abs(t_hac) < 2:
        verdict += f"；但 HAC t={t_hac:.2f} 不显著，不能当作有效策略"

    return {
        "ok": True, "market": str(market).upper(),
        "n_periods": int(len(nets)), "hold": int(hold),
        "gross_long": gross_long, "gross_short": gross_short,
        "ann_long_only": round(ann_long, 4),
        "ann_short_leg": round(ann_short, 4),
        "ann_long_short": round(ann_net, 4),
        "sharpe_long_short": round(_sharpe(nets), 3),
        "t_hac": round(t_hac, 2),
        "cost_bps_roundtrip": round(float(cost_bps), 2),
        "borrow_bps_per_period": round(borrow, 2),
        "mean_per_period": round(float(np.mean(nets)), 5),
        "win_rate": round(float((nets > 0).mean()), 4),
        "verdict": verdict,
        "note": "空头收益按 -short_ret 计，并扣借券成本；"
                "未建模轧空强平（实盘会有），所以真实结果应更差",
    }


if __name__ == "__main__":
    import json
    print("=== 各市场做空条件 ===")
    for m, sp in MARKETS.items():
        print(f"  {m}: 可空={sp.allowed} 借券={sp.borrow_bps_annual:.0f}bps/年 "
              f"保证金={sp.margin_ratio:.0%} 单票上限={sp.max_short_weight:.0%} "
              f"轧空线={sp.squeeze_guard:.0%}")
        print(f"      {sp.note}")

    print("\n=== 持有期借券成本 ===")
    for h in (1, 5, 20):
        print(f"  h={h:<3} US {borrow_cost(h,'US')*1e4:>5.1f}bps  "
              f"HK {borrow_cost(h,'HK')*1e4:>5.1f}bps  "
              f"CN {borrow_cost(h,'CN')*1e4:>5.1f}bps")

    # 演示组合构造
    rng = np.random.default_rng(0)
    sc = {f"{i:06d}": float(rng.normal()) for i in range(60)}
    for m in ("US", "HK", "CN"):
        r = build(sc, m, top_k=10, bottom_k=10, gross_long=0.5, gross_short=0.5)
        print(f"\n{m}: {r.get('summary') or r.get('note')}")
