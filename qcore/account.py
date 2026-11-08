"""
模拟账户 —— 每日结算"预测 vs 实际"，并给出日/周/月/总收益。

## 和 paper.py 的区别

`paper.py` 只做"模拟盘 vs 实盘"的对账（跟踪误差、滑点）。
本模块做的是**一个真正能记账的账户**：

  给一笔模拟资金 → 系统自己选股 → 按市场规则定量 → 每天盯市 →
  记录"当天预测收益"与"当天实际收益" → 汇总成日/周/月/总

## 分市场定量规则（这是最容易做错的地方）

| 市场 | 最小交易单位 | 交割 | 涨跌停 | 最低费用 |
|---|---|---|---|---|
| A 股 | **100 股**（1 手） | T+1 | ±10%（ST ±5%，科创/创业 ±20%） | 佣金最低 5 元 |
| 美股 | **1 股** | T+0 | 无（有熔断） | 零佣金 |
| 港股 | **每手不同**（常见 100/500/1000） | T+2 | 无 | 佣金最低 HK$3 |

**A 股必须整手买**：想买 250 股只能买 200 股，剩下的钱留在现金里。
很多回测忽略这一点，小资金下误差会很大（10 万本金买 10 只，每只 1 万，
高价股如 600519 一手就要 15 万，根本买不起 —— 必须跳过或降权）。

## 预测收益怎么定义

`predicted_return_i = score_i × β_h`
其中 β_h 来自 meta 层对"分数 → 未来 h 日收益"的因果回归。
**没有 β 就不编预测**（返回 NaN），这是纪律。

## 账本

`data/account/ledger_YYYYMMDD.jsonl`，每日一行：
    日期 / 净值 / 现金 / 持仓市值 / 当日实际收益 / 当日预测收益 / 累计收益 / 持仓明细
只追加，不回改 —— 与预测台账同一套规矩。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNT_DIR = os.path.join(ROOT, "data", "account")


# ------------------------------------------------------------------ 市场规则
@dataclass(frozen=True)
class MarketRules:
    market: str
    lot_size: int = 100
    t_plus: int = 1
    price_limit: float = 0.10
    min_commission: float = 5.0
    commission_bps: float = 2.5

    def affordable_lots(self, cash: float, price: float) -> int:
        """按整手计算能买多少股。"""
        if price <= 0 or cash <= 0:
            return 0
        lots = int(cash // (price * self.lot_size))
        return max(lots, 0) * self.lot_size


CN_RULES = MarketRules("CN", lot_size=100, t_plus=1, price_limit=0.10,
                       min_commission=5.0, commission_bps=2.5)
US_RULES = MarketRules("US", lot_size=1, t_plus=0, price_limit=1.0,
                       min_commission=0.0, commission_bps=0.0)
HK_RULES = MarketRules("HK", lot_size=100, t_plus=2, price_limit=1.0,
                       min_commission=3.0, commission_bps=3.0)
RULES = {"CN": CN_RULES, "US": US_RULES, "HK": HK_RULES}


# ------------------------------------------------------------------ 选股 + 定量
def plan_positions(scores: np.ndarray, prices: np.ndarray, sigma: np.ndarray,
                   capital: float, rules: MarketRules = CN_RULES,
                   top_k: int = 10, max_weight: float = 0.15,
                   min_score: float = 0.0) -> List[Dict]:
    """
    给定当日分数与价格，算出"买哪些、买多少"。
    规则：分数 > min_score 的取前 top_k；逆波动配权；整手约束；单票上限。
    """
    sc = np.asarray(scores, dtype=float)
    px = np.asarray(prices, dtype=float)
    sg = np.asarray(sigma, dtype=float)
    ok = np.isfinite(sc) & np.isfinite(px) & (px > 0) & np.isfinite(sg) & (sg > 0) & (sc > min_score)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return []
    pick = idx[np.argsort(-sc[idx])][:top_k]
    w = (1.0 / sg[pick]); w = w / w.sum()
    w = np.minimum(w, max_weight); w = w / w.sum()

    plan, skipped = [], []
    cash_left = capital
    for j, i in enumerate(pick):
        budget = capital * w[j]
        budget = min(budget, cash_left)
        shares = rules.affordable_lots(budget, px[i])
        if shares == 0:
            skipped.append({"idx": int(i), "price": float(px[i]),
                            "reason": f"不足一手（一手 {px[i]*rules.lot_size:.0f}，预算 {budget:.0f}）"})
            continue
        cost = shares * px[i]
        cash_left -= cost
        plan.append({"idx": int(i), "shares": int(shares), "price": float(px[i]),
                     "value": float(cost), "weight": float(cost / capital)})
        if cash_left <= 0:
            break
    for p in plan:
        p["skipped"] = skipped
    return plan


# ------------------------------------------------------------------ 账户
@dataclass
class SimAccount:
    initial_capital: float = 1_000_000.0
    rules: MarketRules = CN_RULES
    cash: float = field(init=False)
    positions: Dict[str, Dict] = field(default_factory=dict)   # code -> {shares, cost}
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.cash = float(self.initial_capital)

    def market_value(self, prices: Dict[str, float]) -> float:
        mv = 0.0
        for code, p in self.positions.items():
            px = prices.get(code)
            if px is None or not np.isfinite(px):
                px = p.get("last_price", p["cost"])
            mv += p["shares"] * px
        return mv

    def nav(self, prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def rebalance(self, date: str, codes: Sequence[str], plan: List[Dict],
                  prices: Dict[str, float]) -> Dict:
        """按目标持仓调仓（先卖后买），返回本次交易明细。"""
        target = {codes[p["idx"]]: p for p in plan}
        trades = []

        # 卖出不在目标里的
        for code in list(self.positions):
            if code not in target:
                p = self.positions.pop(code)
                px = prices.get(code, p["cost"])
                proceeds = p["shares"] * px * (1 - self.rules.commission_bps / 1e4)
                self.cash += proceeds
                trades.append({"side": "sell", "code": code, "shares": p["shares"],
                               "price": float(px), "amount": float(proceeds)})

        # 买入/调整
        for code, t in target.items():
            cur = self.positions.get(code)
            cur_sh = cur["shares"] if cur else 0
            want = t["shares"]
            if want <= cur_sh:
                continue
            add = want - cur_sh
            cost = add * t["price"]
            fee = max(cost * self.rules.commission_bps / 1e4, self.rules.min_commission)
            if cost + fee > self.cash:
                continue
            self.cash -= cost + fee
            if cur:
                tot = cur_sh + add
                cur["cost"] = (cur["cost"] * cur_sh + t["price"] * add) / tot
                cur["shares"] = tot
            else:
                self.positions[code] = {"shares": add, "cost": t["price"]}
            trades.append({"side": "buy", "code": code, "shares": add,
                           "price": t["price"], "amount": float(cost), "fee": float(fee)})

        for code, p in self.positions.items():
            p["last_price"] = prices.get(code, p.get("last_price", p["cost"]))
        return {"date": date, "trades": trades, "cash": round(self.cash, 2),
                "n_positions": len(self.positions)}


# ------------------------------------------------------------------ β 估计
def estimate_beta(scores: np.ndarray, RET: np.ndarray, h: int,
                  warmup: int = 120) -> np.ndarray:
    """
    因果估计 β：分数 → 未来 h 日收益的横截面斜率。
    每天把分数与收益各自去均值后累积回归（与 IC 同口径），只用已到期样本。
    返回长度 T 的 β 序列；最后一位即当前可用 β。
    """
    T, N = scores.shape
    betas = np.zeros(T)
    num = den = 0.0
    for t in range(T):
        s = t - 1 - h
        if s >= 0:
            x = scores[s]; y = RET[s]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 5:
                xm = x[m] - x[m].mean()
                ym = y[m] - y[m].mean()
                num += float(xm @ ym)
                den += float(xm @ xm)
        betas[t] = (num / den) if den > 1e-12 else 0.0
    return betas


# ------------------------------------------------------------------ 逐日回放
def simulate_account(p3d, scores: np.ndarray, dates, codes: List[str],
                     beta_h: float = 0.0, horizon: int = 5,
                     initial_capital: float = 1_000_000.0,
                     rules: MarketRules = CN_RULES, top_k: int = 10,
                     max_weight: float = 0.15, rebalance_days: Optional[int] = None,
                     warmup: int = 120) -> pd.DataFrame:
    """
    逐日盯市，产出账本。scores 为 (N,T)。
    beta_h：分数→预期收益的斜率（来自 meta 层）。为 0 时预测收益记 NaN（不编）。
    """
    T = len(dates)
    acct = SimAccount(initial_capital=initial_capital, rules=rules)
    step = rebalance_days or horizon
    rows = []
    prev_nav = initial_capital

    for t in range(warmup, T):
        prices = {codes[n]: float(p3d.CLOSE[t, n]) for n in range(len(codes))
                  if np.isfinite(p3d.CLOSE[t, n])}

        # 调仓（每 step 天一次）
        if (t - warmup) % step == 0:
            plan = plan_positions(scores[:, t], p3d.CLOSE[t], p3d.SIGMA[t],
                                  acct.nav(prices), rules, top_k, max_weight)
            acct.rebalance(str(pd.Timestamp(dates[t]).date()), codes, plan, prices)

        nav = acct.nav(prices)
        actual = nav / prev_nav - 1.0 if prev_nav > 0 else np.nan
        # 预测收益：持仓加权 score × beta
        pred = np.nan
        if beta_h and acct.positions:
            num = den = 0.0
            for code, p in acct.positions.items():
                n = codes.index(code) if code in codes else -1
                if n < 0:
                    continue
                s = scores[n, t]
                if not np.isfinite(s):
                    continue
                num += p["shares"] * p["cost"] * s
                den += p["shares"] * p["cost"]
            if den > 0:
                pred = float(num / den) * beta_h

        rows.append({
            "date": str(pd.Timestamp(dates[t]).date()),
            "nav": round(nav, 2),
            "cash": round(acct.cash, 2),
            "market_value": round(nav - acct.cash, 2),
            "daily_return": actual,
            "predicted_return": pred,
            "cum_return": nav / initial_capital - 1.0,
            "n_positions": len(acct.positions),
            "holdings": ",".join(list(acct.positions)[:12]),
        })
        prev_nav = nav

    df = pd.DataFrame(rows)
    if len(df) and beta_h:
        m = df["predicted_return"].notna() & df["daily_return"].notna()
        df.attrs["pred_hit"] = float(
            (np.sign(df.loc[m, "predicted_return"]) == np.sign(df.loc[m, "daily_return"])).mean()
        ) if m.sum() else float("nan")
        df.attrs["pred_corr"] = float(
            df.loc[m, "predicted_return"].corr(df.loc[m, "daily_return"])
        ) if m.sum() > 5 else float("nan")
    return df


# ------------------------------------------------------------------ 周期汇总
def period_summary(ledger: pd.DataFrame) -> Dict:
    """日 / 周 / 月 / 总 收益汇总。"""
    if ledger is None or len(ledger) == 0:
        return {"ok": False, "reason": "empty_ledger"}
    df = ledger.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df = df.set_index("dt").sort_index()
    nav = df["nav"]

    def _ret(days: Optional[int]) -> Optional[float]:
        if days is None:
            s = nav
        else:
            s = nav[nav.index >= nav.index[-1] - pd.Timedelta(days=days)]
        if len(s) < 2:
            return None
        return float(s.iloc[-1] / s.iloc[0] - 1.0)

    daily = df["daily_return"].dropna()
    out = {
        "ok": True,
        "as_of": str(nav.index[-1].date()),
        "start": str(nav.index[0].date()),
        "n_days": int(len(nav)),
        "nav": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "today_return": float(daily.iloc[-1]) if len(daily) else None,
        "week_return": _ret(7),
        "month_return": _ret(30),
        "quarter_return": _ret(90),
        "year_return": _ret(365),
        "ann_vol": float(daily.std() * np.sqrt(252)) if len(daily) > 2 else None,
        "ann_sharpe": (float(daily.mean() / daily.std() * np.sqrt(252))
                       if len(daily) > 2 and daily.std() > 0 else None),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "win_rate": float((daily > 0).mean()) if len(daily) else None,
        "n_positions": int(df["n_positions"].iloc[-1]) if "n_positions" in df else None,
    }
    if "predicted_return" in df:
        m = df["predicted_return"].notna() & df["daily_return"].notna()
        if m.sum() > 5:
            pr = df.loc[m, "predicted_return"]
            ar = df.loc[m, "daily_return"]
            out["pred_actual_corr"] = float(pr.corr(ar))
            out["pred_sign_hit"] = float((np.sign(pr) == np.sign(ar)).mean())
            out["pred_mae"] = float((pr - ar).abs().mean())
            out["n_pred_days"] = int(m.sum())
    return out


def save_ledger(ledger: pd.DataFrame, day: Optional[str] = None) -> str:
    """只追加落盘。"""
    os.makedirs(ACCOUNT_DIR, exist_ok=True)
    day = day or (ledger["date"].iloc[-1] if len(ledger) else pd.Timestamp.now().strftime("%Y-%m-%d"))
    path = os.path.join(ACCOUNT_DIR, f"ledger_{day[:7]}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for r in ledger.to_dict("records"):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def format_report(summary: Dict, ledger: Optional[pd.DataFrame] = None) -> str:
    """人类可读的账户日报。"""
    if not summary.get("ok"):
        return "账户尚无数据，先运行一次模拟。"
    f = lambda x, p=2: "—" if x is None else f"{x*100:+.{p}f}%"
    L = [f"## 模拟账户 · {summary['as_of']}", ""]
    L.append(f"**净值 {summary['nav']:,.0f}（累计 {f(summary['total_return'])}）**")
    L.append("")
    L.append("| 周期 | 收益 |")
    L.append("|---|---|")
    L.append(f"| 当日 | {f(summary['today_return'])} |")
    L.append(f"| 本周 | {f(summary['week_return'])} |")
    L.append(f"| 本月 | {f(summary['month_return'])} |")
    L.append(f"| 本季 | {f(summary['quarter_return'])} |")
    L.append(f"| 本年 | {f(summary['year_return'])} |")
    L.append(f"| **累计** | **{f(summary['total_return'])}** |")
    L.append("")
    L.append(f"- 持仓 {summary['n_positions']} 只｜最大回撤 {f(summary['max_drawdown'])}"
             f"｜日胜率 {summary['win_rate']:.1%}" if summary.get('win_rate') is not None else "")
    if "pred_sign_hit" in summary:
        L.append(f"- **预测 vs 实际**：方向命中 {summary['pred_sign_hit']:.1%}，"
                 f"相关系数 {summary['pred_actual_corr']:+.3f}，"
                 f"平均绝对误差 {summary['pred_mae']:.4%}（{summary['n_pred_days']} 个交易日）")
    L.append("")
    L.append("> 预测收益 = 持仓加权 score × β，β 来自 meta 层因果回归。")
    L.append("> 没有 β 时不编预测（显示 —）。")
    return "\n".join(L)
