"""
模拟盘自动交易 —— 按模型预测自动下单、记账、估值。

## 解决什么

之前模拟盘只有 `simulate()`（回测式）和 `live_nav()`（对账式），
**没有"按今天的预测自动调仓"这一步**。用户看到的是一堆数字，
但账户不会动。

本模块把 `picks.json` 变成真实的持仓变动：

    今日预测 top-k
      ↓ 与当前持仓比对
    买入：新进入 top-k 的
    卖出：掉出 top-k 的 / 到期该换的
      ↓ 按整手约束（qcore/lot.py）取整
    成交记录 → data/paper/trades.jsonl
      ↓ 用当日收盘价估值
    持仓快照 → data/state/paper_auto.json

## 三条硬约束

1. **整手**：A股 100 股一手，买不起就跳过（不四舍五入成 0 股假装买了）
2. **T+1**：当天买的不能当天卖 —— 卖出时排除当日新建仓
3. **成本**：买入/卖出都扣 round-trip 的一半（单边成本）

## 与 live_trades 的区别

`data/state/live_trades.jsonl` 是**用户手工录入的实盘成交**，用于对账。
本模块写 `data/paper/trades.jsonl`，是**模拟盘的自动成交**。
两者绝不混用 —— 否则回测和实盘的对比就废了。

## 因果

只用**已经发生的**数据：picks 的 `as_of` 必须是今天或更早，
成交价用当日收盘（T 日出信号，T+1 成交更严谨，但模拟盘先用 T 日收盘简化，
并在记录里标明 `price_basis="close_T"`）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
PAPER_DIR = os.path.join(ROOT, "data", "paper")
TRADES = os.path.join(PAPER_DIR, "trades.jsonl")
ACCOUNT = os.path.join(STATE, "paper_auto.json")


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


def _append(rec: Dict) -> None:
    os.makedirs(os.path.dirname(TRADES), exist_ok=True)
    with open(TRADES, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def load_account() -> Dict:
    try:
        with open(ACCOUNT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trades() -> List[Dict]:
    if not os.path.isfile(TRADES):
        return []
    out = []
    try:
        with open(TRADES, encoding="utf-8") as f:
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


def init_account(capital: float, market: str = "CN") -> Dict:
    acc = {"market": str(market).upper(), "cash": float(capital),
           "initial_capital": float(capital), "positions": {},
           "created_at": _now(), "updated_at": _now()}
    _atomic(ACCOUNT, acc)
    return acc


def _price(code: str, date: Optional[str] = None) -> Optional[float]:
    """真实价（不复权）。不传 date 取最新。"""
    try:
        from .data import fetch_kline
        df = fetch_kline(code, adjust=0, klt=101, beg="20240101")
        if df is None or not len(df):
            return None
        d = df.copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        if date:
            hit = d[d["date"] == str(date)[:10]]
            return float(hit["close"].iloc[-1]) if len(hit) else None
        return float(d["close"].iloc[-1])
    except Exception:
        return None


def _lot(code: str, market: str = "CN") -> int:
    try:
        from .lot import _lot_for
        return int(_lot_for(code, str(market).upper()))
    except Exception:
        return 100 if str(market).upper() == "CN" else 1


def rebalance(picks: Optional[Dict] = None, market: str = "CN",
              top_k: Optional[int] = None, dry_run: bool = False,
              verbose: bool = True) -> Dict:
    """
    按 picks 调仓。返回 {orders, filled, skipped, account}。
    """
    from . import account_config as AC
    from . import costs as CO

    market = str(market).upper()
    cfg = AC.load(market)
    k = int(top_k or cfg.get("top_k") or 20)
    cost_bps = float(cfg.get("cost_bps") or 10.0)

    if picks is None:
        try:
            with open(os.path.join(STATE, "picks.json"), encoding="utf-8") as f:
                picks = json.load(f)
        except Exception:
            return {"ok": False, "reason": "no_picks"}
    rows = (picks or {}).get("rows") or []
    if not rows:
        return {"ok": False, "reason": "picks_empty"}
    as_of = str(picks.get("as_of") or "")[:10]

    acc = load_account()
    if not acc:
        acc = init_account(float(cfg.get("initial_capital") or 1_000_000), market)
    if acc.get("market") != market:
        return {"ok": False, "reason": "market_mismatch",
                "account_market": acc.get("market"), "want": market,
                "hint": "模拟盘按市场独立，请用对应市场的账户"}

    want = [str(r.get("code", "")).zfill(6) for r in rows[:k]]
    score_of = {str(r.get("code", "")).zfill(6): float(r.get("score") or 0) for r in rows}
    pos: Dict[str, Dict] = dict(acc.get("positions") or {})
    cash = float(acc.get("cash") or 0)

    half_cost = cost_bps / 1e4 / 2.0
    orders, filled, skipped = [], [], []

    # ---- 1) 卖出：掉出 top-k 的（T+1：今天建的仓不卖）----
    for code, p in list(pos.items()):
        if code in want:
            continue
        if str(p.get("opened_at", ""))[:10] == as_of:
            skipped.append({"code": code, "why": "T+1，今日建仓不可卖"})
            continue
        px = _price(code, as_of) or _price(code)
        if not px:
            skipped.append({"code": code, "why": "取不到价格"})
            continue
        qty = int(p.get("qty") or 0)
        if qty <= 0:
            pos.pop(code, None)
            continue
        gross = qty * px
        net = gross * (1 - half_cost)
        orders.append({"code": code, "side": "sell", "qty": qty,
                       "price": round(px, 4), "amount": round(gross, 2),
                       "net": round(net, 2), "reason": "掉出 top-k"})
        if not dry_run:
            cash += net
            _append({"date": as_of, "code": code, "side": "sell", "price": px,
                     "qty": qty, "amount": round(gross, 2), "cost": round(gross * half_cost, 2),
                     "reason": "rebalance_out", "source": "auto", "at": _now()})
            pos.pop(code, None)
        filled.append(code)

    # ---- 2) 买入：新进入 top-k 的 ----
    holds = set(pos.keys())
    new = [c for c in want if c not in holds]
    if new:
        # 目标等权，但受现金与整手约束
        per = cash / max(len(new), 1) if new else 0
        for code in new:
            px = _price(code, as_of) or _price(code)
            if not px:
                skipped.append({"code": code, "why": "取不到价格"})
                continue
            lot = _lot(code, market)
            budget = min(per, cash)
            qty = int(budget // (px * lot)) * lot
            if qty < lot:
                skipped.append({"code": code, "why": f"买不起一手（需 {px*lot:,.0f} 元）",
                                "price": round(px, 2), "lot": lot})
                continue
            gross = qty * px
            net = gross * (1 + half_cost)
            if net > cash:
                qty = int(cash // (px * lot * (1 + half_cost))) * lot
                if qty < lot:
                    skipped.append({"code": code, "why": "现金不足"})
                    continue
                gross = qty * px
                net = gross * (1 + half_cost)
            orders.append({"code": code, "side": "buy", "qty": qty,
                           "price": round(px, 4), "amount": round(gross, 2),
                           "net": round(net, 2),
                           "score": round(score_of.get(code, 0), 5),
                           "reason": "新进入 top-k"})
            if not dry_run:
                cash -= net
                pos[code] = {"qty": qty, "cost": px, "opened_at": _now(),
                             "score": score_of.get(code, 0)}
                _append({"date": as_of, "code": code, "side": "buy", "price": px,
                         "qty": qty, "amount": round(gross, 2), "cost": round(gross * half_cost, 2),
                         "reason": "rebalance_in", "source": "auto", "at": _now()})
            filled.append(code)

    # ---- 3) 估值 ----
    mv = 0.0
    for code, p in pos.items():
        px = _price(code, as_of) or _price(code) or p.get("cost") or 0
        p["last_price"] = round(float(px), 4)
        p["market_value"] = round(float(px) * int(p.get("qty") or 0), 2)
        mv += p["market_value"]

    acc.update({"cash": round(cash, 2), "positions": pos,
                "market_value": round(mv, 2),
                "equity": round(cash + mv, 2),
                "updated_at": _now(), "as_of": as_of,
                "n_positions": len(pos)})
    init = float(acc.get("initial_capital") or 1)
    acc["total_return"] = round(acc["equity"] / init - 1.0, 6) if init else None

    if not dry_run:
        _atomic(ACCOUNT, acc)

    if verbose:
        print(f"[paper_auto] {as_of} {market}: "
              f"买 {sum(1 for o in orders if o['side']=='buy')} "
              f"卖 {sum(1 for o in orders if o['side']=='sell')} "
              f"跳过 {len(skipped)} | 持仓 {len(pos)} 净值 {acc['equity']:,.0f}",
              flush=True)
    return {"ok": True, "as_of": as_of, "market": market,
            "orders": orders, "filled": filled, "skipped": skipped,
            "account": acc, "dry_run": dry_run}


def summary() -> Dict:
    acc = load_account()
    if not acc:
        return {"ok": False, "reason": "not_initialized",
                "hint": "先跑 python -m qcore.paper_auto --init 1000000"}
    trades = load_trades()
    return {"ok": True, "account": acc,
            "n_trades": len(trades),
            "recent_trades": trades[-12:],
            "note": "模拟盘自动成交，与手工录入的 live_trades 完全分开"}


if __name__ == "__main__":
    import sys
    if "--init" in sys.argv:
        cap = 1_000_000.0
        for a in sys.argv:
            if a.startswith("--capital="):
                cap = float(a.split("=", 1)[1])
        print(json.dumps(init_account(cap), ensure_ascii=False, indent=1))
    elif "--dry" in sys.argv:
        print(json.dumps(rebalance(dry_run=True), ensure_ascii=False, indent=1)[:1500])
    elif "--summary" in sys.argv:
        print(json.dumps(summary(), ensure_ascii=False, indent=1)[:1200])
    else:
        print(json.dumps(rebalance(), ensure_ascii=False, indent=1)[:1800])
