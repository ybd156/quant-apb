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
def _account_path(market: str) -> str:
    """
    按市场分账户文件。三个市场各自独立的资金、持仓、成交，
    绝不混在一起（成本口径和币种都不同）。
    """
    m = str(market).upper()
    return os.path.join(STATE, "paper_auto.json" if m == "CN"
                        else f"paper_auto_{m.lower()}.json")


ACCOUNT = os.path.join(STATE, "paper_auto.json")   # CN 的兼容路径


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


def load_account(market: str = "CN") -> Dict:
    p = _account_path(market)
    # 老版本只有 paper_auto.json（CN），非 CN 市场不要误读
    if not os.path.isfile(p) and str(market).upper() != "CN":
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trades(market: Optional[str] = None) -> List[Dict]:
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
    if market:
        m = str(market).upper()
        out = [t for t in out if str(t.get("market") or "CN").upper() == m]
    return out


def init_account(capital: float, market: str = "CN") -> Dict:
    m = str(market).upper()
    acc = {"market": m, "cash": float(capital),
           "initial_capital": float(capital), "positions": {},
           "created_at": _now(), "updated_at": _now()}
    _atomic(_account_path(m), acc)
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
    from . import fees as FEES

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

    # fees 模块算每笔明细（佣金有最低收费、印花税仅卖出、各市场不同）。
    # 不用扁平 bps —— 买卖不对称，用平均值会系统性低估卖出成本。
    def _fee(side, amount, qty, price):
        try:
            return FEES.trade_fees(market, side, amount, qty=qty, price=price)
        except Exception:
            v = amount * cost_bps / 1e4 / 2.0
            return {'items': [], 'total': v, 'bps': cost_bps / 2,
                    'source': 'fallback'}

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
        fee = _fee("sell", gross, qty, px)
        fee_amt = float(fee.get("total") or 0)
        net = gross - fee_amt
        orders.append({"code": code, "side": "sell", "qty": qty,
                       "price": round(px, 4), "amount": round(gross, 2),
                       "fee": round(fee_amt, 4),
                       "fee_items": fee.get("items") or [],
                       "fee_bps": fee.get("bps"),
                       "net": round(net, 2), "reason": "掉出 top-k"})
        if not dry_run:
            cash += net
            _append({"date": as_of, "market": market, "code": code, "side": "sell",
                     "price": px, "qty": qty, "amount": round(gross, 2),
                     "cost": round(fee_amt, 4),
                     "fee_items": fee.get("items") or [],
                     "fee_bps": fee.get("bps"),
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
            fee = _fee("buy", gross, qty, px)
            fee_amt = float(fee.get("total") or 0)
            net = gross + fee_amt
            if net > cash:
                est = fee_amt / max(gross, 1.0)
                qty = int(cash // (px * lot * (1 + est))) * lot
                if qty < lot:
                    skipped.append({"code": code, "why": "现金不足（含费用）"})
                    continue
                gross = qty * px
                fee = _fee("buy", gross, qty, px)
                fee_amt = float(fee.get("total") or 0)
                net = gross + fee_amt
            orders.append({"code": code, "side": "buy", "qty": qty,
                           "price": round(px, 4), "amount": round(gross, 2),
                           "fee": round(fee_amt, 4),
                           "fee_items": fee.get("items") or [],
                           "fee_bps": fee.get("bps"),
                           "net": round(net, 2),
                           "score": round(score_of.get(code, 0), 5),
                           "reason": "新进入 top-k"})
            if not dry_run:
                cash -= net
                pos[code] = {"qty": qty, "cost": px, "opened_at": _now(),
                             "score": score_of.get(code, 0),
                             "fee_paid": round(fee_amt, 4)}
                _append({"date": as_of, "market": market, "code": code, "side": "buy",
                         "price": px, "qty": qty, "amount": round(gross, 2),
                         "cost": round(fee_amt, 4),
                         "fee_items": fee.get("items") or [],
                         "fee_bps": fee.get("bps"),
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
        _atomic(_account_path(market), acc)

    if verbose:
        print(f"[paper_auto] {as_of} {market}: "
              f"买 {sum(1 for o in orders if o['side']=='buy')} "
              f"卖 {sum(1 for o in orders if o['side']=='sell')} "
              f"跳过 {len(skipped)} | 持仓 {len(pos)} 净值 {acc['equity']:,.0f}",
              flush=True)
    return {"ok": True, "as_of": as_of, "market": market,
            "orders": orders, "filled": filled, "skipped": skipped,
            "account": acc, "dry_run": dry_run}


def summary(market: str = "CN") -> Dict:
    """单市场摘要。三个市场各自独立的本金、成本、币种、成交。"""
    m = str(market or "CN").upper()
    acc = load_account(m)
    if not acc:
        return {"ok": False, "reason": "not_initialized", "market": m,
                "hint": f"先跑 python -m qcore.paper_auto --init --market={m}"}
    trades = load_trades(m)
    return {"ok": True, "market": m, "account": acc,
            "n_trades": len(trades),
            "recent_trades": trades[-20:],
            "note": "模拟盘自动成交，与手工录入的 live_trades 完全分开"}


def summary_all() -> Dict:
    """三个市场一起看。各自独立的本金、成本、币种、成交。"""
    return {"ok": True,
            "markets": {m: summary(m) for m in ("CN", "US", "HK")},
            "note": "按市场独立；未初始化的市场会返回 not_initialized"}


def by_period(market: str = "CN", bucket: str = "day") -> Dict:
    """按时间段汇总成交。bucket: day | week | month"""
    trades = load_trades(market)
    if not trades:
        return {"ok": True, "market": str(market).upper(), "bucket": bucket,
                "buckets": [], "n_trades": 0}

    def _key(d) -> str:
        s0 = str(d or "")[:10]
        if bucket == "month":
            return s0[:7]
        if bucket == "week":
            try:
                dt = datetime.strptime(s0, "%Y-%m-%d")
                iso = dt.isocalendar()
                return "%d-W%02d" % (iso[0], iso[1])
            except Exception:
                return s0
        return s0

    agg = {}
    for t in trades:
        k = _key(t.get("date"))
        a = agg.setdefault(k, {"n": 0, "buys": 0, "sells": 0,
                               "buy_amount": 0.0, "sell_amount": 0.0, "fee": 0.0})
        a["n"] += 1
        a["fee"] += float(t.get("cost") or 0)
        if t.get("side") == "buy":
            a["buys"] += 1
            a["buy_amount"] += float(t.get("amount") or 0)
        else:
            a["sells"] += 1
            a["sell_amount"] += float(t.get("amount") or 0)

    rows = []
    for k in sorted(agg, reverse=True):
        a = agg[k]
        rows.append({"period": k, "n": a["n"], "buys": a["buys"], "sells": a["sells"],
                     "buy_amount": round(a["buy_amount"], 2),
                     "sell_amount": round(a["sell_amount"], 2),
                     "fee": round(a["fee"], 2)})
    return {"ok": True, "market": str(market).upper(), "bucket": bucket,
            "n_trades": len(trades), "n_buckets": len(rows), "buckets": rows}


def fee_summary(market: str = "CN") -> Dict:
    """成交费用按项目汇总 —— 佣金/印花税/过户费各花了多少。"""
    trades = load_trades(market)
    if not trades:
        return {"ok": True, "market": str(market).upper(), "items": [],
                "total": 0, "n_trades": 0}
    agg = {}
    for t in trades:
        items = t.get("fee_items") or []
        if items:
            for it in items:
                k = str(it.get("item") or "其他")
                agg[k] = agg.get(k, 0.0) + float(it.get("amount") or 0)
        elif t.get("cost"):
            agg["其他（扁平成本）"] = agg.get("其他（扁平成本）", 0.0) + float(t["cost"])
    total = sum(agg.values())
    return {"ok": True, "market": str(market).upper(), "n_trades": len(trades),
            "items": [{"item": k, "amount": round(v, 2),
                       "share": round(v / total, 4) if total else None}
                      for k, v in sorted(agg.items(), key=lambda x: -x[1])],
            "total": round(total, 2),
            "avg_bps": round(total / max(sum(float(t.get("amount") or 0)
                                             for t in trades), 1) * 1e4, 2)}


def by_horizon(market: str = "CN") -> Dict:
    """
    按交易周期汇总 —— 但成交本身不带周期，得靠 picks 的 horizon 反查。
    取不到就明说取不到，不编。
    """
    return {"ok": True, "market": str(market).upper(), "rows": [],
            "note": ("当前成交记录不带 horizon 字段（同一个账户可以混用多周期信号）。"
                     "要看分周期表现请用 /api/horizons 的台账，那里每个周期独立记账。")}


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
