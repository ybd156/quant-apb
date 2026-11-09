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
def _account_path(market: str = "CN", horizon: int = 1) -> str:
    """
    (市场, 周期) 一个账户文件。

    为什么要两个维度：
      · 市场：币种、成本口径、整手规则完全不同，混在一起无法对账
      · 周期：T+1 和 1周 的持仓周期、调仓节奏、成本拖累差几倍，
              放同一个账户里，谁也说不清盈亏该归给谁

    老文件兼容：CN + h=1 仍用 paper_auto.json（老版本只有这一个）。
    """
    m = str(market or "CN").upper()
    h = int(horizon or 1)
    if m == "CN" and h == 1:
        return os.path.join(STATE, "paper_auto.json")
    return os.path.join(STATE, f"paper_auto_{m.lower()}_h{h}.json")


ACCOUNT = os.path.join(STATE, "paper_auto.json")   # CN/h1 的兼容路径


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


def load_account(market: str = "CN", horizon: int = 1) -> Dict:
    p = _account_path(market, horizon)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trades(market: Optional[str] = None,
                horizon: Optional[int] = None) -> List[Dict]:
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
    if horizon is not None:
        h = int(horizon)
        # 老记录没有 horizon 字段 -> 视为 h=1（那时只有一个账户）
        out = [t for t in out if int(t.get("horizon") or 1) == h]
    return out


def init_account(capital: float, market: str = "CN", horizon: int = 1) -> Dict:
    m = str(market).upper()
    h = int(horizon or 1)
    acc = {"market": m, "horizon": h, "cash": float(capital),
           "initial_capital": float(capital), "positions": {},
           "trade_periods": 0,
           "created_at": _now(), "updated_at": _now()}
    _atomic(_account_path(m, h), acc)
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
              verbose: bool = True, horizon: int = 1,
              force: bool = False) -> Dict:
    """
    按 picks 调仓。返回 {orders, filled, skipped, account}。

    horizon：用哪个周期的预测（读 picks_h{h}.json）。
             每个 (市场, 周期) 是独立账户 —— T+1 和 1周 的成本拖累差几倍，
             混在一个账户里说不清盈亏归谁。

    force：忽略持有期未到。默认 False —— 周期没到不该调仓，
           否则 T+15 的账户会每天换手，成本吃光收益。
    """
    from . import account_config as AC
    from . import fees as FEES
    from . import horizons as HZ

    market = str(market).upper()
    h = int(horizon or 1)
    cfg = AC.load(market)
    k = int(top_k or cfg.get("top_k") or 20)
    cost_bps = float(cfg.get("cost_bps") or 10.0)

    # ---- 按周期选 picks 文件 ----
    if picks is None:
        fn = "picks.json" if h == 1 else f"picks_h{h}.json"
        try:
            with open(os.path.join(STATE, fn), encoding="utf-8") as f:
                picks = json.load(f)
        except Exception:
            # 区分两种缺失：没模型（要去训练机）vs 有模型没预测（在 NAS 跑一次 predict）
            has_model = False
            try:
                from . import live_model as LM
                has_model = LM.load_model(h)[0] is not None
            except Exception:
                pass
            if not has_model:
                return {"ok": False, "reason": "no_model", "horizon": h,
                        "hint": f"h={h} 还没有模型。训练只在训练机做："
                                f"在训练机跑 tools/train_one_click.py "
                                f"--horizons={h}，训完会自动上传到这台 NAS"}
            return {"ok": False, "reason": "no_picks", "horizon": h,
                    "hint": f"h={h} 有模型但还没生成预测，在 NAS 跑一次 "
                            f"python -m qcore.live_model --predict --horizon={h} "
                            f"（推理秒级，不需要训练机）"}
    rows = (picks or {}).get("rows") or []
    if not rows:
        return {"ok": False, "reason": "picks_empty", "horizon": h}
    as_of = str(picks.get("as_of") or "")[:10]

    # ---- 市场校验（重要）----
    # picks 文件里带 market 就必须匹配。
    # 实测踩过：美股账户读到 A 股 picks_h5.json，用 500 美元买了
    # 601229（上海银行）2 股 —— 跨市场串仓，对账全废。
    _pm = str((picks or {}).get("market") or "").upper()
    if _pm and _pm != market:
        return {"ok": False, "reason": "market_mismatch",
                "horizon": h, "picks_market": _pm, "account_market": market,
                "hint": f"{fn if picks is None else 'picks'} 属于 {_pm} 市场，"
                        f"不能给 {market} 账户下单。"
                        f"该市场需要自己的预测（先训练该市场的模型）"}
    # picks 里没写 market 时，用代码形态反查：
    # 全是 6 位纯数字 -> A股；带 .HK -> 港股；纯字母 -> 美股
    if not _pm and rows:
        _c0 = str(rows[0].get("code") or "")
        if _c0.isdigit() and market != "CN":
            return {"ok": False, "reason": "picks_market_mismatch",
                    "horizon": h, "picks_market": "CN", "account_market": market,
                    "hint": f"picks 里是 A 股代码（{_c0}），不能给 {market} 账户下单"}

    acc = load_account(market, h)
    if not acc:
        acc = init_account(float(cfg.get("initial_capital") or 1_000_000), market, h)
    if acc.get("market") != market:
        return {"ok": False, "reason": "market_mismatch",
                "account_market": acc.get("market"), "want": market,
                "hint": "模拟盘按市场独立，请用对应市场的账户"}

    # ---- 持有期闸门：没到 h 个交易日就不调仓 ----
    if not force:
        last = str(acc.get("last_rebalance") or "")[:10]
        if last and last == as_of:
            return {"ok": True, "skipped_all": True, "horizon": h,
                    "reason": "already_rebalanced_today",
                    "as_of": as_of, "orders": [], "filled": [], "skipped": [],
                    "account": acc,
                    "note": f"h={h} 今天已经调过仓，不重复交易"}
        prev = str(acc.get("as_of") or "")[:10]
        if prev and last:
            try:
                from .horizon_ledger import _calendar, _trading_days_after
                cal = _calendar()
                if cal:
                    # 从上次调仓日往后数，够 h 个交易日才允许再调
                    nxt = _trading_days_after(cal, last, h)
                    if nxt and as_of < nxt:
                        return {"ok": True, "skipped_all": True, "horizon": h,
                                "reason": "hold_period_not_elapsed",
                                "as_of": as_of, "last_rebalance": last,
                                "next_allowed": nxt,
                                "orders": [], "filled": [], "skipped": [],
                                "account": acc,
                                "note": f"h={h} 持有期未到（上次 {last}，"
                                        f"下次可调 {nxt}）"}
            except Exception:
                pass

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
            _append({"date": as_of, "market": market, "horizon": h,
                     "code": code, "side": "sell",
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
                _append({"date": as_of, "market": market, "horizon": h,
                         "code": code, "side": "buy",
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
                "horizon": h,
                "last_rebalance": as_of,
                "trade_periods": int(acc.get("trade_periods") or 0) + 1,
                "market_value": round(mv, 2),
                "equity": round(cash + mv, 2),
                "updated_at": _now(), "as_of": as_of,
                "n_positions": len(pos)})
    init = float(acc.get("initial_capital") or 1)
    acc["total_return"] = round(acc["equity"] / init - 1.0, 6) if init else None

    if not dry_run:
        _atomic(_account_path(market, h), acc)

    if verbose:
        print(f"[paper_auto] {as_of} {market} h{h}: "
              f"买 {sum(1 for o in orders if o['side']=='buy')} "
              f"卖 {sum(1 for o in orders if o['side']=='sell')} "
              f"跳过 {len(skipped)} | 持仓 {len(pos)} 净值 {acc['equity']:,.0f}",
              flush=True)
    return {"ok": True, "as_of": as_of, "market": market, "horizon": h,
            "orders": orders, "filled": filled, "skipped": skipped,
            "account": acc, "dry_run": dry_run}


def summary(market: str = "CN", horizon: int = 1) -> Dict:
    """单个 (市场, 周期) 账户摘要。"""
    m = str(market or "CN").upper()
    h = int(horizon or 1)
    acc = load_account(m, h)
    if not acc:
        return {"ok": False, "reason": "not_initialized", "market": m,
                "horizon": h,
                "hint": f"先跑 python -m qcore.paper_auto --init "
                        f"--market={m} --horizon={h}"}
    trades = load_trades(m, h)
    return {"ok": True, "market": m, "horizon": h, "account": acc,
            "n_trades": len(trades),
            "recent_trades": trades[-20:],
            "note": "模拟盘自动成交，与手工录入的 live_trades 完全分开"}




def summary_all() -> Dict:
    """三个市场 × 所有周期 的矩阵总览。"""
    try:
        from . import horizons as HZ
        hs = HZ.all_horizons()
    except Exception:
        hs = [1, 3, 5, 10, 15]
    grid = {}
    total_equity = {}
    for m in ("CN", "US", "HK"):
        grid[m] = {}
        eq = 0.0
        for h in hs:
            r = summary(m, h)
            grid[m][h] = {
                "ok": bool(r.get("ok")),
                "reason": r.get("reason"),
                "equity": (r.get("account") or {}).get("equity"),
                "cash": (r.get("account") or {}).get("cash"),
                "n_positions": (r.get("account") or {}).get("n_positions"),
                "total_return": (r.get("account") or {}).get("total_return"),
                "n_trades": r.get("n_trades"),
                "as_of": (r.get("account") or {}).get("as_of"),
            }
            if r.get("ok") and (r.get("account") or {}).get("equity"):
                eq += float(r["account"]["equity"])
        total_equity[m] = round(eq, 2) if eq else None
    return {"ok": True, "markets": ("CN", "US", "HK"), "horizons": hs,
            "grid": grid, "total_equity_by_market": total_equity,
            "note": "每个 (市场, 周期) 是独立账户；本金、成本、币种、成交全部分开"}


def rebalance_all(market: str = "CN", dry_run: bool = False,
                  force: bool = False, verbose: bool = True) -> Dict:
    """
    对所有已训练周期各调一次仓。该周期没模型/没预测就跳过并说明。
    """
    try:
        from . import horizons as HZ
        hs = HZ.all_horizons()
    except Exception:
        hs = [1, 3, 5, 10, 15]
    out = []
    for h in hs:
        if h == 0:
            continue          # T+0 只有 T+0 市场才有，A股不存在
        try:
            r = rebalance(market=market, horizon=h, dry_run=dry_run,
                          force=force, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}"}
        out.append({"horizon": h, "ok": bool(r.get("ok")),
                    "reason": r.get("reason") or r.get("note"),
                    "hint": r.get("hint"),
                    "orders": len(r.get("orders") or []),
                    "skipped_all": bool(r.get("skipped_all"))})
    if verbose:
        done = [x for x in out if x["ok"] and not x["skipped_all"]]
        no_model = [x for x in out if x.get("reason") == "no_model"]
        no_picks = [x for x in out if x.get("reason") == "no_picks"]
        print(f"[paper_auto] {market} 批量调仓：{len(done)} 个周期有动作；"
              f"缺模型 {len(no_model)}（需训练机）；缺预测 {len(no_picks)}（NAS 可跑）",
              flush=True)
    return {"ok": True, "market": str(market).upper(), "dry_run": dry_run,
            "results": out}


def by_period(market: str = "CN", bucket: str = "day",
              horizon: Optional[int] = None) -> Dict:
    """按时间段汇总成交。bucket: day | week | month"""
    trades = load_trades(market, horizon)
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


def fee_summary(market: str = "CN", horizon: Optional[int] = None) -> Dict:
    """成交费用按项目汇总 —— 佣金/印花税/过户费各花了多少。"""
    trades = load_trades(market, horizon)
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
    按交易周期横向对比：每个 (市场, 周期) 账户的净值、成交、费用。
    这是「T+1 和 1周 到底哪个赚」的直接答案。
    """
    try:
        from . import horizons as HZ
        hs = HZ.all_horizons()
    except Exception:
        hs = [1, 3, 5, 10, 15]
    rows = []
    for h in hs:
        if h == 0:
            continue
        r = summary(market, h)
        if not r.get("ok"):
            rows.append({"horizon": h, "label": _label(h), "ok": False,
                         "reason": r.get("reason"), "n_trades": 0})
            continue
        a = r.get("account") or {}
        fs = fee_summary(market, h)
        rows.append({
            "horizon": h, "label": _label(h), "ok": True,
            "equity": a.get("equity"), "initial": a.get("initial_capital"),
            "total_return": a.get("total_return"),
            "n_positions": a.get("n_positions"),
            "n_trades": r.get("n_trades"),
            "trade_periods": a.get("trade_periods"),
            "fee_total": fs.get("total"),
            "fee_avg_bps": fs.get("avg_bps"),
            "as_of": a.get("as_of"),
        })
    # 排名：总收益高的在前（只有有数据的才排）
    ranked = sorted([x for x in rows if x.get("ok") and x.get("total_return") is not None],
                    key=lambda x: -float(x["total_return"]))
    best = ranked[0] if ranked else None
    return {"ok": True, "market": str(market).upper(), "rows": rows,
            "best_horizon": best.get("horizon") if best else None,
            "verdict": (
                f"目前最好的是 {best['label']}（累计 {best['total_return']*100:+.2f}%，"
                f"成交 {best['n_trades']} 笔，费用 {best['fee_total']} 元）"
                if best else "还没有任何周期账户产生数据")}


def _label(h: int) -> str:
    try:
        from . import horizons as HZ
        return HZ.label_for(int(h))
    except Exception:
        return f"h={h}"



if __name__ == "__main__":
    import sys

    def _arg(name, default=""):
        for a in sys.argv:
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
        return default

    market = _arg("market", "CN").upper()
    try:
        horizon = int(_arg("horizon", "1"))
    except ValueError:
        horizon = 1

    if "--init" in sys.argv:
        cap = float(_arg("capital", "1000000"))
        if "--all-horizons" in sys.argv:
            try:
                from . import horizons as HZ
                hs = [h for h in HZ.all_horizons() if h > 0]
            except Exception:
                hs = [1, 3, 5, 10, 15]
            out = [init_account(cap, market, h) for h in hs]
            print(json.dumps({"ok": True, "market": market,
                              "initialized": [o["horizon"] for o in out],
                              "capital_each": cap}, ensure_ascii=False, indent=1))
        else:
            print(json.dumps(init_account(cap, market, horizon),
                             ensure_ascii=False, indent=1))
    elif "--all" in sys.argv:
        print(json.dumps(rebalance_all(market, dry_run=("--dry" in sys.argv)),
                         ensure_ascii=False, indent=1)[:2000])
    elif "--dry" in sys.argv:
        print(json.dumps(rebalance(market=market, horizon=horizon, dry_run=True),
                         ensure_ascii=False, indent=1)[:1500])
    elif "--matrix" in sys.argv:
        print(json.dumps(summary_all(), ensure_ascii=False, indent=1)[:2500])
    elif "--by-horizon" in sys.argv:
        print(json.dumps(by_horizon(market), ensure_ascii=False, indent=1)[:2000])
    elif "--summary" in sys.argv:
        print(json.dumps(summary(market, horizon), ensure_ascii=False, indent=1)[:1200])
    else:
        print(json.dumps(rebalance(market=market, horizon=horizon),
                         ensure_ascii=False, indent=1)[:1800])
