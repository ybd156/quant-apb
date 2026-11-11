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
    # equity 一开始就等于本金 —— 否则刚建好的账户在 UI 上显示「净值 —」，
    # 用户会以为没生效。n_positions 同理记 0。
    acc = {"market": m, "horizon": h, "cash": float(capital),
           "initial_capital": float(capital), "positions": {},
           "equity": float(capital), "n_positions": 0, "total_return": 0.0,
           "trade_periods": 0, "as_of": None,
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


def _norm6(code) -> str:
    """`sz000988` / `000988.SZ` / `000988` → `000988`。"""
    t = str(code or "").strip().lower()
    for pre in ("sh", "sz", "bj"):
        if t.startswith(pre):
            t = t[len(pre):]
    return t.split(".")[0].zfill(6)


def _live_quotes(codes) -> Dict[str, Dict]:
    """批量取**实时**价。返回 {6位代码: {price, as_of, stale, source, latency_s}}。

    盘中调仓必须用这个，不能拿 as_of 那天的收盘价成交 ——
    实测事故：周一 13:10 用周五收盘价成交 20 笔（每笔虚增约 2.2%）。
    取不到实时价的票由调用方跳过（或显式 allow_backfill 才回退旧收盘价）。
    """
    codes = [str(c) for c in codes if c]
    if not codes:
        return {}
    try:
        from . import tick as T
        r = T.snapshot(codes)
    except Exception:  # noqa: BLE001
        return {}
    if not r.get("ok"):
        return {}
    out: Dict[str, Dict] = {}
    for row in (r.get("rows") or []):
        px = row.get("price")
        try:
            px = float(px)
        except (TypeError, ValueError):
            continue
        if not (px > 0):
            continue
        out[_norm6(row.get("code"))] = {
            "price": px, "as_of": r.get("as_of"), "stale": bool(r.get("stale")),
            "source": r.get("source"), "latency_s": r.get("latency_s"),
        }
    return out


def _px_for(code: str, as_of: str, live: Optional[Dict[str, Dict]] = None,
            allow_backfill: bool = False) -> Tuple[Optional[float], str, Optional[str]]:
    """决定这一笔用哪个价：返回 (price, basis, price_as_of)。

    优先级：
      1. 实时价（盘中）—— basis="live"，price_as_of 是报价时刻
      2. as_of 当天收盘价 —— basis="close"（只用于 dry_run 试算；
         真实成交不会走这条路，因为盘前盘后已被门禁拒绝）
      3. allow_backfill 时才回退「最新一根收盘」（**默认禁止**，那是未来价）
    """
    c6 = _norm6(code)
    if live:
        q = live.get(c6)
        if q and q.get("price"):
            return float(q["price"]), "live", str(q.get("as_of") or "")
    px = _price(code, as_of)
    if px:
        return float(px), "close", str(as_of)
    if allow_backfill:
        px = _price(code)
        if px:
            return float(px), "latest_close_backfill", None
    return None, "none", None


def _lot(code: str, market: str = "CN") -> int:
    try:
        from .lot import _lot_for
        return int(_lot_for(code, str(market).upper()))
    except Exception:
        return 100 if str(market).upper() == "CN" else 1


# 每个周期默认持有几只。
#
# 逻辑：周期越短，换手越频繁、成本占比越高，越该集中。
#   T+1  年换手 250 次，成本拖累 38.6%/年 -> 只做 2 只，减少交易次数
#   15 天 年换手 17 次，成本 2.6%/年 -> 可以分散到 10 只
#
# 实测：5000 元本金下，T+1 分 5 只每只 1000 元，
# 排名靠前的票一手就要上万 —— 一只都买不起，账户永远空仓。
HORIZON_TOP_K = {1: 2, 3: 3, 5: 5, 10: 8, 15: 10}


def lag_trading_days(as_of: str, now: Optional[datetime] = None) -> Optional[int]:
    """`as_of` 相对「最近**已收盘**交易日」落后几个交易日。0 = 就是最近已收盘那天。

    实测踩过的坑：旧实现以 **qlib 日历的最后一根** 为基准，
    而本机 qlib 日历最后一根就停在 2026-11-06 —— 于是 `as_of=11-06` 恒返回 0，
    「picks 陈旧」这道闸门永远不触发（陈旧检测等于没有）。

    正确基准是**墙钟 + 交易日历**推出来的「最近已收盘交易日」：
      · 盘中/盘前 → 上一交易日
      · 盘后 15:05 之后（且今天是交易日）→ 今天
    返回 None 表示算不出来（日历不可用 / 日期非法 / 落在非交易日）。
    """
    if not as_of:
        return None
    import datetime as _dt
    try:
        from . import calendar_cn as CAL
    except Exception:  # noqa: BLE001
        return None
    try:
        target = _dt.date.fromisoformat(str(as_of)[:10])
    except Exception:  # noqa: BLE001
        return None
    if not CAL.is_trading_day_bool(target):
        return None

    now = now or datetime.now()
    today = now.date()
    phase = None
    try:
        from . import session as _S
        phase = _S.in_window(now, "CN")   # lag_trading_days 是 A 股日历口径
    except Exception:  # noqa: BLE001
        phase = None

    if phase == "postclose" and CAL.is_trading_day_bool(today):
        ref = today                      # 今天已收盘，今天的收盘价就是最新的
    elif CAL.is_trading_day_bool(today) and phase in ("morning", "lunch", "afternoon"):
        ref = CAL.prev_trading_day(today)
    else:
        ref = CAL.prev_trading_day(today)

    n, cur = 0, ref
    while cur > target and n < 250:
        cur = CAL.prev_trading_day(cur)
        n += 1
    return n if cur == target else None


def _assert_tradable(as_of_hint=None, dry_run: bool = False,
                     now: Optional[datetime] = None,
                     allow_backfill: bool = False,
                     market: str = "CN",
                     allow_extended: bool = False) -> Dict:
    """交易时间门禁 —— 全项目唯一实现。

    实测事故（2026-11-09）：周一 13:10 用**周五收盘价**成交 20 笔，并记成周五。
    三个条件缺一不可：

      ① `as_of` 必须是交易日（用 calendar_cn，不是数周一~周五）
      ② `as_of` 必须是「最近已收盘交易日」（`lag_trading_days == 0`）
         —— 陈旧信号不许成交，否则就是拿三天前的判断下单
      ③ 执行时刻必须落在「as_of 收盘之后、下一交易日开盘之前」
         —— 盘中执行时 as_of 的收盘价是过去的价格，拿它成交等于回填。
         **成交只能发生在盘中**（09:30–11:30 / 13:00–15:00）。
         A 股收盘价在 15:00 那一刻就定了，15:05 下单拿不到那个价 ——
         所以盘前/盘后一律拒绝（dry_run 试算除外），不存在「收盘后调仓」。

    `dry_run=True` 时跳过 ③（回放/试算不落盘，允许在任意时刻演练）。
    `allow_backfill=True` 显式越过 ③，但会在返回体里留痕 `"backfilled": True`。
    """
    import datetime as _dt
    mkt = str(market or "CN").upper()
    now = now or datetime.now()
    picks = as_of_hint or {}
    as_of = str((picks or {}).get("as_of") or "")[:10]
    if not as_of:
        return {"ok": False, "reason": "no_as_of",
                "hint": "picks 缺 as_of，无法判定交易日"}

    # ① 交易日
    #   A 股：用权威日历（含节假日）
    #   港股/美股：**没有当地节假日日历**，只判周末 —— 这是已知边界，
    #   宁可漏判（放行）也不误判成「节假日不能交易」，因为误判会让用户
    #   在真正的交易日被拦住，且完全不知道原因。
    try:
        from . import calendar_cn as CAL
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "calendar_unavailable", "detail": str(e)[:120]}
    _d = _dt.date.fromisoformat(as_of)
    if mkt == "CN":
        is_td, basis = CAL.is_trading_day(_d)
        if not is_td:
            return {"ok": False, "reason": "not_trading_day", "as_of": as_of,
                    "market": mkt,
                    "hint": f"{as_of} 不是交易日（依据 {basis}），不能成交"}
    else:
        if _d.weekday() >= 5:
            return {"ok": False, "reason": "not_trading_day", "as_of": as_of,
                    "market": mkt,
                    "hint": f"{as_of} 是周末，{mkt} 不能成交"}

    # ② 不许是未来 / 不许是「今天还没收盘」/ 不许陈旧
    target = _dt.date.fromisoformat(as_of)
    if target > _dt.date.today():
        return {"ok": False, "reason": "as_of_in_future", "as_of": as_of}
    # 最近**已收盘**交易日
    try:
        from . import session as _S2
        _ph = _S2.in_window(now, mkt)
        _local = _S2._market_now(now, mkt)
    except Exception:  # noqa: BLE001
        _ph, _local = None, now
    if mkt == "CN":
        if _ph == "postclose" and CAL.is_trading_day_bool(_local.date()):
            _ref = _local.date()
        else:
            _ref = CAL.prev_trading_day(_local.date())
    else:
        # 港股/美股：只按周末退，不用 A 股日历（会把 A 股节假日当成休市）
        _ref = _local.date()
        if _ph not in ("postclose",) and _ph != "closed":
            _ref = _ref if _ph in ("morning", "afternoon", "lunch") else _ref
        while _ref.weekday() >= 5:
            _ref = _ref - _dt.timedelta(days=1)
    if target > _ref:
        return {"ok": False, "reason": "not_closed_yet", "as_of": as_of,
                "latest_closed": str(_ref), "market": mkt,
                "hint": f"{as_of} 还没收盘（{mkt} 最近已收盘交易日是 {_ref}），不能成交"}
    if mkt == "CN":
        # 注意必须把 now 传进去：旧写法用真实时钟，注入的 now 只影响 ③，
        # 于是 ② 和 ③ 会基于两个不同的「现在」，测试与实际行为不一致。
        lag = lag_trading_days(as_of, now)
        if lag is None:
            return {"ok": False, "reason": "stale_unknown", "as_of": as_of,
                    "hint": "算不出 as_of 落后几个交易日（日历不可用），拒绝成交"}
        if lag != 0:
            return {"ok": False, "reason": "stale_picks", "as_of": as_of, "lag": lag,
                    "hint": f"picks 落后 {lag} 个交易日，先刷新预测再调仓"}
    else:
        # 港美股：用「自然日差」近似（没有当地交易日历），超过 5 天算陈旧
        _gap = (target - _ref).days if target <= _ref else 0
        if (now.date() - target).days > 5:
            return {"ok": False, "reason": "stale_picks", "as_of": as_of,
                    "lag": (now.date() - target).days, "market": mkt,
                    "hint": f"picks 已 {as_of}，距今天超过 5 天，先刷新预测再调仓"}

    # ③ 取价方式：**常规盘中**用实时价成交；盘前/盘后/夜盘默认不成交。
    #
    # 为什么默认不认延长时段：那几段流动性薄、价差宽，
    # 用那些价成交会让这台测量仪的读数失真（不是不能下单，是测不准）。
    # 需要时显式 allow_extended=True。
    try:
        from . import session as _S
        seg = _S.segment(now, mkt)
    except Exception:  # noqa: BLE001
        seg = {"phase": "unknown", "tradable": False, "label": mkt,
               "local": now, "segments": []}
    phase = seg.get("phase")
    if seg.get("tradable") or (allow_extended and phase != "closed"):
        price_mode = "live"
        if phase in ("overnight", "premarket", "afterhours",
                     "auction_open", "auction_close"):
            price_mode = "extended"     # 延长时段：实时价但标注口径
    else:
        if dry_run:
            price_mode = "close"        # 试算不落盘，允许用收盘价演练
        else:
            _win = []
            for name, a, b, ok in (seg.get("segments") or []):
                if ok:
                    _win.append(f"{a[0]:02d}:{a[1]:02d}–{b[0]:02d}:{b[1]:02d}")
            _txt = " / ".join(_win) or "交易时段"
            _label = seg.get("label") or mkt
            return {"ok": False, "reason": "market_closed", "as_of": as_of,
                    "phase": phase, "market": mkt,
                    "now": now.isoformat(timespec="seconds"),
                    "hint": f"{_label}当前是「{phase}」，不在可成交时段。"
                            f"请在{_label}常规盘中 {_txt} 调仓"
                            f"（{_S.MARKET_SESSIONS.get(mkt, {}).get('tz', '本地时间')}）；"
                            f"延长时段流动性薄，默认不用于成交，"
                            f"确需成交请显式 allow_extended"}
    return {"ok": True, "as_of": as_of, "lag": 0, "phase": phase,
            "price_mode": price_mode, "market": mkt,
            "extended": price_mode == "extended",
            "backfilled": bool(allow_backfill)}


def top_k_info(horizon: int, market: str = "CN") -> Dict:
    """配置的 top_k 与**实际生效**的 top_k（评审 R28：min() 会静默压小）。"""
    h = int(horizon or 1)
    base = HORIZON_TOP_K.get(h, 5)
    cfg = 0
    try:
        from . import account_config as AC
        cfg = int((AC.load(market) or {}).get("top_k") or 0)
    except Exception:  # noqa: BLE001
        cfg = 0
    eff = min(cfg, base) if (cfg > 0 and cfg > base) else (cfg or base)
    return {"horizon": h, "market": str(market).upper(),
            "top_k_configured": cfg or None, "top_k_period_default": base,
            "top_k_effective": int(eff),
            "overridden": bool(cfg and eff != cfg),
            "note": ("配置被周期默认值压小：%s → %s" % (cfg, eff)) if (cfg and eff != cfg) else ""}


def default_top_k(horizon: int, market: str = "CN") -> int:
    """该周期的默认持仓数；配置里显式设了就用配置的。"""
    h = int(horizon or 1)
    base = HORIZON_TOP_K.get(h, 5)
    try:
        from . import account_config as AC
        cfg = AC.load(market)
        # 配置值明显大于周期默认值时，取小的（小资金/短周期优先）
        ck = int(cfg.get("top_k") or 0)
        if ck > 0:
            return min(ck, base) if ck > base else ck
    except Exception:
        pass
    return base


def _infer_market(code: str) -> str:
    """从代码形态反推市场。认不出就返回空串（不猜）。

    CN: 6 位纯数字；HK: 以 .HK 结尾或 5 位数字；US: 纯字母（可带 . / -）
    """
    c = str(code or "").strip().upper()
    if not c:
        return ""
    if c.endswith(".HK"):
        return "HK"
    if c.isdigit():
        return "CN" if len(c) == 6 else ("HK" if len(c) == 5 else "CN")
    if c.replace(".", "").replace("-", "").isalpha():
        return "US"
    return ""


def rebalance(picks: Optional[Dict] = None, market: str = "CN",
              top_k: Optional[int] = None, dry_run: bool = False,
              verbose: bool = True, horizon: int = 1,
              force: bool = False, allow_backfill: bool = False,
              allow_extended: bool = False) -> Dict:
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
    # 持仓数：显式传入 > 周期默认（短线集中、长线分散）
    k = int(top_k or default_top_k(h, market) or 5)

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
    # ---- 时间门禁（实测事故：2026-11-09 周一 13:10 用周五收盘价成交 20 笔并记成周五）----
    # ① as_of 是交易日 ② as_of 是「最近已收盘交易日」（陈旧信号不许成交）
    # ③ 执行时刻落在「as_of 收盘之后、下一交易日开盘之前」（盘中不许拿旧收盘价成交）
    _gate = _assert_tradable(as_of_hint=picks, dry_run=dry_run,
                             allow_backfill=allow_backfill, market=market,
                             allow_extended=allow_extended)
    if not _gate["ok"]:
        return {**_gate, "market": market, "horizon": h}

    _pm = str((picks or {}).get("market") or "").upper()
    if _pm and _pm != market:
        return {"ok": False, "reason": "market_mismatch",
                "horizon": h, "picks_market": _pm, "account_market": market,
                "hint": f"{fn if picks is None else 'picks'} 属于 {_pm} 市场，"
                        f"不能给 {market} 账户下单。"
                        f"该市场需要自己的预测（先训练该市场的模型）"}
    # picks 里没写 market 时，用代码形态反查 —— 必须**三个市场都覆盖**。
    # 旧实现只拦「数字代码给非 A 股账户」，反方向（美股 ticker 给港股账户）
    # 直接放行：AAPL 不是数字 → 不拦 → 跨市场串仓。
    if not _pm and rows:
        _c0 = str(rows[0].get("code") or "")
        _inferred = _infer_market(_c0)
        if _inferred and _inferred != market:
            return {"ok": False, "reason": "picks_market_mismatch",
                    "horizon": h, "picks_market": _inferred, "account_market": market,
                    "hint": (f"picks 里是 {_inferred} 代码（{_c0}），"
                             f"不能给 {market} 账户下单；该市场需要自己的预测")}
        # 混装（一行 A 股一行美股）也必须拒 —— 只看第一行会漏
        _mixed = {_infer_market(str(r.get("code") or "")) for r in rows} - {"", _inferred}
        if _mixed:
            return {"ok": False, "reason": "picks_mixed_markets",
                    "horizon": h, "markets": sorted(_mixed | {_inferred}),
                    "hint": "同一份 picks 里混了多个市场的代码，拒绝下单"}

    acc = load_account(market, h)
    if not acc:
        acc = init_account(float(AC.capital_for(market, h) or 5_000), market, h)
    if acc.get("market") != market:
        return {"ok": False, "reason": "market_mismatch",
                "account_market": acc.get("market"), "want": market,
                "hint": "模拟盘按市场独立，请用对应市场的账户"}
    # ---- 资金约束：每只至少要有 MIN_PER_STOCK 元，否则一手都买不起 ----
    # 注意：这段必须在 acc 赋值**之后**。之前放在前面引用了未定义的 acc，
    # UnboundLocalError 被 except 吞掉，约束从未生效 —— 实测 5000 元本金
    # 仍然按 k=5 分（每只 1000 元），账户照样空仓。
    MIN_PER_STOCK = 2500.0
    _equity = float(acc.get("equity") or acc.get("cash") or 0)
    if _equity > 0:
        _max_k = max(1, int(_equity // MIN_PER_STOCK))
        if k > _max_k:
            k = _max_k


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

    pos: Dict[str, Dict] = dict(acc.get("positions") or {})
    cash = float(acc.get("cash") or 0)
    score_of = {str(r.get("code", "")).zfill(6): float(r.get("score") or 0) for r in rows}

    # ---- 选目标：小资金必须优先选「买得起」的 ----
    # 直接取排名前 k 是错的：5000 元本金下，排名第 1 的票 131 元/股，
    # 一手就要 13100 元 —— 一只也买不了，账户永远空仓。
    # 做法：按排名往下扫，能买得起的才要；已持有的无条件保留（否则会被误卖）。
    def _affordable(code: str, budget: float) -> bool:
        try:
            px = _price(code)
            if not px or px <= 0:
                return False
            unit = float(_lot(code, market) or 1)
            return px * unit <= budget * 1.05      # 允许 5% 超出，买入时再按实际算
        except Exception:
            return False

    budget = cash / max(k, 1)
    want: List[str] = []
    # 1) 先放已持有的（保住，不因"买不起"被误判掉出 top-k）
    for code in pos:
        if code in score_of:
            want.append(code)
    # 2) 再按排名补，跳过买不起的
    for r in rows:
        if len(want) >= k:
            break
        code = str(r.get("code", "")).zfill(6)
        if code in want:
            continue
        if _affordable(code, budget):
            want.append(code)
    # 3) 一只都买不起：不要静默 0 成交，明确告诉用户为什么
    if not want:
        _cheapest = None
        for r in rows[:20]:
            _c = str(r.get("code", "")).zfill(6)
            _p = _price(_c)
            _u = _lot(_c, market)
            if _p and _u:
                _cost = _p * _u
                if _cheapest is None or _cost < _cheapest[1]:
                    _cheapest = (_c, _cost)
        hint = ""
        if _cheapest:
            hint = (f"当前本金 {cash:,.0f} 元，每只预算 {budget:,.0f} 元；"
                    f"候选里最便宜的一手也要 {_cheapest[1]:,.0f} 元（{_cheapest[0]}）。"
                    f"建议提高本金，或减少持仓数")
        return {"ok": False, "reason": "no_affordable_target", "horizon": h,
                "market": market, "cash": round(cash, 2),
                "per_stock_budget": round(budget, 2),
                "cheapest_lot_cost": round(_cheapest[1], 2) if _cheapest else None,
                "hint": hint or "本金太小，买不起任何候选标的的一手",
                "orders": [], "filled": [], "skipped": [], "account": acc}

    # fees 模块算每笔明细（佣金有最低收费、印花税仅卖出、各市场不同）。
    # 不用扁平 bps —— 买卖不对称，用平均值会系统性低估卖出成本。
    def _fee(side, amount, qty, price):
        try:
            return FEES.trade_fees(market, side, amount, qty=qty, price=price)
        except Exception:
            v = amount * cost_bps / 1e4 / 2.0
            return {'items': [], 'total': v, 'bps': cost_bps / 2,
                    'source': 'fallback'}

    # 盘中（含午休）：预取一次实时价，全部用同一个时刻的报价成交。
    # 必须放在 pos / want 都算好之后 —— 早一版写在前面，`pos` 还没赋值，
    # 直接 UnboundLocalError（陷阱 20：const/变量先用后定义）。
    _price_mode = str(_gate.get("price_mode") or "close")
    _live: Dict[str, Dict] = {}
    if _price_mode == "live":
        _need = list(pos.keys()) + list(want)
        _live = _live_quotes(_need)
        if not _live:
            return {"ok": False, "reason": "no_live_price", "market": market,
                    "horizon": h, "phase": _gate.get("phase"),
                    "hint": "盘中取不到实时价（行情源不可达），稍后重试。"
                            "注意：A股收盘后不能成交，不存在「盘后用收盘价调仓」"}

    orders, filled, skipped = [], [], []

    # ---- 1) 卖出：掉出 top-k 的（T+1：今天建的仓不卖）----
    for code, p in list(pos.items()):
        if code in want:
            continue
        # T+1 必须用**数据日**比数据日。旧代码拿 opened_at（墙钟）比 as_of（数据日），
        # 在回填场景下会失效 —— 实测出现过「11-09 建仓、被 as_of=11-07 的调仓卖出」。
        _open_day = str(p.get("open_trade_date") or str(p.get("opened_at", ""))[:10])
        if _open_day == as_of:
            skipped.append({"code": code, "why": "T+1，今日建仓不可卖"})
            continue
        # 盘中用实时价；dry_run 时用 as_of 收盘价（见 _px_for）。
        # 旧代码 `or _price(code)` 会退回「最新一根」—— 那是相对 as_of 的未来价。
        px, _basis, _px_asof = _px_for(code, as_of, _live, allow_backfill)
        if not px:
            skipped.append({"code": code,
                            "why": "取不到可成交价（停牌 / 盘中无实时报价）"})
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
            _append({"date": as_of, "signal_as_of": as_of,
                     "price_as_of": _px_asof or as_of, "price_basis": _basis,
                     "market": market, "horizon": h,
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
            px, _basis, _px_asof = _px_for(code, as_of, _live, allow_backfill)
            if not px:
                skipped.append({"code": code, "why": "取不到可成交价"})
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
                             "open_trade_date": as_of,
                             "score": score_of.get(code, 0),
                             "fee_paid": round(fee_amt, 4)}
                _append({"date": as_of, "signal_as_of": as_of,
                         "price_as_of": _px_asof or as_of, "price_basis": _basis,
                         "market": market, "horizon": h,
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
        px, _vbasis, _vasof = _px_for(code, as_of, _live, allow_backfill)
        if px:
            p["last_price"] = round(float(px), 4)
            p["last_price_source"] = _vbasis
            p["last_price_as_of"] = _vasof
        else:
            # 取不到就用成本价，但**必须留痕**：旧代码这里用最新价，
            # 导致持仓市值永远等于成本（20/20 last_price==cost），
            # 账户收益结构性地恒等于「只亏手续费」。
            p["last_price"] = round(float(p.get("cost") or 0), 4)
            p["last_price_source"] = "cost_fallback"
        p["market_value"] = round(float(p["last_price"]) * int(p.get("qty") or 0), 2)
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


def reset(market: str = "CN", horizon: int = 1,
          capital: Optional[float] = None,
          backup: bool = True) -> Dict:
    """
    清空某个 (市场, 周期) 的模拟盘，用新金额重开。

    ## 清什么

      · 该账户的持仓、现金、净值（重建为初始本金）
      · `trades.jsonl` 里属于该 (market, horizon) 的成交记录
      · **不动**其他市场/周期的数据

    ## 备份

    默认备份到 `data/paper/backup/<时间戳>_<market>_h<horizon>/`，
    含账户 JSON + 该账户的成交记录。清空是危险操作，万一清错了还能找回。

    ## 金额

    capital 为空时沿用原账户的 initial_capital；
    传了就用新值（这就是「重新按照现有的金额模拟」）。
    """
    m = str(market or "CN").upper()
    h = int(horizon or 1)

    old_acc = load_account(m, h)
    old_trades = load_trades(m, h)

    # 金额优先级（实测踩过两个相反方向的 bug，最终定这个顺序）：
    #   ① 显式传入的 capital
    #   ② **用户在各周期本金里显式设过的值**  ← 必须是它，否则「设置不生效」
    #   ③ 旧账户的 initial_capital          ← 保证「沿用原值」不会把 80 万变 5000
    #   ④ 配置兜底 capital_for
    # 曾经 ②③ 顺序颠倒：用户把 h=5 设成 20 万后点清空重投，本金变成旧账户净值 ——
    # 界面上明明设了，重投却不跟随。
    if capital is None:
        try:
            from . import account_config as AC
            capital = AC.capital_explicit(m, h)
        except Exception:  # noqa: BLE001
            capital = None
    if capital is None:
        capital = float((old_acc or {}).get("initial_capital") or 0) or None
    if capital is None:
        try:
            from . import account_config as AC
            capital = float(AC.capital_for(m, h) or 0) or 5_000.0
        except Exception:
            capital = 5_000.0
    capital = float(capital)
    if capital <= 0:
        return {"ok": False, "reason": "bad_capital", "capital": capital}

    # ---- 备份 ----
    backup_dir = ""
    if backup and (old_acc or old_trades):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(PAPER_DIR, "backup", f"{ts}_{m}_h{h}")
            os.makedirs(backup_dir, exist_ok=True)
            if old_acc:
                _atomic(os.path.join(backup_dir, "account.json"), old_acc)
            if old_trades:
                with open(os.path.join(backup_dir, "trades.jsonl"), "w",
                          encoding="utf-8") as f:
                    for t in old_trades:
                        f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
        except Exception:
            backup_dir = ""

    # ---- 清成交记录（只删本市场+本周期的）----
    removed = 0
    if os.path.isfile(TRADES):
        try:
            keep = []
            with open(TRADES, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tm = str(t.get("market") or "CN").upper()
                    th = int(t.get("horizon") or 1)
                    if tm == m and th == h:
                        removed += 1
                    else:
                        keep.append(line)
            tmp = TRADES + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for line in keep:
                    f.write(line + "\n")
            os.replace(tmp, TRADES)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "clear_trades_failed", "detail": str(e)[:120]}

    # ---- 重建账户 ----
    acc = init_account(capital, m, h)
    try:
        from . import account_config as AC
        AC.save({"horizon_capital": {str(h): capital}}, m)
    except Exception:
        pass

    return {"ok": True, "market": m, "horizon": h,
            "capital": round(capital, 2),
            "cleared_trades": removed,
            "backup_dir": backup_dir,
            "account": acc,
            "note": (f"已清空 {m} / h{h}，用 {capital:,.2f} 重新开始"
                     + (f"；原数据备份在 {backup_dir}" if backup_dir else ""))}


def reset_all(market: str = "CN", capital: Optional[float] = None,
              backup: bool = True) -> Dict:
    """清空某市场的所有周期账户。"""
    try:
        from . import horizons as HZ
        hs = [x for x in HZ.all_horizons() if x > 0]
    except Exception:
        hs = [1, 3, 5, 10, 15]
    out = []
    for h in hs:
        r = reset(market, h, capital=capital, backup=backup)
        out.append({"horizon": h, "ok": bool(r.get("ok")),
                    "cleared_trades": r.get("cleared_trades"),
                    "reason": r.get("reason")})
    return {"ok": True, "market": str(market).upper(), "capital": capital,
            "results": out,
            "total_cleared": sum(x.get("cleared_trades") or 0 for x in out)}


def summary(market: str = "CN", horizon: int = 1) -> Dict:
    """单个 (市场, 周期) 账户摘要。"""
    m = str(market or "CN").upper()
    h = int(horizon or 1)
    acc = load_account(m, h)
    _cfg_cap = None
    try:
        from . import account_config as AC
        _cfg_cap = float(AC.capital_for(m, h) or 0) or None
        _cfg_explicit = AC.capital_explicit(m, h)
    except Exception:  # noqa: BLE001
        _cfg_explicit = None
    if not acc:
        return {"ok": False, "reason": "not_initialized", "market": m,
                "horizon": h,
                "hint": f"先跑 python -m qcore.paper_auto --init "
                        f"--market={m} --horizon={h}"}
    # 老账户可能缺 equity / n_positions（早期版本没有这两个字段）。
    # 用现金 + 持仓成本兜底，避免 UI 显示「净值 —」看起来像没生效。
    if acc.get("equity") is None:
        cash = float(acc.get("cash") or 0)
        pos = acc.get("positions") or {}
        mv = 0.0
        for c, p in pos.items():
            try:
                mv += float(p.get("qty") or 0) * float(p.get("price") or p.get("cost") or 0)
            except (TypeError, ValueError):
                pass
        acc["equity"] = round(cash + mv, 2)
        acc.setdefault("n_positions", len(pos))
        init = float(acc.get("initial_capital") or 0)
        if init > 0:
            acc["total_return"] = round(acc["equity"] / init - 1.0, 6)
    trades = load_trades(m, h)
    # 账户实际本金与「各周期本金」配置不一致时必须显式说出来。
    # 实测：界面写「A股 80万」，五个周期账户却是 100万/100万/100万/100万/30万 ——
    # 用户以为设置生效了，实际每个账户都在用别的数跑。
    _acc_cap = float(acc.get("initial_capital") or 0)
    _mismatch = bool(_cfg_cap and _acc_cap
                     and abs(_acc_cap - float(_cfg_cap)) > 1.0)
    return {"ok": True, "market": m, "horizon": h, "account": acc,
            "capital_configured": _cfg_cap, "capital_explicit": _cfg_explicit,
            "account_capital": _acc_cap or None,
            "capital_mismatch": _mismatch,
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
                # 账户本金 vs 配置本金：不一致必须随行透出，否则用户以为设置生效了
                "account_capital": r.get("account_capital"),
                "capital_configured": r.get("capital_configured"),
                "capital_mismatch": bool(r.get("capital_mismatch")),
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
            # 账户本金 vs「各周期本金」配置：不一致必须随行透出，
            # 否则用户以为设了 80 万，五个周期其实各跑 100 万/30 万。
            "account_capital": r.get("account_capital"),
            "capital_configured": r.get("capital_configured"),
            "capital_mismatch": bool(r.get("capital_mismatch")),
        })
    # 排名：只算真正交易过的账户。
    # 没成交的账户收益率恒为 0，会排在亏损账户前面 —— 那是假象，
    # 用户会以为「T+3 最好」，其实它根本没建仓。
    ranked = sorted([x for x in rows
                     if x.get("ok") and x.get("total_return") is not None
                     and (x.get("n_trades") or 0) > 0],
                    key=lambda x: -float(x["total_return"]))
    best = ranked[0] if ranked else None
    return {"ok": True, "market": str(market).upper(), "rows": rows,
            "best_horizon": best.get("horizon") if best else None,
            "verdict": (
                f"目前最好的是 {best['label']}（累计 {best['total_return']*100:+.2f}%，"
                f"成交 {best['n_trades']} 笔，费用 {best['fee_total']} 元）"
                if best else
                ("各周期账户都还没成交 —— 先跑一次调仓" if any(
                    (x.get("ok") and (x.get("n_trades") or 0) == 0) for x in rows)
                 else "还没有任何周期账户产生数据"))}


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
        # 不再默认 100 万。实测：界面上「A股 80万」，五个周期账户却各自是
        # 100万/100万/100万/100万/30万 —— 既不是配置本金，彼此也不一致，
        # 用户看到的「各周期本金」设置等于没生效。
        _cap_arg = _arg("capital", "")
        cap = float(_cap_arg) if str(_cap_arg).strip() else None
        if "--all-horizons" in sys.argv:
            try:
                from . import horizons as HZ
                hs = [h for h in HZ.all_horizons() if h > 0]
            except Exception:
                hs = [1, 3, 5, 10, 15]
            try:
                from . import account_config as _AC
            except Exception:  # noqa: BLE001
                _AC = None
            out = []
            for h in hs:
                _c = cap
                if _c is None:
                    try:
                        _c = float(_AC.capital_for(market, h) or 0) or 5_000.0
                    except Exception:  # noqa: BLE001
                        _c = 5_000.0
                out.append(init_account(_c, market, h))
            print(json.dumps({"ok": True, "market": market,
                              "initialized": [o["horizon"] for o in out],
                              "capital_each": {o["horizon"]: o["initial_capital"]
                                               for o in out}},
                             ensure_ascii=False, indent=1))
        else:
            if cap is None:
                try:
                    from . import account_config as _AC2
                    cap = float(_AC2.capital_for(market, horizon) or 0) or 5_000.0
                except Exception:  # noqa: BLE001
                    cap = 5_000.0
            print(json.dumps(init_account(cap, market, horizon),
                             ensure_ascii=False, indent=1))
    elif "--all" in sys.argv:
        print(json.dumps(rebalance_all(market, dry_run=("--dry" in sys.argv)),
                         ensure_ascii=False, indent=1)[:2000])
    elif "--dry" in sys.argv:
        print(json.dumps(rebalance(market=market, horizon=horizon, dry_run=True),
                         ensure_ascii=False, indent=1)[:1500])
    elif "--reset" in sys.argv:
        cap = _arg("capital", "")
        capv = float(cap) if cap else None
        if "--all-horizons" in sys.argv:
            print(json.dumps(reset_all(market, capital=capv),
                             ensure_ascii=False, indent=1))
        else:
            print(json.dumps(reset(market, horizon, capital=capv),
                             ensure_ascii=False, indent=1))
    elif "--matrix" in sys.argv:
        print(json.dumps(summary_all(), ensure_ascii=False, indent=1)[:2500])
    elif "--by-horizon" in sys.argv:
        print(json.dumps(by_horizon(market), ensure_ascii=False, indent=1)[:2000])
    elif "--summary" in sys.argv:
        print(json.dumps(summary(market, horizon), ensure_ascii=False, indent=1)[:1200])
    else:
        print(json.dumps(rebalance(market=market, horizon=horizon),
                         ensure_ascii=False, indent=1)[:1800])
