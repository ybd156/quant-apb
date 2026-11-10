"""
三段式交易时段 —— 盘前预测 / 盘中监控 / 收盘复盘。

## 三个时段各自能做什么（诚实版）

| 时段 | 可用信息 | 能做什么 | 能不能赚钱 |
|---|---|---|---|
| **盘前** 09:00-09:25 | T−1 收盘 + 隔夜美股（凌晨已收） | **打分，生成今日建议** | ✅ 这是模型训练时的口径，最匹配 |
| **盘中** 09:30-15:00 | 3 秒快照 + 5 分钟线 | 偏离监控、风控触发、执行 | ❌ 已证伪：毛命中 33.5%，净 -19.39bps/笔 |
| **收盘后** 15:10- | T 日完整日线 | 复盘对账 + 生成 T+1 建议 | ✅ 唯一能验证预测准不准的时点 |

## 为什么盘前才是真正的预测时点

Alpha158 的标签是 `close[T+1]/close[T]−1`（T 日出信号，T+1 成交）。
所以模型**本来就该在 T 日开盘前用 T−1 收盘数据打分**——
盘中或收盘后再打分，用的是模型没见过的信息组合，属于误用。

## 盘中为什么不做预测

不是技术做不到，是**赚不到钱**。实测（`qcore/intraday.py`）：
240 根/日、毛命中 33.5%、净 -19.39 bps/笔、年化成本拖累 10,929%。
盘中真正有价值的是：**执行质量**和**风控**，不是择时。

## 文件契约

- `data/state/premarket.json`   盘前建议（for_date = 今日）
- `data/state/intraday.json`    盘中偏离（实时刷新）
- `data/state/postclose.json`   收盘复盘（含昨日建议的实际表现）
- `data/state/picks.json`       最新推荐（收盘后生成，供 UI 模型页）
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
os.makedirs(STATE, exist_ok=True)


def _now() -> datetime:
    # 必须 aware：naive 的「现在」在 UTC 机器与东八区机器上差 8 小时，
    # 而时段判断完全依赖它（实测 CI 跑在 UTC，本地过、CI 红）。
    return datetime.now().astimezone()


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _now()).astimezone().isoformat(timespec="seconds")


def _write(name: str, payload: Dict) -> str:
    p = os.path.join(STATE, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)
    return p


def _read(name: str) -> Dict:
    try:
        with open(os.path.join(STATE, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _names() -> Dict[str, str]:
    try:
        from . import names as N
        out = {}
        for k, v in (N.load() or {}).items():
            bare = str(k)
            if bare[:2] in ("sh", "sz", "bj"):
                bare = bare[2:]
            out[bare.zfill(6)] = v.get("name") if isinstance(v, dict) else str(v)
        return out
    except Exception:
        return {}


def _latest_trade_date() -> Optional[pd.Timestamp]:
    """数据里最新的交易日（不是今天，是行情最后一天）。"""
    try:
        from . import qlib_data as QD
        cal = QD.load_calendar()
        return pd.Timestamp(cal[-1])
    except Exception:
        return None


# ------------------------------------------------------------------ 盘前
def premarket(top_k: int = 20, verbose: bool = True) -> Dict:
    """
    盘前打分：用**数据里最新的交易日**（通常是 T−1）的完整特征，
    产出 for_date = 下一个交易日的建议。

    不猜未来：特征只用到 T−1 收盘，和训练口径一致。
    """
    from . import live_model as LM

    model, meta = LM.load_model()
    if model is None:
        return {"ok": False, "reason": "model_not_trained"}

    data_date = _latest_trade_date()
    if data_date is None:
        return {"ok": False, "reason": "no_data"}

    # 复用 live_model.predict 的打分逻辑，但写独立文件
    r = LM.predict(top_k=top_k, verbose=verbose)
    if not r.get("ok"):
        return r

    nm = _names()
    rows = []
    for it in r.get("rows", []):
        code = str(it.get("code", "")).zfill(6)
        rec = {**it, "name": it.get("name") or nm.get(code, "")}
        # 补参考价：盘中偏离监控要靠它算「相对盘前涨了多少」
        try:
            from . import qlib_data as QD
            q = ("sh" + code) if code[0] in "69" else ("bj" + code if code[:2] in ("43", "83", "87", "92") else "sz" + code)
            df = QD.load_stock(q, start=(_now() - timedelta(days=30)).strftime("%Y-%m-%d"))
            if df is not None and len(df):
                rec["ref_close"] = float(df["close"].iloc[-1])
                rec["ref_date"] = str(pd.Timestamp(df["date"].iloc[-1]).date())
        except Exception:
            pass
        rows.append(rec)

    # for_date 必须是**下一个交易日**，不能简单 +1 天（会落到周末）
    nxt = None
    try:
        from . import qlib_data as QD
        cal = QD.load_calendar()
        later = cal[cal > data_date]
        if len(later):
            nxt = pd.Timestamp(later[0])
    except Exception:
        pass
    if nxt is None:
        # 日历里没有未来日期（数据只到最新交易日）→ 手算下一个工作日
        nxt = data_date + pd.Timedelta(days=1)
        while nxt.weekday() >= 5:          # 5=周六 6=周日
            nxt += pd.Timedelta(days=1)
    payload = {
        "session": "premarket",
        "data_date": str(data_date.date()),
        "for_date": str(nxt.date()),
        "generated_at": _iso(),
        "n": len(rows), "top_k": top_k,
        "model_age_days": r.get("model_age_days"),
        "rows": rows,
        "note": "盘前建议：特征截至 data_date 收盘，与训练口径一致（T 日信号 → T+1 成交）",
    }
    _write("premarket.json", payload)
    if verbose:
        print(f"[premarket] {payload['for_date']} 建议 {len(rows)} 只 "
              f"（数据截至 {payload['data_date']}）", flush=True)
    return {"ok": True, **payload}


# ------------------------------------------------------------------ 盘中
def intraday(verbose: bool = True) -> Dict:
    """
    盘中偏离监控：对比盘前建议的参考价与当前实时价。
    **不做预测**——只回答「现在已经偏离预测多少」，供执行和风控用。
    """
    pm = _read("premarket.json")
    rows = pm.get("rows") or []
    if not rows:
        return {"ok": False, "reason": "no_premarket",
                "hint": "先跑盘前预测（qcore.session premarket）"}

    codes = [str(r.get("code", "")).zfill(6) for r in rows]
    try:
        from . import tick
        snap = tick.snapshot(codes)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"tick_failed: {type(e).__name__}"}
    if not snap.get("ok"):
        return {"ok": False, "reason": "tick_unavailable"}

    live = {}
    for r in snap.get("rows", []):
        c = str(r.get("code", ""))
        live[c[2:] if c[:2] in ("sh", "sz", "bj") else c] = r

    out = []
    for r in rows:
        c = str(r.get("code", "")).zfill(6)
        t = live.get(c) or {}
        px = t.get("price")
        # 参考价必须用**真实价**口径：Qlib 的 close 是归一化价（首日=1.0），
        # 与 tick 的真实价直接相除会得到几千 % 的假偏离。
        # tick 自带 prev_close（真实昨收），这才是「相对盘前涨了多少」的正确基准。
        ref = t.get("prev_close")
        dev = None
        if px and ref and ref > 0:
            dev = (px / ref - 1.0) * 100.0
        out.append({"code": c, "name": r.get("name"), "score": r.get("score"),
                    "ref_prev_close": ref, "now": px,
                    "chg_pct": t.get("chg_pct"), "dev_pct": dev,
                    "weight": r.get("weight")})

    devs = [x["dev_pct"] for x in out if x["dev_pct"] is not None]
    payload = {
        "session": "intraday",
        "as_of": _iso(),
        "tick_as_of": snap.get("as_of"),
        "stale": snap.get("stale"),
        "n": len(out),
        "n_live": len(devs),
        "mean_dev_pct": round(float(np.mean(devs)), 4) if devs else None,
        "worst_dev": sorted([x for x in out if x["dev_pct"] is not None],
                            key=lambda x: x["dev_pct"])[:3] if devs else [],
        "rows": out,
        "note": "盘中只做偏离监控（相对昨收的当日涨跌），不做择时（盘中预测已证伪：净 -19.39bps/笔）",
    }
    _write("intraday.json", payload)
    if verbose:
        print(f"[intraday] {payload['n_live']}/{payload['n']} 只算得出偏离，"
              f"平均 {payload['mean_dev_pct']}%", flush=True)
    return {"ok": True, **payload}


# ------------------------------------------------------------------ 收盘
def postclose(verbose: bool = True) -> Dict:
    """
    收盘复盘：
      1. 用**今日**完整数据重算分数
      2. 对账：盘前建议的股票今天实际涨跌多少、命中没有
      3. 生成明日建议（写 picks.json，供 UI）
    """
    from . import live_model as LM

    data_date = _latest_trade_date()
    pm = _read("premarket.json")
    pm_rows = pm.get("rows") or []
    pm_data_date = pm.get("data_date")

    r = LM.predict(top_k=20, verbose=verbose)
    if not r.get("ok"):
        return r
    nm = _names()

    # 对账：盘前建议的票，**下一个交易日**的实际表现
    #
    # ⚠️ 因果红线：盘前建议是用 data_date 收盘数据算出来的，
    # 只能拿 data_date 之后的数据来评价。若行情最新日 == 建议的数据日，
    # 说明「下一交易日还没发生」，此时复盘任何数字都是前视。
    review = []
    _pm_date = pd.Timestamp(pm_data_date) if pm_data_date else None
    _can_review = (_pm_date is not None and data_date is not None
                   and data_date > _pm_date)
    if pm_rows and not _can_review:
        review = [{"pending": True,
                   "reason": f"盘前建议基于 {pm_data_date}，行情最新仅到 "
                             f"{data_date.date() if data_date is not None else '?'}，"
                             f"下一交易日尚未发生，暂不复盘"}]
    elif pm_rows:
        codes = [str(x.get("code", "")).zfill(6) for x in pm_rows]
        try:
            from . import qlib_data as QD
            for x in pm_rows:
                c = str(x.get("code", "")).zfill(6)
                df = QD.load_stock("sh" + c if c[0] in "69" else "sz" + c,
                                   start=(_now() - timedelta(days=30)).strftime("%Y-%m-%d"))
                if df is None or len(df) < 2:
                    continue
                ref = x.get("close")
                last = float(df["close"].iloc[-1])
                # 归一化价与真实价口径不同，用同源比值
                prev = float(df["close"].iloc[-2])
                ret = last / prev - 1.0
                review.append({"code": c, "name": x.get("name"),
                               "pred_score": x.get("score"),
                               "ret_pct": round(ret * 100, 3),
                               "hit": bool(ret > 0)})
        except Exception as e:  # noqa: BLE001
            review = [{"error": f"{type(e).__name__}: {e}"[:80]}]

    hits = [x["hit"] for x in review if "hit" in x]
    rets = [x["ret_pct"] for x in review if "ret_pct" in x]
    payload = {
        "session": "postclose",
        "data_date": str(data_date.date()) if data_date is not None else None,
        "premarket_data_date": pm_data_date,
        "generated_at": _iso(),
        "review": {
            "n": len(review),
            "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
            "mean_ret_pct": round(float(np.mean(rets)), 3) if rets else None,
            "rows": review,
        },
        "next_picks": {"n": r.get("n"), "as_of": r.get("as_of"),
                       "rows": [{**it, "name": it.get("name") or nm.get(str(it.get("code", "")).zfill(6), "")}
                                for it in r.get("rows", [])]},
        "note": "收盘复盘：盘前建议 vs 今日实际；同时生成明日建议",
    }
    _write("postclose.json", payload)
    if verbose:
        rv = payload["review"]
        print(f"[postclose] 复盘 {rv['n']} 只 命中率 {rv['hit_rate']} "
              f"平均收益 {rv['mean_ret_pct']}%", flush=True)
    return {"ok": True, **payload}


# ------------------------------------------------------------------ 调度
#: 各市场的时段表（**各自本地时间**）。
#:
#: 每个市场都是多段的，美股还有一段**跨午夜**的夜盘（20:00–04:00 ET），
#: 所以窗口用「分钟数区间」表示，start > end 表示跨午夜。
#:
#: `tradable` 决定这一段算不算「可以成交」：
#:   · 只有**常规盘中**默认 tradable=True
#:   · 盘前/盘后/夜盘 tradable=False —— 不是不能下单，而是流动性薄、
#:     价差宽，用那些价成交会污染这台测量仪的读数。
#:     真要在这几段成交，显式传 `allow_extended=True`。
MARKET_SESSIONS: Dict[str, Dict] = {
    "CN": {
        "label": "A股", "tz": "Asia/Shanghai", "has_calendar": True,
        # (名称, 起, 止, 是否可成交)  起止都是 (时, 分)，左闭右开
        # **顺序有讲究**：segment() 倒序检查，越具体的段放越后面。
        "segments": [
            ("premarket",    (8, 30), (9, 30), False),
            ("auction_open", (9, 15), (9, 25), False),
            ("morning",      (9, 30), (11, 30), True),
            ("lunch",        (11, 30), (13, 0), False),
            ("afternoon",    (13, 0), (15, 0), True),
            ("auction_close", (14, 57), (15, 0), False),
            ("postclose",    (15, 5), (17, 0), False),
        ],
    },
    "HK": {
        "label": "港股", "tz": "Asia/Hong_Kong", "has_calendar": False,
        "segments": [
            ("auction_open", (9, 0), (9, 30), False),
            ("morning",      (9, 30), (12, 0), True),
            ("lunch",        (12, 0), (13, 0), False),
            ("afternoon",    (13, 0), (16, 0), True),
            ("auction_close", (16, 0), (16, 10), False),
        ],
    },
    "US": {
        "label": "美股", "tz": "America/New_York", "has_calendar": False,
        "segments": [
            # 夜盘跨午夜：20:00 → 次日 04:00
            ("overnight",  (20, 0), (4, 0), False),
            ("premarket",  (4, 0), (9, 30), False),
            ("regular",    (9, 30), (16, 0), True),
            ("afterhours", (16, 0), (20, 0), False),
        ],
    },
}


def _market_now(now: Optional[datetime], market: str) -> datetime:
    """把「现在」换算到该市场的本地时间。

    用 `zoneinfo`（标准库，3.9+）：美东有夏令时，写死 UTC-5 会在夏天错一小时。
    naive 时间按**本机时区**解释。
    """
    n = now or _now()
    cfg = MARKET_SESSIONS.get(str(market).upper()) or MARKET_SESSIONS["CN"]
    tzname = cfg.get("tz")
    if not tzname:
        return n
    try:
        from zoneinfo import ZoneInfo
        if n.tzinfo is None:
            # naive 按**本机时区**解释。注意这意味着「同一串 naive 时间
            # 在 UTC 机器与东八区机器上代表不同时刻」——
            # 调用方（尤其测试）应当传 tz-aware 的 datetime。
            n = n.astimezone()
        return n.astimezone(ZoneInfo(tzname))
    except Exception:  # noqa: BLE001
        return n


def _mins(hm) -> int:
    return int(hm[0]) * 60 + int(hm[1])


def _seg_hit(m: int, seg) -> bool:
    """分钟数 m 是否落在**左闭右开** [起, 止) 内。起 > 止 表示跨午夜。

    必须左闭右开：美股「盘后 16:00–20:00」与「夜盘 20:00–04:00」在 20:00 处
    首尾相接，两端都闭会让 20:00 同时命中两段（实测：20:00 被判成盘后，
    夜盘整段被吞掉）。
    """
    a, b = _mins(seg[0]), _mins(seg[1])
    if a == b:
        return False
    if a < b:
        return a <= m < b
    return m >= a or m < b           # 跨午夜


def segment(now: Optional[datetime] = None, market: str = "CN") -> Dict:
    """返回当前所处**段**的完整信息（含是否可成交）。

    {"phase", "tradable", "label", "market", "local", "segments": [...]}
    """
    m = str(market or "CN").upper()
    cfg = MARKET_SESSIONS.get(m) or MARKET_SESSIONS["CN"]
    n = _market_now(now, m)

    if cfg.get("has_calendar"):
        try:
            from . import calendar_cn as CAL
            if not CAL.is_trading_day_bool(n.date()):
                return {"phase": "closed", "tradable": False, "market": m,
                        "label": cfg.get("label", m), "local": n, "segments": []}
        except Exception:  # noqa: BLE001
            if n.weekday() >= 5:
                return {"phase": "closed", "tradable": False, "market": m,
                        "label": cfg.get("label", m), "local": n, "segments": []}
    else:
        if n.weekday() >= 5:
            return {"phase": "closed", "tradable": False, "market": m,
                    "label": cfg.get("label", m), "local": n, "segments": []}

    cur = _mins((n.hour, n.minute))
    # 倒序检查：后面的段更具体（如 A 股 14:57 的收盘竞价覆盖 afternoon）
    hit = None
    for name, a, b, ok in reversed(cfg.get("segments") or []):
        if _seg_hit(cur, (a, b)):
            hit = (name, ok)
            break
    if hit is None:
        return {"phase": "closed", "tradable": False, "market": m,
                "label": cfg.get("label", m), "local": n,
                "segments": cfg.get("segments") or []}
    return {"phase": hit[0], "tradable": bool(hit[1]), "market": m,
            "label": cfg.get("label", m), "local": n,
            "segments": cfg.get("segments") or []}


#: 旧调用方兼容：把新段名映射回老的粗分类
_PHASE_ALIAS = {
    "morning": "morning", "afternoon": "afternoon",
    "regular": "morning", "lunch": "lunch",
    "premarket": "premarket", "auction_open": "premarket",
    "afterhours": "postclose", "overnight": "overnight",
    "auction_close": "postclose",
    "postclose": "postclose",        # ← 漏了这条会让 A 股盘后全被判成 closed
    "closed": "closed",
}


def in_window(now: Optional[datetime] = None, market: str = "CN") -> str:
    """当前时段（粗分类，向后兼容）：premarket|morning|lunch|afternoon|
    postclose|overnight|closed

    需要「这一段能不能成交」用 `segment()`，它给的是原始段名 + tradable。
    """
    return _PHASE_ALIAS.get(segment(now, market)["phase"], "closed")


def in_session(now: Optional[datetime] = None, market: str = "CN",
               allow_extended: bool = False) -> bool:
    """是否**可以成交**。

    默认只认常规盘中（美股 regular / A 股 morning+afternoon / 港股同）。
    盘前盘后夜盘流动性薄、价差宽，默认不算 —— 用那些价成交会污染读数。
    需要时显式 `allow_extended=True`。
    """
    seg = segment(now, market)
    if seg["phase"] == "closed":
        return False
    return bool(seg["tradable"]) or bool(allow_extended)


def run(session: str = "auto", verbose: bool = True) -> Dict:
    """跑指定时段。session=auto 时按当前时间判断。"""
    s = session
    if s == "auto":
        s = in_window()
    if s == "premarket":
        return premarket(verbose=verbose)
    if s in ("intraday", "morning", "afternoon"):
        return intraday(verbose=verbose)
    if s == "postclose":
        return postclose(verbose=verbose)
    return {"ok": False, "reason": "outside_session", "session": s,
            "hint": "盘前 08:30-09:30 / 盘中 09:30-15:00 / 收盘 15:05-17:00"}


if __name__ == "__main__":
    import sys
    mode = "auto"
    for a in sys.argv[1:]:
        if a in ("premarket", "intraday", "postclose"):
            mode = a
    r = run(mode)
    print(json.dumps({k: v for k, v in r.items() if k not in ("rows", "review", "next_picks")},
                     ensure_ascii=False, indent=1)[:900])
