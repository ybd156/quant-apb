"""
个股行情数据 —— 给「个股」页提供 K 线 + 均线 + 量能 + 区间统计。

（注意：`qcore/chart.py` 是 Pillow 图片渲染器，两者不是一回事。
 本模块只出**数据**，画图交给前端 SVG。）

## 为什么单独一个模块

AGENTS.md INV-8：`webui/server.py` 只做 IO，不计算指标。
所以均线、涨跌幅、区间高低必须在这里算完，server 只负责序列化。

## 复权口径（这里最容易出错）

| adjust | 含义 | 用途 |
|---|---|---|
| 0 | 不复权（真实价） | **和券商软件一致**，用户看着不会困惑 |
| 1 | 前复权（减法） | 历史连续，但可能产生负价（2013 年前的票） |
| 2 | 后复权（乘法） | 算因子用，价格是现价的几十倍，**不能给人看** |

默认 `adjust=0`：用户输入 600519 想看的是 1442 元，不是后复权的 4 万。
同时返回 `adjust_note` 说明口径，避免误读。

## 周期

daily / weekly / monthly / quarterly / halfyear / yearly
周线月线由东财直接返回（klt=102/103…），**不是自己聚合** ——
自己聚合会和交易所的周切分不一致（节假日周、半周）。

## 均线

MA5 / MA10 / MA20 / MA60 / MA120，按收盘价滚动均值。
**不足窗口的位置给 null**，不填充、不前视。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

PERIOD_KLT = {"daily": 101, "weekly": 102, "monthly": 103,
              "quarterly": 104, "halfyear": 105, "yearly": 106}
PERIOD_LABEL = {"daily": "日K", "weekly": "周K", "monthly": "月K",
                "quarterly": "季K", "halfyear": "半年K", "yearly": "年K"}
MA_WINDOWS = (5, 10, 20, 60, 120)


def _f(x) -> Optional[float]:
    """转 float，NaN/inf -> None（JSON 里必须是 null，不能是 NaN）。"""
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _ma(closes: np.ndarray, w: int) -> List[Optional[float]]:
    """滚动均值。不足窗口给 null，不填充、不前视。"""
    n = len(closes)
    out: List[Optional[float]] = [None] * n
    if n < w or w <= 0:
        return out
    c = np.asarray(closes, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(c, nan=0.0))])
    cnt = np.concatenate([[0], np.cumsum(np.isfinite(c).astype(float))])
    for i in range(w - 1, n):
        s = cs[i + 1] - cs[i + 1 - w]
        k = cnt[i + 1] - cnt[i + 1 - w]
        if k >= w * 0.8:                      # 允许 20% 缺失（停牌）
            out[i] = round(float(s / k), 4)
    return out


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return round((a / b - 1.0) * 100.0, 3)


def _resample(df, period: str):
    """
    把日线聚合成周/月/季/半年/年线。
    A 股用东财原生周期（和交易所切分一致）；港美股只有日线，只能本地聚合。
    本地聚合按自然周/月切分，节假周可能和交易所略有差异 —— 会标注出来。
    """
    if period == "daily":
        return df
    import pandas as pd
    rule = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE",
            "halfyear": "6ME", "yearly": "YE"}.get(period)
    if rule is None:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).set_index("date").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    if "amount" in d.columns:
        agg["amount"] = "sum"
    out = d.resample(rule).agg(agg).dropna(subset=["open", "close"]).reset_index()
    return out


def build(code: str, period: str = "daily", count: int = 180,
          adjust: int = 0) -> Dict:
    """返回一只股票的图表数据。取不到就明确失败，不编造。"""
    period = str(period).lower()
    if period not in PERIOD_KLT:
        return {"ok": False, "reason": "bad_period",
                "supported": list(PERIOD_KLT.keys())}
    count = max(30, min(int(count or 180), 1000))
    code = str(code).strip()

    df = None
    fetch_err = None
    # A 股走东财（支持日/周/月/季/半年/年）
    # 港股/美股走 us_data（只有日线，周期由本地聚合）
    _c = str(code).upper()
    _is_hk = _c.endswith(".HK")
    _is_us = (not _is_hk) and (not str(code).isdigit()) and _c.isalpha()
    if _is_hk or _is_us:
        try:
            from . import us_data as U
            # adjust 必须透传：us_data 默认 ADJUST=2（后复权），
            # 直接调会让 AAPL 显示 79066 而不是 270 —— 用户会以为数据错了。
            _adj = int(adjust)
            raw = (U.load_hk_kline(code, adjust=_adj) if _is_hk
                   else U.load_us_kline(code, adjust=_adj))
            if raw is not None and len(raw):
                df = _resample(raw, period)
                if df is None:
                    return {"ok": False, "reason": "unsupported_period_for_market",
                            "detail": "港美股只有日线，周期由本地聚合；"
                                      "请用 daily/weekly/monthly"}
        except Exception as e:  # noqa: BLE001
            fetch_err = f"{type(e).__name__}: {str(e)[:100]}"
    else:
        try:
            from . import data as D
            df = D.fetch_kline(code, adjust=int(adjust), klt=PERIOD_KLT[period],
                               beg="19900101")
        except Exception as e:  # noqa: BLE001
            fetch_err = f"{type(e).__name__}: {str(e)[:100]}"
    if df is None:
        return {"ok": False, "reason": "fetch_failed", "detail": fetch_err or ""}

    if df is None or len(df) == 0:
        return {"ok": False, "reason": "no_data", "code": code}

    df = df.tail(count).reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=float)
    if len(closes) and np.nanmin(closes) <= 0:
        if int(adjust) != 0:            # 前复权产生负价 -> 退回不复权
            return build(code, period, count, 0)
        return {"ok": False, "reason": "nonpositive_price",
                "detail": "该股历史价格出现非正值，数据源异常"}

    bars: List[Dict] = []
    for r in df.itertuples(index=False):
        bars.append({"d": str(getattr(r, "date"))[:10],
                     "o": _f(getattr(r, "open", None)),
                     "h": _f(getattr(r, "high", None)),
                     "l": _f(getattr(r, "low", None)),
                     "c": _f(getattr(r, "close", None)),
                     "v": _f(getattr(r, "volume", None)),
                     "a": _f(getattr(r, "amount", None))})

    ma = {f"ma{w}": _ma(closes, w) for w in MA_WINDOWS}

    # ---- 实时报价（拿不到就用最后一根 bar）----
    quote: Dict = {}
    try:
        from . import tick
        snap = tick.snapshot([code])
        row = (snap.get("rows") or [{}])[0] if snap.get("ok") else {}
        if row and row.get("price"):
            quote = {"price": _f(row.get("price")),
                     "chg_pct": _f(row.get("chg_pct")),
                     "open": _f(row.get("open")),
                     "high": _f(row.get("high")),
                     "low": _f(row.get("low")),
                     "prev_close": _f(row.get("prev_close")),
                     "volume": _f(row.get("volume_lot")),
                     "amount": _f(row.get("amount")),
                     "as_of": snap.get("as_of"), "stale": snap.get("stale"),
                     "source": snap.get("source")}
    except Exception:
        quote = {}

    last = bars[-1] if bars else {}
    if not quote.get("price"):
        quote = {"price": last.get("c"), "chg_pct": None,
                 "open": last.get("o"), "high": last.get("h"),
                 "low": last.get("l"),
                 "prev_close": bars[-2]["c"] if len(bars) > 1 else None,
                 "volume": last.get("v"), "amount": last.get("a"),
                 "as_of": last.get("d"), "stale": True, "source": "kline"}

    n = len(closes)
    stats = {
        "n_bars": n, "first": bars[0]["d"] if bars else None,
        "last": bars[-1]["d"] if bars else None,
        "high_all": _f(np.nanmax(df["high"])) if n else None,
        "low_all": _f(np.nanmin(df["low"])) if n else None,
        "chg_5": _pct(_f(closes[-1]), _f(closes[-6])) if n >= 6 else None,
        "chg_20": _pct(_f(closes[-1]), _f(closes[-21])) if n >= 21 else None,
        "chg_60": _pct(_f(closes[-1]), _f(closes[-61])) if n >= 61 else None,
    }
    if "volume" in df:
        v = df["volume"].to_numpy(dtype=float)
        stats["vol_ma5"] = _f(np.nanmean(v[-5:])) if n >= 5 else None
        stats["vol_ma20"] = _f(np.nanmean(v[-20:])) if n >= 20 else None
        if stats.get("vol_ma5") and stats.get("vol_ma20"):
            stats["vol_ratio"] = round(stats["vol_ma5"] / stats["vol_ma20"], 3)

    name = None
    try:
        from . import names as NAMES
        m = NAMES.load() or {}
        rec = m.get(str(code)) or m.get(str(code).zfill(6)) or {}
        name = rec.get("name") if isinstance(rec, dict) else (str(rec) if rec else None)
    except Exception:
        pass

    adjust_note = {0: "不复权（真实成交价，与券商软件一致）",
                   1: "前复权（历史连续，仅展示用）",
                   2: "后复权（算因子用，价格远高于现价，不要当现价看）"}
    return {"ok": True, "code": code, "name": name,
            "period": period, "period_label": PERIOD_LABEL[period],
            "adjust": int(adjust), "adjust_note": adjust_note.get(int(adjust), ""),
            "count": len(bars), "quote": quote, "bars": bars, "ma": ma,
            "ma_windows": list(MA_WINDOWS), "stats": stats,
            "as_of": datetime.now().astimezone().isoformat(timespec="seconds")}


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    for p in ("daily", "weekly", "monthly"):
        r = build(code, p, count=60)
        if not r.get("ok"):
            print(p, "FAIL", r.get("reason"))
            continue
        q = r["quote"]
        print(f"{p:<8} {r['name'] or code:<8} {r['count']:>3} 根 | "
              f"现价 {q.get('price')} ({q.get('chg_pct')}%) | "
              f"MA5={r['ma']['ma5'][-1]} MA20={r['ma']['ma20'][-1]} | "
              f"5日 {r['stats']['chg_5']}% 20日 {r['stats']['chg_20']}%")
