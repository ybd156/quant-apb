"""
A 股交易日历 —— 判断今天到底是不是交易日。

## 为什么不能只看周末

`session.in_window()` 原来只判断 `weekday() >= 5`。但 A 股有春节、国庆这类
**落在工作日的长假**，那几天系统会照常"盘前预测"，用陈旧数据生成假建议。

## 数据源优先级（重要）

1. **Qlib 社区数据的交易日历**（`calendars/day.txt`）—— 权威、真实、已含节假日
2. 硬编码节假日表 —— 仅用于日历覆盖不到的未来日期
3. 工作日判断 —— 最后的兜底

**永远优先用 1**。硬编码的节假日表会过期（2026 年后的安排国务院每年才公布），
所以它只作为"日历不够用"时的近似，并在返回里标注 `source`。

## 用法

```python
from qcore.calendar_cn import is_trading_day, next_trading_day, prev_trading_day
is_trading_day()              # 今天
is_trading_day("2026-10-01")  # False（国庆）
next_trading_day()            # 下一个交易日
```
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 硬编码节假日（仅作日历覆盖不到时的近似）。
# 2026 及以后为**估算**，国务院每年年底才公布次年安排，届时需更新。
HOLIDAYS: Dict[int, Dict[str, Tuple[str, str]]] = {
    2024: {"元旦": ("01-01", "01-01"), "春节": ("02-09", "02-17"),
           "清明": ("04-04", "04-06"), "劳动": ("05-01", "05-05"),
           "端午": ("06-08", "06-10"), "中秋": ("09-15", "09-17"),
           "国庆": ("10-01", "10-07")},
    2025: {"元旦": ("01-01", "01-01"), "春节": ("01-28", "02-04"),
           "清明": ("04-04", "04-06"), "劳动": ("05-01", "05-05"),
           "端午": ("05-31", "06-02"), "国庆中秋": ("10-01", "10-08")},
    2026: {"元旦": ("01-01", "01-03"), "春节": ("02-16", "02-22"),
           "清明": ("04-04", "04-06"), "劳动": ("05-01", "05-05"),
           "端午": ("06-19", "06-21"), "中秋": ("09-25", "09-27"),
           "国庆": ("10-01", "10-07")},
    2027: {"元旦": ("01-01", "01-03"), "春节": ("02-06", "02-12"),
           "清明": ("04-04", "04-06"), "劳动": ("05-01", "05-05"),
           "端午": ("06-09", "06-11"), "中秋": ("09-15", "09-17"),
           "国庆": ("10-01", "10-07")},
}

_cache: Dict[str, object] = {}


def _as_date(d) -> date:
    if d is None:
        return date.today()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    return date.today()


def calendar_days() -> Optional[set]:
    """Qlib 交易日历（权威）。取不到返回 None。"""
    key = "cal"
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]
    try:
        from . import qlib_data as QD
        if not QD.available():
            _cache[key] = None
            return None
        idx = QD.load_calendar()
        s = {pd_ts.date() for pd_ts in idx}
        _cache[key] = s
        return s
    except Exception:
        _cache[key] = None
        return None


def in_hardcoded_holiday(d) -> Optional[str]:
    """命中硬编码节假日则返回节日名。"""
    dt = _as_date(d)
    year = HOLIDAYS.get(dt.year)
    if not year:
        return None
    md = dt.strftime("%m-%d")
    for name, (lo, hi) in year.items():
        if lo <= md <= hi:
            return name
    return None


def is_trading_day(d=None) -> Tuple[bool, str]:
    """
    返回 (是否交易日, 依据)。
    依据 ∈ {"calendar", "holiday", "weekend", "weekday"}
    """
    dt = _as_date(d)
    cal = calendar_days()
    if cal is not None:
        # 日历覆盖范围内：直接以日历为准（它已包含所有节假日）
        if cal:
            lo, hi = min(cal), max(cal)
            if lo <= dt <= hi:
                return (dt in cal), "calendar"
    # 日历覆盖不到（未来日期）：工作日 + 硬编码节假日
    if dt.weekday() >= 5:
        return False, "weekend"
    h = in_hardcoded_holiday(dt)
    if h:
        return False, "holiday"
    return True, "weekday"


def is_trading_day_bool(d=None) -> bool:
    return is_trading_day(d)[0]


def next_trading_day(d=None, max_scan: int = 30) -> date:
    dt = _as_date(d) + timedelta(days=1)
    for _ in range(max_scan):
        if is_trading_day_bool(dt):
            return dt
        dt += timedelta(days=1)
    return dt


def prev_trading_day(d=None, max_scan: int = 30) -> date:
    dt = _as_date(d) - timedelta(days=1)
    for _ in range(max_scan):
        if is_trading_day_bool(dt):
            return dt
        dt -= timedelta(days=1)
    return dt


def summary(d=None) -> Dict:
    ok, src = is_trading_day(d)
    dt = _as_date(d)
    cal = calendar_days()
    return {"date": str(dt), "is_trading_day": ok, "basis": src,
            "weekday": dt.weekday(),
            "holiday": in_hardcoded_holiday(dt),
            "calendar_range": (f"{min(cal)}..{max(cal)}" if cal else None),
            "next": str(next_trading_day(dt)), "prev": str(prev_trading_day(dt)),
            "note": "优先用 Qlib 交易日历（权威）；未来日期用硬编码节假日近似"}


if __name__ == "__main__":
    import json
    for d in ("2026-11-09", "2026-11-07", "2026-10-01", "2026-10-05",
              "2027-02-08", "2026-01-01"):
        print(json.dumps(summary(d), ensure_ascii=False))
    print()
    print("今天:", json.dumps(summary(), ensure_ascii=False))
