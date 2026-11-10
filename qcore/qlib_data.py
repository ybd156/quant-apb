"""
Qlib 二进制数据读取 —— 对接 `chenditc/investment_data`（Apache-2.0）。

## 为什么换这套数据

我们原来的池子是「今天市值前 300 的现存股票」，有严重生存者偏差
（AGENTS.md INV-6，实测约 +22pp/年）。
这套社区数据是 **6170 只股票、2000-2026、含退市股**，且提供
**point-in-time 的成分股名单**（`instruments/csi300.txt` 每行带起止日期）。

## 目录结构

```
qlib_bin/
  calendars/day.txt            每行一个交易日
  instruments/all.txt          code \t 起始日 \t 结束日
  instruments/csi300.txt       CSI300 成分股（带起止日期 = 点对点）
  features/sh600519/
      close.day.bin            float32: [起始索引, 值...]
      factor.day.bin           复权因子：原始价 = close / factor
      open/high/low/volume/amount/vwap/change/adjclose.day.bin
```

## 关键口径

1. **`.bin` 第一个 float 是「在 calendar 中的起始索引」**，不是数据。
2. **close 是归一化价**（首日 = 1.0）。但**所有比值都保留**，
   所以算技术因子可以直接用归一化序列，不需要还原。
3. 要还原原始价：`raw = close / factor`（实测 600519 首日 35.55、末日 1453.00，正确）。
4. `volume` 单位是股；`amount` 单位是元（不复权）。

## 性能

单只股票 10 个字段约 0.6 MB，全市场 6170 只约 3.7 GB 原始。
所以**默认只加载需要的池子**（如 csi300 的 300 只），不要全量读。
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "data", "qlib", "x", "qlib_bin")

FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor")

_cache: Dict[str, object] = {}


def base_dir() -> str:
    return os.environ.get("APB_QLIB_DIR", DEFAULT_DIR)


def available() -> bool:
    d = base_dir()
    return os.path.isfile(os.path.join(d, "calendars", "day.txt"))


def load_calendar() -> pd.DatetimeIndex:
    key = "calendar"
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]
    p = os.path.join(base_dir(), "calendars", "day.txt")
    with open(p, encoding="utf-8") as f:
        dates = [ln.strip() for ln in f if ln.strip()]
    idx = pd.DatetimeIndex(pd.to_datetime(dates))

    # 合并 overlay 里的日期。
    # 不能只靠「往 day.txt 追加」—— data_sync 重下 tarball 会解压覆盖
    # 整个 qlib 目录，追加的日期就被抹掉了（实测：11-09 加过又消失）。
    # 读取时合并最稳，不依赖文件状态。
    try:
        extra = []
        # overlay 在 <root>/data/qlib/overlay，而 base_dir 是
        # <root>/data/qlib/x/qlib_bin —— 要从 base_dir 往上跳两层
        od = os.path.join(os.path.dirname(os.path.dirname(base_dir())), "overlay")
        if os.path.isdir(od):
            for fn in os.listdir(od):
                if fn.endswith(".json") and len(fn) == 13:
                    extra.append(f"{fn[:4]}-{fn[4:6]}-{fn[6:8]}")
        if extra:
            merged = sorted(set(dates) | set(extra))
            idx = pd.DatetimeIndex(pd.to_datetime(merged))
    except Exception:
        pass

    _cache[key] = idx
    return idx


def load_instruments(name: str = "all") -> pd.DataFrame:
    """返回 DataFrame[code, start, end]。"""
    key = f"inst:{name}"
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]
    p = os.path.join(base_dir(), "instruments", f"{name}.txt")
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) >= 3:
                rows.append({"code": parts[0].lower(),
                             "start": pd.Timestamp(parts[1]),
                             "end": pd.Timestamp(parts[2])})
    df = pd.DataFrame(rows)
    _cache[key] = df
    return df


def _read_bin(path: str) -> Optional[np.ndarray]:
    """读 .bin，返回 (start_idx, values)。文件缺失返回 None。"""
    if not os.path.isfile(path):
        return None
    a = np.fromfile(path, dtype="<f4")
    if a.size < 2:
        return None
    return int(a[0]), a[1:]


def load_stock(code: str, fields: Iterable[str] = FIELDS,
               start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    读单只股票。返回 DataFrame[date, open, high, low, close, volume, amount, vwap, factor]。
    `close` 是归一化价（比值保留，可直接算因子）。
    """
    code = str(code).lower()
    d = os.path.join(base_dir(), "features", code)
    if not os.path.isdir(d):
        return None
    cal = load_calendar()
    cols: Dict[str, np.ndarray] = {}
    start_idx = None
    for f in fields:
        r = _read_bin(os.path.join(d, f"{f}.day.bin"))
        if r is None:
            if f in ("open", "high", "low", "close"):
                return None          # 核心字段缺失 -> 这只票不可用
            continue
        si, vals = r
        start_idx = si if start_idx is None else start_idx
        cols[f] = vals
    if start_idx is None or "close" not in cols:
        return None
    n = min(len(v) for v in cols.values())
    dates = cal[start_idx:start_idx + n]
    df = pd.DataFrame({"date": dates})
    for f, v in cols.items():
        df[f] = v[:n].astype(np.float64)
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # ---- 追加 overlay（收盘后自行抓的当日 K 线）----
    # 上游数据包每天只发布一次，实测晚上 19:47 本地还停在上一交易日。
    # qcore.daily_append 从东财抓当日数据，以「相对前收的比值」存盘，
    # 这里乘到最后一根收盘价上，保证收益序列连续。
    try:
        df = _apply_overlay(code, df)
    except Exception:
        pass

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df if len(df) else None


_OVERLAY_CACHE: Dict[str, object] = {}


def _overlays() -> list:
    """所有 overlay 文件（按日期升序）。带 60 秒缓存。"""
    import time as _t
    key = "list"
    hit = _OVERLAY_CACHE.get(key)
    now = _t.time()
    if hit and (now - hit[0]) < 60:
        return hit[1]
    out = []
    try:
        from . import daily_append as DA
        for d in DA.list_overlays():
            o = DA.load_overlay(d)
            if o and o.get("bars"):
                out.append(o)
    except Exception:
        out = []
    _OVERLAY_CACHE[key] = (now, out)
    return out


def _apply_overlay(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    把 overlay 里的当日比值追加到 df 末尾。

    比值定义（相对前一日收盘）：
        oo=open/prev_close  oh=high/prev_close
        ol=low/prev_close   oc=close/prev_close
    追加行的价格 = 本地最后一根 close × 对应比值。
    """
    ovs = _overlays()
    if not ovs or df is None or len(df) == 0:
        return df
    c = str(code).lower()
    last_date = pd.Timestamp(df["date"].iloc[-1])
    last_close = float(df["close"].iloc[-1])
    if last_close <= 0:
        return df

    rows = []
    for o in ovs:
        try:
            d = pd.Timestamp(o.get("date"))
        except Exception:
            continue
        if d <= last_date:
            continue
        b = (o.get("bars") or {}).get(c)
        if not b:
            continue
        # 连续性校验：overlay 的比值是相对 prev 那天算的。
        # 若本地最后一根不是 prev（中间缺了交易日），硬乘会把
        # 断档日的涨跌整段吞掉 —— 宁可少一根，也不要错一根。
        _prev = b.get("prev")
        if _prev:
            _expect = str(last_date.date()) if not rows else str(rows[-1]["date"].date())
            if str(_prev) != _expect:
                continue
        prev_close = last_close if not rows else float(rows[-1]["close"])
        try:
            rows.append({
                "date": d,
                "open": prev_close * float(b["oo"]),
                "high": prev_close * float(b["oh"]),
                "low": prev_close * float(b["ol"]),
                "close": prev_close * float(b["oc"]),
                "volume": float(b.get("volume") or 0),
                "amount": float(b.get("amount") or 0),
            })
        except Exception:
            continue
    if not rows:
        return df
    add = pd.DataFrame(rows)
    add["vwap"] = (add["high"] + add["low"] + add["close"]) / 3.0
    # factor 是 qlib 的**归一化尺度**（600519 约 0.028），不是复权因子。
    # 硬写 1.0 会让 noon.py 的 close/factor 算出 -97% 的假亏损，
    # 也会让 cleaning 每天报 factor_jump 假警报。沿用上一根。
    try:
        add["factor"] = float(df["factor"].iloc[-1]) if "factor" in df.columns else 1.0
    except Exception:
        add["factor"] = 1.0
    for col in df.columns:
        if col not in add.columns:
            add[col] = df[col].iloc[-1]
    add = add[df.columns]
    return pd.concat([df, add], ignore_index=True)


def build_panel(codes: Iterable[str], start: str = "2008-01-01",
                min_bars: int = 500, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """批量加载。返回 {code: DataFrame}。"""
    out: Dict[str, pd.DataFrame] = {}
    codes = [str(c).lower() for c in codes]
    for i, c in enumerate(codes, 1):
        df = load_stock(c, start=start)
        if df is None or len(df) < min_bars:
            continue
        out[c] = df
        if verbose and i % 100 == 0:
            print(f"  ... {i}/{len(codes)}", flush=True)
    return out


def universe_codes(name: str = "csi300", start: Optional[str] = None) -> List[str]:
    """取成分股代码。start 给定时只保留该时点仍在成分内的。"""
    df = load_instruments(name)
    if start is not None:
        t = pd.Timestamp(start)
        df = df[(df["start"] <= t) & (df["end"] >= t)]
    return df["code"].tolist()


def pit_membership(name: str = "csi300") -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """点对点成分：{code: [(start,end), ...]}，用于构造 UNIV 掩码。"""
    df = load_instruments(name)
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(r.code, []).append((r.start, r.end))
    return out


def summary() -> Dict:
    if not available():
        return {"ok": False, "reason": f"数据目录不存在: {base_dir()}"}
    cal = load_calendar()
    inst = load_instruments("all")
    return {"ok": True, "dir": base_dir(), "calendar_days": len(cal),
            "date_from": str(cal[0].date()), "date_to": str(cal[-1].date()),
            "n_stocks": len(inst),
            "universe_files": sorted(
                f.replace(".txt", "") for f in os.listdir(os.path.join(base_dir(), "instruments"))
                if f.endswith(".txt"))}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
    if available():
        codes = universe_codes("csi300", start="2017-01-01")
        print(f"csi300 @2017-01-01: {len(codes)} 只")
        df = load_stock(codes[0], start="2017-01-01")
        print(f"{codes[0]}: {len(df)} 根  {df['date'].min().date()} -> {df['date'].max().date()}")
        print(df.head(3).to_string())
