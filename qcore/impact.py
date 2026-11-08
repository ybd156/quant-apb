"""
事件研究 —— 每个「事件类型 × 板块 × 持有期」历史上平均赚多少。

## 为什么值得做

政策日历告诉你「什么时候发生了什么」，但交易需要的是
「这类事件发生后，这个板块接下来 5 天平均涨多少、能不能信」。

这一层就是把两者连起来：对每个事件，取它标注的板块，
算 CAR（累计异常收益），再按 (type, sector, horizon) 聚合。

## 口径（严格）

- **signed CAR** = `direction × CAR(0,h)`：按事件方向押注的收益。
  `mean_car > 0` 才叫"这个方向的历史胜率能赚钱"。
- 事件日盘后公告用 `lag=1`（每个事件自带），否则错位一天。
- 同类型事件日期重叠 → HAC lags = h−1。
- `direction == 0` 的事件不入表（没有方向假设就无法定义 signed CAR）。
- 样本不足（`n < MIN_N`）的行标 `insufficient=True`，不假装有结论。

## 板块收益怎么来

优先用东财行业板块指数；拿不到时退化为「池内同行业等权」，
这是**近似口径**，只能看方向与量级，不能当精确回测。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import events as E
from .policy_calendar import CALENDAR


def _market_ret(start: str = "2010-01-01") -> Optional[pd.Series]:
    """基准：沪深300 日收益。取不到返回 None。"""
    try:
        from .data import load_index
        idx = load_index("000300")
        if idx is None or len(idx) < 100:
            return None
        d = idx[idx["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        s = pd.Series(d["close"].to_numpy(), index=pd.DatetimeIndex(d["date"]))
        return s.pct_change()
    except Exception:
        return None


# 事件里用的通用板块名 -> 东财板块名关键词（events.SECTOR_ALIASES 未覆盖的部分）
_EXTRA_ALIASES: Dict[str, List[str]] = {
    "券商": ["证券"],
    "地产": ["房地产"],
    "医药": ["化学制药", "生物制品", "医疗器械", "中药"],
    "传媒": ["文化传媒", "游戏"],
    "有色金属": ["有色金属", "小金属", "工业金属"],
    "汽车": ["汽车整车", "汽车零部件"],
    "公用事业": ["电力", "燃气"],
    "新能源": ["电池", "光伏设备", "风电设备"],
}


def _match_board(name: str, avail: List[str], bmap: Dict[str, str]):
    """事件板块名 -> 板块代码。先走 events 的别名，再走本模块补充表。"""
    for c in E.expand_sector(name, avail):
        if c in bmap:
            return bmap[c]
    for kw in _EXTRA_ALIASES.get(name, []):
        for a in avail:
            if kw in a:
                return bmap[a]
    return None


def _board_map() -> Dict[str, str]:
    """行业名 -> 东财板块代码（BKxxxx）。取不到返回空表。"""
    try:
        from .capital import sector_flow
        df = sector_flow(top=100)
        if df is None or len(df) == 0:
            return {}
        return {str(r.board_name): str(r.board_code)
                for r in df.itertuples(index=False)
                if getattr(r, "board_name", None) and getattr(r, "board_code", None)}
    except Exception:
        return {}


def _board_sector_returns(sectors: Sequence[str], start: str) -> Dict[str, pd.Series]:
    """
    优先用**东财行业板块指数**（市值加权、覆盖全部成分股），
    比池内等权近似准确得多。取不到的行业留给调用方回退。
    """
    bmap = _board_map()
    if not bmap:
        return {}
    out: Dict[str, pd.Series] = {}
    beg = str(start).replace("-", "")
    # 事件里的通用名（券商/地产）要映射到东财板块名（证券/房地产开发）
    avail = list(bmap.keys())
    for name in sectors:
        code = _match_board(str(name), avail, bmap)
        if not code:
            continue
        try:
            ser = E.fetch_board_returns(code, beg=beg)
        except Exception:
            continue
        if ser is not None and len(ser) > 200:
            out[str(name)] = ser
    return out


def _sector_returns(pool: pd.DataFrame, start: str, max_stocks: int) -> Dict[str, pd.Series]:
    """池内同行业等权日收益（近似口径）。"""
    from .data import load_kline
    panel: Dict[str, pd.DataFrame] = {}
    codes = [str(c) for c in pool["code"].tolist()[:max_stocks]]
    for c in codes:
        try:
            df = load_kline(c)
        except Exception:
            continue
        if df is None or len(df) < 200:
            continue
        panel[c] = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    if not panel:
        return {}
    try:
        return E.build_sector_returns(panel, pool, min_names=3)
    except Exception:
        return {}


def compute(pool: pd.DataFrame, start: str = "2016-01-01",
            horizons: Sequence[int] = (1, 5, 20), max_stocks: int = 120,
            min_n: int = 8) -> List[Dict]:
    """
    返回可直接写进 events.json 的 `impact` 数组。
    失败时返回空列表（调用方据此显示"暂无"），不抛异常。
    """
    # 事件里出现过的板块名
    _names = sorted({str(x) for ev in CALENDAR for x in (ev.sectors or [])})
    sec = _board_sector_returns(_names, start)
    _n_board = len(sec)
    if _n_board:
        print(f"      impact: 板块指数 {_n_board}/{len(_names)} 个（东财行业板块）", flush=True)
    # 缺的用池内等权补
    _eq = _sector_returns(pool, start, max_stocks)
    for k, v in _eq.items():
        sec.setdefault(k, v)
    if not sec:
        return []
    mkt = _market_ret(start)
    if mkt is None:
        return []
    try:
        tbl = E.impact_factor(CALENDAR, sec, mkt, horizons=tuple(horizons), min_n=min_n)
    except Exception:
        return []
    if tbl is None or len(tbl) == 0:
        return []

    out: List[Dict] = []
    for r in tbl.to_dict("records"):
        def _f(v):
            try:
                f = float(v)
                return f if np.isfinite(f) else None
            except Exception:
                return None
        out.append({
            "type": r.get("type"),
            "sector": r.get("sector"),
            "horizon": int(r.get("horizon") or 0),
            "n": int(r.get("n") or 0),
            "mean_car": _f(r.get("mean_car")),
            "mean_car_raw": _f(r.get("mean_car_raw")),
            "t": _f(r.get("t")),
            "p": _f(r.get("p")),
            "win_rate": _f(r.get("win_rate")),
            "stability": _f(r.get("stability")),
            "n_years": int(r.get("n_years") or 0),
            "conflict_rate": _f(r.get("conflict_rate")),
            "insufficient": bool(r.get("insufficient")),
            "level": r.get("level"),
        })
    # 只保留有意义的：样本够且 t 有值，按 |t| 排序
    out = [r for r in out if r["n"] >= min_n and r["mean_car"] is not None]
    out.sort(key=lambda r: -(abs(r["t"]) if r["t"] is not None else 0))
    return out


def summary(impact: List[Dict]) -> Dict:
    """给 UI 的一句话摘要。"""
    if not impact:
        return {"ok": False, "reason": "无有效事件样本"}
    strong = [r for r in impact if r["t"] is not None and abs(r["t"]) >= 2.0
              and not r["insufficient"]]
    pos = [r for r in strong if (r["mean_car"] or 0) > 0]
    return {
        "ok": True, "n_rows": len(impact), "n_significant": len(strong),
        "n_positive": len(pos),
        "note": (f"{len(impact)} 组事件×板块×周期，其中 {len(strong)} 组 |t|≥2；"
                 f"按事件方向押注平均为正的 {len(pos)} 组"),
    }


if __name__ == "__main__":
    from .data import get_pool
    p = get_pool(top_n=140)
    rows = compute(p, start="2016-01-01", max_stocks=120)
    print(f"rows: {len(rows)}")
    for r in rows[:12]:
        t = r["t"]
        print(f"  {r['type']:<14} {r['sector']:<8} h={r['horizon']:<3} n={r['n']:<4} "
              f"car={r['mean_car']:+.4f} t={t:+.2f}" if t is not None else "")
    print(summary(rows))
