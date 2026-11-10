"""
情绪与事件数据的读取层：把「散户情绪」和「政策/新闻事件」变成可用的按日序列。

## 两类数据的历史性质完全不同（必须说清）

| 数据 | 历史 | 原因 |
|---|---|---|
| 散户情绪（股吧） | **没有历史**，只能从今天起累积 | 股吧帖子是抓当下的，官方不提供历史快照 |
| 政策/新闻事件 | 有（`data/state/event_ledger.jsonl` 实测 80 条） | 每次采集都追加进台账，带日期 |

所以：
  · 情绪信号族在回测里**前期必然全 0**，这不是 bug，是数据不存在。
    它要几个月才能攒够样本 —— 任何「情绪信号有效」的结论在此之前都不可采信。
  · 事件信号族只有台账覆盖期内的日期有值。

## 为什么单独一层

`sentiment.py` 负责**抓取**（联网、慢），这里负责**读已落盘的快照**（本地、快）。
回测绝不能联网 —— 所以 signals 只读这里，不碰 sentiment.fetch_posts。

## 落盘约定（与 sentiment.save 一致）

    data/sentiment/<YYYY-MM>/<YYYY-MM-DD>.jsonl   每行一个 SentimentResult
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_DIR = os.path.join(ROOT, "data", "sentiment")
EVENT_LEDGER = os.path.join(ROOT, "data", "state", "event_ledger.jsonl")

#: 情绪列（对外承诺）
SENT_COLUMNS = ["date", "snt_bull_ratio", "snt_consensus", "snt_attention_gap"]
#: 事件列
EVENT_COLUMNS = ["date", "evt_direction", "evt_level"]


def _read_jsonl(path: str) -> List[Dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return []
    return out


def load_sentiment(code: str) -> pd.DataFrame:
    """该股票的散户情绪日序列（从已落盘快照读，**绝不联网**）。

    返回 [date, snt_bull_ratio, snt_consensus, snt_attention_gap]。
    没有快照时返回空表 —— 信号族输出 0，这是正确行为。
    """
    code = str(code).zfill(6)
    files = sorted(glob.glob(os.path.join(SENT_DIR, "*", "*.jsonl")))
    if not files:
        return pd.DataFrame(columns=SENT_COLUMNS)
    rows = []
    for path in files:
        for rec in _read_jsonl(path):
            if str(rec.get("code") or "").zfill(6) != code:
                continue
            as_of = str(rec.get("as_of") or "")[:10]
            if not as_of:
                continue
            rows.append({
                "date": pd.Timestamp(as_of),
                "snt_bull_ratio": rec.get("bull_ratio_w", rec.get("bull_ratio")),
                "snt_consensus": rec.get("consensus_w", rec.get("consensus")),
                "snt_attention_gap": rec.get("attention_gap"),
            })
    if not rows:
        return pd.DataFrame(columns=SENT_COLUMNS)
    df = pd.DataFrame(rows).dropna(subset=["date"])
    # 同一天多次采集取最后一次
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    for c in SENT_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[SENT_COLUMNS].reset_index(drop=True)


def load_events(code: str = "", sectors: Optional[List[str]] = None) -> pd.DataFrame:
    """政策/新闻事件日序列（读 event_ledger，本地）。

    事件是**板块级或全市场级**的，不是每只票一份。规则：
      · `sectors` 为空 → 全市场事件，对所有股票生效
      · 否则只有属于这些板块的股票生效（板块名需外部传入，见 `sectors` 参数）

    返回 [date, evt_direction, evt_level]（同日多条取强度最大的方向）。
    """
    rows = _read_jsonl(EVENT_LEDGER)
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    secs = set(sectors or [])
    picked = []
    for rec in rows:
        d = str(rec.get("date") or "")[:10]
        if not d:
            continue
        ev_secs = set(rec.get("sectors") or [])
        if ev_secs and not (ev_secs & secs):
            continue
        try:
            direction = int(rec.get("direction") or 0)
            level = float(rec.get("level") or 1.0)
        except (TypeError, ValueError):
            continue
        if direction == 0:
            continue
        picked.append({"date": pd.Timestamp(d), "dir": direction, "lvl": level})
    if not picked:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(picked)
    # 同日多条：方向按 level 加权求和后取符号，强度取最大
    g = df.groupby("date", as_index=False).apply(
        lambda x: pd.Series({
            "evt_direction": float(pd.Series(x["dir"] * x["lvl"]).sum()),
            "evt_level": float(x["lvl"].max()),
        }), include_groups=False)
    g["evt_direction"] = g["evt_direction"].apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
    return g.reset_index(drop=True)


def coverage() -> Dict:
    """诊断：情绪与事件数据各覆盖多少。"""
    files = sorted(glob.glob(os.path.join(SENT_DIR, "*", "*.jsonl")))
    n_sent = 0
    dates = set()
    codes = set()
    for path in files:
        for rec in _read_jsonl(path):
            n_sent += 1
            d = str(rec.get("as_of") or "")[:10]
            if d:
                dates.add(d)
            c = str(rec.get("code") or "").zfill(6)
            if c and c != "000000":
                codes.add(c)
    ev = _read_jsonl(EVENT_LEDGER)
    ev_dates = sorted({str(r.get("date") or "")[:10] for r in ev if r.get("date")})
    return {
        "ok": True,
        "sentiment": {"files": len(files), "rows": n_sent,
                      "n_days": len(dates), "n_codes": len(codes),
                      "first": min(dates) if dates else None,
                      "last": max(dates) if dates else None},
        "events": {"rows": len(ev), "n_days": len(ev_dates),
                   "first": ev_dates[0] if ev_dates else None,
                   "last": ev_dates[-1] if ev_dates else None},
        "note": ("情绪没有历史，只能从采集日开始累积；"
                 "在此之前任何情绪类结论都不可采信"),
    }


if __name__ == "__main__":
    print(json.dumps(coverage(), ensure_ascii=False, indent=1))
