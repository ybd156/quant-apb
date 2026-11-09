"""
实时特征 —— 把 tick 快照聚成日频特征，供训练使用。

## 为什么这是"加强训练"的正路

Alpha158 全是**日频量价**（开盘/最高/最低/收盘/成交量算出来的）。
它永远不知道：

    上午 10 点就封板了 → 下午的量是虚的
    尾盘 30 分钟暴力拉升 → 次日往往有惯性
    下午量只有上午的三分之一 → 承接不足

这些信息只在**盘中**存在，收盘后就没了。tick 快照是唯一能留住它的东西。

## 数据来源

`qcore/tick.py` 每 3 秒存一次快照到 `data/tick/YYYYMMDD/snapshot.jsonl`：

    {fetched_at, as_of, source, latency_s,
     rows: [{code, price, chg_pct, volume_lot}]}

本模块把一天几百上千个快照压成**每只票一行**的日频特征。

## 提取的特征（每只票每天）

| 特征 | 含义 | 为什么有用 |
|---|---|---|
| `am_ret` | 上午涨跌（11:30 相对首笔） | 早盘方向性 |
| `pm_ret` | 下午涨跌（收盘相对 11:30） | 午后是否延续 |
| `close_vs_open` | 收盘相对开盘 | 日内净强度 |
| `intraday_range` | (最高−最低)/开盘 | 日内波动 |
| `tail_30m` | 最后 30 分钟涨跌 | 尾盘资金态度 |
| `vol_pm_am` | 下午量/上午量 | 承接力度 |
| `vol_share_last` | 最后 10% 时间的成交占比 | 尾盘异动 |
| `n_snap` | 有效快照数 | 数据质量 |
| `px_vs_vwap` | 收盘相对当日均价 | 强弱 |

## 硬约束（必须说清楚）

**只有当天数据**。`data/tick/` 从部署那天开始积累，
所以这些特征**不能回测** —— 只能：

    每天存 → 攒够 60+ 个交易日 → 作为增量特征加进 Alpha158 → 测 IC 有没有提升

在那之前，它们只是"存着"，不参与模型。这是诚实的做法，
不能拿几天的数据假装验证过。

## 用法

```python
from qcore.live_features import extract_day, load_panel
extract_day("20261109")          # 从 tick 提取当天特征
load_panel(days=60)              # 读最近 N 天，返回 {date: {code: {...}}}
```
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(ROOT, "data", "tick")
LIVE_DIR = os.path.join(ROOT, "data", "live")
OUT = os.path.join(ROOT, "data", "state", "live_features.json")

FEATURE_NAMES = [
    "am_ret", "pm_ret", "close_vs_open", "intraday_range",
    "tail_30m", "vol_pm_am", "vol_share_last", "px_vs_vwap", "n_snap",
]


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _bare(code: str) -> str:
    c = str(code).lower()
    return c[2:] if c[:2] in ("sh", "sz", "bj") else c


def available_days() -> List[str]:
    if not os.path.isdir(TICK_DIR):
        return []
    return sorted(d for d in os.listdir(TICK_DIR)
                  if os.path.isdir(os.path.join(TICK_DIR, d)))


def extract_day(day: str, verbose: bool = False) -> Dict:
    """
    从一天的 tick 快照提取每只票的日频特征。
    快照太少（< 20）就返回 insufficient，不硬算。
    """
    p = os.path.join(TICK_DIR, str(day), "snapshot.jsonl")
    if not os.path.isfile(p):
        return {"ok": False, "reason": "no_file", "day": day}

    # 每只票的时间序列
    series: Dict[str, List[Dict]] = {}
    n_snap = 0
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = str(rec.get("as_of") or rec.get("fetched_at") or "")
                if not t:
                    continue
                n_snap += 1
                for r in (rec.get("rows") or []):
                    c = _bare(r.get("code"))
                    px = r.get("price")
                    if not c or px is None:
                        continue
                    series.setdefault(c, []).append({
                        "t": t, "px": float(px),
                        "chg": r.get("chg_pct"),
                        "vol": float(r.get("volume_lot") or 0),
                    })
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}", "day": day}

    if n_snap < 20:
        return {"ok": False, "reason": f"快照太少 {n_snap} < 20", "day": day,
                "n_snap": n_snap}

    rows: Dict[str, Dict] = {}
    for code, arr in series.items():
        if len(arr) < 20:
            continue
        arr.sort(key=lambda x: x["t"])
        px = np.array([x["px"] for x in arr], dtype=float)
        vol = np.array([x["vol"] for x in arr], dtype=float)
        if not np.all(np.isfinite(px)) or px[0] <= 0:
            continue

        # 用成交量的增量算"每段成交量"（volume_lot 是累计量）
        dv = np.diff(vol, prepend=vol[0])
        dv[dv < 0] = 0                       # 隔日/重置保护
        total_v = float(dv.sum())

        # 时间切分：上午/下午以 11:30 为界；尾盘取最后 10% 时间
        hours = np.array([int(x["t"][11:13]) if len(x["t"]) > 13 else 0 for x in arr])
        mins = np.array([int(x["t"][14:16]) if len(x["t"]) > 16 else 0 for x in arr])
        hhmm = hours * 60 + mins
        am_mask = hhmm <= (11 * 60 + 30)
        i_am = int(np.where(am_mask)[0][-1]) if am_mask.any() else len(arr) // 2
        i_tail = int(len(arr) * 0.9)

        o, c_ = px[0], px[-1]
        hi, lo = float(px.max()), float(px.min())
        vwap = float(np.sum(px * dv) / total_v) if total_v > 0 else float(px.mean())

        am_ret = px[i_am] / o - 1.0 if i_am > 0 else 0.0
        pm_ret = c_ / px[i_am] - 1.0 if i_am > 0 else 0.0
        v_am = float(dv[:i_am + 1].sum())
        v_pm = float(dv[i_am + 1:].sum())

        rows[code] = {
            "am_ret": round(float(am_ret), 6),
            "pm_ret": round(float(pm_ret), 6),
            "close_vs_open": round(float(c_ / o - 1.0), 6),
            "intraday_range": round(float((hi - lo) / o), 6),
            "tail_30m": round(float(c_ / px[i_tail] - 1.0), 6) if i_tail < len(px) else 0.0,
            "vol_pm_am": round(v_pm / v_am, 4) if v_am > 0 else None,
            "vol_share_last": round(float(dv[i_tail:].sum() / total_v), 4) if total_v > 0 else None,
            "px_vs_vwap": round(float(c_ / vwap - 1.0), 6) if vwap > 0 else None,
            "n_snap": int(len(arr)),
        }

    if not rows:
        return {"ok": False, "reason": "没有足够的每股快照", "day": day}

    payload = {"day": str(day), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "n_snap": n_snap, "n_codes": len(rows),
               "features": FEATURE_NAMES, "rows": rows,
               "note": ("从 tick 快照提取；只有当天数据，不能回测。"
                        "攒够 60+ 交易日再作为增量特征加进模型并测 IC。")}
    _atomic(os.path.join(LIVE_DIR, f"{day}.json"), payload)
    if verbose:
        print(f"[live_features] {day}: {n_snap} 快照 -> {len(rows)} 只", flush=True)
    return {"ok": True, **{k: v for k, v in payload.items() if k != "rows"},
            "n_codes": len(rows)}


def extract_all(verbose: bool = True) -> Dict:
    days = available_days()
    done = []
    for d in days:
        r = extract_day(d, verbose=verbose)
        if r.get("ok"):
            done.append({"day": d, "n_codes": r.get("n_codes"), "n_snap": r.get("n_snap")})
    # 汇总
    payload = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "days": len(done), "detail": done[-30:],
               "feature_names": FEATURE_NAMES,
               "ready_for_training": len(done) >= 60,
               "note": (f"已积累 {len(done)} 天。"
                        f"{'可以加进模型测 IC 了' if len(done) >= 60 else '需要 60 天以上才够验证'}")}
    _atomic(OUT, payload)
    if verbose:
        print(f"[live_features] 共 {len(done)} 天", flush=True)
    return {"ok": True, **payload}


def load_panel(days: int = 60) -> Dict[str, Dict[str, Dict]]:
    """读最近 N 天的特征，返回 {day: {code: {feat: val}}}。训练用。"""
    if not os.path.isdir(LIVE_DIR):
        return {}
    fs = sorted((f for f in os.listdir(LIVE_DIR) if f.endswith(".json")), reverse=True)
    out = {}
    for f in fs[:days]:
        try:
            with open(os.path.join(LIVE_DIR, f), encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("rows"):
                out[str(d.get("day"))] = d["rows"]
        except Exception:
            continue
    return out


def summary() -> Dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return extract_all(verbose=False)


if __name__ == "__main__":
    import sys
    if "--extract" in sys.argv:
        d = None
        for a in sys.argv:
            if a.startswith("--day="):
                d = a.split("=", 1)[1]
        print(json.dumps(extract_day(d, verbose=True) if d else extract_all(),
                         ensure_ascii=False, indent=1)[:1200])
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1)[:900])
