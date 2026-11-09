"""
事件学习 —— 预测是利好，实际怎么走？把答案记下来，下次更准。

## 为什么这是必须的

现在的事件系统有两层：
- 人工整理的 56 条历史事件（带 `direction` 标注）
- 实时快讯 41 条（规则分类，`direction` 由关键词猜）

**但没有任何一层在验证"猜对了没有"。**
一条新闻被判成「政策·利好·银行」，5 天后银行板块到底涨没涨？没人记录。

这正是用户要的：「虽然预测是利好，但是要看实际的反应，也要这样自我学习」。

## 设计：事件台账 + 到期对账 + 校准

### 1. 登记（record）
每条事件登记时写：type / direction / sectors / 日期 / 验证窗口 h。
写入 `data/state/event_ledger.jsonl`（只追加，INV-7）。

### 2. 对账（verify_due）
到期后（date + h 个交易日已过），算受影响板块的**实际前向收益**：
- 板块收益 = 池内同行业等权
- `signed_ret = direction × 实际收益`（按预测方向押注的收益）
- `hit = signed_ret > 0`

**因果红线**：只有 date + h 之后的数据才用来评价（和 session.postclose 同一原则）。

### 3. 校准（calibrate）
按 (type, sector) 聚合历史命中率，产出**可靠性倍数**：
- 命中率 > 60% 且样本 ≥ 10 → ×1.2（这类事件可信）
- 命中率 < 40% 且样本 ≥ 10 → ×0.5（这类事件在反向）
- 样本 < 10 → ×1.0（不调，样本不够）

这个倍数供未来预测时使用：**预测一个"政策·利好·银行"事件时，
先查历史上这类事件的命中率，再决定信多少。**

### 4. 输出
- `data/state/event_learning.json`：命中率表 + 校准倍数 + 最近对账
- `hit_rate(type, sector)` 供其他模块查询

## 与 impact.py 的区别

`impact.py` 是**一次性历史回测**（用 56 条人工事件算 CAR）。
本模块是**持续台账**：每条新事件都会进账，到期自动对账，
命中率随样本增加而更新。前者是快照，后者是流水。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
LEDGER = os.path.join(STATE, "event_ledger.jsonl")
OUT = os.path.join(STATE, "event_learning.json")

DEFAULT_H = 5          # 默认验证窗口（交易日）
MIN_N_CALIB = 10       # 校准所需最少样本


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


def load_ledger() -> List[Dict]:
    out = []
    if not os.path.isfile(LEDGER):
        return out
    try:
        with open(LEDGER, encoding="utf-8") as f:
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
    return out


def _append(rec: Dict) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _key(rec: Dict) -> str:
    """去重键：事件 id 或 (日期+标题前20字)。"""
    return str(rec.get("eid") or (str(rec.get("date")) + "|" + str(rec.get("title"))[:20]))


# ---------------------------------------------------------------- 登记
def record_events(events: List[Dict], h: int = DEFAULT_H,
                  source: str = "auto", verbose: bool = True) -> Dict:
    """
    把事件登记进台账。已存在的跳过（按 _key 去重）。
    events 可以是 events.json 的 rows（人工）或 live（实时）。
    """
    have = {_key(r) for r in load_ledger()}
    added = 0
    for e in events:
        if not isinstance(e, dict):
            continue
        k = _key(e)
        if k in have:
            continue
        d = e.get("direction")
        try:
            d = int(d)
        except Exception:
            d = 0
        rec = {
            "eid": k, "date": str(e.get("date") or "")[:10],
            "time": e.get("time"), "type": e.get("type"),
            "direction": d, "sectors": list(e.get("sectors") or [])[:4],
            "level": e.get("level"), "title": str(e.get("title") or "")[:120],
            "source": source if source != "auto" else (e.get("source") or "manual"),
            "reliability": e.get("reliability"),
            "h": int(h), "recorded_at": _now(),
            "status": "pending", "verified_at": None,
            "signed_ret": None, "hit": None, "detail": None,
        }
        _append(rec)
        have.add(k)
        added += 1
    if verbose:
        print(f"[event_learning] 新登记 {added} 条（已有 {len(have) - added} 条）", flush=True)
    return {"ok": True, "added": added, "total": len(have)}


# ---------------------------------------------------------------- 板块收益
def _sector_stocks() -> Dict[str, List[str]]:
    """板块 -> 池内股票代码列表。"""
    try:
        from .data import get_pool
        pool = get_pool(top_n=800)
        out: Dict[str, List[str]] = {}
        for r in pool.itertuples(index=False):
            c = str(getattr(r, "code", "")).zfill(6)
            ind = getattr(r, "industry", None)
            if c and isinstance(ind, str) and ind:
                out.setdefault(ind, []).append(c)
        return out
    except Exception:
        return {}


def _match_sector(name: str, avail: List[str]) -> Optional[str]:
    """事件板块名 -> 池内行业名（模糊）。"""
    if name in avail:
        return name
    for a in avail:
        if name in a or a in name:
            return a
    return None


def _forward_return(codes: List[str], start_date: str, h: int) -> Optional[float]:
    """
    等权前向收益：从 start_date 之后 h 个交易日的等权收益。
    用 Qlib 归一化价的比值（比值与真实价一致）。
    """
    if not codes:
        return None
    from . import qlib_data as QD
    rets = []
    for c in codes[:40]:                       # 最多 40 只，够代表板块
        q = ("sh" + c) if c[0] in "69" else (
            "bj" + c if c[:2] in ("43", "83", "87", "92") else "sz" + c)
        try:
            df = QD.load_stock(q, start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"))
        except Exception:
            continue
        if df is None or len(df) < h + 3:
            continue
        dates = df["date"].to_numpy()
        # 找 start_date 之后的第一个交易日
        idx = None
        for i, d in enumerate(dates):
            if str(d)[:10] > start_date:
                idx = i
                break
        if idx is None or idx + h >= len(df):
            continue
        px = df["close"].to_numpy(dtype=float)
        if px[idx] <= 0 or not np.isfinite(px[idx + h]):
            continue
        rets.append(float(px[idx + h] / px[idx] - 1.0))
    if len(rets) < 3:
        return None
    return float(np.mean(rets))


# ---------------------------------------------------------------- 对账
def verify_due(verbose: bool = True) -> Dict:
    """
    对到期事件算实际反应。只处理 date + h 已经过去的，绝不提前。
    """
    recs = load_ledger()
    if not recs:
        return {"ok": True, "checked": 0, "note": "台账为空"}

    # 最新交易日
    latest = None
    try:
        from . import qlib_data as QD
        cal = QD.load_calendar()
        latest = cal[-1]
    except Exception:
        pass
    if latest is None:
        return {"ok": False, "reason": "no_calendar"}

    sec_map = _sector_stocks()
    avail = list(sec_map.keys())

    changed = 0
    newly = []
    out_recs = []
    for r in recs:
        if r.get("status") != "pending":
            out_recs.append(r)
            continue
        d = str(r.get("date") or "")[:10]
        h = int(r.get("h") or DEFAULT_H)
        if not d:
            out_recs.append(r)
            continue
        # 到期判定：date 之后第 h 个交易日 <= 最新交易日
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            out_recs.append(r)
            continue
        # 粗略：自然日 1.6 倍覆盖 h 个交易日，再要求 latest 已过
        need = dt + timedelta(days=int(h * 1.6) + 2)
        if latest.date() < need:
            out_recs.append(r)          # 还没到期
            continue
        if int(r.get("direction") or 0) == 0:
            r["status"] = "skipped"
            r["detail"] = "direction=0，无方向假设，无法定义 signed_ret"
            out_recs.append(r)
            changed += 1
            continue
        if not r.get("sectors"):
            r["status"] = "skipped"
            r["detail"] = "无板块标注"
            out_recs.append(r)
            changed += 1
            continue

        # 取第一个能匹配上的板块
        ret = None
        used_sec = None
        for s in r["sectors"]:
            m = _match_sector(str(s), avail)
            if not m:
                continue
            ret = _forward_return(sec_map.get(m, []), d, h)
            if ret is not None:
                used_sec = m
                break
        if ret is None:
            r["status"] = "skipped"
            r["detail"] = "板块收益取不到（池内该行业股票不足）"
            out_recs.append(r)
            changed += 1
            continue

        signed = float(r["direction"]) * ret
        r["status"] = "verified"
        r["verified_at"] = _now()
        r["raw_ret"] = round(ret, 6)
        r["signed_ret"] = round(signed, 6)
        r["hit"] = bool(signed > 0)
        r["detail"] = f"板块 {used_sec} {h} 日 {ret*100:+.2f}%，按 {r['direction']:+d} 方向押注 {signed*100:+.2f}%"
        newly.append(r)
        out_recs.append(r)
        changed += 1

    if changed:
        tmp = LEDGER + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in out_recs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, LEDGER)

    if verbose:
        print(f"[event_learning] 对账 {changed} 条（新到期 {len(newly)}）", flush=True)
        for r in newly[:5]:
            print(f"  {r['date']} {r['type']} {r['direction']:+d} "
                  f"{'命中' if r['hit'] else '打脸'} | {r['detail']}", flush=True)
    return {"ok": True, "checked": changed, "newly": len(newly),
            "rows": [{k: r.get(k) for k in ("date", "type", "direction", "signed_ret", "hit", "detail")}
                     for r in newly[:20]]}


# ---------------------------------------------------------------- 校准
def hit_rate_by() -> Dict:
    """按 (type) 和 (type, sector) 聚合命中率。"""
    recs = [r for r in load_ledger() if r.get("status") == "verified"]
    if not recs:
        return {"ok": True, "n": 0, "note": "还没有已对账的事件"}

    def _agg(rows: List[Dict]) -> Dict:
        if not rows:
            return {"n": 0, "hit_rate": None, "mean_signed_ret": None}
        hits = [bool(r.get("hit")) for r in rows if r.get("hit") is not None]
        rets = [float(r["signed_ret"]) for r in rows if r.get("signed_ret") is not None]
        return {"n": len(hits),
                "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
                "mean_signed_ret": round(float(np.mean(rets)), 6) if rets else None}

    by_type: Dict[str, Dict] = {}
    for t in sorted({str(r.get("type")) for r in recs}):
        by_type[t] = _agg([r for r in recs if str(r.get("type")) == t])

    by_ts: Dict[str, Dict] = {}
    for r in recs:
        for s in (r.get("sectors") or ["*"]):
            k = f"{r.get('type')}|{s}"
            by_ts.setdefault(k, {"rows": []})
            by_ts[k]["rows"].append(r)
    by_ts = {k: _agg(v["rows"]) for k, v in by_ts.items()}

    return {"ok": True, "n": len(recs), "by_type": by_type, "by_type_sector": by_ts,
            "overall": _agg(recs)}


def calibrate(min_n: int = MIN_N_CALIB) -> Dict:
    """
    产出可靠性倍数。供未来预测时使用：
    预测「政策·利好·银行」前，先查这类事件历史命中率。
    """
    hr = hit_rate_by()
    if not hr.get("n"):
        return {"ok": False, "reason": "no_verified_events",
                "hint": "先登记事件并等待验证窗口过去"}
    mult: Dict[str, float] = {}
    detail: Dict[str, Dict] = {}
    for k, v in (hr.get("by_type_sector") or {}).items():
        n, r = v.get("n") or 0, v.get("hit_rate")
        if n < min_n or r is None:
            mult[k] = 1.0
            detail[k] = {"n": n, "hit_rate": r, "mult": 1.0,
                         "why": f"样本 {n} < {min_n}，不调"}
            continue
        if r >= 0.60:
            m, why = 1.2, f"命中率 {r:.0%} ≥ 60%，这类事件可信"
        elif r <= 0.40:
            m, why = 0.5, f"命中率 {r:.0%} ≤ 40%，这类事件可能在反向"
        else:
            m, why = 1.0, f"命中率 {r:.0%} 接近随机，不调"
        mult[k] = m
        detail[k] = {"n": n, "hit_rate": r, "mult": m, "why": why}
    return {"ok": True, "multipliers": mult, "detail": detail,
            "n_verified": hr.get("n"), "overall": hr.get("overall")}


def confidence(type_: str, sector: str = "*") -> float:
    """查某类事件的历史可靠性倍数（默认 1.0）。"""
    c = calibrate()
    if not c.get("ok"):
        return 1.0
    m = c["multipliers"]
    return float(m.get(f"{type_}|{sector}", m.get(f"{type_}|*", 1.0)))


def run(verbose: bool = True) -> Dict:
    """完整流程：登记 → 对账 → 校准 → 落盘。"""
    from . import news_live
    from . import policy_calendar

    added = 0
    # 1) 实时事件
    live = news_live.load()
    if live.get("rows"):
        r = record_events(live["rows"], source="live", verbose=False)
        added += r.get("added", 0)
    # 2) 人工历史事件（只登记，用于验证历史命中率）
    try:
        cal_events = [e.__dict__ if hasattr(e, "__dict__") else dict(e)
                      for e in policy_calendar.CALENDAR]
        r2 = record_events(cal_events, source="manual", verbose=False)
        added += r2.get("added", 0)
    except Exception:
        pass
    # 3) 对账
    v = verify_due(verbose=verbose)
    # 4) 校准
    c = calibrate()
    hr = hit_rate_by()

    payload = {"generated_at": _now(), "added": added,
               "verify": {"checked": v.get("checked"), "newly": v.get("newly")},
               "hit_rate": hr, "calibration": c}
    _atomic(OUT, payload)
    if verbose:
        if hr.get("n"):
            print(f"[event_learning] 已验证 {hr['n']} 条，总命中率 "
                  f"{hr['overall']['hit_rate']:.1%}，"
                  f"平均按方向押注 {hr['overall']['mean_signed_ret']*100:+.2f}%", flush=True)
            for t, v2 in (hr.get("by_type") or {}).items():
                if v2.get("n"):
                    print(f"  {t:<12} n={v2['n']:<3} 命中率 {v2['hit_rate']:.1%} "
                          f"收益 {v2['mean_signed_ret']*100:+.2f}%", flush=True)
        else:
            print("[event_learning] 还没有可对账的事件（需等验证窗口过去）", flush=True)
    return {"ok": True, **payload}


def summary() -> Dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"ok": False, "reason": "no_report",
                "hint": "跑一次 qcore.event_learning.run() 生成"}


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        print(json.dumps(verify_due(), ensure_ascii=False, indent=1)[:1500])
    elif "--calibrate" in sys.argv:
        print(json.dumps(calibrate(), ensure_ascii=False, indent=1)[:1500])
    else:
        r = run()
        print()
        print("台账:", len(load_ledger()), "条")
        if r["hit_rate"].get("n"):
            print(json.dumps(r["hit_rate"].get("by_type_sector"), ensure_ascii=False, indent=1)[:900])
