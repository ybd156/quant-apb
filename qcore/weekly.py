"""
周报 —— 每周自动汇总这一周系统学到了什么、赚没赚、哪里错了。

## 为什么需要

系统每天在跑：预测、对账、归因、调仓。但**没有人看**就等于没跑。
周报的作用是把一周的散点收成一句话：

    这周模拟盘 +2.1%，主要靠选股不是靠板块；
    T+1 周期仍然亏钱（成本 38%/年），建议关掉；
    实时特征攒到 12 天，还差 48 天才能验证。

## 汇总什么

| 板块 | 数据来源 | 看什么 |
|---|---|---|
| 模拟盘 | paper_auto.json + trades.jsonl | 净值变化、成交笔数、盈亏归因 |
| 多周期台账 | horizon_ledger | 每周期新增对账数、命中率、超额 |
| 归因 | attribution | 命中里板块驱动 vs 选股驱动 |
| 实时特征 | live_features | 积累天数、够不够训练 |
| 事件学习 | event_learning | 命中率变化、哪些类型失效 |
| 数据质量 | data_quality.json | 有没有新的脏数据 |
| 模型健康 | model_health | healthy / watch / degraded / failed |

## 输出

- `data/state/weekly.json` —— 结构化，给 UI 用
- 控制台打印一段人话摘要

## 诚实原则

**样本不足就明说不足**，不给"看起来不错"的假象。
命中率 3/5 就是 3/5，不吹成"60% 命中"。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
OUT = os.path.join(STATE, "weekly.json")


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


def _read(name: str) -> Dict:
    try:
        with open(os.path.join(STATE, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 各板块
def _paper(days: int) -> Dict:
    try:
        from . import paper_auto as PA
        acc = PA.load_account()
        trades = PA.load_trades()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}"}
    if not acc:
        return {"ok": False, "reason": "账户未初始化"}

    cut = _days_ago(days)
    week = [t for t in trades if str(t.get("date") or "") >= cut]
    buys = [t for t in week if t.get("side") == "buy"]
    sells = [t for t in week if t.get("side") == "sell"]
    cost = sum(float(t.get("cost") or 0) for t in week)
    eq = float(acc.get("equity") or 0)
    init = float(acc.get("initial_capital") or 1)
    return {"ok": True, "equity": round(eq, 2), "initial": round(init, 2),
            "total_return": round(eq / init - 1.0, 6) if init else None,
            "cash": round(float(acc.get("cash") or 0), 2),
            "n_positions": acc.get("n_positions"),
            "week_trades": len(week), "week_buys": len(buys),
            "week_sells": len(sells), "week_cost": round(cost, 2),
            "as_of": acc.get("as_of")}


def _ledger(days: int) -> Dict:
    try:
        from . import horizon_ledger as HL
        from . import horizons as HZ
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}"}
    cut = _days_ago(days)
    rows = []
    for h in HL._all_horizons():
        recs = HL._read(h)
        done = [r for r in recs if r.get("status") == "done" and r.get("result")]
        week = [r for r in done if str(r.get("as_of")) >= cut]
        pending = [r for r in recs if r.get("status") == "pending"]
        item = {"horizon": h, "label": HZ.label_for(h),
                "n_total": len(done), "n_week": len(week),
                "n_pending": len(pending)}
        if week:
            tops = [r["result"]["top_ret"] for r in week]
            alls = [r["result"]["all_ret"] for r in week]
            hits = [bool(r["result"]["hit"]) for r in week]
            item.update({
                "week_hit_rate": round(float(np.mean(hits)), 4),
                "week_top": round(float(np.mean(tops)), 6),
                "week_excess": round(float(np.mean(tops)) - float(np.mean(alls)), 6),
            })
        rows.append(item)
    return {"ok": True, "horizons": rows}


def _attribution() -> Dict:
    d = _read("attribution.json")
    if not d.get("horizons"):
        return {"ok": False, "reason": "还没有归因数据"}
    tot_hit = tot_board = tot_stock = tot_head = tot_pick = 0
    for h in d["horizons"]:
        if not h.get("n"):
            continue
        tot_hit += h.get("hit_n") or 0
        tot_board += h.get("hit_board") or 0
        tot_stock += h.get("hit_stock") or 0
        tot_head += h.get("miss_headwind") or 0
        tot_pick += h.get("miss_pickerr") or 0
    if tot_hit + tot_head + tot_pick == 0:
        return {"ok": False, "reason": "样本为 0"}
    return {"ok": True, "hits": tot_hit,
            "hit_board": tot_board, "hit_stock": tot_stock,
            "miss_headwind": tot_head, "miss_pickerr": tot_pick,
            "verdict": (
                f"命中 {tot_hit} 次里，板块驱动 {tot_board} / 选股驱动 {tot_stock}；"
                f"未中里行业逆风 {tot_head} / 选股失误 {tot_pick}")}


def _live() -> Dict:
    d = _read("live_features.json")
    if not d:
        return {"ok": False, "reason": "还没有提取"}
    return {"ok": True, "days": d.get("days", 0),
            "ready": bool(d.get("ready_for_training")),
            "need": max(0, 60 - int(d.get("days") or 0)),
            "note": d.get("note")}


def _events() -> Dict:
    d = _read("event_learning.json")
    hr = (d.get("hit_rate") or {})
    if not hr.get("n"):
        return {"ok": False, "reason": "还没有事件对账"}
    ov = hr.get("overall") or {}
    return {"ok": True, "n": hr.get("n"),
            "hit_rate": ov.get("hit_rate"),
            "mean_signed_ret": ov.get("mean_signed_ret"),
            "by_type": hr.get("by_type")}


def _quality() -> Dict:
    d = _read("data_quality.json")
    if not d:
        return {"ok": False, "reason": "还没有跑数据体检"}
    return {"ok": True, "n_input": d.get("n_input"),
            "n_dropped": d.get("n_dropped"),
            "issue_kinds": d.get("issue_kinds")}


def _model_health() -> Dict:
    try:
        from . import model_health as MH
        return MH.summary()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}"}


# ---------------------------------------------------------------- 主流程
def build(days: int = 7, verbose: bool = True) -> Dict:
    p = _paper(days)
    lg = _ledger(days)
    at = _attribution()
    lv = _live()
    ev = _events()
    q = _quality()
    mh = _model_health()

    # 一句话总结
    lines: List[str] = []
    if p.get("ok"):
        tr = p.get("total_return")
        lines.append(
            f"模拟盘净值 {p['equity']:,.0f}（累计 {tr*100:+.2f}%），"
            f"本周成交 {p['week_trades']} 笔、成本 {p['week_cost']:,.0f} 元，"
            f"持仓 {p['n_positions']} 只。")
    else:
        lines.append(f"模拟盘：{p.get('reason')}")

    if lg.get("ok"):
        act = [h for h in lg["horizons"] if h.get("n_week")]
        if act:
            best = max(act, key=lambda x: x.get("week_excess") or -9)
            lines.append(
                f"本周有对账的周期 {len(act)} 个，表现最好的是 {best['label']}"
                f"（超额 {best.get('week_excess', 0)*100:+.2f}%，"
                f"命中率 {best.get('week_hit_rate', 0)*100:.0f}%）。")
        else:
            lines.append("本周没有新的周期对账（持有期还没到）。")

    if at.get("ok"):
        lines.append(at["verdict"] + "。")
    if lv.get("ok"):
        lines.append(f"实时特征已积累 {lv['days']} 天，"
                     + ("可以加进模型了。" if lv["ready"]
                        else f"还差 {lv['need']} 天才能验证。"))
    if mh.get("level"):
        lines.append(f"模型健康度：{mh.get('level')} —— {mh.get('note')}")

    payload = {
        "generated_at": _now(), "window_days": int(days),
        "since": _days_ago(days),
        "paper": p, "ledger": lg, "attribution": at,
        "live": lv, "events": ev, "quality": q, "model_health": mh,
        "summary_lines": lines,
        "note": ("样本不足时只报数字不下结论。命中率 3/5 就是 3/5，"
                 "不吹成『60% 命中』。"),
    }
    _atomic(OUT, payload)

    if verbose:
        print("=" * 66)
        print(f"  APB 周报  {payload['since']} ~ {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 66)
        for l in lines:
            print("  - " + l)
        print()
        if lg.get("ok"):
            print(f"  {'周期':<14}{'总对账':>7}{'本周':>6}{'命中率':>8}{'超额':>9}{'待对账':>7}")
            for h in lg["horizons"]:
                if not h.get("n_total") and not h.get("n_pending"):
                    continue
                wr = h.get("week_hit_rate")
                we = h.get("week_excess")
                print(f"  {h['label']:<14}{h['n_total']:>7}{h.get('n_week', 0):>6}"
                      f"{(f'{wr*100:.0f}%' if wr is not None else '—'):>8}"
                      f"{(f'{we*100:+.2f}%' if we is not None else '—'):>9}"
                      f"{h.get('n_pending', 0):>7}")
        print()
    return {"ok": True, **payload}


def summary() -> Dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return build(verbose=False)


if __name__ == "__main__":
    import sys
    d = 7
    for a in sys.argv:
        if a.startswith("--days="):
            d = int(a.split("=", 1)[1])
    build(days=d, verbose=True)
