"""
多周期预测台账 —— 每个持有期独立预测、独立对账、独立学。

## 为什么必须按周期分开

T+1 和 T+15 是**两个完全不同的问题**：
- T+1 的收益主要是噪声，命中率接近 50%，看的是"明天谁更强"
- T+15 的收益由趋势主导，命中率可以明显偏离 50%

用同一本台账记所有周期，等于把两件事混在一起，学出来的东西既不准也不可解释。
所以每个 horizon 一本账：`data/pred_ledger/horizon_{h}.jsonl`。

## 自学习闭环（每个周期独立跑）

    预测（T 日收盘后，h 日持有）
      ↓ 封存
    台账（只追加，INV-7）
      ↓ 等 h 个交易日过去
    对账（实际涨没涨、top 组 vs bottom 组）
      ↓ 累积
    周期战绩（命中率 / 平均收益 / 多空价差 / 逐月衰减）
      ↓
    影响下一次的仓位与信心

## 因果红线

对账只处理「entry_date + h 个交易日 <= 最新交易日」的记录。
提前对账 = 用未来数据评价过去，这是本项目反复强调的红线。

## 与 pred_ledger 旧台账的关系

`data/pred_ledger/2026.jsonl` 是单周期（h=5）的老台账，保留不动。
新台账按 `horizon_{h}.jsonl` 命名，两者互不干扰。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
LEDGER_DIR = os.path.join(ROOT, "data", "pred_ledger")
OUT = os.path.join(STATE, "horizon_perf.json")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path(h: int) -> str:
    os.makedirs(LEDGER_DIR, exist_ok=True)
    return os.path.join(LEDGER_DIR, f"horizon_{int(h)}.jsonl")


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(h: int) -> List[Dict]:
    p = _path(h)
    if not os.path.isfile(p):
        return []
    out = []
    try:
        with open(p, encoding="utf-8") as f:
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


def _write_all(h: int, rows: List[Dict]) -> None:
    p = _path(h)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


# ---------------------------------------------------------------- 交易日工具
def _calendar() -> Optional[List]:
    try:
        from . import qlib_data as QD
        return [str(d)[:10] for d in QD.load_calendar()]
    except Exception:
        return None


def _trading_days_after(cal: List[str], date: str, n: int) -> Optional[str]:
    """返回 date 之后第 n 个交易日（1-based）。不存在则 None。"""
    try:
        i = cal.index(str(date)[:10])
    except ValueError:
        # 日期不在日历里：找第一个 >= date 的位置
        i = next((k for k, d in enumerate(cal) if d >= str(date)[:10]), None)
        if i is None:
            return None
    j = i + int(n)
    return cal[j] if 0 <= j < len(cal) else None


# ---------------------------------------------------------------- 登记
def record(h: int, picks: Dict, verbose: bool = False) -> Dict:
    """
    登记一次预测。picks 是 live_model.predict 的 payload。
    同一天同一周期只留一条（重复调用会覆盖）。
    """
    h = int(h)
    as_of = str(picks.get("as_of") or "")[:10]
    if not as_of:
        return {"ok": False, "reason": "no_as_of"}
    # 评估用 rows_all（固定 30 只），展示/下单用 rows（按周期 2~10 只）。
    # 只有 rows 的话 h1(2只)/h3(3只) 永远凑不到 5 个样本 -> 永久 skipped。
    rows_display = picks.get("rows") or []
    rows_eval = picks.get("rows_all") or rows_display
    if not rows_eval:
        return {"ok": False, "reason": "no_rows"}
    rows = rows_eval

    cal = _calendar()
    entry = _trading_days_after(cal, as_of, 1) if cal else None
    exit_ = _trading_days_after(cal, as_of, 1 + h) if cal else None

    rec = {
        "horizon": h,
        "as_of": as_of,                      # 出信号的日子
        "entry_date": entry,                 # 成交日（T+1）
        "exit_date": exit_,                  # 到期日（T+1+h）
        "recorded_at": _now(),
        "n": len(rows),                       # 评估样本数（rows_all）
        "n_display": len(rows_display),       # 展示/下单数（按周期收窄）
        "items": [{"code": str(r.get("code", "")).zfill(6),
                   "name": r.get("name"),
                   "score": float(r.get("score") or 0.0),
                   "weight": r.get("weight"),
                   "price": r.get("price") or r.get("close")}
                  for r in rows],
        "status": "pending",
        "result": None,
    }

    existing = _read(h)
    prev = next((x for x in existing
                 if str(x.get("as_of")) == as_of), None)
    if prev and str(prev.get("status")) not in ("pending", "", "None"):
        # 已经对过账了：不要用新的 pending 覆盖它（INV-7 台账只追加）。
        # 之前这里无条件重建记录，同一天重复预测会把 result 清空。
        if verbose:
            print(f"[h{h}] {as_of} 已对账（{prev.get('status')}），跳过登记", flush=True)
        return {"ok": True, "horizon": h, "as_of": as_of,
                "skipped": "already_reconciled",
                "status": prev.get("status")}
    old = [x for x in existing if str(x.get("as_of")) != as_of]
    old.append(rec)
    old.sort(key=lambda x: str(x.get("as_of")))
    _write_all(h, old)
    if verbose:
        print(f"[h{h}] 登记 {as_of} 的 {len(rows)} 只，到期 {exit_}", flush=True)
    return {"ok": True, "horizon": h, "as_of": as_of, "exit_date": exit_,
            "n": len(rows)}


# ---------------------------------------------------------------- 对账
def _price_on(code: str, date: str) -> Optional[float]:
    """取某只票在某交易日的收盘价（真实价，不复权）。"""
    try:
        from . import data as D
        df = D.fetch_kline(code, adjust=0, klt=101, beg="20240101")
        if df is None or not len(df):
            return None
        d = df.copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        hit = d[d["date"] == str(date)[:10]]
        if hit.empty:
            return None
        return float(hit["close"].iloc[-1])
    except Exception:
        return None


def reconcile(h: Optional[int] = None, verbose: bool = True) -> Dict:
    """
    对到期的预测算实际收益。只处理 exit_date 已到的。
    """
    hs = [int(h)] if h else _all_horizons()
    cal = _calendar()
    if not cal:
        return {"ok": False, "reason": "no_calendar"}
    latest = cal[-1]

    total_done, total_skip = 0, 0
    detail = []
    for hh in hs:
        rows = _read(hh)
        if not rows:
            continue
        changed = False
        for r in rows:
            if r.get("status") != "pending":
                continue
            ex = r.get("exit_date")
            if not ex or ex > latest:
                total_skip += 1
                continue

            rets, missing = [], 0
            for it in (r.get("items") or []):
                code = it.get("code")
                p0 = _price_on(code, r.get("entry_date") or r.get("as_of"))
                p1 = _price_on(code, ex)
                if p0 and p1 and p0 > 0:
                    rets.append({"code": code, "name": it.get("name"),
                                 "score": it.get("score"),
                                 "entry": round(p0, 4), "exit": round(p1, 4),
                                 "ret": round(p1 / p0 - 1.0, 6)})
                else:
                    missing += 1

            if len(rets) < 5:
                r["status"] = "skipped"
                r["result"] = {"reason": f"可对账样本不足 {len(rets)}",
                               "missing": missing}
                changed = True
                total_skip += 1
                continue

            rets.sort(key=lambda x: -(x.get("score") or 0))
            k = max(1, len(rets) // 3)
            top = rets[:k]
            bot = rets[-k:]
            top_ret = float(np.mean([x["ret"] for x in top]))
            bot_ret = float(np.mean([x["ret"] for x in bot]))
            all_ret = float(np.mean([x["ret"] for x in rets]))
            hit = bool(top_ret > all_ret)

            r["status"] = "done"
            r["result"] = {
                "n": len(rets), "missing": missing,
                "top_k": k,
                "top_ret": round(top_ret, 6),
                "bottom_ret": round(bot_ret, 6),
                "all_ret": round(all_ret, 6),
                "spread": round(top_ret - bot_ret, 6),
                "hit": hit,
                "top_codes": [x["code"] for x in top],
                "bottom_codes": [x["code"] for x in bot],
                "items": rets,
            }
            changed = True
            total_done += 1
            detail.append({"horizon": hh, "as_of": r.get("as_of"),
                           "exit": ex, "top_ret": round(top_ret * 100, 2),
                           "all_ret": round(all_ret * 100, 2),
                           "hit": hit})

        if changed:
            _write_all(hh, rows)

    if verbose:
        print(f"[horizon_ledger] 对账完成 {total_done} 条，未到期 {total_skip} 条",
              flush=True)
        for d in detail[:6]:
            print(f"  h{d['horizon']} {d['as_of']}->{d['exit']} "
                  f"top {d['top_ret']:+.2f}% 全体 {d['all_ret']:+.2f}% "
                  f"{'命中' if d['hit'] else '未中'}", flush=True)
    return {"ok": True, "done": total_done, "pending": total_skip, "rows": detail}


# ---------------------------------------------------------------- 战绩
def _all_horizons() -> List[int]:
    try:
        from . import horizons as HZ
        return HZ.all_horizons()
    except Exception:
        return [1, 3, 5, 10, 15]


def _main_ledger_path() -> str:
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    return _os.path.join(root, "data", "state", "ledger.json")


def _model_horizon() -> Optional[int]:
    """
    主台账属于哪个周期。

    优先读 ledger_meta.json（pipeline 显式写下的）。
    没有就退回 pipeline 默认 horizon=5 —— 主台账是 pipeline 产出的，
    而 pipeline 默认就是 5。不靠猜模型文件（h=1 和 h=5 的 train_end 一样，
    靠它区分会归错）。
    """
    import json as _json
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    p = _os.path.join(root, "data", "state", "ledger_meta.json")
    try:
        if _os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                d = _json.load(f) or {}
            h = d.get("horizon")
            if h is not None:
                return int(h)
    except Exception:
        pass
    # 老数据没有 ledger_meta.json：主台账由 pipeline 默认 h=5 产出
    return 5


def _stats_from_main_ledger(h: int) -> Optional[Dict]:
    """
    从主台账（data/state/ledger.json）算某个周期的战绩。

    ## 为什么要这个回退

    `horizon_N.jsonl` 只有 `live_model.predict` 之后才开始累积，
    新部署时几乎是空的（实测 2~3KB）。

    但主台账里有 pipeline 逐日重放产生的几千条已对账记录
    （实测 2994 条 / 30 个交易日）。界面上「已对账」显示 0，
    不是因为没有数据，是**读错了文件**。

    主台账记录字段：as_of / code / score / entry_price / exit_price /
                    ret / hit / status
    口径与 horizon_ledger 一致：按天分组，score 前 1/3 为 top 组，
    top 组平均收益 > 全体平均 => 命中。
    """
    import json as _json
    import os as _os
    from collections import defaultdict

    path = _main_ledger_path()
    if not _os.path.isfile(path):
        return None

    # 主台账属于哪个周期？只在该周期匹配时采用
    mh = _model_horizon()
    if mh is not None and int(mh) != int(h):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return None
    rows = data if isinstance(data, list) else (data.get("rows") or [])
    if not rows:
        return None

    by_day = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("status") or "") != "scored":
            continue
        ret = r.get("ret")
        score = r.get("score")
        if ret is None or score is None:
            continue
        try:
            by_day[str(r.get("as_of"))[:10]].append((float(score), float(ret)))
        except (TypeError, ValueError):
            continue

    if len(by_day) < 3:
        return None

    tops, alls, spreads, hits, days = [], [], [], [], []
    for d, pairs in sorted(by_day.items()):
        if len(pairs) < 6:
            continue
        pairs.sort(key=lambda x: -x[0])
        k = max(1, int(len(pairs) / 3))
        top = [p[1] for p in pairs[:k]]
        allr = [p[1] for p in pairs]
        tm, am = float(np.mean(top)), float(np.mean(allr))
        tops.append(tm); alls.append(am)
        spreads.append(tm - am)
        hits.append(tm > am)
        days.append(d)

    if not days:
        return None

    return {
        "horizon": int(h),
        "n": len(days) * 100,          # 近似：每天约 100 只
        "n_days": len(days),
        "hit_rate": round(float(np.mean(hits)), 4),
        "top_mean": round(float(np.mean(tops)), 6),
        "all_mean": round(float(np.mean(alls)), 6),
        "excess_mean": round(float(np.mean(tops)) - float(np.mean(alls)), 6),
        "spread_mean": round(float(np.mean(spreads)), 6),
        "top_win_rate": round(float(np.mean([t > 0 for t in tops])), 4),
        "recent90_n": len(days),
        "recent90_top": round(float(np.mean(tops[-20:])), 6) if tops else None,
        "first": days[0], "last": days[-1],
        "pending": 0,
        "source": "main_ledger",
        "note": "从主台账（pipeline 逐日重放）统计；horizon_N.jsonl 还在累积",
    }


def _stats_for(h: int) -> Dict:
    rows = _read(h)
    done = [r for r in rows if r.get("status") == "done" and r.get("result")]
    if not done:
        # 本周期台账还空 —— 回退到主台账（pipeline 产出的几千条）
        fb = _stats_from_main_ledger(h)
        if fb:
            return fb
        return {"horizon": h, "n": 0, "note": "还没有到期的预测"}
    tops = [r["result"]["top_ret"] for r in done]
    alls = [r["result"]["all_ret"] for r in done]
    spreads = [r["result"]["spread"] for r in done]
    hits = [bool(r["result"]["hit"]) for r in done]

    # 逐月衰减（最近 3 个月 vs 更早）
    def _recent(days: int) -> List[Dict]:
        cut = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [r for r in done if str(r.get("as_of")) >= cut]

    r90 = _recent(90)
    return {
        "horizon": h,
        "n": len(done),
        "hit_rate": round(float(np.mean(hits)), 4),
        "top_mean": round(float(np.mean(tops)), 6),
        "all_mean": round(float(np.mean(alls)), 6),
        "excess_mean": round(float(np.mean(tops)) - float(np.mean(alls)), 6),
        "spread_mean": round(float(np.mean(spreads)), 6),
        "top_win_rate": round(float(np.mean([t > 0 for t in tops])), 4),
        "recent90_n": len(r90),
        "recent90_top": round(float(np.mean([r["result"]["top_ret"] for r in r90])), 6) if r90 else None,
        "first": done[0].get("as_of"), "last": done[-1].get("as_of"),
        "pending": sum(1 for r in rows if r.get("status") == "pending"),
    }


def stats(verbose: bool = False) -> Dict:
    """所有周期的战绩汇总，落盘给 UI。"""
    out = []
    for h in _all_horizons():
        s = _stats_for(h)
        try:
            from . import horizons as HZ
            s["label"] = HZ.label_for(h)
            s["cost_drag_annual"] = round(HZ.annual_cost_drag(max(h, 1), "CN") * 100, 2)
        except Exception:
            s["label"] = f"h={h}"
        out.append(s)
    payload = {"generated_at": _now(), "horizons": out,
               "note": "每个周期独立台账、独立对账；命中 = top 三分之一跑赢全体平均"}
    _atomic(OUT, payload)
    if verbose:
        print(f"{'周期':<16}{'样本':>6}{'命中率':>8}{'top均':>9}{'超额':>9}{'成本':>8}")
        for s in out:
            if not s.get("n"):
                print(f"  {s.get('label',''):<14}{0:>6}{'-':>8}{'-':>9}{'-':>9}{s.get('cost_drag_annual',0):>7.1f}%")
                continue
            print(f"  {s['label']:<14}{s['n']:>6}{s['hit_rate']*100:>7.1f}%"
                  f"{s['top_mean']*100:>8.2f}%{s['excess_mean']*100:>8.2f}%"
                  f"{s.get('cost_drag_annual',0):>7.1f}%")
    return {"ok": True, **payload}


def summary() -> Dict:
    """
    各周期战绩。**每次都重算**，不读缓存。

    踩过：以前读 OUT 缓存文件，改了统计逻辑（比如加主台账回退）
    但缓存还在，界面一直显示旧结果（已对账恒为 0）。
    重算成本很低（读两个 jsonl + 一个主台账），没必要缓存。
    """
    try:
        return stats(verbose=False)
    except Exception:
        try:
            with open(OUT, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"ok": False, "horizons": []}


if __name__ == "__main__":
    import sys
    if "--reconcile" in sys.argv:
        print(json.dumps(reconcile(), ensure_ascii=False, indent=1)[:1200])
    print()
    stats(verbose=True)
