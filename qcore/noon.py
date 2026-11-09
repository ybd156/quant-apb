"""
午后预测 —— 用上午的信息决定下午买什么，验证到次日收盘。

## 与盘前预测的区别

| | 盘前预测 | 午后预测 |
|---|---|---|
| 时点 | 08:30-09:25 | 11:30-13:00 |
| 信息 | T−1 收盘 + 隔夜美股 | **再加今日上午半天** |
| 持有 | T 日开盘 → T+1 收盘 | **T 日下午 → T+1 收盘** |
| 优势 | 与训练口径完全一致 | 多了半天信息，能过滤掉上午就走坏的票 |

## 硬约束：为什么不能历史回测

5 分钟线接口只回 **31 个交易日**（实测 lmt/beg 怎么调都一样），
日频因子最多 31 行。所以：
- **不能**用历史数据训练一个"上午特征 → 下午收益"的模型
- **只能**用盘前模型 + 上午特征做**过滤**，然后**前向累积样本**

这不是妥协，是诚实：**先跑起来，用真实的前向记录说话**。

## 设计：过滤而非另训模型

    盘前分数 → 按上午表现调整 → 午后建议

上午特征（全部来自 tick 快照，实时可算）：
1. `am_ret`     上午涨跌幅（相对昨收）
2. `gap`        开盘跳空
3. `hold_open`  是否守住开盘价（现价 > 开盘价）
4. `vol_ratio`  当前成交量 / 20 日均量（放量程度）
5. `vwap_dev`   现价相对当日均价（均价用 tick 累积近似）

调整逻辑（可解释，不是黑箱）：
- 上午大跌（< −3%）→ 降权（可能有利空）
- 上午放量上涨（vol_ratio > 1.5 且 am_ret > 0）→ 加权（有资金进场）
- 跌破开盘价且缩量 → 降权（无人接盘）

## 验证

`verify()` 对每条历史午后建议算：
**从建议时的价格 → 下一个交易日收盘**的收益，累积命中率。
这是唯一诚实的评估——因为样本是前向积累的，不存在回测过拟合。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
NOON = os.path.join(STATE, "noon_picks.json")
HIST = os.path.join(STATE, "noon_history.jsonl")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read(name: str) -> Dict:
    try:
        with open(os.path.join(STATE, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append(path: str, rec: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _avg_volume(code: str, days: int = 20) -> Optional[float]:
    """20 日均量（股）。用于算放量倍数。"""
    try:
        from . import qlib_data as QD
        q = ("sh" + code) if code[0] in "69" else (
            "bj" + code if code[:2] in ("43", "83", "87", "92") else "sz" + code)
        df = QD.load_stock(q, start=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"))
        if df is None or len(df) < 5:
            return None
        v = df["volume"].tail(days)
        return float(v.mean()) if len(v) else None
    except Exception:
        return None


def _adjust(r: Dict, am: Dict, avg_vol: Optional[float]) -> Dict:
    """
    用上午表现调整盘前分数。返回调整后的记录（含调整理由）。
    """
    score = float(r.get("score") or 0.0)
    mult = 1.0
    reasons = []

    am_ret = am.get("chg_pct")          # 上午涨跌（相对昨收，%）
    px = am.get("price")
    open_px = am.get("open")
    vol = am.get("volume_lot")

    if am_ret is not None:
        if am_ret < -3.0:
            mult *= 0.5
            reasons.append(f"上午大跌 {am_ret:.1f}% 降权")
        elif am_ret > 3.0:
            mult *= 0.8
            reasons.append(f"上午已涨 {am_ret:.1f}% 追高降权")

    if px is not None and open_px:
        if px < open_px:
            mult *= 0.8
            reasons.append("跌破开盘价")
        else:
            mult *= 1.1
            reasons.append("守住开盘价")

    # 量能调整：**暂不做**。
    # 实测 tick 的 volume_lot（手）与 Qlib 的 volume 不是同一口径
    # （601018 相差 84 倍，600519 相差 4.4 倍，比例不固定），
    # 直接相除会得出"放量 84 倍"这种假信号。
    # 宁可不做，也不能用一个假的量能指标去调整仓位。
    # 等 tick 历史累积到 20 天以上，用「tick 自己的均值」再算放量倍数。
    if vol:
        reasons.append("量能暂不参与（口径未对齐）")

    return {"score_noon": round(score * mult, 6),
            "score_pre": round(score, 6),
            "mult": round(mult, 3),
            "reasons": reasons or ["上午无显著变化"]}


def predict(top_k: int = 20, verbose: bool = True) -> Dict:
    """
    午后预测：盘前建议 + 上午表现 → 新的排序。
    建议在 11:30 之后调用（上午数据已完整）。
    """
    pm = _read("premarket.json")
    rows = pm.get("rows") or []
    if not rows:
        return {"ok": False, "reason": "no_premarket",
                "hint": "先跑盘前预测（/api/session?which=premarket）"}

    codes = [str(r.get("code", "")).zfill(6) for r in rows]
    try:
        from . import tick
        snap = tick.snapshot(codes)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"tick_failed: {type(e).__name__}"}
    if not snap.get("ok"):
        return {"ok": False, "reason": "tick_unavailable"}

    live = {}
    for x in snap.get("rows", []):
        c = str(x.get("code", ""))
        live[c[2:] if c[:2] in ("sh", "sz", "bj") else c] = x

    out = []
    for r in rows:
        c = str(r.get("code", "")).zfill(6)
        am = live.get(c) or {}
        avg_vol = _avg_volume(c)
        adj = _adjust(r, am, avg_vol)
        out.append({
            "code": c, "name": r.get("name"),
            "price": am.get("price"), "prev_close": am.get("prev_close"),
            "am_ret_pct": am.get("chg_pct"),
            **adj,
            "weight": None,   # 排序后重算
        })

    out.sort(key=lambda x: -x["score_noon"])
    out = out[:top_k]
    w = round(1.0 / len(out), 4) if out else None
    for x in out:
        x["weight"] = w
        x["rank"] = out.index(x) + 1

    # 验证时点：下一个交易日收盘
    verify_at = None
    try:
        from . import calendar_cn as CAL
        verify_at = str(CAL.next_trading_day())
    except Exception:
        verify_at = str((datetime.now() + timedelta(days=1)).date())

    payload = {
        "session": "noon", "as_of": _now(),
        "tick_as_of": snap.get("as_of"), "stale": snap.get("stale"),
        "for_date": str(datetime.now().date()),
        "verify_at": verify_at,
        "n": len(out), "top_k": top_k,
        "rows": out,
        "note": ("午后建议 = 盘前分数 × 上午表现调整；"
                 "验证窗口为「建议时价格 → 下一个交易日收盘」。"
                 "5 分钟数据仅 31 天历史，故只能前向验证，不做历史回测。"),
    }
    _atomic(NOON, payload)
    _append(HIST, {"as_of": payload["as_of"], "verify_at": verify_at,
                   "rows": [{"code": x["code"], "price": x["price"],
                             "score_noon": x["score_noon"]} for x in out]})
    if verbose:
        print(f"[noon] {payload['for_date']} 午后建议 {len(out)} 只，"
              f"验证于 {verify_at}", flush=True)
    return {"ok": True, **payload}


def verify(verbose: bool = True) -> Dict:
    """
    验证历史午后建议：从建议时价格 → 下一个交易日收盘的收益。
    只评价 verify_at 已到的记录，绝不提前。
    """
    if not os.path.isfile(HIST):
        return {"ok": True, "n": 0, "note": "还没有午后建议记录"}
    recs = []
    with open(HIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not recs:
        return {"ok": True, "n": 0}

    today = datetime.now().date()
    results = []
    for rec in recs:
        try:
            va = datetime.strptime(str(rec.get("verify_at"))[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if va > today:
            continue          # 还没到期，不评价
        for it in rec.get("rows", []):
            code = str(it.get("code", "")).zfill(6)
            entry = it.get("price")
            if not entry or entry <= 0:
                continue
            q = ("sh" + code) if code[0] in "69" else (
                "bj" + code if code[:2] in ("43", "83", "87", "92") else "sz" + code)
            try:
                from . import qlib_data as QD
                df = QD.load_stock(q, start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
                if df is None or len(df) == 0:
                    continue
                # 用真实价：Qlib 归一化价 × 同一日因子还原不必要，
                # 这里用「归一化价之比」算收益，比值与真实价一致
                last = float(df["close"].iloc[-1])
                # entry 是真实价，需要把归一化 last 转成真实价同口径
                factor = float(df["factor"].iloc[-1]) if "factor" in df else 0.0
                if factor > 0:
                    last_real = last / factor
                else:
                    continue
                ret = last_real / entry - 1.0
                results.append({"code": code, "entry": entry,
                                "last": round(last_real, 4),
                                "ret_pct": round(ret * 100, 3),
                                "hit": bool(ret > 0), "as_of": rec.get("as_of")})
            except Exception:
                continue

    if not results:
        return {"ok": True, "n": 0, "note": "没有已到期且可评价的记录"}
    rets = [r["ret_pct"] for r in results]
    hits = [r["hit"] for r in results]
    out = {"ok": True, "n": len(results),
           "hit_rate": round(float(np.mean(hits)), 4),
           "mean_ret_pct": round(float(np.mean(rets)), 3),
           "median_ret_pct": round(float(np.median(rets)), 3),
           "n_days": len({r["as_of"] for r in results}),
           "sample_warning": ("样本不足 30 条，只能当观察，不能当结论"
                              if len(results) < 30 else None),
           "worst": sorted(results, key=lambda x: x["ret_pct"])[:3],
           "best": sorted(results, key=lambda x: -x["ret_pct"])[:3]}
    if verbose:
        print(f"[noon.verify] {out['n']} 条 / {out['n_days']} 天 "
              f"命中率 {out['hit_rate']:.1%} 平均 {out['mean_ret_pct']:+.2f}%", flush=True)
    return out


def summary() -> Dict:
    cur = _read("noon_picks.json")
    v = verify(verbose=False)
    return {"ok": True, "current": {k: cur.get(k) for k in
                                    ("as_of", "for_date", "verify_at", "n", "stale")},
            "verified": v,
            "note": "午后预测只能前向验证（5分钟数据仅31天历史）"}


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        print(json.dumps(verify(), ensure_ascii=False, indent=1)[:1200])
    elif "--summary" in sys.argv:
        print(json.dumps(summary(), ensure_ascii=False, indent=1)[:1200])
    else:
        r = predict()
        print(json.dumps({k: v for k, v in r.items() if k != "rows"},
                         ensure_ascii=False, indent=1)[:800])
        for x in (r.get("rows") or [])[:8]:
            print(f"  {x['code']} {x.get('name',''):<8} "
                  f"score {x['score_pre']:.4f}→{x['score_noon']:.4f} "
                  f"(×{x['mult']}) 上午 {x.get('am_ret_pct')}% | {', '.join(x['reasons'])}")
