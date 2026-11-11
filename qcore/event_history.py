"""
历史事件库 —— 给"今天这条新闻"找历史参照，并把参照的**真实结果**量出来。

## 为什么要有它

`autoinfo` 能自己找新闻、能打分，但打分只回答"这条新闻有多相关"，
不回答"**上一次发生这种事，市场实际怎么走**"。
没有历史参照，"利空"就只是一个词；有了历史参照，它才变成一个可以被打脸的预测：

    "这次的关税消息像 2018-03-23 / 2025-04-03 那两次 → 历史上 5 日 -3.2% / -5.1%"
    → 5 天后拿真实收益来对账 → 差多少，记下来，下次修。

## 数据从哪来（不编）

每条事件只存**日期 + 类型 + 影响方向 + 关键词**，**不存收益**。
收益在 `measure()` 里用本地行情现算：

    CAR(h) = close[t0+1+h] / close[t0+1] - 1        （与 qcore.dataset.RET 同口径）
    max_dd = 窗口内 close 相对起点的最大回撤

市场级用 `data/raw/index/000300.csv`（本地就有，2005 起）。
板块级没有本地板块指数 → 一律标 `basis="market_proxy"`，**不假装是板块自己的数**。

## 边界（必须写清）

- 这是"参照"，不是"因果"：同一天 A 股下跌可能是别的原因；
- 事件日按**新闻见报日**记，盘后消息的公告效应按 `lag=1` 处理（`measure` 里用 t0+1 起算）；
- 目录是**人工 curated** 的，规模有限（~20 条），`analogs()` 命中率靠关键词，
  不追求覆盖全部新闻 —— 宁缺毋滥，避免把噪声当历史。

## 用法

    python -m qcore.event_history            # 打印目录 + 实测收益
    python -m qcore.event_history --match "关税"   # 查像哪几次
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state", "event_history.json")
INDEX = os.path.join(ROOT, "data", "raw", "index", "000300.csv")

#: 事件类型
KINDS = ("black_swan", "policy_tighten", "policy_ease", "geopolitics",
         "regulatory", "liquidity", "price_hike", "price_cut")

#: 人工目录：date / name / kind / direction / sectors / keywords
#: direction: +1 利多（对 A 股整体或所列板块）/ -1 利空
CATALOG: List[Dict] = [
    {"id": "lehman2008", "date": "2008-09-16", "name": "雷曼破产（全球金融危机）",
     "kind": "black_swan", "direction": -1, "sectors": ["银行", "非银金融"],
     "keywords": ["雷曼", "金融危机", "次贷", "全球衰退"]},
    {"id": "property2010", "date": "2010-04-19", "name": "国十条（地产调控）",
     "kind": "policy_tighten", "direction": -1, "sectors": ["房地产", "银行", "建筑材料"],
     "keywords": ["国十条", "限购", "地产调控", "房贷"]},
    {"id": "crash2015", "date": "2015-06-15", "name": "去杠杆股灾",
     "kind": "liquidity", "direction": -1, "sectors": ["非银金融", "计算机"],
     "keywords": ["去杠杆", "两融", "强平", "股灾", "千股跌停"]},
    {"id": "circuit2016", "date": "2016-01-04", "name": "熔断机制触发",
     "kind": "black_swan", "direction": -1, "sectors": [],
     "keywords": ["熔断", "暂停交易"]},
    {"id": "trade2018", "date": "2018-03-23", "name": "中美贸易战开打",
     "kind": "geopolitics", "direction": -1, "sectors": ["电子", "机械设备", "家用电器"],
     "keywords": ["贸易战", "关税", "301调查", "加征"]},
    {"id": "relief2018", "date": "2018-10-22", "name": "民企纾困（高层表态）",
     "kind": "policy_ease", "direction": 1, "sectors": ["非银金融", "计算机"],
     "keywords": ["纾困", "民营企业座谈会", "股权质押"]},
    {"id": "covid2020", "date": "2020-02-03", "name": "新冠首日开盘",
     "kind": "black_swan", "direction": -1, "sectors": ["社会服务", "交通运输", "商贸零售"],
     "keywords": ["疫情", "新冠", "封城", "停业"]},
    {"id": "easing2020", "date": "2020-07-06", "name": "流动性宽松牛市启动",
     "kind": "liquidity", "direction": 1, "sectors": ["非银金融", "电子"],
     "keywords": ["降准", "流动性宽松", "牛市", "成交额破万亿"]},
    {"id": "edu2021", "date": "2021-07-26", "name": "双减政策（教育）",
     "kind": "regulatory", "direction": -1, "sectors": ["传媒", "社会服务"],
     "keywords": ["双减", "教培", "校外培训", "学科类"]},
    {"id": "power2021", "date": "2021-09-27", "name": "能耗双控 / 限电",
     "kind": "policy_tighten", "direction": -1, "sectors": ["基础化工", "有色金属", "钢铁"],
     "keywords": ["限电", "能耗双控", "停产", "拉闸"]},
    {"id": "coal2021", "date": "2021-10-19", "name": "动力煤价格调控",
     "kind": "price_cut", "direction": -1, "sectors": ["煤炭"],
     "keywords": ["动力煤", "煤价", "保供", "限价"]},
    {"id": "ukraine2022", "date": "2022-02-24", "name": "俄乌冲突",
     "kind": "geopolitics", "direction": -1, "sectors": ["石油石化", "有色金属", "国防军工"],
     "keywords": ["俄乌", "冲突", "制裁", "原油"]},
    {"id": "fmc2022", "date": "2022-03-16", "name": "金融委会议稳预期",
     "kind": "policy_ease", "direction": 1, "sectors": ["非银金融", "房地产"],
     "keywords": ["金融委", "稳增长", "稳预期", "积极信号"]},
    {"id": "covidreopen2022", "date": "2022-11-14", "name": "防疫优化二十条",
     "kind": "policy_ease", "direction": 1, "sectors": ["社会服务", "交通运输", "食品饮料"],
     "keywords": ["防疫优化", "二十条", "放开", "复常"]},
    {"id": "stamp2023", "date": "2023-08-28", "name": "印花税减半",
     "kind": "policy_ease", "direction": 1, "sectors": ["非银金融"],
     "keywords": ["印花税", "减半", "活跃资本市场"]},
    {"id": "micro2024", "date": "2024-02-06", "name": "微盘踩踏后汇金增持",
     "kind": "policy_ease", "direction": 1, "sectors": ["银行", "非银金融"],
     "keywords": ["汇金", "增持", "微盘", "流动性危机"]},
    {"id": "nine2024", "date": "2024-04-15", "name": "新国九条（严监管）",
     "kind": "regulatory", "direction": -1, "sectors": ["综合", "轻工制造"],
     "keywords": ["国九条", "退市", "分红", "严监管"]},
    {"id": "tariff2025", "date": "2025-04-07", "name": "关税冲击",
     "kind": "geopolitics", "direction": -1, "sectors": ["电子", "电力设备", "机械设备"],
     "keywords": ["对等关税", "关税", "反制", "出口"]},
]


def _read_index() -> Optional[List[Dict]]:
    try:
        import csv
        with open(INDEX, encoding="utf-8") as f:
            return [{"date": r["date"][:10], "close": float(r["close"])}
                    for r in csv.DictReader(f) if r.get("close")]
    except Exception:  # noqa: BLE001
        return None


def measure(h: int = 5, lag: int = 1) -> Dict:
    """用本地 000300 给每个事件量出真实反应。**不编数字**：算不出来就标 None。"""
    idx = _read_index()
    if not idx:
        return {"ok": False, "reason": "no_index_data", "path": INDEX}
    dates = [r["date"] for r in idx]
    closes = [r["close"] for r in idx]

    def pos_on_or_after(d: str) -> Optional[int]:
        for i, x in enumerate(dates):
            if x >= d:
                return i
        return None

    rows = []
    for ev in CATALOG:
        p = pos_on_or_after(ev["date"])
        if p is None:
            rows.append({**ev, "car": None, "max_dd": None, "basis": "unavailable"})
            continue
        i0 = p + lag
        i1 = i0 + h
        if i1 >= len(closes) or i0 >= len(closes):
            rows.append({**ev, "car": None, "max_dd": None, "basis": "out_of_range"})
            continue
        c0 = closes[i0]
        car = closes[i1] / c0 - 1.0
        peak, mdd = -1e9, 0.0
        for k in range(i0, i1 + 1):
            peak = max(peak, closes[k])
            mdd = min(mdd, closes[k] / peak - 1.0)
        rows.append({**ev, "entry_date": dates[i0], "exit_date": dates[i1],
                     "car": round(car, 5), "max_dd": round(mdd, 5),
                     "basis": "market_proxy"})

    out = {"ok": True, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "h": int(h), "lag": int(lag), "index": INDEX,
           "n": len(rows), "n_measured": sum(1 for r in rows if r.get("car") is not None),
           "rows": rows,
           "note": ("CAR 用沪深300（market_proxy）现算，不是板块自身的收益；"
                    "event 日按新闻见报日，lag=1 视为次日开盘起算")}
    try:
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        tmp = STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATE)
    except Exception:  # noqa: BLE001
        pass
    return out


def load(force: bool = False, h: int = 5) -> Dict:
    if not force and os.path.isfile(STATE):
        try:
            with open(STATE, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("h") == int(h) and d.get("n_measured"):
                return d
        except Exception:  # noqa: BLE001
            pass
    return measure(h=h)


def analogs(text: str, top: int = 3, h: int = 5) -> List[Dict]:
    """给一句话找历史参照：关键词命中为主，板块/类型为辅。"""
    t = str(text or "")
    if not t:
        return []
    d = load(h=h)
    rows = d.get("rows") or []
    scored = []
    for r in rows:
        hit = sum(1 for k in (r.get("keywords") or []) if k and k in t)
        if hit <= 0:
            continue
        scored.append({**r, "_hit": hit})
    scored.sort(key=lambda r: (-r["_hit"], -abs(float(r.get("car") or 0))))
    return scored[: max(1, int(top))]


def implied(analogs_list: Sequence[Dict]) -> Optional[Dict]:
    """把多条参照合成一个"预期幅度"：中位数 + 方向一致率。"""
    cars = [float(a["car"]) for a in analogs_list if a.get("car") is not None]
    if not cars:
        return None
    cars.sort()
    mid = cars[len(cars) // 2] if len(cars) % 2 else (cars[len(cars) // 2 - 1] + cars[len(cars) // 2]) / 2
    dirs = [1 if c > 0 else (-1 if c < 0 else 0) for c in cars]
    agree = sum(1 for x in dirs if x == dirs[0]) / len(dirs) if dirs else 0.0
    return {"implied_car": round(mid, 5), "n_analogs": len(cars),
            "direction_agree": round(agree, 3), "basis": "market_proxy",
            "lo": round(cars[0], 5), "hi": round(cars[-1], 5)}


def summary() -> Dict:
    d = load()
    rows = d.get("rows") or []
    by_kind: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("car") is None:
            continue
        by_kind.setdefault(r["kind"], []).append(float(r["car"]))
    agg = {k: {"n": len(v), "mean": round(sum(v) / len(v), 5)}
           for k, v in by_kind.items()}
    return {"ok": d.get("ok", False), "n": d.get("n"), "n_measured": d.get("n_measured"),
            "h": d.get("h"), "by_kind": agg,
            "note": "各类型事件在沪深300上的 5 日平均反应（market_proxy，非板块自身）"}


if __name__ == "__main__":
    import sys
    if "--match" in sys.argv:
        q = sys.argv[sys.argv.index("--match") + 1] if len(sys.argv) > sys.argv.index("--match") + 1 else ""
        print(json.dumps(analogs(q), ensure_ascii=False, indent=1))
    else:
        d = measure()
        for r in (d.get("rows") or []):
            print(f"{r['date']}  {r['name'][:18]:<20} {r['kind']:<16} "
                  f"car={r.get('car')} dd={r.get('max_dd')}")
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
