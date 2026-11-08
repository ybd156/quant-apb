"""
实时行情与市场情绪 —— 为 UI 提供"现在市场怎么样"。

## 设计原则

- **UI 不做计算**（INV-8）：本模块把结果写成 `data/state/market.json`，Web UI 只读
- **采集独立进程**：由容器启动脚本后台跑，每 N 秒刷新一次，不阻塞 Web 服务
- **接口失败要降级**：某个源挂了，其它字段照常出，`missing` 里写清楚

## 采集内容

| 类别 | 字段 |
|---|---|
| 指数 | 上证/深证/创业板/科创50/沪深300/中证500/恒生/纳指/标普 |
| 宽度 | 涨家数、跌家数、平盘、涨停数、跌停数、上涨占比 |
| 量能 | 沪深两市成交额（亿） |
| 情绪 | 涨跌比、涨停/跌停比、情绪温度（0-100） |

## 情绪温度怎么算

不是拍脑袋，是三个分位的加权（全部可解释）：

    温度 = 100 × (0.4·上涨占比分位 + 0.3·涨停数分位 + 0.3·成交额分位)

分位用**滚动 250 日**，只用历史。冷启动阶段（样本不足）返回 None，不编。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
HISTORY_PATH = os.path.join(ROOT, "data", "market_history.jsonl")
os.makedirs(STATE_DIR, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://quote.eastmoney.com/"}
_MIN = 0.25
_last = [0.0]

# 主要指数（secid, 显示名, 市场）
INDICES = [
    ("1.000001", "上证指数", "CN"),
    ("0.399001", "深证成指", "CN"),
    ("0.399006", "创业板指", "CN"),
    ("1.000688", "科创50", "CN"),
    ("1.000300", "沪深300", "CN"),
    ("1.000905", "中证500", "CN"),
    ("100.HSI", "恒生指数", "HK"),
    ("100.NDX", "纳斯达克100", "US"),
    ("100.SPX", "标普500", "US"),
]


def _get(url: str, retries: int = 3, timeout: int = 12) -> str:
    last = None
    for i in range(retries):
        wait = _MIN - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
            _last[0] = time.time()
            return body
        except Exception as e:  # noqa: BLE001
            _last[0] = time.time()
            last = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} :: {last}")


def fetch_indices() -> Dict:
    """批量取指数实时行情。"""
    secids = ",".join(s for s, _, _ in INDICES)
    url = ("https://push2.eastmoney.com/api/qt/ulist.np/get"
           f"?fltt=2&invt=2&secids={secids}"
           "&fields=f1,f2,f3,f4,f5,f6,f12,f13,f14,f15,f16,f17,f18")
    d = json.loads(_get(url))
    diff = (d.get("data") or {}).get("diff") or []
    out = []
    for it in diff:
        code = str(it.get("f12") or "")
        name = it.get("f14") or code
        out.append({"code": code, "name": name,
                    "price": _f(it.get("f2")), "chg_pct": _f(it.get("f3")),
                    "change": _f(it.get("f4")), "volume": _f(it.get("f5")), "amount": _f(it.get("f6"))})
    return {"ok": bool(out), "rows": out}


def fetch_breadth() -> Dict:
    """
    市场宽度：涨跌家数。
    东财单只指数接口（stock/get 的 f104/f105/f106）实测恒为 0/0/100，不可用；
    这里用 ulist.np 逐市场取。
    """
    result = {}
    total_up = total_down = total_flat = 0
    for mk, secid in (("SH", "1.000001"), ("SZ", "0.399001")):
        try:
            url = ("https://push2.eastmoney.com/api/qt/ulist.np/get"
                   f"?fltt=2&invt=2&secids={secid}&fields=f1,f12,f14,f104,f105,f106")
            d = json.loads(_get(url))
            row = ((d.get("data") or {}).get("diff") or [{}])[0]
            up, down, flat = int(row.get("f104") or 0), int(row.get("f105") or 0), int(row.get("f106") or 0)
            if up + down + flat > 0:
                result[mk] = {"up": up, "down": down, "flat": flat}
                total_up += up; total_down += down; total_flat += flat
        except Exception as e:  # noqa: BLE001
            result[mk] = {"error": str(e)[:60]}
    return {"ok": total_up + total_down > 0, "by_market": result,
            "up": total_up, "down": total_down, "flat": total_flat}


def fetch_limit_up() -> Dict:
    """涨停/跌停家数（东财涨停池，只保留最近约 20 个交易日）。"""
    try:
        url = ("https://push2ex.eastmoney.com/getTopicZTPool"
               "?ut=7eea3edcaed734bea9cbfc24409ed989&dpt=wz.ztzt"
               "&Pageindex=0&pagesize=500&sort=fbt%3Aasc&date="
               + time.strftime("%Y%m%d"))
        d = json.loads(_get(url))
        data = d.get("data") or {}
        pool = data.get("pool") or []
        return {"ok": True, "limit_up": len(pool), "date": data.get("date")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": str(e)[:60]}


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return None if not np.isfinite(v) else v
    except Exception:
        return None


def _pct_rank(series: List[float], v: float) -> Optional[float]:
    arr = np.array([x for x in series if x is not None and np.isfinite(x)], dtype=float)
    if arr.size < 20 or v is None:
        return None
    # 历史完全没有变化时分位恒为 0，会误导成"冰点"，直接判为不可用
    if float(arr.std()) < 1e-9:
        return None
    return float((arr < v).mean())


def sentiment_temperature(up_ratio: float, limit_up: int, amount_yi: float) -> Dict:
    """
    情绪温度 0-100。分位用**滚动历史**（data/market_history.jsonl），冷启动返回 None。
    """
    hist = _load_history()
    if len(hist) < 20:
        return {"ok": False, "reason": f"历史样本 {len(hist)} < 20，暂不计算温度",
                "temperature": None}
    # 去重后仍无变化 -> 分位无意义（例如采集到的都是同一时刻的重复快照）
    _u = {round(float(h.get("up_ratio") or 0), 6) for h in hist}
    if len(_u) < 5:
        return {"ok": False,
                "reason": f"历史值几乎无变化（{len(_u)} 个不同取值），分位不可用",
                "temperature": None}
    p1 = _pct_rank([h.get("up_ratio") for h in hist], up_ratio)
    p2 = _pct_rank([h.get("limit_up") for h in hist], limit_up)
    p3 = _pct_rank([h.get("amount_yi") for h in hist], amount_yi)
    parts = [(p1, 0.4), (p2, 0.3), (p3, 0.3)]
    avail = [(p, w) for p, w in parts if p is not None]
    if not avail:
        return {"ok": False, "reason": "分位不可用", "temperature": None}
    wsum = sum(w for _, w in avail)
    temp = 100.0 * sum(p * w for p, w in avail) / wsum
    level = ("过热" if temp >= 80 else "偏热" if temp >= 60 else
             "中性" if temp >= 40 else "偏冷" if temp >= 20 else "冰点")
    return {"ok": True, "temperature": round(temp, 1), "level": level,
            "parts": {"up_ratio_pct": p1, "limit_up_pct": p2, "amount_pct": p3}}


def _load_history() -> List[Dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    out = []
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return []
    return out[-500:]


def _append_history(rec: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def refresh(write: bool = True) -> Dict:
    """采集一次并写 state/market.json。"""
    t0 = time.time()
    idx = fetch_indices()
    br = fetch_breadth()
    lu = fetch_limit_up()

    up, down = br.get("up", 0), br.get("down", 0)
    up_ratio = (up / (up + down)) if (up + down) > 0 else None
    amount_yi = None
    # 只取上证综指 + 深证成指，避免与沪深300/创业板等重叠重复计算
    _two = {r["code"]: r for r in idx.get("rows", []) if r["code"] in ("000001", "399001")}
    if len(_two) == 2:
        amount_yi = round(sum((r.get("amount") or 0) for r in _two.values()) / 1e8, 1)

    limit_up = lu.get("limit_up") if lu.get("ok") else None
    temp = sentiment_temperature(up_ratio, limit_up, amount_yi)

    payload = {
        "as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
        "indices": idx.get("rows", []),
        "breadth": {"up": up, "down": down, "flat": br.get("flat", 0),
                    "up_ratio": round(up_ratio, 4) if up_ratio is not None else None,
                    "by_market": br.get("by_market")},
        "limit_up": limit_up,
        "amount_yi": amount_yi,
        "sentiment": temp,
        "missing": [k for k, v in (("indices", idx.get("ok")), ("breadth", br.get("ok")),
                                   ("limit_up", lu.get("ok"))) if not v],
        "elapsed_ms": int((time.time() - t0) * 1000),
    }

    # 只在交易时段写历史，避免非交易日重复污染
    if up_ratio is not None and amount_yi:
        _append_history({"date": time.strftime("%Y-%m-%d"),
                         "up_ratio": up_ratio, "limit_up": limit_up,
                         "amount_yi": amount_yi})

    if write:
        path = os.path.join(STATE_DIR, "market.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        payload["_path"] = path
    return payload


def loop(interval: int = 60) -> None:
    """后台常驻：每 interval 秒刷新一次（交易时段才刷）。"""
    while True:
        try:
            now = datetime.now()
            # 9:15-15:10 之间才刷（含集合竞价与收盘后几分钟）
            in_session = (now.weekday() < 5 and
                          ((9, 15) <= (now.hour, now.minute) <= (15, 10)))
            if in_session:
                r = refresh()
                print(f"[market] {r['as_of']} idx={len(r['indices'])} "
                      f"up={r['breadth']['up']} down={r['breadth']['down']} "
                      f"temp={(r['sentiment'] or {}).get('temperature')}", flush=True)
            else:
                print(f"[market] off-session {now:%H:%M}, sleep", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[market] ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    import sys
    if "--loop" in sys.argv:
        interval = 60
        for a in sys.argv:
            if a.startswith("--interval="):
                interval = int(a.split("=")[1])
        loop(interval)
    else:
        print(json.dumps(refresh(), ensure_ascii=False, indent=1)[:1500])
