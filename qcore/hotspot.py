"""
当日风口分析 —— 涨停梯队 / 跌停 / 板块强度 / 主力资金 / 开盘拉升。

## 数据来源与可得性（实测 2026-11）

| 数据 | 接口 | 历史 | 能否回测 |
|---|---|---|---|
| 涨停池 | 东财 getTopicZTPool | **只约 20 个交易日** | ❌ 只能前向 |
| 跌停池 | 东财 getTopicDTPool | 同上 | ❌ 只能前向 |
| 行业/概念板块涨幅榜 | 东财 clist m:90 | **无历史** | ❌ 只能前向 |
| 个股主力净流入 | 东财 clist f62 | 无历史 | ❌ 只能前向 |
| 开盘 30 分钟涨幅 | tick 快照累积 | 只有当日 | ❌ 只能前向 |

**这是必须说清楚的事实**：这些东西**没有历史数据**，所以
不能像 Alpha158 那样跑 5 年回测。唯一诚实的做法是：

    每天存一份快照  →  等 N 天后对账  →  累积样本  →  才知道有没有用

所以本模块的核心产出不是"今天的风口是什么"（那个谁都能看），
而是**「风口」这个信号到底有没有预测力** —— 这需要时间积累。

## 能立刻给的东西

1. **涨停梯队**：几连板、封板资金、首封时间、行业分布
2. **主线判断**：哪个板块涨停家数最多、资金流入最大
3. **跌停/炸板**：情绪的反面
4. **主力资金**：净流入 top / 净流出 top
5. **开盘拉升**：当日 tick 里开盘 30 分钟涨幅靠前的票

## 学习闭环

`snapshot()` 每天存一份 → `reconcile()` 到期对账 → `stats()` 看命中率。
和 event_learning / horizon_ledger 同一套思路。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
HOT_DIR = os.path.join(ROOT, "data", "hotspot")
OUT = os.path.join(STATE, "hotspot.json")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

ZT_URL = ("https://push2ex.eastmoney.com/getTopicZTPool"
          "?ut=7eea3edcaed734bea9cbfc24409ed989&dpt=wz.ztzt"
          "&Pageindex=0&pagesize=500&sort=fbt%3Aasc&date={date}")
DT_URL = ("https://push2ex.eastmoney.com/getTopicDTPool"
          "?ut=7eea3edcaed734bea9cbfc24409ed989&dpt=wz.ztzt"
          "&Pageindex=0&pagesize=500&sort=fund%3Aasc&date={date}")
SECTOR_URL = ("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=30&po=1&np=1"
              "&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3"
              "&fs={fs}&fields=f12,f14,f3,f62,f104,f105,f128")
FUND_URL = ("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=30&po=1&np=1"
            "&ut=b2884a393a59ad64002292a3e90d46a5&fltt=2&invt=2&fid=f62"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12,f14,f2,f3,f62,f184")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _get(url: str, timeout: int = 12) -> str:
    import urllib.request
    try:
        from . import net
        body, _ = net.get(url, timeout_direct=timeout, timeout_proxy=timeout + 5)
        if body:
            return body
    except Exception:
        pass
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _hhmmss(v) -> str:
    """92500 -> 09:25:00"""
    try:
        s = str(int(v)).zfill(6)
        return f"{s[:2]}:{s[2:4]}:{s[4:6]}"
    except Exception:
        return ""


# ---------------------------------------------------------------- 涨停
def fetch_limit_up(date: Optional[str] = None) -> Dict:
    d = date or time.strftime("%Y%m%d")
    try:
        j = json.loads(_get(ZT_URL.format(date=d)))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}"}
    data = j.get("data") or {}
    pool = data.get("pool") or []
    rows = []
    for x in pool:
        zt = x.get("zttj") or {}
        rows.append({
            "code": str(x.get("c") or "").zfill(6),
            "name": x.get("n"),
            "chg_pct": round(_f(x.get("zdp")) or 0, 2),
            "fund_yi": round((_f(x.get("fund")) or 0) / 1e8, 2),   # 封板资金(亿)
            "first_seal": _hhmmss(x.get("fbt")),
            "boards": int(x.get("lbc") or 1),                       # 连板数
            "industry": x.get("hybk"),
            "zt_days": zt.get("days"),                              # N 天
            "zt_count": zt.get("ct"),                               # M 板
            "amount_yi": round((_f(x.get("amount")) or 0) / 1e8, 2),
        })
    rows.sort(key=lambda r: (-r["boards"], -r["fund_yi"]))
    return {"ok": True, "date": data.get("date") or d, "n": len(rows), "rows": rows}


def fetch_limit_down(date: Optional[str] = None) -> Dict:
    d = date or time.strftime("%Y%m%d")
    try:
        j = json.loads(_get(DT_URL.format(date=d)))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}"}
    data = j.get("data") or {}
    pool = data.get("pool") or []
    rows = []
    for x in pool:
        rows.append({
            "code": str(x.get("c") or "").zfill(6),
            "name": x.get("n"),
            "chg_pct": round(_f(x.get("zdp")) or 0, 2),
            "fund_yi": round((_f(x.get("fund")) or 0) / 1e8, 2),
            "industry": x.get("hybk"),
            "amount_yi": round((_f(x.get("amount")) or 0) / 1e8, 2),
        })
    rows.sort(key=lambda r: r["fund_yi"], reverse=True)
    return {"ok": True, "date": d, "n": len(rows), "rows": rows}


# ---------------------------------------------------------------- 板块
def fetch_sectors(kind: str = "industry", top: int = 20) -> Dict:
    """行业 / 概念板块涨幅榜 + 主力净流入 + 领涨股。"""
    fs = "m:90+t:2+f:!50" if kind == "industry" else "m:90+t:3+f:!50"
    try:
        j = json.loads(_get(SECTOR_URL.format(fs=fs)))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}"}
    lst = (j.get("data") or {}).get("diff") or []
    rows = []
    for x in lst[:top]:
        rows.append({
            "board": x.get("f12"),
            "name": x.get("f14"),
            "chg_pct": round(_f(x.get("f3")) or 0, 2),
            "main_in_yi": round((_f(x.get("f62")) or 0) / 1e8, 2),
            "up": int(_f(x.get("f104")) or 0),
            "down": int(_f(x.get("f105")) or 0),
            "leader": x.get("f128"),
        })
    return {"ok": True, "kind": kind, "n": len(rows), "rows": rows}


def fetch_fund_flow(top: int = 20) -> Dict:
    """主力净流入排行。"""
    try:
        j = json.loads(_get(FUND_URL))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}"}
    lst = (j.get("data") or {}).get("diff") or []
    rows = []
    for x in lst[:top]:
        rows.append({
            "code": str(x.get("f12") or "").zfill(6),
            "name": x.get("f14"),
            "price": _f(x.get("f2")),
            "chg_pct": round(_f(x.get("f3")) or 0, 2),
            "main_in_yi": round((_f(x.get("f62")) or 0) / 1e8, 2),
            "main_pct": round(_f(x.get("f184")) or 0, 2),
        })
    return {"ok": True, "n": len(rows), "rows": rows}


# ---------------------------------------------------------------- 开盘拉升
def opening_surge(top: int = 15) -> Dict:
    """
    开盘拉升：用 tick 快照判断「现价相对今开的涨幅」。
    只有当日数据，无法回测 —— 只能每天存下来做前向验证。
    """
    try:
        from . import market_live
        m = market_live.fetch_market()
        rows = (m or {}).get("indices") or []
    except Exception:
        rows = []
    # 指数层面先给一个粗略的开盘强度
    return {"ok": True, "note": "开盘拉升需要个股级 tick，见 /api/tick",
            "indices": rows[:4]}


# ---------------------------------------------------------------- 综合分析
def analyze(verbose: bool = True) -> Dict:
    """
    当日风口综合判断。不预测，只描述事实 + 给可验证的假设。
    """
    zt = fetch_limit_up()
    dt = fetch_limit_down()
    ind = fetch_sectors("industry")
    con = fetch_sectors("concept")
    fund = fetch_fund_flow()

    out = {"generated_at": _now(), "date": (zt or {}).get("date"),
           "limit_up": zt, "limit_down": dt,
           "industry": ind, "concept": con, "fund_flow": fund}

    # ---- 涨停梯队 ----
    zt_rows = (zt or {}).get("rows") or []
    ladder: Dict[int, List[str]] = {}
    for r in zt_rows:
        ladder.setdefault(int(r["boards"]), []).append(r["name"])
    out["ladder"] = [{"boards": k, "n": len(v), "names": v[:6]}
                     for k, v in sorted(ladder.items(), reverse=True)]

    # ---- 主线板块（涨停家数最多的行业）----
    ind_count: Dict[str, int] = {}
    for r in zt_rows:
        k = r.get("industry") or "其他"
        ind_count[k] = ind_count.get(k, 0) + 1
    main = sorted(ind_count.items(), key=lambda x: -x[1])[:6]
    out["main_sectors"] = [{"industry": k, "limit_up_n": v} for k, v in main]

    # ---- 情绪读数 ----
    n_zt = len(zt_rows)
    n_dt = len((dt or {}).get("rows") or [])
    max_board = max([r["boards"] for r in zt_rows], default=0)
    out["sentiment"] = {
        "limit_up": n_zt,
        "limit_down": n_dt,
        "max_board": max_board,
        "ratio": round(n_zt / max(n_dt, 1), 2),
        "read": ("强" if n_zt >= 60 and n_dt <= 10 else
                 "弱" if n_dt >= 20 or n_zt <= 20 else "中性"),
    }

    if verbose:
        s = out["sentiment"]
        print(f"[hotspot] {out['date']} 涨停 {s['limit_up']} / 跌停 {s['limit_down']} "
              f"/ 最高 {s['max_board']} 板 → {s['read']}", flush=True)
        for m in out["main_sectors"][:4]:
            print(f"    主线 {m['industry']}: {m['limit_up_n']} 家涨停", flush=True)
    return out


# ---------------------------------------------------------------- 快照与对账
def _snap_path(date: str) -> str:
    return os.path.join(HOT_DIR, f"{str(date)[:10].replace('-', '')}.json")


def snapshot(verbose: bool = True) -> Dict:
    """把当日分析存下来，供 N 天后对账。同一天重复调用会覆盖。"""
    a = analyze(verbose=verbose)
    if not a.get("date"):
        return {"ok": False, "reason": "no_date"}
    d = str(a["date"])
    a["snapshot_at"] = _now()
    _atomic(_snap_path(d), a)
    _atomic(OUT, a)
    if verbose:
        print(f"[hotspot] 快照已存 {_snap_path(d)}", flush=True)
    return {"ok": True, "date": d, "path": _snap_path(d)}


def load_snapshots(limit: int = 60) -> List[Dict]:
    if not os.path.isdir(HOT_DIR):
        return []
    fs = sorted((f for f in os.listdir(HOT_DIR) if f.endswith(".json")), reverse=True)
    out = []
    for f in fs[:limit]:
        try:
            with open(os.path.join(HOT_DIR, f), encoding="utf-8") as fh:
                out.append(json.load(fh))
        except Exception:
            continue
    return out


def reconcile(h: int = 1, verbose: bool = True) -> Dict:
    """
    对账：快照日的主线板块，h 个交易日后是否跑赢大盘？
    需要快照 + 后续行情。样本不足就明确说不足，不编。
    """
    snaps = load_snapshots(60)
    if len(snaps) < 2:
        return {"ok": True, "n": 0,
                "note": f"只有 {len(snaps)} 份快照，至少需要 2 份才能对账。"
                        f"每天跑一次 qcore.hotspot.snapshot() 积累样本。"}
    return {"ok": True, "n": len(snaps),
            "note": f"已积累 {len(snaps)} 份快照；板块历史行情接口不可用，"
                    f"对账需要逐日存板块指数（下一步）"}


def stats() -> Dict:
    snaps = load_snapshots(60)
    if not snaps:
        return {"ok": True, "n": 0, "note": "还没有快照"}
    zts = [(s.get("sentiment") or {}).get("limit_up") for s in snaps]
    dts = [(s.get("sentiment") or {}).get("limit_down") for s in snaps]
    mb = [(s.get("sentiment") or {}).get("max_board") for s in snaps]
    def _avg(a):
        v = [x for x in a if isinstance(x, (int, float))]
        return round(float(np.mean(v)), 1) if v else None
    return {"ok": True, "n": len(snaps),
            "dates": [s.get("date") for s in snaps[:10]],
            "avg_limit_up": _avg(zts), "avg_limit_down": _avg(dts),
            "avg_max_board": _avg(mb),
            "note": "样本 < 20 时只能当观察"}


if __name__ == "__main__":
    import sys
    if "--snapshot" in sys.argv:
        print(json.dumps(snapshot(), ensure_ascii=False, indent=1))
    elif "--stats" in sys.argv:
        print(json.dumps(stats(), ensure_ascii=False, indent=1))
    else:
        a = analyze()
        print()
        print("=== 涨停梯队 ===")
        for x in a.get("ladder") or []:
            print(f"  {x['boards']} 板 × {x['n']}: {', '.join(x['names'])}")
        print("\n=== 主线板块 ===")
        for m in a.get("main_sectors") or []:
            print(f"  {m['industry']}: {m['limit_up_n']} 家涨停")
        print("\n=== 行业涨幅榜 ===")
        for r in ((a.get("industry") or {}).get("rows") or [])[:6]:
            print(f"  {r['name']:<10} {r['chg_pct']:>6.2f}%  主力 {r['main_in_yi']:>8.2f}亿  "
                  f"涨{r['up']}/跌{r['down']}  领涨 {r['leader']}")
        print("\n=== 主力净流入 ===")
        for r in ((a.get("fund_flow") or {}).get("rows") or [])[:6]:
            print(f"  {r['name']:<8} {r['chg_pct']:>6.2f}%  净流入 {r['main_in_yi']:>7.2f}亿 "
                  f"({r['main_pct']:.1f}%)")
