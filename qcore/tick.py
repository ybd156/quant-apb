"""
秒级实时行情 —— 多源轮询 + 落盘 + 延迟监控。

## 为什么是"轮询"而不是"推送"

A 股没有免费的逐笔推送接口。实际能做到的秒级是：
**每隔 1~3 秒轮询一次快照接口**，用接口返回的**时间戳字段**判断是否真的更新了。

必须区分两件事：
  · **采集频率**（我们多久问一次）
  · **数据新鲜度**（交易所多久更新一次）
腾讯/新浪的快照在交易时段约 **3 秒** 更新一次，所以轮询快于 3 秒没有意义，
只会拿到重复数据。本模块用时间戳去重，并在返回里标注 `stale`。

## 三个源（自动降级）

| 源 | 接口 | 时间戳精度 | 字段 |
|---|---|---|---|
| 腾讯 | `qt.gtimg.cn/q=` | **秒** | 最全（五档 + 换手 + 市值） |
| 新浪 | `hq.sinajs.cn/list=` | 秒 | 五档 + 成交额 |
| 东财 | `push2/stock/get` | 秒（f86 Unix） | 单只，需逐个请求 |

腾讯支持**批量**（一次几十只），所以作为首选。

## 落盘

`data/tick/YYYYMMDD/<code>.jsonl`，每行一个快照。用于事后分析
"信号发出后 N 秒价格怎么走"——这是评估执行滑点的唯一诚实办法。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(ROOT, "data", "tick")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}


def _get(url: str, encoding: str = "gbk", timeout: int = 6,
         referer: Optional[str] = None) -> str:
    h = dict(UA)
    if referer:
        h["Referer"] = referer
    with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                timeout=timeout) as r:
        return r.read().decode(encoding, errors="replace")


def _norm(code: str) -> str:
    """
    统一成腾讯的代码格式。支持 A股 / 港股 / 美股。

        600519      -> sh600519
        000001      -> sz000001
        00700.HK    -> hk00700      （港股）
        AAPL        -> usAAPL       （美股）
        hk00700     -> hk00700      （已带前缀原样）

    多市场是必须的：分钟级预测要在所有市场都试一遍，
    不同市场的波动结构和成本差异很大，不能假设结论通用。
    """
    raw = str(code).strip()
    c = raw.lower()
    if not c:
        return c
    if c[:2] in ("sh", "sz", "bj", "hk", "us"):
        # 已带前缀：港股补零对齐，美股保持原样（含 BRK.B 这种双类）
        if c.startswith("hk"):
            return "hk" + c[2:].zfill(5)
        if c.startswith("us"):
            return "us" + raw[2:].upper()
        return c

    # 港股：00700.HK / 0700.HK / 700.HK
    if c.endswith(".hk"):
        num = c[:-3].lstrip("0") or "0"
        return "hk" + num.zfill(5)

    # 美股：AAPL / TSLA / BRK.B —— 带字母。双类股保留后面的 .X
    # （BRK.B 在腾讯里就是 usBRK.B，不能截断成 usBRK）
    if c and not c[0].isdigit():
        head = c.split(".")[0]
        tail = c[len(head):]        # 保留 .B / .A 之类
        return "us" + head.upper() + tail.upper()

    c6 = c.zfill(6)
    # 上交所：股票 6/9 开头；ETF/LOF：51/52/56/58 开头
    #   510300 沪深300ETF  588000 科创50ETF  512880 证券ETF
    if c6[0] in ("6", "9") or c6[:2] in ("51", "52", "56", "58"):
        return "sh" + c6
    # 北交所
    if c6[:2] in ("43", "83", "87", "92"):
        return "bj" + c6
    # 深交所：股票 0/3 开头；ETF/LOF：15/16/18 开头
    #   159915 创业板ETF  159919 沪深300ETF  161725 白酒LOF
    return "sz" + c6


def is_etf(code: str) -> bool:
    """
    判断是不是 ETF/LOF。**这直接影响成本**：
    A股 ETF 免印花税（股票卖出要 0.05%）、免过户费，
    往返成本比股票低约一半 —— 分钟级交易能不能成立，差别就在这。
    """
    c = str(code).strip().lower()
    if c[:2] in ("sh", "sz", "bj"):
        c = c[2:]
    c6 = c.zfill(6)
    return c6[:2] in ("51", "52", "56", "58", "15", "16", "18")


def _bare(code: str) -> str:
    """sh600519 -> 600519；hk00700 -> 00700.HK；usAAPL -> AAPL"""
    c = str(code).strip().lower()
    if c.startswith("hk"):
        return c[2:].zfill(5) + ".HK"
    if c.startswith("us"):
        return c[2:].upper()        # usBRK.B -> BRK.B
    if c[:2] in ("sh", "sz", "bj"):
        return c[2:]
    return c


# ------------------------------------------------------------------ 腾讯
def _tencent(codes: List[str]) -> Dict[str, Dict]:
    """腾讯批量快照。字段用 ~ 分隔，位置见下。"""
    q = ",".join(_norm(c) for c in codes)
    txt = _get(f"https://qt.gtimg.cn/q={q}")
    out: Dict[str, Dict] = {}
    for line in txt.split(";"):
        line = line.strip()
        if not line.startswith("v_"):
            continue
        head, _, payload = line.partition("=")
        sym = head[2:]
        f = payload.strip().strip('"').split("~")
        if len(f) < 35:
            continue
        def g(i, cast=float):
            try:
                v = f[i]
                return cast(v) if v not in ("", "-") else None
            except Exception:
                return None
        ts = f[30] if len(f) > 30 else ""
        out[sym] = {
            "code": sym, "name": f[1],
            "price": g(3), "prev_close": g(4), "open": g(5),
            "volume_lot": g(6), "amount": (g(37) * 1e4 if g(37) is not None else None),
            "high": g(33), "low": g(34),
            "chg": g(31), "chg_pct": g(32),
            "turnover_pct": g(38), "pe": g(39),
            "bid1": g(9), "bid1_vol": g(10), "ask1": g(19), "ask1_vol": g(20),
            "ts_raw": ts,
            "ts": _parse_ts(ts),
            "source": "tencent",
        }
    return out


def _parse_ts(s: str) -> Optional[float]:
    """20261106161458 -> Unix 秒。"""
    s = (s or "").strip()
    if len(s) < 14:
        return None
    try:
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S").timestamp()
    except Exception:
        return None


# ------------------------------------------------------------------ 新浪
def _sina(codes: List[str]) -> Dict[str, Dict]:
    q = ",".join(_norm(c) for c in codes)
    txt = _get(f"https://hq.sinajs.cn/list={q}", referer="https://finance.sina.com.cn")
    out: Dict[str, Dict] = {}
    for line in txt.split(";"):
        line = line.strip()
        if not line.startswith("var hq_str_"):
            continue
        head, _, payload = line.partition("=")
        sym = head.replace("var hq_str_", "").strip()
        f = payload.strip().strip('"').split(",")
        if len(f) < 32:
            continue
        def g(i):
            try:
                return float(f[i]) if f[i] not in ("", "-") else None
            except Exception:
                return None
        ts = f"{f[30]} {f[31]}" if len(f) > 31 else ""
        out[sym] = {
            "code": sym, "name": f[0],
            "open": g(1), "prev_close": g(2), "price": g(3),
            "high": g(4), "low": g(5),
            "volume_lot": (g(8) / 100.0 if g(8) is not None else None),
            "amount": g(9),
            "bid1": g(11), "bid1_vol": g(10), "ask1": g(21), "ask1_vol": g(20),
            "chg": (g(3) - g(2)) if (g(3) is not None and g(2)) else None,
            "chg_pct": ((g(3) / g(2) - 1) * 100) if (g(3) is not None and g(2)) else None,
            "ts_raw": ts,
            "ts": _parse_ts(ts.replace("-", "").replace(":", "").replace(" ", "")),
            "source": "sina",
        }
    return out


# ------------------------------------------------------------------ 东财
def _eastmoney(codes: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for c in codes:
        n = _norm(c)
        # 东财这条路只覆盖 A 股；港美股走腾讯/新浪
        if not (n.startswith("sh") or n.startswith("sz") or n.startswith("bj")):
            continue
        secid = f"{'1' if n.startswith('sh') else '0'}.{n[2:]}"
        try:
            txt = _get("https://push2.eastmoney.com/api/qt/stock/get"
                       f"?secid={secid}&fields=f43,f44,f45,f46,f47,f48,f57,f58,f60,f86,f169,f170",
                       encoding="utf-8", referer="https://quote.eastmoney.com/")
            d = (json.loads(txt).get("data") or {})
            if not d:
                continue
            out[n] = {
                "code": n, "name": d.get("f58"),
                "price": (d.get("f43") or 0) / 100.0 or None,
                "prev_close": (d.get("f60") or 0) / 100.0 or None,
                "open": (d.get("f46") or 0) / 100.0 or None,
                "high": (d.get("f44") or 0) / 100.0 or None,
                "low": (d.get("f45") or 0) / 100.0 or None,
                "volume_lot": d.get("f47"), "amount": d.get("f48"),
                "chg_pct": (d.get("f170") or 0) / 100.0,
                "ts_raw": str(d.get("f86")), "ts": d.get("f86"),
                "source": "eastmoney",
            }
        except Exception:
            continue
    return out


# ------------------------------------------------------------------ 对外
def snapshot(codes: Iterable[str]) -> Dict:
    """
    取一次快照。按 腾讯 -> 新浪 -> 东财 依次降级。
    返回 {"ok","rows","as_of","latency_s","stale","source","missing"}
    """
    codes = [str(c) for c in codes]
    if not codes:
        return {"ok": False, "reason": "empty"}
    rows: Dict[str, Dict] = {}
    used = None
    for fn, name in ((_tencent, "tencent"), (_sina, "sina"), (_eastmoney, "eastmoney")):
        try:
            r = fn(codes)
        except Exception:
            r = {}
        if r:
            rows = r
            used = name
            break
    if not rows:
        return {"ok": False, "reason": "all_sources_failed"}

    now = time.time()
    ts_list = [v["ts"] for v in rows.values() if v.get("ts")]
    newest = max(ts_list) if ts_list else None
    latency = (now - newest) if newest else None
    missing = [c for c in codes if _norm(c) not in rows]
    return {"ok": True, "source": used, "n": len(rows),
            "rows": list(rows.values()),
            # 必须 .astimezone()：`fromtimestamp()` 返回 naive，
            # 而同一个 dict 里的 `fetched_at` 是 aware —— 同一 payload 两种口径，
            # 下游 `as_of or fetched_at` 再切字符串就依赖「两者恰好都是本地」这个
            # 没写下来的约定。server.py:225 同样的操作是写对的，这里照抄。
            "as_of": datetime.fromtimestamp(newest).astimezone().isoformat(timespec="seconds")
            if newest else None,
            "latency_s": round(latency, 1) if latency is not None else None,
            "stale": (latency is not None and latency > 15),
            "missing": missing,
            "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def save_tick(payload: Dict) -> Optional[str]:
    """把一次快照追加到当天文件。"""
    if not payload.get("ok"):
        return None
    day = datetime.now().strftime("%Y%m%d")
    d = os.path.join(TICK_DIR, day)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "snapshot.jsonl")
    rec = {"fetched_at": payload["fetched_at"], "as_of": payload["as_of"],
           "source": payload["source"], "latency_s": payload["latency_s"],
           "rows": [{"code": r["code"], "price": r.get("price"),
                     "chg_pct": r.get("chg_pct"), "volume_lot": r.get("volume_lot")}
                    for r in payload["rows"]]}
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return p
    except Exception:
        return None


def in_session(now: Optional[datetime] = None) -> bool:
    """是否在 A 股交易时段（含集合竞价）。"""
    n = now or datetime.now()
    if n.weekday() >= 5:
        return False
    hm = (n.hour, n.minute)
    return (9, 15) <= hm <= (11, 30) or (13, 0) <= hm <= (15, 5)


def poll(codes: List[str], interval: float = 3.0, save: bool = True,
         max_rounds: int = 0, verbose: bool = True) -> None:
    """
    常驻轮询。interval 建议 ≥3 秒——腾讯/新浪的快照本身约 3 秒更新一次，
    轮询更快只会拿到重复数据（本模块用时间戳去重）。
    """
    rounds = 0
    last_ts = None
    while True:
        if not in_session():
            if verbose:
                print(f"[tick] off-session {datetime.now():%H:%M:%S}", flush=True)
            time.sleep(30)
            continue
        try:
            p = snapshot(codes)
            if p.get("ok"):
                if p["as_of"] != last_ts:
                    last_ts = p["as_of"]
                    if save:
                        save_tick(p)
                    if verbose:
                        print(f"[tick] {p['as_of']} src={p['source']} n={p['n']} "
                              f"lat={p['latency_s']}s stale={p['stale']}", flush=True)
            elif verbose:
                print(f"[tick] failed: {p.get('reason')}", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[tick] ERROR {type(e).__name__}: {e}", flush=True)
        rounds += 1
        if max_rounds and rounds >= max_rounds:
            return
        time.sleep(interval)


if __name__ == "__main__":
    import sys
    codes = ["600519", "000001", "300750"]
    interval = 3.0
    for arg in sys.argv[1:]:
        if arg.startswith("--codes="):
            codes = [c.strip() for c in arg.split("=", 1)[1].split(",") if c.strip()]
        elif arg.startswith("--interval="):
            try:
                interval = float(arg.split("=", 1)[1])
            except ValueError:
                pass
    p = snapshot(codes)
    print(json.dumps({k: v for k, v in p.items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    for r in p.get("rows", []):
        print(f"  {r['name']:<8} {r.get('price')}  {r.get('chg_pct')}%  "
              f"ts={r.get('ts_raw')} src={r['source']}")
    if "--loop" in sys.argv:
        poll(codes, interval=interval)
