"""
数据源健康度 —— 记录每个接口的可用性、延迟、失败原因，写 `data/state/source_health.json`。

## 为什么必须落盘

UI 上「数据源健康度」一栏长期显示"暂无"，因为健康度只在**真正发请求时**
才被记录，而数据命中缓存时根本不会请求 → 文件永远不生成。

本模块独立于数据抓取：**定期主动探活**，无论有没有命中缓存都记录。
这样用户随时能看到"现在哪个源是好的"。

## 格式（UI 契约）

```json
{
  "as_of": "2026-11-08T21:40:00+08:00",
  "sources": {
    "eastmoney.kline": {"ok": 12, "fail": 0, "latency_ms": 210, "last_ok": "...", "last_error": ""},
    "tencent.tick":    {"ok": 8,  "fail": 1, "latency_ms": 95,  "last_ok": "...", "last_error": "timeout"}
  },
  "summary": {"n_sources": 2, "n_ok": 2, "n_fail": 0, "worst_latency_ms": 210}
}
```
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
PATH = os.path.join(STATE, "source_health.json")

# 探活清单：(key, 说明, 探测函数)
PROBES: List[tuple] = []


def _register():
    """延迟注册，避免 import 期的循环依赖。"""
    global PROBES
    if PROBES:
        return
    from . import qlib_data, tick
    from .data import _get_json  # noqa: F401

    def probe_kline():
        import urllib.request
        u = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
             "?secid=1.600519&fields1=f1&fields2=f51,f53&klt=101&fqt=2&lmt=5")
        r = urllib.request.urlopen(urllib.request.Request(
            u, headers={"User-Agent": "apb"}), timeout=8)
        d = json.loads(r.read().decode("utf-8", "replace"))
        n = len(((d.get("data") or {}).get("klines") or []))
        return n > 0

    def probe_index():
        import urllib.request
        u = ("https://push2.eastmoney.com/api/qt/ulist.np/get"
             "?fltt=2&invt=2&secids=1.000001&fields=f2,f3,f14")
        r = urllib.request.urlopen(urllib.request.Request(
            u, headers={"User-Agent": "apb"}), timeout=8)
        d = json.loads(r.read().decode("utf-8", "replace"))
        return bool((d.get("data") or {}).get("diff"))

    def probe_tick():
        p = tick.snapshot(["600519"])
        return bool(p.get("ok"))

    def probe_qlib():
        return qlib_data.available()

    PROBES = [
        ("eastmoney.kline", "个股日K（后复权）", probe_kline),
        ("eastmoney.index", "指数实时", probe_index),
        ("tencent.tick", "秒级快照", probe_tick),
        ("qlib.local", "Qlib 社区数据（本地）", probe_qlib),
    ]


def load() -> Dict:
    try:
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def check_all(write: bool = True) -> Dict:
    """逐个探活，累加计数后写盘。"""
    _register()
    prev = load().get("sources") or {}
    out: Dict[str, Dict] = {}
    for key, desc, fn in PROBES:
        old = prev.get(key) or {}
        ok_n = int(old.get("ok") or 0)
        fail_n = int(old.get("fail") or 0)
        t0 = time.time()
        err = ""
        good = False
        try:
            good = bool(fn())
            if not good:
                err = "empty"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:60]}"
        ms = round((time.time() - t0) * 1000, 1)
        if good:
            ok_n += 1
        else:
            fail_n += 1
        out[key] = {
            "desc": desc, "ok": ok_n, "fail": fail_n,
            "latency_ms": ms,
            "last_ok": datetime.now().astimezone().isoformat(timespec="seconds") if good
            else old.get("last_ok"),
            "last_error": "" if good else err,
        }
    lats = [v["latency_ms"] for v in out.values() if v.get("last_error") == ""]
    payload = {
        "as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": out,
        "summary": {"n_sources": len(out),
                    "n_ok": sum(1 for v in out.values() if not v["last_error"]),
                    "n_fail": sum(1 for v in out.values() if v["last_error"]),
                    "worst_latency_ms": max(lats) if lats else None},
    }
    if write:
        os.makedirs(STATE, exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, PATH)
    return payload


if __name__ == "__main__":
    r = check_all()
    print(json.dumps(r["summary"], ensure_ascii=False))
    for k, v in r["sources"].items():
        flag = "OK  " if not v["last_error"] else "FAIL"
        print(f"  [{flag}] {k:<20} {v['latency_ms']:>7.1f}ms  ok={v['ok']} fail={v['fail']}"
              f"  {v['last_error']}")
