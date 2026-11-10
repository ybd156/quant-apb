"""
预热外部数据缓存（资金流 / 大宗 / 龙虎榜 / 融资融券 / 增减持 / 解禁 / 业绩预告）。

## 为什么必须独立于 pipeline

实测（2026-11-10）：每只票要打 5~6 个外部接口，单只约 **4.8 秒**。
300 只 = 约 **24 分钟**。如果放在 `build_dataset` 里逐票联网：

  · pipeline 卡在 `[1/7] data` 十几分钟不动（我实际踩到了）
  · 每次回测都重新联网，等于把回测变成网络压力测试
  · 断网时回测直接不可用

所以正确分工是：
  · **这个脚本**（独立、可后台、可定时）负责联网 + 落盘
  · `build_dataset` 只读缓存（`external_loader(cache_only=True)`），**绝不联网**

## 用法

    python -m tools.prewarm_external                 # 默认 300 只
    python -m tools.prewarm_external --pool 50       # 只预热前 50 只
    python -m tools.prewarm_external --codes 600519,000001
    python -m tools.prewarm_external --force         # 忽略缓存强制重取

## 输出

    data/events/ext/<code>.csv      事件类（大宗/龙虎榜/两融/增减持/解禁/预告）
    data/flow/...                   资金流（由 flow_factor 自己管）
    data/state/prewarm_status.json  进度（可被 UI / 定时任务读）
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STATE = os.path.join(ROOT, "data", "state")
STATUS_PATH = os.path.join(STATE, "prewarm_status.json")


def _codes(pool: int, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [str(c).zfill(6) for c in explicit]
    from qcore.data import get_pool
    return get_pool(top_n=pool)["code"].astype(str).tolist()


def run(pool: int = 300, codes: Optional[List[str]] = None,
        force: bool = False, verbose: bool = True) -> Dict:
    from qcore import event_data as ED
    from qcore import flow_factor as FF

    todo = _codes(pool, codes)
    t0 = time.time()
    ok_ev = ok_flow = fail = 0
    detail = []
    for i, code in enumerate(todo, 1):
        c6 = str(code).zfill(6)
        ev_ok = fl_ok = False
        try:
            ED.load_events(c6, refresh=force)
            ev_ok = True
        except Exception as e:  # noqa: BLE001
            detail.append({"code": c6, "stage": "events", "error": str(e)[:80]})
        try:
            FF.fetch_flow(c6, use_cache=not force)
            fl_ok = True
        except Exception as e:  # noqa: BLE001
            detail.append({"code": c6, "stage": "flow", "error": str(e)[:80]})
        ok_ev += int(ev_ok)
        ok_flow += int(fl_ok)
        if not (ev_ok or fl_ok):
            fail += 1
        if verbose and i % 20 == 0:
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)}  已用 {el:.0f}s  预计还要 {eta:.0f}s  "
                  f"（事件 {ok_ev} / 资金流 {ok_flow} / 全失败 {fail}）", flush=True)
            _write_status({"running": True, "done": i, "total": len(todo),
                           "ok_events": ok_ev, "ok_flow": ok_flow, "failed": fail,
                           "elapsed_s": round(el, 1)})
    out = {"ok": True, "total": len(todo), "ok_events": ok_ev,
           "ok_flow": ok_flow, "failed": fail,
           "seconds": round(time.time() - t0, 1), "errors": detail[:20]}
    _write_status({**out, "running": False,
                   "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return out


def _write_status(d: Dict) -> None:
    try:
        os.makedirs(STATE, exist_ok=True)
        tmp = STATUS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATUS_PATH)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    a = sys.argv[1:]
    pool = 300
    if "--pool" in a:
        pool = int(a[a.index("--pool") + 1])
    codes = None
    if "--codes" in a:
        codes = [x.strip() for x in a[a.index("--codes") + 1].split(",") if x.strip()]
    r = run(pool=pool, codes=codes, force=("--force" in a))
    print(json.dumps(r, ensure_ascii=False, indent=1))
