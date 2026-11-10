#!/usr/bin/env python3
"""
同步代理 —— 跑在训练机上，自动领取 NAS 下发的任务、训练、上传。

## 用法

    # 领一个任务就退出（适合手动跑）
    python tools/sync_agent.py --nas http://192.168.250.3:8765 --once

    # 常驻，每 60 秒问一次
    python tools/sync_agent.py --nas http://192.168.250.3:8765 --loop 60

    # 不走任务队列，直接把现有模型推上去
    python tools/sync_agent.py --nas http://192.168.250.3:8765 --push-only

## 它做什么

    1. GET  /api/sync/task        领任务（没有就等）
    2. 本地训练指定 horizons
    3. POST /api/sync/push        上传模型 zip（服务端做四项校验）
    4. POST /api/sync/result      回报成功/失败

## Token

NAS 设了 APB_TOKEN 就要带 `--token`；只在内网用可以不设。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
from datetime import datetime
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _req(url: str, method: str = "GET", payload=None, token: str = "",
         timeout: int = 60) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json", "User-Agent": "apb-sync-agent"}
    if data:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-APB-Token"] = token
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return json.loads(body) if body.strip() else {"ok": True}
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8", "replace"))
        except Exception:
            return {"ok": False, "error": f"http_{e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def _worker_name() -> str:
    try:
        return f"{socket.gethostname()}-{os.getpid()}"
    except Exception:
        return f"worker-{os.getpid()}"


def push_models(nas: str, horizons, token: str) -> dict:
    """把本地指定 horizon 的模型导出并上传。"""
    from qcore import model_transfer as MT
    pushed = []
    for h in horizons:
        ex = MT.export_bundle(int(h))
        if not ex.get("ok"):
            pushed.append({"horizon": h, "ok": False, "reason": ex.get("reason")})
            continue
        with open(ex["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        r = _req(f"{nas}/api/sync/push", "POST",
                 {"data": b64, "horizon": int(h)}, token, timeout=120)
        pushed.append({"horizon": h, "ok": bool(r.get("ok")),
                       "reason": r.get("error") or r.get("detail"),
                       "size_kb": ex.get("size_mb", 0) * 1024})
    return {"ok": all(p["ok"] for p in pushed) if pushed else False,
            "pushed": pushed}


def _journal(rec: dict) -> None:
    """训练机本地流水（JSONL）。服务端看不到机器，出了事只能靠这本账。"""
    try:
        import os as _os
        d = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                          "data", "state")
        _os.makedirs(d, exist_ok=True)
        rec = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), **rec}
        with open(_os.path.join(d, "train_journal.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + chr(10))
    except Exception:  # noqa: BLE001
        pass


def _report(nas: str, tid, ok: bool, detail: dict, token: str,
            retries: int = 3, verbose: bool = True) -> bool:
    """把结果回报给服务端 —— **必须送达**。

    实测事故：任务 11-09 19:01 被领走，模型 19:14 已上传成功，
    但 `finished_at` 一直是 null、状态最后变 `expired`。
    服务端因此「根本不知道训练机训了什么」。
    旧实现只在上传成功后顺序调用一次 `_req`，任何异常/网络抖动都会让回报丢失，
    而 `_one()` 的 `except` 只打印一行就继续下一轮。
    """
    payload = {"id": tid, "ok": bool(ok), "worker": _worker_name(), "detail": detail or {}}
    for i in range(max(1, retries)):
        try:
            r = _req(f"{nas}/api/sync/result", "POST", payload, token, timeout=60)
            if r.get("ok"):
                _journal({"event": "reported", "task": tid, "ok": bool(ok), "attempt": i + 1})
                return True
            if verbose:
                print(f"    回报被拒：{r.get('error') or r.get('detail')}", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"    回报失败（第 {i+1}/{retries} 次）：{type(e).__name__}: {str(e)[:60]}",
                      flush=True)
        time.sleep(min(2 ** i, 8))
    _journal({"event": "report_failed", "task": tid, "ok": bool(ok), "detail": detail})
    if verbose:
        print(f"    ⚠ 结果未能回传服务端（已写入 data/state/train_journal.jsonl）", flush=True)
    return False


def run_task(nas: str, task: dict, token: str, verbose: bool = True) -> dict:
    """执行一个任务：训练 → 上传 → 回报。"""
    from qcore import live_model as LM
    hs = task.get("horizons") or [1]
    trees = int(task.get("trees") or 250)
    stocks = int(task.get("stocks") or 300)
    tid = task.get("id")

    if verbose:
        print(f"  任务 {tid}: horizons={hs} trees={trees} stocks={stocks}", flush=True)

    _journal({"event": "claimed", "task": tid, "horizons": hs,
              "trees": trees, "stocks": stocks})
    results = []
    for h in hs:
        t0 = time.time()
        try:
            r = LM.train(horizon=int(h), trees=trees, stocks=stocks, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:80]}"}
        dt = time.time() - t0
        # 服务端排的规模 vs 实际训出来的规模：不一致必须显式记录，
        # 否则「排 300 只、训 200 只」这种事在产物里看不出来（实测已发生）。
        _got = r.get("n_stocks")
        rec = {"horizon": int(h), "ok": bool(r.get("ok")),
               "seconds": round(dt, 1), "reason": r.get("reason"),
               "requested_stocks": int(stocks), "n_stocks": _got}
        if r.get("ok") and _got is not None and int(_got) < int(stocks):
            rec["stocks_shortfall"] = int(stocks) - int(_got)
        results.append(rec)
        _journal({"event": "trained", **rec})
        if verbose:
            mark = "OK" if r.get("ok") else "FAIL"
            print(f"    h={h} [{mark}] {dt/60:.1f} 分钟"
                  + ("" if r.get("ok") else f"  {r.get('reason')}"), flush=True)

    good = [x["horizon"] for x in results if x["ok"]]
    pushed = {"pushed": []}
    if good:
        pushed = push_models(nas, good, token)
        if verbose:
            for p in pushed.get("pushed", []):
                print(f"    上传 h={p['horizon']}: "
                      f"{'OK' if p['ok'] else 'FAIL ' + str(p.get('reason'))}", flush=True)

    ok_all = bool(good) and pushed.get("ok", False)
    _report(nas, tid, ok_all, {"train": results, "push": pushed.get("pushed", [])},
            token, verbose=verbose)
    return {"ok": ok_all, "train": results, "push": pushed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="APB 同步代理（跑在训练机上）")
    ap.add_argument("--nas", default="",
                    help="NAS 地址；不填就用 apb_defaults.json 里的，再没有就用 "
                         "https://apbcdn.683868.xyz")
    ap.add_argument("--token", default=os.environ.get("APB_TOKEN", ""),
                    help="访问 token（NAS 设了才需要）")
    ap.add_argument("--once", action="store_true", help="领一个任务就退出")
    ap.add_argument("--loop", type=int, default=0, help="常驻，每隔 N 秒轮询")
    ap.add_argument("--push-only", action="store_true", help="不领任务，直接推送现有模型")
    ap.add_argument("--horizons", default="", help="--push-only 时指定，如 1,5")
    ap.add_argument("--auto-update", action="store_true",
                    help="每轮先检查服务端版本，比自己新就自动覆盖代码")
    ap.add_argument("--update-only", action="store_true",
                    help="只检查更新，不领任务")
    a = ap.parse_args(argv)

    try:
        import defaults as _DEF
    except Exception:
        _DEF = None
    nas = (a.nas or (_DEF.nas() if _DEF else "https://apbcdn.683868.xyz")).rstrip("/")
    token = a.token or ""
    worker = _worker_name()

    # 连通性
    h = _req(f"{nas}/api/version", token=token, timeout=15)
    if not h.get("ok"):
        print(f"[错误] 连不上 {nas}：{h.get('error')}")
        print("       检查地址、端口、以及 NAS 是否在跑")
        return 1
    print(f"已连接 {nas}  版本 {h.get('data', {}).get('version', '?')}")

    def _maybe_update() -> None:
        """服务端下发更新：版本比自己新就覆盖代码（不动 data/dist/runtime）。"""
        if not (a.auto_update or a.update_only):
            return
        try:
            from qcore import kit_update as KU
            r = KU.check_and_update(nas, token, verbose=True)
            if r.get("action") == "updated":
                print(f"  已更新到 {r.get('to')}（重启后生效）")
        except Exception as e:  # noqa: BLE001
            print(f"  更新检查失败：{type(e).__name__}: {str(e)[:60]}")

    _maybe_update()
    if a.update_only:
        return 0

    if a.push_only:
        hs = ([int(x) for x in a.horizons.split(",") if x.strip()]
              or (_DEF.horizons() if _DEF else [1, 5]))
        print(f"推送 horizons={hs} …")
        r = push_models(nas, hs, token)
        for p in r.get("pushed", []):
            print(f"  h={p['horizon']}: {'OK' if p['ok'] else 'FAIL ' + str(p.get('reason'))}")
        return 0 if r.get("ok") else 1

    def _one() -> bool:
        r = _req(f"{nas}/api/sync/task", "POST",
                 {"worker": worker, "claim": True}, token, timeout=30)
        t = r.get("task")
        if not t:
            print("  队列为空，等 NAS 排任务（网页「设置 → 训练任务」）")
            return False
        print(f"领到任务 {t.get('id')}")
        try:
            res = run_task(nas, t, token)
        except Exception as e:  # noqa: BLE001
            # 领了任务却崩掉 = 服务端永远等不到结果，最后变成 expired。
            # 必须回报失败，让队列状态诚实地显示「失败」而不是「过期」。
            print(f"  任务异常：{type(e).__name__}: {str(e)[:120]}", flush=True)
            _journal({"event": "crashed", "task": t.get("id"),
                      "error": f"{type(e).__name__}: {str(e)[:200]}"})
            _report(nas, t.get("id"), False,
                    {"error": f"{type(e).__name__}: {str(e)[:200]}"}, token)
            return True
        print(f"  任务完成：{'全部成功' if res.get('ok') else '有失败'}")
        return True

    if a.loop and a.loop > 0:
        print(f"常驻模式，每 {a.loop} 秒轮询一次（Ctrl+C 退出）")
        while True:
            try:
                _maybe_update()
                _one()
            except KeyboardInterrupt:
                print("\n退出")
                return 0
            except Exception as e:  # noqa: BLE001
                print(f"  异常：{type(e).__name__}: {str(e)[:80]}")
            time.sleep(a.loop)

    _one()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
