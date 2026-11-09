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
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


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


def run_task(nas: str, task: dict, token: str, verbose: bool = True) -> dict:
    """执行一个任务：训练 → 上传 → 回报。"""
    from qcore import live_model as LM
    hs = task.get("horizons") or [1]
    trees = int(task.get("trees") or 250)
    stocks = int(task.get("stocks") or 300)
    tid = task.get("id")

    if verbose:
        print(f"  任务 {tid}: horizons={hs} trees={trees} stocks={stocks}", flush=True)

    results = []
    for h in hs:
        t0 = time.time()
        try:
            r = LM.train(horizon=int(h), trees=trees, stocks=stocks, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:80]}"}
        dt = time.time() - t0
        results.append({"horizon": int(h), "ok": bool(r.get("ok")),
                        "seconds": round(dt, 1),
                        "reason": r.get("reason")})
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
    _req(f"{nas}/api/sync/result", "POST",
         {"id": tid, "ok": ok_all, "worker": _worker_name(),
          "detail": {"train": results, "push": pushed.get("pushed", [])}},
         token, timeout=60)
    return {"ok": ok_all, "train": results, "push": pushed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="APB 同步代理（跑在训练机上）")
    ap.add_argument("--nas", required=True, help="NAS 地址，如 http://192.168.250.3:8765")
    ap.add_argument("--token", default=os.environ.get("APB_TOKEN", ""),
                    help="访问 token（NAS 设了才需要）")
    ap.add_argument("--once", action="store_true", help="领一个任务就退出")
    ap.add_argument("--loop", type=int, default=0, help="常驻，每隔 N 秒轮询")
    ap.add_argument("--push-only", action="store_true", help="不领任务，直接推送现有模型")
    ap.add_argument("--horizons", default="", help="--push-only 时指定，如 1,5")
    a = ap.parse_args(argv)

    nas = a.nas.rstrip("/")
    token = a.token or ""
    worker = _worker_name()

    # 连通性
    h = _req(f"{nas}/api/version", token=token, timeout=15)
    if not h.get("ok"):
        print(f"[错误] 连不上 {nas}：{h.get('error')}")
        print("       检查地址、端口、以及 NAS 是否在跑")
        return 1
    print(f"已连接 {nas}  版本 {h.get('data', {}).get('version', '?')}")

    if a.push_only:
        hs = [int(x) for x in a.horizons.split(",") if x.strip()] or [1, 5]
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
        res = run_task(nas, t, token)
        print(f"  任务完成：{'全部成功' if res.get('ok') else '有失败'}")
        return True

    if a.loop and a.loop > 0:
        print(f"常驻模式，每 {a.loop} 秒轮询一次（Ctrl+C 退出）")
        while True:
            try:
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
