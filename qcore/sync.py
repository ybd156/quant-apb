"""
控制面 —— NAS 下发任务，训练机领取、执行、回传。模型也能直接推上来。

## 解决什么问题

之前是纯手工：
    别的电脑训练 → 导出 zip → scp → docker cp → 容器里 import

每一步都可能出错，而且你不在电脑前就干不了。本模块把它变成：

    NAS 上排一个任务  →  训练机（或本机）自动领取 → 训练 → 自动上传 → 结果回传

## 两种角色

**NAS（服务端）**
- `POST /api/sync/task`    排一个训练任务（horizons / trees / stocks）
- `GET  /api/sync/task`    查看队列
- `POST /api/sync/push`    直接上传模型（base64 zip），等价于 model/import

**训练机（客户端）**
- `python tools/sync_agent.py --nas http://192.168.250.3:8765 --once`
  领一个任务 → 训练 → 上传模型 → 回报

## 安全

所有写操作要 `X-APB-Token`。token 从 `APB_TOKEN` 环境变量读；
**未设置时只允许内网访问**（和现有 /api/run 同一策略）。

## 状态

任务队列落 `data/state/sync_tasks.json`，只追加不改历史（INV-7 精神）。
每个任务有 id / 状态 / 领取时间 / 结果，超时未回传的标 `expired`。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
TASKS = os.path.join(STATE, "sync_tasks.json")

# 任务超过这么久没回传就算过期（分钟）
EXPIRE_MIN = 240

#: 按任务类型设领取超时（分钟）。一刀切 4 小时的问题：
#: pipeline 只要 20 分钟，卡 4 小时才标过期 → 用户看到「已领走」以为在跑，
#: 实际早就死了（实测 98ac761c229d 卡了 3 小时还是 claimed）。
EXPIRE_MIN_BY_JOB = {"train": 240, "pipeline": 90, "prewarm": 90}

#: 失败/超时后允许重领的次数。没有它，一条任务挂掉就是永久废掉，
#: 用户只能重新排一条 —— 队列里 6 条 expired 就是这么堆出来的。
MAX_RETRIES = 2


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load() -> Dict:
    try:
        with open(TASKS, encoding="utf-8") as f:
            d = json.load(f)
            if isinstance(d, dict) and "tasks" in d:
                return d
    except Exception:
        pass
    return {"tasks": [], "updated_at": None}


def _save(d: Dict) -> None:
    d["updated_at"] = _now()
    _atomic(TASKS, d)


def _expire(d: Dict) -> int:
    """把领取超时的任务标成 expired；还有重试次数的退回 pending。

    返回被处理的任务数。
    """
    n = 0
    now = datetime.now().astimezone()
    for t in d.get("tasks", []):
        if t.get("status") != "claimed":
            continue
        job = str(t.get("job") or "train")
        win = int(EXPIRE_MIN_BY_JOB.get(job, EXPIRE_MIN))
        try:
            at = datetime.fromisoformat(str(t.get("claimed_at")))
        except Exception:
            continue
        if at >= now - timedelta(minutes=win):
            continue
        tries = int(t.get("retries") or 0)
        if tries < MAX_RETRIES:
            # 退回队列重领，而不是直接判死 —— 一次网络抖动不该废掉整条任务
            t["status"] = "pending"
            t["retries"] = tries + 1
            t["claimed_by"] = None
            t["claimed_at"] = None
            t["last_expired_at"] = _now()
            t["last_expire_reason"] = f"claimed 超过 {win} 分钟未回报（第 {tries + 1} 次重试）"
        else:
            t["status"] = "expired"
            t["expired_at"] = _now()
            t["expire_reason"] = f"已重试 {tries} 次仍未回报"
        n += 1
    return n


# ---------------------------------------------------------------- 任务
def create_task(horizons: List[int], trees: int = 250, stocks: int = 300,
                note: str = "", created_by: str = "api",
                job: str = "train", params: Dict = None) -> Dict:
    """排一个任务。`job` 决定训练机干什么：

      train     —— 训练模型（默认）
      pipeline  —— 跑全流程回测（`python -m qcore.pipeline`），用于测信号效果
      prewarm   —— 预热外部数据缓存（`tools.prewarm_external`）

    为什么要有 pipeline：102 个信号的完整回测在本机要几分钟到几十分钟，
    而训练机本来就是干重活的 —— 把它外包出去，NAS 只负责取数与展示。
    """
    d = _load()
    _expire(d)
    tid = uuid.uuid4().hex[:12]
    task = {
        "id": tid,
        "job": str(job or "train")[:20],
        "params": dict(params or {}),
        "horizons": [int(h) for h in horizons],
        "trees": int(trees), "stocks": int(stocks),
        "note": str(note)[:200],
        "created_by": created_by, "created_at": _now(),
        "status": "pending", "claimed_by": None, "claimed_at": None,
        "finished_at": None, "result": None, "retries": 0,
    }
    d.setdefault("tasks", []).append(task)
    _save(d)
    return {"ok": True, "task": task}


def set_progress(task_id: str, phase: str = "", horizon=None, done=None,
                 total=None, note: str = "", eta_s=None) -> Dict:
    """更新任务进度。**只写进度字段，不改状态机**。

    ## 为什么需要它

    原来任务只有 pending/claimed/done/expired 五个状态，
    训练机领走之后界面上就只剩「claimed」—— 用户完全不知道
    训到第几个周期、还剩多久（用户原话：「训练到哪都不知道」）。

    进度是**可观测性**数据，不是状态：写失败也不能影响任务本身，
    所以这里永远返回 dict，不抛异常给调用方。
    """
    try:
        d = _load()
        now = _now()
        for t in d.get("tasks", []):
            if t.get("id") != task_id:
                continue
            p = t.get("progress") or {}
            p.update({"updated_at": now})
            if phase:
                p["phase"] = str(phase)[:40]
            if horizon is not None:
                p["horizon"] = int(horizon)
            if done is not None:
                p["done"] = int(done)
            if total is not None:
                p["total"] = int(total)
            if note:
                p["note"] = str(note)[:200]
            if eta_s is not None:
                p["eta_s"] = int(eta_s)
            if p.get("total"):
                p["pct"] = round(100.0 * float(p.get("done") or 0) / p["total"], 1)
            t["progress"] = p
            _save(d)
            return {"ok": True, "progress": p}
        return {"ok": False, "reason": "not_found", "id": task_id}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:80]}"}


def claim_task(worker: str = "unknown") -> Dict:
    """领一个待办任务（FIFO）。没有就返回 pending=0。"""
    d = _load()
    _expire(d)
    for t in d.get("tasks", []):
        if t.get("status") == "pending":
            t["status"] = "claimed"
            t["claimed_by"] = str(worker)[:60]
            t["claimed_at"] = _now()
            _save(d)
            return {"ok": True, "task": t}
    _save(d)
    return {"ok": True, "task": None, "pending": 0,
            "note": "队列为空，等 NAS 排新任务"}


def report_task(task_id: str, ok: bool, detail: Dict = None,
                worker: str = "") -> Dict:
    """训练机回传结果。"""
    d = _load()
    for t in d.get("tasks", []):
        if t.get("id") == task_id:
            t["status"] = "done" if ok else "failed"
            t["finished_at"] = _now()
            t["result"] = detail or {}
            if worker:
                t["claimed_by"] = str(worker)[:60]
            _save(d)
            return {"ok": True, "task": t}
    return {"ok": False, "reason": "task_not_found", "id": task_id}


def list_tasks(limit: int = 30) -> Dict:
    d = _load()
    _expire(d)
    _save(d)
    ts = list(reversed(d.get("tasks", [])))[:limit]
    counts = {}
    for t in d.get("tasks", []):
        counts[t.get("status")] = counts.get(t.get("status"), 0) + 1
    return {"ok": True, "tasks": ts, "counts": counts,
            "total": len(d.get("tasks", [])),
            "expire_min": EXPIRE_MIN,
            "expire_min_by_job": EXPIRE_MIN_BY_JOB,
            "max_retries": MAX_RETRIES}


def cancel_task(task_id: str) -> Dict:
    d = _load()
    for t in d.get("tasks", []):
        if t.get("id") == task_id and t.get("status") in ("pending", "claimed"):
            t["status"] = "cancelled"
            t["finished_at"] = _now()
            _save(d)
            return {"ok": True, "task": t}
    return {"ok": False, "reason": "not_cancellable"}


def summary() -> Dict:
    r = list_tasks(limit=10)
    return {"ok": True, "counts": r["counts"], "total": r["total"],
            "recent": [{"id": t["id"], "status": t["status"],
                        "horizons": t.get("horizons"), "trees": t.get("trees"),
                        "claimed_by": t.get("claimed_by"),
                        "created_at": t.get("created_at"),
                        "finished_at": t.get("finished_at")}
                       for t in r["tasks"][:10]]}


if __name__ == "__main__":
    import sys
    if "--add" in sys.argv:
        hs = [1, 5]
        for a in sys.argv:
            if a.startswith("--horizons="):
                hs = [int(x) for x in a.split("=", 1)[1].split(",")]
        print(json.dumps(create_task(hs), ensure_ascii=False, indent=1))
    elif "--list" in sys.argv:
        print(json.dumps(list_tasks(), ensure_ascii=False, indent=1)[:1500])
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
