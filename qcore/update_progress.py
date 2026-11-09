"""
更新进度上报 —— 让用户知道「开始没开始、到哪了」。

## 为什么需要

之前点「确认更新」后只显示一个转圈和「约 1-3 分钟」，用户完全不知道：
  · 到底开始了吗
  · 在拉镜像还是在重建容器
  · 卡住了还是正常在跑
  · 失败了没

## 怎么做

辅助容器每做一步就写 `data/state/update_progress.json`：

    {"run_id": "...", "stage": "pulling", "stage_label": "拉取新镜像",
     "percent": 35, "started_at": "...", "updated_at": "...",
     "steps": [{"stage": ..., "at": ..., "ok": true, "detail": "..."}],
     "target": "v0.22.0", "error": null}

这个文件在 data 挂载里，所以**主容器被停掉后新容器还能读到**，
用户刷新页面就知道上次更新成功没有。

## 阶段

    queued    排队中（辅助容器还没起来）
    pulling   拉取新镜像
    stopping  停止旧容器
    removing  删除旧容器
    creating  创建新容器
    starting  启动新容器
    health    健康检查
    done      完成
    failed    失败（附 error）
    rolled_back 失败后已回滚
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "state", "update_progress.json")

# 阶段 -> (中文, 累计进度百分比)
STAGES = {
    "queued": ("排队中", 5),
    "pulling": ("拉取新镜像", 25),
    "stopping": ("停止旧容器", 55),
    "removing": ("清理旧容器", 62),
    "creating": ("创建新容器", 72),
    "starting": ("启动新容器", 82),
    "health": ("健康检查", 92),
    "done": ("更新完成", 100),
    "failed": ("更新失败", 100),
    "rolled_back": ("已回滚到旧版本", 100),
}

ORDER = ["queued", "pulling", "stopping", "removing",
         "creating", "starting", "health", "done"]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read() -> Dict:
    try:
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write(d: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, PATH)
    except Exception:
        pass


def begin(target: str, from_version: str = "", run_id: str = "") -> Dict:
    """开始一次更新。"""
    d = {
        "run_id": run_id or datetime.now().strftime("%Y%m%d%H%M%S"),
        "target": target, "from_version": from_version,
        "stage": "queued", "stage_label": STAGES["queued"][0],
        "percent": STAGES["queued"][1],
        "started_at": _now(), "updated_at": _now(),
        "finished_at": None, "error": None,
        "steps": [{"stage": "queued", "label": STAGES["queued"][0],
                   "at": _now(), "ok": True, "detail": ""}],
    }
    _write(d)
    return d


def step(stage: str, detail: str = "", ok: bool = True,
         error: Optional[str] = None) -> Dict:
    """推进到下一阶段。"""
    d = _read()
    if not d:
        return {}
    label, pct = STAGES.get(stage, (stage, d.get("percent") or 0))
    d["stage"] = stage
    d["stage_label"] = label
    d["percent"] = pct
    d["updated_at"] = _now()
    if error:
        d["error"] = error
    d.setdefault("steps", []).append(
        {"stage": stage, "label": label, "at": _now(),
         "ok": bool(ok), "detail": str(detail)[:300]})
    if stage in ("done", "failed", "rolled_back"):
        d["finished_at"] = _now()
    _write(d)
    return d


def finish(detail: str = "") -> Dict:
    return step("done", detail=detail)


def fail(error: str, detail: str = "") -> Dict:
    return step("failed", detail=detail, ok=False, error=str(error)[:300])


def read() -> Dict:
    """
    读进度，并算出可读状态：
      running   正在跑（updated_at 在 10 分钟内）
      stale     超过 10 分钟没更新 —— 可能辅助容器挂了
      done / failed / rolled_back
    没有记录时返回 {}。
    """
    d = _read()
    if not d:
        return {}
    stage = str(d.get("stage") or "")
    try:
        upd = datetime.fromisoformat(str(d.get("updated_at")))
        age = (datetime.now().astimezone() - upd).total_seconds()
    except Exception:
        age = 10 ** 6

    if stage in ("done", "finished"):
        state = "done"
    elif stage in ("failed",):
        state = "failed"
    elif stage in ("rolled_back",):
        state = "rolled_back"
    elif age < 600:
        state = "running"
    else:
        state = "stale"

    d["state"] = state
    d["age_seconds"] = int(age)
    # 预计剩余时间：按已用时间线性推算
    pct = float(d.get("percent") or 0)
    if state == "running" and pct > 0:
        try:
            s0 = datetime.fromisoformat(str(d.get("started_at")))
            used = (datetime.now().astimezone() - s0).total_seconds()
            d["elapsed_seconds"] = int(used)
            d["eta_seconds"] = int(max(0, used / (pct / 100.0) - used)) if pct < 100 else 0
        except Exception:
            d["elapsed_seconds"] = None
            d["eta_seconds"] = None
    return d


def clear() -> None:
    try:
        os.remove(PATH)
    except Exception:
        pass
