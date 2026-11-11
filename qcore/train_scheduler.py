"""
训练任务调度器 —— 让**服务端自己决定**什么时候该训、训哪个周期。

## 现状（为什么要它）

现在训练任务只有一个人工入口：`POST /api/sync/task`。也就是说：

- 模型 7 天没重训，没人点按钮 → 就一直用旧模型；
- 数据已经走到 11-11，模型训练截止还停在 2026-10-10 → 没人发现；
- 某个周期的模型退化（IC 掉到 0 以下）→ 没有任何东西会去修。

`qcore.sync` 有队列、`qcore.model_health` 有健康度、`tools/sync_agent.py` 会领任务，
**唯独缺"谁来判断该不该训"这一层**。本模块补的就是它。

## 判据（全部可解释，写在产物里）

对每个支持的周期 h：

| 条件 | 动作 | 优先级 |
|---|---|---|
| 没有模型文件（`meta_h{h}.json` 缺失） | 训练 | 1 |
| `model_health.level in (failed, degraded)` | 训练 | 1 |
| 模型超龄（`trained_at` 超过 `MAX_AGE_DAYS`） | 训练 | 2 |
| 数据比训练截止新 ≥ `STALE_TRADING_DAYS` 个交易日 | 训练 | 2 |
| 队列里已有该周期的 pending/claimed 任务 | 跳过（不重复排） | — |
| 距离上次为该周期排任务 < `COOLDOWN_HOURS` | 跳过（防抖） | — |
| 本轮已排满 `MAX_ENQUEUE` 个 | 停止 | — |

## 产物

`data/state/train_plan.json`（UI 只读）：

```json
{"generated_at": "...", "items": [{"horizon": 5, "action": "train",
  "reason": "模型超龄 9 天", "priority": 2, "enqueued": true, "task_id": "..."}],
 "summary": {"checked": 3, "enqueued": 1, "skipped": 2}}
```

## 用法

    python -m qcore.train_scheduler --plan          # 只看判断，不排任务
    python -m qcore.train_scheduler                 # 判断 + 排任务
    python -m qcore.train_scheduler --dry-run       # 等价于 --plan
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
MODEL_DIR = os.path.join(ROOT, "data", "model")
PLAN_PATH = os.path.join(STATE_DIR, "train_plan.json")

#: 模型超过这么多天没重训 → 排一单
MAX_AGE_DAYS = 7
#: 数据比训练截止新这么多交易日 → 排一单（太小会天天训）
STALE_TRADING_DAYS = 3
#: 同一周期两次"排任务"的最小间隔（小时），防抖
COOLDOWN_HOURS = 12
#: 单轮最多排几个任务（别把训练机一次塞满）
MAX_ENQUEUE = 2
#: 默认支持的周期（与 horizons.HORIZONS 对齐，这里只排真正在用的）
HORIZONS = (1, 5)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _age_days(stamp: Optional[str]) -> Optional[float]:
    if not stamp:
        return None
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.astimezone()
        return (datetime.now().astimezone() - t).total_seconds() / 86400.0
    except Exception:  # noqa: BLE001
        return None


def _trading_lag(newer: Optional[str], older: Optional[str]) -> Optional[int]:
    """older → newer 之间隔了几个交易日（取不到日历就退化为自然日//1.4）。"""
    if not newer or not older:
        return None
    try:
        from . import calendar_cn as CAL
        d0 = datetime.strptime(str(older)[:10], "%Y-%m-%d").date()
        d1 = datetime.strptime(str(newer)[:10], "%Y-%m-%d").date()
        if d1 <= d0:
            return 0
        n, cur = 0, d0
        while cur < d1:
            cur = CAL.next_trading_day(cur)
            n += 1
            if n > 400:
                break
        return n
    except Exception:  # noqa: BLE001
        try:
            d0 = datetime.strptime(str(older)[:10], "%Y-%m-%d")
            d1 = datetime.strptime(str(newer)[:10], "%Y-%m-%d")
            return max(0, int((d1 - d0).days * 5 / 7))
        except Exception:  # noqa: BLE001
            return None


def _model_meta(horizon: int) -> Dict:
    name = "meta.json" if int(horizon) == 1 else f"meta_h{int(horizon)}.json"
    return _read(os.path.join(MODEL_DIR, name), {}) or {}


def _pending_horizons() -> Dict[int, str]:
    """队列里已经在等/在跑的周期 → {horizon: task_id}。"""
    out: Dict[int, str] = {}
    try:
        from . import sync as SY
        d = SY.list_tasks(limit=100) or {}
        for t in (d.get("tasks") or []):
            if t.get("status") not in ("pending", "claimed"):
                continue
            if t.get("job") not in (None, "", "train"):
                continue
            for h in (t.get("horizons") or []):
                out.setdefault(int(h), t.get("id"))
    except Exception:  # noqa: BLE001
        pass
    return out


#: 双轨复核多久重跑一次（天）。它是重活（每期 5–8 分钟 × 2 口径），不必频繁。
AB_EVERY_DAYS = 30


def _ab_last() -> Optional[str]:
    """上一次双轨复核的时间（读训练机回传的 ab_train.json）。"""
    d = _read(os.path.join(STATE_DIR, "ab_train.json"), {}) or {}
    return d.get("generated_at")


def _ab_due(now=None) -> bool:
    """双轨复核该不该排：没跑过 → 该；跑过但太旧 → 该。"""
    last = _ab_last()
    if not last:
        return True
    age = _age_days(last)
    return age is None or age > AB_EVERY_DAYS


def _last_enqueued() -> Dict[int, str]:
    """上次为某周期排任务的时间（读 train_plan.json，用于冷却）。"""
    d = _read(PLAN_PATH, {}) or {}
    out: Dict[int, str] = {}
    for it in (d.get("items") or []):
        if it.get("enqueued") and it.get("at"):
            out[int(it.get("horizon"))] = str(it.get("at"))
    return out


def plan(horizons: Optional[List[int]] = None) -> Dict:
    """只做判断，不排任务。返回 items + summary。"""
    hs = [int(h) for h in (horizons or HORIZONS)]
    health = {}
    try:
        from . import model_health as MH
        health = MH.summary() or {}
    except Exception:  # noqa: BLE001
        health = {}
    level = str(health.get("level") or "unknown")

    data_date = None
    try:
        from . import data_sync as DS
        data_date = DS.local_date()
    except Exception:  # noqa: BLE001
        data_date = None

    pending = _pending_horizons()
    last = _last_enqueued()
    now = datetime.now().astimezone()

    items: List[Dict] = []
    for h in hs:
        meta = _model_meta(h)
        age = _age_days(meta.get("trained_at") or meta.get("saved_at"))
        lag = _trading_lag(data_date, meta.get("train_end"))
        item = {"horizon": h, "action": "skip", "reason": "", "priority": 9,
                "model_age_days": None if age is None else round(age, 1),
                "train_end": meta.get("train_end"), "data_date": data_date,
                "data_lag_trading_days": lag}

        if not meta:
            item.update(action="train", priority=1, reason="该周期还没有模型")
        elif level in ("failed", "degraded"):
            item.update(action="train", priority=1,
                        reason=f"模型健康度 {level}（{health.get('note') or ''}"[:60] + "）")
        elif age is not None and age > MAX_AGE_DAYS:
            item.update(action="train", priority=2,
                        reason=f"模型超龄 {age:.0f} 天（阈值 {MAX_AGE_DAYS}）")
        elif lag is not None and lag >= STALE_TRADING_DAYS:
            item.update(action="train", priority=2,
                        reason=f"数据比训练截止新 {lag} 个交易日（阈值 {STALE_TRADING_DAYS}）")
        else:
            item.update(reason="模型新鲜且数据未落后")

        if item["action"] == "train":
            if h in pending:
                item.update(action="skip", reason=f"队列里已有该周期任务 {pending[h]}")
            else:
                _ln = last.get(h)
                try:
                    _lt = datetime.fromisoformat(_ln) if _ln else None
                    if _lt and _lt.tzinfo is None:
                        _lt = _lt.astimezone()
                    if _lt and (now - _lt) < timedelta(hours=COOLDOWN_HOURS):
                        item.update(action="skip",
                                    reason=f"冷却中（{COOLDOWN_HOURS}h 内已排过）")
                except Exception:  # noqa: BLE001
                    pass
        items.append(item)

    # 双轨复核（决策 1C）：与 train 分开判，不占 MAX_ENQUEUE 名额。
    ab_item = {"kind": "train_ab", "action": "skip",
               "reason": f"上次双轨复核 {_ab_last() or '从未跑过'}"}
    if _ab_due():
        if "ab" in {str(k) for k in pending.values()}:
            ab_item["reason"] = "队列里已有双轨复核任务"
        else:
            ab_item.update(action="train_ab", priority=2,
                           reason=("尚未做过双轨复核" if not _ab_last()
                                   else f"上次复核超过 {AB_EVERY_DAYS} 天"))
    items.append(ab_item)

    todo = [i for i in items if i["action"] == "train"]
    todo.sort(key=lambda x: x["priority"])
    return {"ok": True, "generated_at": _now(),
            "items": items,
            "queue_waiting": {str(k): v for k, v in pending.items()},
            "model_health": level,
            "summary": {"checked": len(items),
                        "want_train": len(todo),
                        "will_enqueue": min(len(todo), MAX_ENQUEUE),
                        "skipped": len(items) - len(todo)}}


def run(dry_run: bool = False, verbose: bool = True) -> Dict:
    """判断 + （可选）真的排任务。写 data/state/train_plan.json 供 UI 读。"""
    p = plan()
    enq: List[Dict] = []
    n = 0
    for it in p["items"]:
        if it["action"] == "train_ab" and not dry_run:
            try:
                from . import sync as SY
                r = SY.create_task([5], trees=250, stocks=300, job="train_ab",
                                   note="[auto] 双轨复核：快照 vs 点对点",
                                   created_by="train_scheduler",
                                   params={"horizon": 5, "stocks": 300, "trees": 250})
                it["enqueued"] = bool(r.get("ok"))
                it["task_id"] = (r.get("task") or {}).get("id")
            except Exception as e:  # noqa: BLE001
                it["enqueued"] = False
                it["error"] = f"{type(e).__name__}: {str(e)[:80]}"
            it["at"] = _now()
            continue
        if it["action"] != "train":
            continue
        if n >= MAX_ENQUEUE:
            it["reason"] += "；本轮名额已满"
            continue
        if dry_run:
            it["enqueued"] = False
            it["would_enqueue"] = True
            n += 1
            continue
        try:
            from . import sync as SY
            r = SY.create_task([int(it["horizon"])], trees=250, stocks=300,
                               note=f"[auto] {it['reason']}"[:200], created_by="train_scheduler",
                               job="train")
            ok = bool(r.get("ok"))
            it["enqueued"] = ok
            it["task_id"] = (r.get("task") or {}).get("id")
            if ok:
                n += 1
                enq.append(it)
        except Exception as e:  # noqa: BLE001
            it["enqueued"] = False
            it["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        it["at"] = _now()

    out = {**p, "dry_run": bool(dry_run),
           "summary": {**p["summary"], "enqueued": n},
           "generated_at": _now()}
    # 保留上一轮的 at（冷却判据要用），所以读旧文件合并
    old = _read(PLAN_PATH, {}) or {}
    if not dry_run:
        merged_items = []
        seen = set()
        for it in (out.get("items") or []):
            merged_items.append(it)
            if it.get("horizon") is not None:
                seen.add(int(it["horizon"]))
        for it in (old.get("items") or []):
            # 双轨复核条目没有 horizon，不参与"按周期冷却"，跳过
            if it.get("horizon") is None:
                continue
            if it.get("enqueued") and int(it["horizon"]) not in seen:
                merged_items.append(it)          # 旧的成功记录留着给冷却用
        out["items"] = merged_items
    try:
        _atomic(PLAN_PATH, out)
    except Exception:  # noqa: BLE001
        pass

    if verbose:
        print(f"[train_scheduler] checked={out['summary']['checked']} "
              f"want={out['summary']['want_train']} enqueued={n}"
              f"{' (dry-run)' if dry_run else ''}", flush=True)
        for it in out["items"]:
            _tag = ("ab  " if it.get("kind") == "train_ab" else f"h={it.get('horizon')}")
            print(f"  {_tag:<5} {it['action']:<8} {it.get('reason','')}", flush=True)
    return out


def summary() -> Dict:
    d = _read(PLAN_PATH, {}) or {}
    return {"ok": bool(d), "generated_at": d.get("generated_at"),
            "items": d.get("items") or [],
            "summary": d.get("summary") or {},
            "note": "训练任务调度器的最近一次判断（服务端自己决定该不该训）"}


if __name__ == "__main__":
    import sys
    args = set(sys.argv[1:])
    if "--plan" in args or "--dry-run" in args:
        print(json.dumps(run(dry_run=True), ensure_ascii=False, indent=1)[:2000])
    elif "--summary" in args:
        print(json.dumps(summary(), ensure_ascii=False, indent=1)[:1500])
    else:
        print(json.dumps(run(), ensure_ascii=False, indent=1)[:2000])
