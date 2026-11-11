# 25 · 训练任务自动调度（服务端自己排单）

> 模块：`qcore/train_scheduler.py`　测试：`tests/test_train_scheduler.py`（10 条断言）
> 产物：`data/state/train_plan.json`　接口：`GET /api/train/plan`（只读）
> 调度：`docker/start.sh` §6.6，每小时一次（`APB_TRAIN_AUTO=1`）

---

## 一、它解决什么问题

在此之前，训练任务只有**一个人工入口**：`POST /api/sync/task`。于是三件事没人管：

| 现象 | 后果 |
|---|---|
| 模型 7 天没重训 | 一直在用旧模型，没有任何东西会提醒或修复 |
| 数据已到 11-11、模型训练截止停在 2026-10-10 | 模型"看起来正常"，实际用的是 22 个交易日前的数据 |
| 某周期模型退化（IC 转负） | 没有自动修复路径 |

`qcore.sync` 有队列，`qcore.model_health` 有健康度，`tools/sync_agent.py` 会领任务，
**缺的是"谁来判断该不该训"这一层** —— 本模块补的正是它。

---

## 二、判据（全部可解释、全部写进产物）

对每个支持的周期 `h`（默认 1 和 5）：

| 顺序 | 条件 | 动作 | 优先级 |
|---|---|---|---|
| 1 | 没有模型文件（`meta_h{h}.json` / `meta.json` 缺失） | 训练 | **1** |
| 2 | `model_health.level ∈ {failed, degraded}` | 训练 | **1** |
| 3 | 模型超龄：`trained_at` 距今 > `MAX_AGE_DAYS=7` 天 | 训练 | **2** |
| 4 | 数据落后：`data_sync.local_date()` 比 `train_end` 新 ≥ `STALE_TRADING_DAYS=3` 个交易日 | 训练 | **2** |
| — | 队列里已有该周期的 `pending` / `claimed` 任务 | **跳过**（不重复排） | — |
| — | 距上次为该周期排单 < `COOLDOWN_HOURS=12` 小时 | **跳过**（防抖） | — |
| — | 本轮已排满 `MAX_ENQUEUE=2` 个 | 停止 | — |

> 为什么阈值是 3 个交易日而不是 1 天：`daily_append` 每天都会把数据往前推，
> 阈值取 1 会变成"天天训"。3 天 ≈ 每周两次，和 `LIVE_MAX_AGE=7` 形成双保险。

同一周期被判定要训时，还会记录**为什么**（`reason`），例如：

```
数据比训练截止新 22 个交易日（阈值 3）
模型超龄 9 天（阈值 7）
模型健康度 degraded（近 20 日 IC 均值 -0.004）
该周期还没有模型
```

---

## 三、产物 `data/state/train_plan.json`

```json
{
  "generated_at": "2026-11-11T19:04:52+08:00",
  "items": [
    {"horizon": 1, "action": "train", "priority": 2,
     "reason": "数据比训练截止新 22 个交易日（阈值 3）",
     "model_age_days": 2.3, "train_end": "2026-10-10",
     "data_date": "2026-11-10", "data_lag_trading_days": 22,
     "enqueued": true, "task_id": "7e7bb6b8361d", "at": "..."},
    {"horizon": 5, "action": "skip", "reason": "队列里已有该周期任务 9030c747dfb7"}
  ],
  "queue_waiting": {"1": "7e7bb6b8361d"},
  "model_health": "unknown",
  "summary": {"checked": 2, "want_train": 1, "will_enqueue": 1,
              "skipped": 1, "enqueued": 1}
}
```

UI 读 `/api/train/plan`（INV-8：server 只读 JSON，不做判断）。
排出来的任务带着 `created_by="train_scheduler"` 和 `note="[auto] <原因>"`，事后可追。

---

## 四、用法

```bash
python -m qcore.train_scheduler --plan      # 只判断，不排单（安全）
python -m qcore.train_scheduler --dry-run   # 同上
python -m qcore.train_scheduler             # 判断 + 真排单
python -m qcore.train_scheduler --summary   # 看最近一次判断
python -m tests.test_train_scheduler        # 契约测试（10 条）
```

容器里由 `start.sh` §6.6 每小时跑一次（`APB_TRAIN_AUTO=1`，默认开）；
关掉就 `APB_TRAIN_AUTO=0`。

---

## 五、端到端验证（本机实测）

```
$ python -m qcore.train_scheduler
[train_scheduler] checked=2 want=2 enqueued=2
  h=1   train  数据比训练截止新 22 个交易日（阈值 3）
  h=5   train  数据比训练截止新 22 个交易日（阈值 3）

$ python -c "from qcore import sync as S; print(S.summary()['counts'])"
{'done': 1, 'pending': 2}          # 队列里多了两张单

$ python -m qcore.train_scheduler --plan
[train_scheduler] checked=2 want=0 enqueued=0
  h=1   skip   队列里已有该周期任务 7e7bb6b8361d
  h=5   skip   队列里已有该周期任务 9030c747dfb7
```

第二次一定跳过 —— 这是"不重复排单"的实证，也是测试里第 2 条断言的现场版。

---

## 六、与训练机的关系

```
服务端（NAS）                       训练机
─────────────────────────────      ─────────────────────────────
train_scheduler  判断 → 排单
      ↓
sync_tasks.json  队列
      ↓  POST /api/sync/task {claim:true}
                              tools/sync_agent.py 领取 → train → 上传模型
      ↑  POST /api/model/import（校验通过才装，留快照）
      ↓
versioning 快照 + current.json
```

队列侧本轮还加了**跨平台文件锁**（`sync._FileLock`）：多台训练机同时轮询时，
不会两台领到同一张单，`create`/`claim` 交错也不会互相覆盖。

---

## 七、边界（写清楚，避免又变成"看起来在跑"）

1. **它只排单，不训练** —— 训练仍然只在训练机上做（NAS 不做重活）；
2. **它不判断模型好坏** —— 只判断"该不该训"。质量门是另一件事（评审 R36：
   `model_transfer._validate` 目前只校验形状/能 predict，不比较新旧模型的样本外表现）；
3. **`model_health` 现在多半是 `unknown`** —— 因为 `model_health.record()` 需要每日
   IC 样本，样本 < 20 天时不会判定。所以当前主要靠"超龄 + 数据落后"两条硬判据；
4. **不排 `pipeline` / `prewarm` 任务** —— 那两类目前仍由人工在 UI 上排；
5. 冷却状态存在 `train_plan.json` 里（保留最近的成功记录），**删掉这个文件等于清空冷却**。
