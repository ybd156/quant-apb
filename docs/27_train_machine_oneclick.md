# 27 · 训练机「一键」——双击之后到底发生了什么

> 入口：Windows `APB.bat auto`　Linux/macOS `bash apb.sh auto`
> 代码：`tools/train_machine_auto.py`　排单侧：`qcore/train_scheduler.py`

---

## 一、为什么之前不算"一键"

仓库里本来就有三块能用的东西，但它们各自为政：

| 已有 | 干什么 | 缺什么 |
|---|---|---|
| `tools/kit_setup.py --cli` | 填 NAS 地址 + 令牌，测通后保存 | 用户得知道**先**跑它 |
| `tools/agent_daemon.py` | 把 `sync_agent` 挂到后台 | 没配置时启动直接失败退出 |
| `tools/train_one_click.py` | 本地训完打包、推模型 | **不领队列任务** |

于是新机器的真实流程是：猜一下该跑哪个 → 报错 → 翻文档。现在收成一步。

---

## 二、`APB.bat auto` 做的五件事

```
[1/5] 环境自检    Python 3.10+ / numpy / pandas / 磁盘 / 训练数据是否就位
[2/5] 配置        没有 apb_defaults.json → 就地拉起配置向导（填→测→存）
[3/5] 连通        kit_status 探一次 NAS，把最后几行原样打给你看
[4/5] 训练数据    已就位就跳过；未下载则**明确提示**（不会偷偷下 545MB）
[5/5] 启动        已在跑就不重复起；否则 start 后台守护
```

最后打印「接下来会发生什么」+ 一屏状态，然后退出。**它自己不留窗口**——
守护在后台，随时 `APB.bat status` 看。

### 其它形态

```bash
APB.bat auto              # 一键（推荐）
bash apb.sh auto          # Linux/Mac 同上
python tools/train_machine_auto.py --check    # 只体检，什么都不动
python tools/train_machine_auto.py --once     # 领一个任务就退出（调试用）
python tools/train_machine_auto.py --no-daemon # 前台常驻，日志打在屏幕上
python tools/train_machine_auto.py --data      # 显式下载训练数据（545MB）
```

---

## 三、领到任务之后（队列侧）

```
NAS  train_scheduler        判断该不该训 → 排单（train / train_ab）
      ↓ sync_tasks.json
训练机 sync_agent            每 60s 问一次「有没有任务」
      ↓ claim
      ├─ job=train       → live_model --train → export_bundle → POST /api/sync/push
      ├─ job=train_ab    → scripts/ab_train_pit → 结果随 report 回传
      ├─ job=pipeline    → qcore.pipeline（全流程回测）
      └─ job=prewarm     → tools.prewarm_external
      ↓ POST /api/sync/result（含 ab_train 时 NAS 落 data/state/ab_train.json）
NAS  GET /api/ab_train       只读最近一次双轨复核结果
```

失败**不静默**：`_report(ok=false, detail)` 会把 returncode、stderr 尾部带回去，
NAS 侧任务状态变 `failed` 并留下原因。

---

## 四、它明确不做的三件事

1. **不装 Python**（要管理员权限，脚本不该替你决定）；
2. **不偷偷下数据包**（545MB，必须 `--data` 显式触发）；
3. **不碰 NAS 以外的东西**（只按队列领任务、按结果回报）。

---

## 五、验收（本机实测）

```
$ python tools/train_machine_auto.py --check
==================================================================
  APB 训练机 · 一键启动
==================================================================

[1/2] 环境自检
  Python 3.12.14
  磁盘剩余 1247.0 GB · 训练数据已就位
```

启动器约束（AGENTS §7 陷阱 14）：`APB.bat` 断言 **纯 ASCII + CRLF**，
`apb.sh` 断言 `bash -n` 通过 —— 两条都在改完后立刻验过。

---

## 六、和 NAS 侧的关系（别搞混）

| 谁 | 干什么 |
|---|---|
| **NAS** | 取数、算信号、出名单、跑模拟盘、判断"该不该训"、**展示**结果 |
| **训练机** | 只干重活：训练 / 双轨复核 / 全流程回测 / 预热缓存 |

所以「一键」只解决**训练机上手**，不改变职责划分：NAS 不会因为一键而开始训练。
