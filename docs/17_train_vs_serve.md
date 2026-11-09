# 训练与推理的分工

> **一句话：训练只在训练机做，NAS 和开发机都不训练。**

---

## 为什么必须分开

| | 训练 | 推理 |
|---|---|---|
| 耗时 | 5-6 分钟/周期 | **毫秒级** |
| 资源 | CPU/内存密集 | 几乎不占 |
| 频率 | 每周或按需 | 每 5 分钟 |
| 在哪做 | **训练机** | **NAS** |

把训练放在 NAS 上有三个问题：

1. **NAS 是长期在线的服务节点**，6 核 N100 被训练占满时，
   API 响应、盘前预测、盘中监控全部变慢
2. **模型文件会覆盖**，训练中途失败可能留下半个模型
3. **没有 GPU**，核显实测比 CPU 还慢（带宽打平、算力只有 37%）

而推理本来就快——`predict` 用的是已保存的模型，一次 30 只票不到 1 秒。

---

## 三个角色

### 训练机（任意一台能跑 Python 的电脑）

```bash
# 1. 解压 apb-train-kit_<version>.zip
# 2. 双击 train.bat（Windows）或 bash train.sh
```

第一次运行会自动装便携 Python（不需要系统装 Python）。

训练完**自动上传**到 NAS，不需要手动拷贝。

**什么时候跑**：
- 每周一次（模型会随市场变化衰减）
- 改了因子/超参之后
- 某个周期表现明显变差时

### NAS（部署节点）

只做三件事：

1. **推理**：`python -m qcore.live_model --predict --horizon=N`
2. **模拟盘**：按 picks 自动调仓、记账
3. **对账学习**：台账、归因、事件学习、周报

**不做**：训练、调参、蒸馏。

`start.sh` 里原来的「每周自动重训」已改成**只告警**：

```
[model] 模型已 8.3 天未更新，建议在训练机重训后上传
```

### 开发机（写代码的机器）

**也不训练**。改完代码只跑 `preflight.py` 和单元测试，
不产出模型文件。

---

## 完整链路

```
训练机
  ├─ 下载行情数据（545MB，一次性）
  ├─ 训练 h=1 / h=3 / h=5 / h=10 / h=15
  └─ 每个周期训完就上传 ────────┐
                                │  POST /api/sync/push
                                ↓
                              NAS
                                ├─ 收到即生效（四项校验：反序列化/特征数/feature_idx/能否预测）
                                ├─ 推理：生成 picks_hN.json
                                ├─ 模拟盘：各周期账户独立调仓
                                └─ 对账 → 归因 → 周报
```

---

## 训练机怎么知道 NAS 地址

`apb_defaults.json`：

```json
{
  "nas_url": "https://apbcdn.683868.xyz",
  "default_horizons": [1, 3, 5, 10, 15],
  "default_trees": 250,
  "default_stocks": 300
}
```

改这一行就能指向你自己的 NAS。

命令行也能覆盖：

```bash
python tools/train_one_click.py --nas https://your-nas --token xxx
```

---

## 服务端下发更新（训练机自更新）

训练机的代码版本可能落后。NAS 会告诉它最新版本：

```bash
# 训练机执行一次检查
python tools/sync_agent.py --nas https://apbcdn.683868.xyz --update-only

# 常驻模式：每 60 秒检查一次更新 + 领任务
python tools/sync_agent.py --nas https://apbcdn.683868.xyz --loop 60 --auto-update
```

更新是**安全的**：
- 只覆盖 `qcore/ tools/ scripts/ *.bat *.sh VERSION`
- **绝不动** `data/ dist/ runtime/ apb_defaults.json`
- 先解压到临时目录校验通过才覆盖，任何一步失败保持原样

---

## 如果某个周期一直没数据

`by_horizon` 表会明确告诉你缺什么：

| 状态 | 含义 | 怎么办 |
|---|---|---|
| `no_model` | 该周期没训练过 | **训练机**跑 `--horizons=N` 后上传 |
| `no_picks` | 有模型但没生成预测 | 在 **NAS** 跑 `--predict --horizon=N`（秒级） |
| `hold_period_not_elapsed` | 持有期没到，不该调仓 | 正常，等 |

**这两个不能混**：前者要训练机，后者 NAS 自己就能做。

---

## 相关代码

| 文件 | 职责 |
|---|---|
| `tools/train_one_click.py` | 训练机主程序，训完自动上传 |
| `tools/bootstrap.ps1` | 便携 Python 运行时引导（多镜像） |
| `tools/sync_agent.py` | 领任务 / 上传模型 / 自更新 |
| `qcore/kit_update.py` | 自更新逻辑（只覆盖代码） |
| `qcore/model_transfer.py` | 模型导出/导入 + 四项校验 |
| `qcore/live_model.py` | `train()` 和 `predict()` 都在这里 |
| `docker/start.sh` | NAS 启动脚本（训练已改成只告警） |
