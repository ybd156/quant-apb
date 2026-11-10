# 22 · 泄漏告警豁免台账（2026-11-10）

> 目的：把「704 条 MEDIUM 里 608 条从没人看过」变成一个**可复核的判定**。
> 生成器：`python -m audit.gen_waivers`（`--check` 校验清单是否与代码一致，CI 会跑）。
> 机器可读版：`audit/waivers.json`。

## 一、两层规则

### 第一层（机械）：是否在打分/建模链路闭包里

```
闭包根 = signals / learner / engine / dataset
       + gbdt / alpha158 / feature_select / sample_weight / online_calib /
         intraday_online / live_model / horizons / ensemble / ensemble_model /
         distill / ranker / residual / neutral / behavior_report /
         event_learning / meta
→ 传递展开后共 45 个模块
```

**闭包外**的文件：全样本 `mean/std/quantile` 只影响事后指标，不可能进入信号或
模型参数 → **文件级豁免**。

**闭包内**的文件：默认**不豁免**，进入第二层人判。

> ⚠️ 第一版只把「打分链路」当根，闭包只有 6 个模块，会把 `gbdt`/`alpha158`/
> `online_calib` 这类**训练**代码误判成"纯评估层"。训练代码里的全样本统计
> 会污染模型参数，必须留在闭包内。

### 第二层（人判）：闭包内的文件逐个核实

理由必须指到**代码行为**，不许写「看起来没问题」。

| 文件 | 判定 | 依据 |
|---|---|---|
| `horizon_ledger` | 豁免 | 全是台账展示聚合（mean/hit_rate/spread），只写 `horizon_perf.json` 供 UI，不回喂预测 |
| `intraday_online` | 豁免 | `train_online` 只接收 `y is not None` 的**已到期**样本，标准化在该集合内做，训练时刻天然因果 |
| `analysis` | 豁免 | 优势体检模块，demean/秩相关/分位差本身就是评估指标 |
| `events` | 豁免 | 事件研究 CAR 统计，产物写 `events.json` 供 UI |
| `behavior_report` | 豁免 | 仅 pipeline 调用的报告脚本 |
| `sources` | 豁免 | 数据源交叉验证健康度，只影响 `source_health.json` |
| `ensemble` / `ranker` / `residual` / `distill` | 豁免 | **零 importer**，不可能进入主链路 |
| `meta` | 豁免 | 告警点全是评估量（`meta_ic` 相关、`frac_inverted`、`hit_rate`）；`final_sign` 由 `final_t` 判定，不由这些均值决定 |
| `online_calib` | 豁免 | 对（预测, 已实现收益）做 OLS 求 a/b，样本是已到期配对；参数只作用于未来 |
| `reward` | 豁免 | hit_rate/加权均值/max_dd 都是对已实现序列的评价指标 |
| `event_learning` | 豁免 | 事件后收益均值与命中率，产物供 UI |
| `live_model` | 豁免 | 索引越界检查、样本数门槛、应用在线校准；无全样本统计进入打分 |
| **`gbdt`** | **保留** | 分箱边界只用训练矩阵拟合、节点内梯度/海森求和、L2/R² 指标 —— 非泄漏，但属训练核心，保留告警当提醒 |
| **`engine`** | **保留** | 告警点全部是 `axis=1`（信号维）归一化与横截面 mean，非时序统计；核心代码保留告警 |
| **`learner`** | **保留** | softmax 在信号维归一化 |
| **`signed`** | **保留** | L56 横截面去均值；L88-89 的 mu/sd 在 fit 时用训练矩阵拟合并存盘、预测时复用 |
| **`dataset`** | **保留** | 面板构造核心 |

## 二、结果

```
scanned 119 file(s)
RESULT: HIGH=0  MEDIUM=33  INFO=100  WAIVED=781（行内 96 / 文件级 685）
```

- **705 → 33**：从「704 条不知道要不要管」变成「33 条已知、已理解、仍在盯」。
- 33 条全部落在 5 个核心/训练文件里，每次改动都值得重看一遍 —— 这是**故意的**，
  不是没处理完。
- 文件级豁免 685 条，每条都带具体理由，写进 `audit/waivers.json`。

## 三、生成器的一个坑（已修）

第一版 `gen_waivers` 扫描时**沿用了上一次的 waivers.json**，于是已豁免的文件扫出
0 条 MEDIUM → 不进候选 → 新清单把它们洗掉。表现为 `MEDIUM 246 → 207 → 归零`。
修法：扫描前 `LS._FILE_WAIVERS = {}`（先关掉豁免再统计）。

## 四、什么时候必须重跑

- 新增/删除 `qcore/*.py`
- 改动某个文件里 `mean/std/quantile/sum` 的语义（例如把截面归一化改成时序）
- CI 会在 `gen_waivers --check` 不一致时变红

```bash
python -m audit.gen_waivers          # 刷新
python -m audit.gen_waivers --check  # 只校验（CI 用这个）
```
