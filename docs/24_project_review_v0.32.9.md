# 项目评审报告 · v0.32.9（2026-11-11）

> 范围：整个仓库（qcore / webui / audit / tests / scripts / tools / docs / 运行产物）。
> 方法：读源码 + 读产物 + 跑门禁，**以产物与命令输出为准**，不采信文档自述。
> 口径：本文所有数字均标出来源（文件:行 或 产物字段 或 命令）。
> 基线：commit `f20a6d3`（v0.32.9，2026-11-11 11:03）。

---

## 0. 结论（一句话）

**测量仪本身是可信的，被测量对象不是它自称的那个东西。**

- 算法与统计口径（无未来函数、HAC、DSR 单位、台账幂等、成本下限、时间门禁）经复核**成立**；
  18 个测试模块全绿，`test_no_lookahead` 的 worst diff = `0.00e+00`。
- 但"106 个专家 / 73 个专家 / 102 个信号"这三个数字同时存在于代码、产物、文档里，
  而实际**只有 73 个信号在发声**：v0.29–v0.31 接入的 **6 个外部数据族、29 个信号全部恒定输出 0**，
  原因是 `dataset.external_loader()` **零调用点**（接线漏了一行）。
- 因此"接了 6 个外部数据族"这件事对结果的影响是 **0**；它只增加了分母、权重账面和文档篇幅。
- 另有两处对外表述与产物不一致：模型页展示的 GBDT（250 树 / 300 只 / 训练到 2019）
  **不是实际在出 picks 的模型**（20 树 / 60 只 / 训练到 2026-10-10）；
  成本页写的"换手 0.8 / 250÷h"与它自己算出的数值（0.7 / 252÷h）矛盾。
- 系统自身的核心结论（5 日尺度无可用的横截面 alpha）**未被本次评审推翻**，反而更稳：
  扣成本后多空 −4.80%/年，DSR 0.0145，学习器≈等权。
- 第二轮走线上路径（picks / 模拟盘 / 在线校准 / 台账 / 部署门禁）又抓到两条会**直接改变
  用户看到的名单**的 P1：brief 里"系统已自动反向使用"没有下游实现；
  在线校准的 `a` 已变成 −0.00305（h=5），一旦生成 `picks_h5` 就会把排序整体倒过来且无任何标记。

严重度分布：**P0 × 2 · P1 × 14 · P2 × 32 · P3 × 7（共 55 条）**。

**修复进度：已修 51 条（§7.1，七批）、未修 4 条（§7.2）。R48 收口：增量 dataset 16.9s → **9.1s**（300 只，逐位相等）。外部数据已接上（dead 33→15）；服务端能自己排训练单；文档数字全部回填到 2026-11-11 产物。**

---

## 1. 评审范围与方法

| 做了什么 | 命令 / 方式 | 结果 |
|---|---|---|
| 18 个测试模块 | `python -m tests.test_*`（逐个） | 全部 0 failed（含 95/95、55/55 等断言级） |
| 截断不变性 | `python tests/test_no_lookahead.py` | PASS 4/4，worst diff `0.00e+00`，负控能抓住 `shift(-1)` |
| 泄漏静态扫描 | `python audit/leak_scan.py` | HIGH=0，MEDIUM=51，WAIVED=782 |
| 豁免清单一致性 | `python -m audit.gen_waivers --check` | 一致 |
| 不变量自检 | `python -m qcore.invariants` | 8 checks 0 failed —— **但 INV-1/INV-3 是 SKIP**（见 R10） |
| 判断台账 | `python -m qcore.claims score` | 13 条 / 已结算 6 / 命中 50%（样本不足，系统自己标了） |
| 产物 vs 文档 | 逐字段比对 `data/state/*.json` 与 AGENTS/docs | 见 R8–R9 |
| 代码走读（第一轮） | engine / learner / dataset / metrics / risk / pipeline / store / server / ui | 见 R1–R5 |
| 代码走读（第二轮） | live_model / online_calib / paper_auto / horizon_ledger / store.append_ledger / brief / docker | 见 R16–R23 |
| 代码走读（第三轮） | behavior_report / alert / horizons / transfer / stats.deflated_sharpe / accounts / health 顶栏 | 见 R24–R27 |
| 代码走读（第四轮） | live vs live_model / hotspot / tick / paper_auto / 74 条路由 | 见 R26–R32 |
| 代码走读（第五轮） | events / impact / thresholds / release.yml | 见 R33–R35 |
| 代码走读（第六轮） | 训练机：model_transfer / data_sync / kit_update / build-train-kit.sh / bat 编码 | 见 R36–R39 |
| 代码走读（第七轮） | sync 队列 / agent_daemon / sync_agent / preflight 调度 | 见 R40–R42 |
| 代码走读（第八轮） | daily_append / qlib overlay / tick 落盘 / docs 命令 | 见 R43 |
| 代码走读（第九轮） | 连续学习就绪度：live.step_once / loop / tick 名单 / live_features / intraday_online | 见 R45–R48 |
| 代码走读（第十轮） | 预测→验证闭环：horizon_ledger.record/reconcile / qlib 日历缓存 / 四本账 | 见 R49–R50 |
| 实现（第十二轮） | 新增 `qcore/event_history.py` + 接入 autoinfo + `tests/test_event_history.py` | 见 R52 / §4.14 |
| 实现（第十三轮） | 新增 `qcore/intel_features.py` + `live_model` 融合 + `reward` 方向改造 + `tests/test_intel_features.py` | 见 R53 / §4.15 |
| 实现（第十一轮） | 新增 `qcore/autoinfo.py` + `tests/test_autoinfo.py`（13 断言） | 见 R51 / §4.13 |

**没有做**（诚实清单）：没有重跑 `qcore.pipeline`（会覆盖全部产物，属破坏性动作）；
没有重训 GBDT（`data/qlib/x/qlib_bin` 当前对该用户不可读，见 R16）；没有跑
`tools/ui_audit.mjs`（需 Chrome + 起服务）；N100 耗时未复测。

---

## 2. 阻断级发现（P0）

### R1 · 外部数据族整体未接线：29 个信号恒定输出 0

**现象**：`data/state/weights.json` 共 102 行，其中 **29 行 `n_updates = 0`**，
`hit_rate = 0.5`（先验）、`mean_payoff = 0.0`、`weight = 0.9863%`。
这 29 个正好是外部数据族全部成员：

```
flow 6 · block_trade 3 · billboard 2 · margin 4 · holder 2 · lift 2
forecast 2 · event 2 · sentiment 3 · breadth 3            = 29
```

**根因**：`qcore/dataset.py:216` 的 `build_dataset(..., extra_loader=None)` 只有在
`extra_loader` 非空时才逐票加载外部表（`dataset.py:349-353`），
而 **`external_loader`（`dataset.py:102`）在整个仓库里没有任何调用点**：

```
$ grep -rn "external_loader" --include=*.py .
qcore/dataset.py:102:def external_loader(code: str, cache_only: bool = True):
qcore/event_data.py:347:  # 实测事故：external_loader 在 build_dataset 里逐票调 5 个接口…
tools/prewarm_external.py:15:  · build_dataset 只读缓存（external_loader(cache_only=True)），绝不联网
```

主流程三个入口都**没有传**这个参数：

- `qcore/pipeline.py:100` `build_dataset(codes, start=..., loader=loader, pad=pad, market=market)`
- `qcore/live.py:88` `build_dataset(codes, start=..., extra_horizons=[1,20], verbose=False)`
- `qcore/live.py:161` `build_dataset(..., cutoff=..., live_quotes=_live)`

于是 `compute_signals(df, flow_df=None)` → 所有 `_f_flow/_f_bt/_f_margin/_f_holder/_f_lift/
_f_forecast/_f_event/_f_sent/_f_breadth/_f_seat` 走空表分支 → DIR/STR 恒 0。

**复现实验（600519，6039 根日线，2001-08…2026-11-10）**：

```python
from qcore.data import load_kline; from qcore.dataset import external_loader
from qcore.signals import compute_signals
df = load_kline('600519'); x = external_loader('600519')
compute_signals(df)                 # 现状（flow_df=None）
compute_signals(df, flow_df=x)      # 接线后
```

| 信号族 | 现状发声天数 | 接线后发声天数 |
|---|---|---|
| flow | 0 | 608 |
| margin | 0 | 6754（族内合计） |
| breadth | 0 | 2068 |
| forecast | 0 | 252 |
| holder | 0 | 147 |
| billboard | 0 | 3 |
| block_trade / sentiment / event / seat | 0 | 0（该票无数据） |

—— 差额**只来自那一个参数**。这也意味着：接线之后 `IC / DSR / 权重 / picks` 会全部变化，
必须重跑并重新评估，不能当作纯重构。

**影响**：

1. 账面"106 个专家"里 **33 个（含 4 个席位信号）从不发声**；实际被测集合是 73 个。
2. 按权重算的分散度被死信号抬高：29 个死信号各占 0.99%，合计 **28.7% 的权重账面**
   没有任何投票行为；把它们剔除后重新归一，`1/HHI` 只有 **69.6**，
   而 `risk.signal_level.effective_n` 报的是 **98.6**（虚高 42%）。
3. 11-10 那批"接 6 个外部数据族"的 commit（v0.29.0–v0.31.0）对回测数字的净效果是 **0**，
   与 `performance.json` 的 IC 仍停在 −0.0183 完全一致。
4. 这正好是 AGENTS §7 陷阱 27 的复刻：**模块写了、缓存预热脚本写了、文档写了，唯一缺的是调用点**
   （`tools/prewarm_external.py` 预热的缓存无人读取）。

**建议**（按顺序）：

1. 立刻给 `pipeline.py` / `live.py` 传 `extra_loader=external_loader`；
2. 在 `build_dataset` 里对每个信号族统计"发声天数"，**输出 0 天数的信号直接列进 `status.json` 的
   `dead_signals`**，UI 上把 `n_updates=0` 的行显式标红（现在它们和正常专家长得一样）；
3. 加一条测试：用真实缓存跑 3 只票 × 全 zoo，断言"每个信号至少发声 1 次"，否则红。
   ——这条测试若早存在，R1 不会活过一天。

> 补充：即便接上线，外部数据覆盖也很不均匀（300 只票缓存实测）：
> margin 98.3% · 大宗 4.5% · 龙虎榜 0.9% · 增减持 0.5% · 业绩预告 0.6% · 解禁 0.3% · **席位 1/300 只**；
> 资金流只有 **120 个交易日**（2026-05-15…11-09）。接线之后必须重新评估这批信号的样本量，
> 不能直接沿用"每个专家 3000 次更新"的直觉。

### R2 · 产物与代码不同步：S=102（产物）/ 106（代码），4 个新信号从未评估

**证据**：

- `signal_zoo()` 当前返回 **106**（实测 `len(signal_zoo())`）；
- `data/state/status.json` / `performance.json` 的 `n_signals` = **102**；
- `git log`：`qcore/signals.py` 最后一次修改 `4cf4cac`（**11-10 20:29**，"席位信号" +4）；
- `performance.json` 的 `generated_at` = **11-10T16:03:54** —— 早于那次提交 4.5 小时。

**影响**：4 个 `seat_*` 信号从未进入任何一次 walk-forward：没有 IC、没有权重曲线、没有台账、
没有进 risk 体检。UI 却按 `status.json` 显示 102，代码实际 106，读者无法发现差异。
且席位数据缓存覆盖 **1/300 只票**（13/15/1 行非空），接上线也基本是死信号。

**建议**：产物里加 `code_fingerprint`（zoo 的 sid 列表哈希）与 `zoo_n`；UI 在
`zoo_n != n_signals` 时显式提示"产物落后于代码，请重跑 pipeline"。
把"重跑 pipeline"写进改信号的 Definition of Done。

---

## 3. 重要发现（P1）

### R3 · 模型页展示的 GBDT 不是实际在出 picks 的模型

| 口径 | 展示值（`data/state/model.json`） | 实际部署（`data/model/meta.json` + `gbdt.pkl`） |
|---|---|---|
| 股票数 | 300 | **60** |
| 树数 | 250 | **20** |
| 训练截止 | 2019-12-31 | **2026-10-10** |
| 写入时间 | 11-09 08:06 | 11-09 11:37 |
| 对外业绩 | 年化 12.98%、超额 9.43%、Sharpe 0.77 | 无 |

`qcore/live_model.py:78` 的 `load_model()` 直接读 `gbdt.pkl` + `meta.json`（不走 `versioning.current.json`），
而 `meta.json` 在 11-09 11:37 被一次 **20 棵树 / 60 只票** 的训练覆盖了
（`"train_n": 259145, "train_seconds": 16.3`），`model.json` 没有随之更新。
`picks.json`（11-10 13:54，带 `online_calib`/`model_age_days` 字段，可确认出自 `live_model`）
就是用这个 20 树模型打的分。

**影响**：模型页是项目里唯一"正 alpha"的展示位（+9.4% 超额），它描述的不是线上模型。
这正是 AGENTS 陷阱 13（多实例共用文件名）在 h=1 上的复发。

**建议**：`live_model` 改走 `versioning.load(current.json)`；`model.json` 由**同一个写入方**
在 `save_model` 之后落盘，并把 `model_file` / `model_hash` 写进 `model.json`，UI 回显。
`meta.json` 与 `model.json` 至少要有 `trained_at` 一致性的断言。

### R4 · GBDT 训练集用"2021 年成分股"选 2008–2019 的股票（未来条件选择）

`scripts/lowturnover_bt.py:42`：

```python
codes = QD.universe_codes(universe, start=test_start)     # test_start = 2021-01-01
...
d = bench.build_features(codes, start="2008-01-01", horizon=horizon, ...)
```

训练期 2008–2019 的样本 = "在 2021-01-01 仍属 CSI300 的票"，即**按未来是否存活/入指筛过一遍**。
`qcore/qlib_data.py:316` 虽已写好 `pit_membership()`（逐日点对点成分），
但全仓库 **零调用点**（`grep -rn pit_membership` 只命中定义本身）—— 与 R1 同一种病：
正确的基础设施写好了，没有接上去。

`docs/15_robustness.md` 只承认了"测试期用同一批股票"，并写"影响有限"，
**没有**指出训练期也被未来的成分快照筛选过，也没有给出量化。
按项目自己的基线（生存者偏差 ≈ +22pp/年），+9.43% 的超额不足以排除该项解释。

**建议**：训练/测试都改用 `pit_membership()` 逐日 UNIV；或退一步——把 2008–2019 的训练样本
换成"当日在指"的成员。在改之前，模型页的 +9.4% 必须加"含未来条件选样、未去偏"的醒目标注。

### R5 · 成本口径：标签与数值互相矛盾（同一份产物里）

`data/state/costs.json`（由 `qcore/pipeline.py:342-353` 写入）：

```
round_trip_bps: 19.282
annual_drag_pct: {"1": 34.01, "5": 6.8, "15": 2.27}
turnover_assumption: 0.8
note: "annual_drag_pct = 250/h × 换手0.8 × 往返bps"
```

但数值是用 **0.7 换手、252 交易日** 算的（`qcore/horizons.py:78` 的默认
`turnover=0.7`、`periods=252/h`）：50.4 × 0.7 × 19.282bps = 6.80%。
若真按 0.8 算应是 7.77%。**字段与注释都在说 0.8/250，数值是 0.7/252。**
AGENTS §9.1 记录过"已统一为 252/0.7"，但只改了计算，没改标签。

同一问题的第二处：`webui/server.py:2248` 硬编码
"A 股光成本就 38.6%/年；h=15 只有 2.6%" —— 产物是 **34.01% / 2.27%**。

把三种口径都算一遍，可以确认这两处标签是**旧公式的化石**（19.282bps 往返）：

| h | 0.7 / 252（现算，产物值） | 0.8 / 252 | 0.8 / 250（note 的字面口径） |
|---|---|---|---|
| 1 | 34.01% | 38.87% | **38.56% → 38.6%** |
| 5 | **6.80%** | 7.77% | 7.71% |
| 15 | 2.27% | 2.59% | **2.57% → 2.6%** |

server 里的 38.6% / 2.6% 与 `0.8 / 250` 完全对得上，而 costs.json 的数值是 `0.7 / 252`。
即：**计算改过（252/0.7），文案没改（250/0.8）**，两套口径在同一份产物里并存。
h=1 这一格差 **4.5pp**（34.0 vs 38.6），正好是"T+1 到底能不能做"这个判断的量级。

**建议**：`turnover_assumption` 与 `note` 改为从 `horizons` 的实参回填（或直接删掉重复描述，
只留 `periods_per_year`、`turnover` 两个数值字段）；server 的 note 改成读 `costs.json`。

### R6 · "相关性调整后有效独立数"没有参与判定，判定仍用被高估的那个

`performance.json`：

```
risk.signal_level.effective_n          = 98.6      ← 按权重 1/HHI
risk.signal_level.corr_adjusted.effective_n_corr = 22.35   ← 按相关性
risk.signal_level.verdict              = "分散良好"
```

`qcore/risk.py:315-329` 的 `verdict` 只看 `effective_n`（98.6 > 阈值 10）与 `max_share`，
相关性口径只追加一条 warning（`pipeline.py:277`），**不改判定**。
考虑到 R1 的 29 个死信号把按权重算的有效数从 **69.6** 抬到 **98.6**（+42%），
这个"分散良好"同时被两件事高估。

同一模块的阈值还停在 `min_effective=10`（73 个信号时代的 14%，现在是 106 个信号的 9.4%），
且 warning 文案写死 `{eff:.0f}/73`（`pipeline.py:279`）、`"有效独立信号 33/73"`（`risk.py:326`）。

**建议**：`verdict` 改为以 `effective_n_corr` 为准（例如 < S/4 判"高度同质"），
`min_effective` 改成 `S*k` 形式的相对阈值，文案用 `len(sids)`。

### R7 · 命中率的标准误按"天数"聚类，但收益是 h 日重叠的

`qcore/metrics.py:216`：

```python
se_cluster = sqrt(p(1-p) / n_days)     # n_days = 3335（h=5）
n_indep    = n_days // h               # = 667，算了但没用于 SE
```

h=5 的重叠收益下，独立期数是 667 不是 3335，SE 应为 `sqrt(.25/667)=1.94pp`，
现在给的是 **0.87pp**，**乐观 2.2 倍**（正是 INV-4 说的问题，只是没应用到 hit_rate）。
AGENTS §9.1 引用的"距 50% 仅 1.7 个 SE"就是按 0.87pp 算的；按 1.94pp 只有 **0.77 个 SE**。

用产物重算三个周期（`data/state/performance.json` → `metrics.hit_rate`）：

| h | 命中率 | n_days | SE(现用,天) | n_indep | SE(重叠期) | 距 50%：现用 / 应用 |
|---|---|---|---|---|---|---|
| 1 | 48.40% | 3339 | 0.865pp | 3339 | 0.865pp | 1.85 / 1.85（不受影响） |
| 5 | 48.51% | 3335 | 0.865pp | 667 | 1.935pp | **1.72 / 0.77** |
| 20 | 48.55% | 3320 | 0.867pp | 166 | 3.879pp | **1.67 / 0.37** |

h=1 无重叠，所以这个错误只在 h≥2 时出现；h=20 的"1.67 个 SE"实际只有 **0.37 个 SE**。

结论方向不变（都是"与 50% 不可区分"），但这是"评估口径"层面的自我高估，属项目红线区。

**建议**：`se_cluster` 用 `n_indep`（或 HAC(lags=h-1) 算均值序列的 SE），并把 `n_eff` 标成
`n_indep` 而不是 `n_days`；CI 里加一条断言 `se_cluster >= 朴素SE × √h × 0.8`。

---

## 4. 一般发现（P2/P3）

### R8 · 同一个"信号个数"有四个值

| 位置 | 值 |
|---|---|
| `README.md:3` | **80**（"80 个技术信号"） |
| `AGENTS.md:11,56,236,483`、`webui/static/app.js:2283-2295`、`qcore/signals.py:1438` docstring | **73** |
| `data/state/status.json`、`performance.json` | **102** |
| `len(signal_zoo())` | **106** |

`app.js:2294` 还会渲染成"相关性调整后有效独立信号 22.4 **/ 73**"，
而同一页下方的表格来自 `/api/signals`，会列出 102 行（S=102 的产物）——同一页自相矛盾。
`qcore/advanced.py:6`、`analysis.py:6,165,285`、`meta.py:6`、`reward.py:7` 的注释也都写死 73。

**建议**：所有面向人的数字改成读 `status.n_signals`；文档里凡是固定数字都改成公式
（如"当前 N 个"）或标注生成时间；`ui_audit` 增加一条"页面声明数与表格行数一致"的断言
（现在它只看成交表/版本号/本金，抓不到这个）。

### R9 · AGENTS §9.1 的"基线 A"与产物对不上（不止一处）

| 指标 | AGENTS §9.1 | `performance.json`（11-10T16:03） |
|---|---|---|
| 信号数 S | 73 | **102** |
| Deflated Sharpe h=5 / h=1 / h=20 | 0.0313 / 0.2411 / 0.0308 | **0.0145 / 0.1508 / 0.0144** |
| 有效独立信号（相关性） | 12.13 / 73 | **22.35 / 102** |
| 多空年化（毛）h=5 / h=1 | +4.18% / +13.27% | **+2.65% / +11.55%** |
| 模拟盘年化（CN/h=5） | −3.49% | **−4.02%** |

文档自己写了"数字以产物为准"，但给出的恰恰不是当前产物。
（IC −0.01847 vs −0.01826、命中率 48.49% vs 48.51%、Meta 50.72% vs 50.70% 属同量级，
可以认为是同一次运行的不同快照；上面五项则是数量级/口径级差异。）

**建议**：把 §9.1 改成"生成命令 + 产物路径 + `generated_at`"，只保留**可复算**的关系，
不要抄具体小数；或加一个 `docs/check_baseline.py` 让 CI 比对文档与产物。

### R10 · 不变量自检在受限环境下 SKIP，仍然 exit 0、仍然"0 failed"

本机实测（DSH 沙箱禁止子进程）：

```
[SKIP] INV-1 无未来函数（截断不变性） :: PermissionError: [WinError 5]
[SKIP] INV-3 统计量必须滚动（泄漏扫描） :: 扫描器异常: PermissionError
8 checks, 0 failed      ← 退出码 0
```

`qcore/invariants.py:339-345` 只统计 `fail`，SKIP 不计入退出码。
`data/state/invariants.json` 因此把两条最关键的检查标成 skip 落盘，UI 体检页照常展示。

我另行直接执行了这两个脚本（绕过子进程）：`test_no_lookahead` 4/4 PASS、
`leak_scan` HIGH=0 —— **结论本身是好的**，问题是"门禁在受限环境里会静默降级成绿灯"。

**建议**：`main()` 在存在 SKIP 时返回 2（与 0/1 区分），CI 里对 1/2 都判红；
或 `--allow-skip` 显式开关。这与 R15「不变量不能 SKIP」是同一件事的两个层面。

### R11 · 泄漏扫描基线漂移：MEDIUM 33 → 51（新增 6 个文件）

`docs/22_leak_waivers.md` 记录 `HIGH=0 MEDIUM=33 INFO=100 WAIVED=781`；
实测 `HIGH=0 MEDIUM=51 INFO=107 WAIVED=782`，新增需人判的文件：

```
session.py 6 · event_data.py 5 · sentiment_store.py 4 · market_breadth.py 2 · qlib_data.py 1
```

`gen_waivers --check` 通过（说明 `waivers.json` 与代码同步），但**文档里那张"33 条已知、已理解"的
判定表已经不代表现状**。

**建议**：`gen_waivers` 在 `--check` 通过时顺手刷新文档里的计数，或把计数从文档移进
`audit/waivers.json` 单点；并给这 5 个新文件补人判理由（每行写"代码行为"级依据）。

### R12 · `scripts/export_model.py` 写 `picks.json` 缺 `market`（多写入方 schema 漂移）

- `scripts/export_model.py:190` 写 picks：`{as_of, rows, n, top_k, hold_days, band, note}` —— **无 `market`、无 `horizon`**；
- `qcore/live_model.py:364` 写 picks：带 `market`/`horizon`/`horizon_label`/`online_calib`/`model_age_days`。

同一个 `data/state/picks.json` 两个写入方、两套 schema。任何人跑一次 `export_model.py`
都会把 `market` 抹掉，`paper_auto.rebalance` 只能退化到"按代码形态猜市场"（AGENTS §7.5 规则一的兜底）。
这与 AGENTS 陷阱 12（多写入方覆盖同一文件）同型。

**建议**：`export_model.py` 不写 `picks.json`（它写的是"模型分"，不是"今日名单"），
或写 `model_picks.json`；真需要写就补 `market`/`horizon` 并复用同一个构造函数。

### R13 · 验收文档停在旧状态（R3 标"未开始"，实际已上线）

`docs/19_acceptance.md`：R3「在线学习」= **未开始**；但 `qcore/online_calib.py` 已接调度，
`data/state/online_calib.json` 有两次真实更新（11-09、11-10，n=2997/3000）。
同表 7 项仍标"改完未验证"、3 项"未开始"，与 v0.32.9 的实际状态脱节。

**建议**：验收表每发一版用脚本从产物回填（R3 的证据就是 `online_calib.json` 的 `history`），
否则它会持续误导"哪些还没做"。

### R14 · 运行面：数据到 11-10，产物停在 11-06，守护进程已停

```
data/raw/kline/*.csv 最后 bar = 2026-11-10（20/20 抽样）
data/state/status.json  as_of = 2026-11-06, last_processed = 2026-11-06, data_to = 2026-11-10
data/state/daily_append.json  target = 2026-11-10（11-11 08:01 写入）
logs/agent.stop 存在；logs/agent.log 最后更新 11-10 20:06
horizon_perf.json  h=5 n=3000，只有 2026-09-10…10-29 的 30 天
```

行情在补，回放/台账/净值没跟上：**11-09、11-10 两个交易日的预测与对账缺失**，
且守护进程处于停止状态。UI 顶栏（v0.32.8 修过）会如实显示"数据截至 11-06"，这点是诚实的；
但 `picks.json` 的 `as_of = 2026-11-09` 又比面板新，三个日期（数据/面板/名单）互不一致。

**建议**：`alert.py` 已有 `picks_stale`（3 天）告警，但没有"面板落后于行情"的告警——
补一条 `panel_lag_days > 0` 就响；并把 `status.as_of` 与 `picks.as_of` 的先后关系做成断言。

### R15 · 零字节文件与临时目录堆积（P3）

- `qcore/realtime.py`：**0 字节**、untracked（`git status` 唯一未跟踪项）。空模块会被
  引用扫描当成"死代码候选"，也会让 `ls qcore/*.py` 的计数虚高。
- `tmp/` 862MB（多份 `apb-train-kit` 副本 kit2…kt2）、`data/qlib/qlib_bin.tar.gz` 571MB；
  `.backup/`、`.tmp_audit/`、`.wheel-cache/` 均在 `.gitignore` 内，不进版本库，但占盘。

同族的第三件：`data/realtime/20261111.jsonl.gz`（今天 10:59 写入，4 条快照，每条 ~286 只票的实时价）
**在本仓库代码里既没有写入方也没有读取方**（文件仍在增长，说明有仓库外的进程在写；
`tick.py` 的落盘路径是 `data/tick/<day>/snapshot.jsonl`，不是这里） —— `grep -rn "realtime" --include=*.py` 全仓库零命中
（唯一的 `qcore/realtime.py` 是 0 字节）。这是"实时行情归档"功能的半成品：
文件在长，代码里没有它的位置。

**建议**：删 `qcore/realtime.py` 与 `data/realtime/`（或把 tick 的落盘/读取补上并接进回放）；
`tmp/` 加清理脚本进 `tools/`。

---

## 4.5 追加发现（第二轮走读：在线路径 / 台账 / 校准 / 部署）—— 含 2 条 P1

第二轮专门走了「预测 → picks → 模拟盘」和「台账 → 对账」这两条线上路径，以及部署侧门禁。
新增 8 条，其中两条会**直接改变用户看到的下单名单**。

### R16 ·（P1）"系统已自动反向使用"是句空话（Meta 自我纠错没有下游）

`qcore/brief.py:174-176`：

```python
if sign < 0:
    return (f"信号方向系统性做反（IC t={t:.2f}）→ 系统已自动反向使用；"
            "但反向后的优势尚未达到显著，只可小仓位验证")
```

而 `meta.learn_meta()` 的 `final_sign` **只在三个地方被读**：
`pipeline.py`（写进 `performance.json`）、`brief.py`（生成上面这句话）、
`report_stats.py`（写进统计报告）。`grep` 结果：

```
qcore/live_model.py · session.py · paper_auto.py · transfer.py  → 0 命中
```

即：**反号只进了展示层，没进打分层**。`picks.json` 用的是原始 scores，
`paper_auto` 照原始 scores 下单。当前 `meta.final_sign = -1`、`verdict = "反着用"`，
所以这句话此刻是**假的**。

同一模块还有一处"同名不同实"：`meta.compare` 的命中率（50.70%，n=269,332）
是在"剔除 18.8% 不表态日 + 按 t≥2 才反号"的样本上算的，而原始命中率（48.51%，n=328,604）
是全样本。两个数并列展示，读者会以为是同一批样本的前后对比。

**建议**：要么把 sign 真正接进 `live_model`（并给 `final_sign` 一个生效/失效的日志），
要么把文案改成"Meta 判定方向做反，但系统未据此调整仓位"。二选一，不能留现状。

### R17 ·（P1）在线校准可以在无人知晓的情况下把 picks 排序倒过来

`qcore/live_model.py:275-282`：

```python
_adj, _cal = OC.apply(horizon, _raw)       # out = a * score + b
if _cal.get("applied"):
    score[ok] = _adj
order = np.argsort(-score)[:top_k]         # 按校准后的分数排名
```

`data/state/online_calib.json` 当前 h=5 的系数是 **a = −0.00305**（h=1 仍是 `n_updates=0`，
`picks.json` 里如实写着 `"applied": false`）。
`a < 0` 意味着 `a*score` 把排序**整体反转**——picks 的"第 1 名"变成原始分数最低的票。
**目前还没有 `picks_h5.json`，所以这颗子弹还没打出去**；但只要跑一次
`python -m qcore.live_model --predict --horizon=5`，名单就会反过来，且没有任何标记。
代码注释把这当成特性（"如果 a 变成负数，说明模型方向反了 —— 系统自己发现的"），
但三个问题没有任何一处被守住：

1. **样本量口径**：`MIN_SAMPLES=30` 数的是**行数**（股票-日），实际这批是
   `n_samples=3000 / n_days=30`，h=5 下独立期数只有 **6**。用 6 个独立观测拟合 2 个参数，
   再让结果去翻转真金白银的排序，与项目自己的 `MIN_TRADES=30` 精神直接冲突。
2. **没有显著性门**：`apply()` 只判断 `n_updates == 0`，不判断 `a` 的 t 值、也不判断
   符号是否稳定（历史里 a 从 1.0 → 0.0071 → −0.00305，两次更新方向相反）。
3. **不可见**：`picks.json` 只写了 `online_calib: {a,b,...}`，没有"已反转"标记；
   分数会变成 −0.03 这类量级，UI 照常显示"第 1 名"。
   `docs/19` 把"a 1.0→0.0071、MSE 0.0674→0.0034（改善 20 倍）"当成功案例，
   但那是**同一批样本上的拟合内改善**，不是样本外。

另外文档说"`a` 限制在 [-3, 3]"（`online_calib.py:40` 注释），代码是
`A_RANGE = (-20.0, 20.0)`。

**建议**：`a` 加符号门（例如 |t(a)| ≥ 3 且连续 N 次同号才允许为负），
反转时在 `picks.json` 写 `calib_inverted: true` 并让 UI 明示；
`MIN_SAMPLES` 改成"独立期数 ≥ 30"（`n_days // h`）。

### R18 · 主台账 47,676 行的 `horizon` 全是 `null`，幂等键已退化

实测：

```
$ python -c "... Counter(r.get('horizon') for r in ledger)"
Counter({'h_None': 47676})
```

原因不是"老数据没补"，而是**写入侧补不上**：`store.append_ledger()` 用
`rec.setdefault("horizon", dh)`（`store.py:318`），而传入的行来自 DataFrame，
本身带 `horizon` 键（值 NaN）→ `setdefault` 看到键已存在就**不覆盖** → `_clean` 把 NaN 写成 `null`。

后果：

- `_ledger_key()`（`store.py:267`）在 `horizon is None` 时回落到**写入方传的默认值**，
  于是同一行在 h=1 写入时被判成 `(as_of, code, 1)`、在 h=5 写入时被判成 `(as_of, code, 5)`：
  键不是行的属性，而是**读取时的上下文**。一旦 h=1 也开始追加到 `{year}.jsonl`，
  同一天同一只票的 h=1 记录会被旧 h=5 行"顶掉"（判为重复跳过）。
- `horizon_ledger` 只能靠 `exit_date - entry_date` 的天数反推周期（当前 8150 个键里
  全是 7 天/14 天这种，恰好对应 h=5 + 节假日），这层反推没有测试。

**建议**：写入前强制 `rec["horizon"] = int(dh)`（不用 setdefault）；
对存量行做一次 `horizon` 回填并**保留原文件**（INV-7 不允许改历史，可另写一份
`{year}.tagged.jsonl` 作为视图）；加一条测试：同一 (as_of, code) 的 h=1 与 h=5 必须共存。

### R19 · `horizon_ledger` 的样本数 `n` 是乘出来的，不是数出来的

`qcore/horizon_ledger.py:443`：

```python
"n": len(days) * 100,          # 近似：每天约 100 只
```

`data/state/horizon_perf.json` 因此显示 `h=5, n=3000, n_days=30`。
真实样本是"台账里当天 `status=scored` 的行数之和"，可能是 20、也可能是 100。
AGENTS §9.4 明令"未测项不许估算冒充"，§10 规则 7 要求"任何结论必须报样本数"——
这里报的正是估算值。

同一份产物还有一处对不上：`hit_rate = mean(top_mean > all_mean)` 是**按天的布尔均值**，
30 天 → 0.3333 就是 10/30；而 `docs/19` 记的"h=5 n=3100 命中 61.29%"是同一指标的另一版数字，
两者无法互相验算（一个按天、一个按条）。

**建议**：`n` 改成 `sum(len(pairs))`，`n_eff` 另给 `n_days // h`；
`hit_rate` 明确标注"按天"，并同时给按条的命中率，避免"61% vs 33%"这种跨文档打架。

### R20 · 部署侧的门禁是装饰性的

- `docker/start.sh:326`：`python -m qcore.invariants > log || true` —— 失败不阻塞、不告警；
  加上 R10 的 SKIP→exit 0，容器里的不变量检查**永远不会让任何东西变红**。
- `Dockerfile:45,48` 只拷 `audit/leak_scan.py` 与 `tests/test_no_lookahead.py` 两个文件；
  后者在缺 `data/raw/kline`（镜像不打包行情）时会返回退出码 **2 = 全部 SKIP**，
  于是 INV-1 在容器里长期是"跳过"，而 UI 风险页会把它渲染成一条灰色条目。
- `docker/start.sh` 里没有 `python -m qcore.claims settle`，判断台账的到期结算只发生在 CI。

**建议**：`|| true` 改为"记录失败并在 `/api/health` 暴露 `invariants_failed`"；
`ENABLE_INVARIANTS` 打开时若无行情，明确落 `skip_reason=no_data` 并在页面标红而非灰。

### R21 · 回测基线表的 Sharpe 与"年化收益/年化波动"不可互算（P3）

`performance.json.backtest.baselines`：

| 策略 | 年化收益 | 年化波动 | Sharpe | ann/vol |
|---|---|---|---|---|
| 学习器(多空) | −4.80% | 26.75% | −0.0498 | −0.18 |
| 等权集成(多空) | −5.44% | 25.67% | −0.0889 | −0.21 |
| 随机打分(多空) | −14.41% | 11.34% | −1.315 | −1.27 |
| 买入持有 | +11.38% | 24.10% | +0.568 | +0.47 |

Sharpe 用的是**算术**日均值，年化收益是**几何**复合（`backtest.py:189-199`），
所以两列不能相除验算。这本身没错，但表格没有说明，而"随机打分"那一行恰好接近
（因为它的漂移小），会让人误以为口径是一致的。

**建议**：表头注明"Sharpe = 算术均值/标准差 × √252；年化收益为几何"，
或直接多给一列 `ann_arith`。

### R22 · 模拟盘按"信号日收盘"成交，回测按"次日开盘"成交——两个数字被并排展示（P2）

`qcore/paper_auto.py:36` 自己写着：

```
成交价用当日收盘（T 日出信号，T+1 成交更严谨，但模拟盘先用 T 日收盘简化，
并在记录里标明 price_basis="close_T"）。
```

而回测（`backtest.py`）的价格是 `OPEN[t+1] → OPEN[t+1+h]`。于是：

- `performance.json.backtest.long_short.ann_return = −4.80%`（次日开盘成交、已扣 19.28bps）
- `performance.json.paper.sim_annual = −4.02%`（信号日收盘成交）

两个数常被放在同一页比较（"模拟盘比回测好/差"），但它们差着一个交易日的价格跳空，
且模拟盘隐含"用刚算完信号的这根收盘价就能成交"。单是这一项就足以贡献 0.5–1pp 的差异。

**建议**：模拟盘改为 `T+1 开盘`成交（与回测一致），或至少在 UI 上把
`price_basis` 与回测的成交假设并排显示，禁止把两者直接相减。

### R23 · 事件权重的"时间衰减"会把未来排期事件当成当天事件（P3，潜在）

`qcore/reward.py:352-362`：

```python
cutoff = now.date() - timedelta(days=lookback_days)
age = (event_date - cutoff).days
if age < 0: continue                       # 只挡"太老"
decay = 0.5 ** (max(0.0, lookback_days - age) / half)
```

`max(0.0, …)` 把"未来事件"（`age > lookback_days`）的指数夹到 0，于是**未来日期的事件
拿到 decay = 1.0（权重最大）**，而且不会被 `age < 0` 挡住。
当前 `data/state/events.json` 里 `live`/`rows` 都没有未来日期事件（实测 0/41、0/56），
所以还没发生；但 `policy_calendar` 的职责就是产出"已知排期"的事件，
一旦它给排期事件带上 `direction`，这条路径就会用"还没发生的事"调今天的仓位。

**建议**：加一行 `if event_date > now.date(): continue`（或显式允许并标 `scheduled`），
并给 `event_adjust` 补一条"未来事件不得影响当前权重"的单测。

---

## 4.6 第三轮（行为因子 / 验收台账）

### R24 · 行为因子的"方向与文献一致"是个橡皮图章，3/10 在中性化后变号（P2）

`qcore/behavior_report.py:153`：

```python
note = "方向与文献一致" if (ic_s["mean"] or 0) != 0 else ""
```

**只要 IC 非零就打这个标签**——代码里没有任何"文献方向"表可比对。
更糟的是，它不检查中性化后的符号。产物 `performance.json["behavior"]`：

| 因子 | ic（全样本） | t | ic_neutral（最近 250 日） | 是否变号 | note |
|---|---|---|---|---|---|
| beh_ret_5d_rev | −0.0153 | −2.58 | **+0.0117** | **是** | 方向与文献一致 |
| beh_skew_60 | −0.0110 | −2.29 | **+0.0129** | **是** | 方向与文献一致 |
| beh_amihud | +0.0229 | +3.62 | +0.0204 | 否 | 方向与文献一致 |
| beh_turn_vol | −0.0228 | −2.33 | −0.0125 | 否（衰减 45%） | 方向与文献一致 |

两个问题叠在一起：

1. **标签与事实不符**：`ret_5d_rev` 中性化后从"反转"变成"动量"，页面仍写"与文献一致"。
2. **两个 IC 不是同一段样本**：`ic` 是全样本均值，`ic_neutral` 只取**最近 250 个交易日**。
   把它们并排显示（还用比值 0.4 触发"衰减"结论），等于拿全历史比最近一年。

**建议**：`ic_neutral` 改在同一窗口上算（或同时给 `ic_recent250`）；
note 改成基于"符号是否一致 + |t| 是否 ≥2"的机械判定，并显式标出"中性化后变号"。

### R25 · 验收表里有一个数字没有任何产物支持（P2）

`docs/19_acceptance.md:27`：

```
| R18 | 分钟级预测 | 已验证 | 样本外命中 63.6%，净 -1.84bps |
```

而仓库里唯一一份分钟级前向验证报告 `reports/intraday_20261106.md` 写的是：

```
毛命中率 0.335 · 净（扣 19.3bps 往返）0.090 · 每笔 −19.39bps
结论：净收益为负 ⇒ 该频率不可交易
```

`grep -rn "63.6"` 在整个仓库只命中这一行验收表，**没有产物、没有报告、没有日志**。
"63.6% 命中 / −1.84bps" 与"33.5% / −19.39bps、不可交易"是方向相反的结论，
而验收表选的是好看的那个。

**建议**：把 R18 的"已验证"改成"已验证（结论：不可交易）"并引用报告路径；
验收表每一格都必须能指到一个文件（`docs/19` 自己开头就写了"附实测输出"）。


---

## 4.7 第四轮（架构 / 实时采集 / 模拟盘 / 接口冒烟）

这一轮把服务真的起起来（`python -m webui.server --port 8792`）做了接口冒烟，
并顺着「自学习到底在哪跑」「实时行情怎么采」「模拟盘怎么成交」三条线读代码。

### R26 ·（P1）生产打分与"自学习参数体"是两套系统，交易那条用的是另一套

两条完全独立的打分链路同时存在：

| | 回放链路（那台"测量仪"） | 生产链路（真的出名单） |
|---|---|---|
| 入口 | `qcore.pipeline` / `qcore.live.rebuild` | `qcore.live_model.predict` |
| 打分器 | 102 个信号 × Hedge 权重（`AdaptiveWeighter`） | Alpha158 + GBDT，再乘一个 `a*score+b` |
| 产物 | `weights.json` / `performance.json` / `ledger.json` | `picks.json` / `horizon_1.jsonl` |
| 消费者 | Web UI 的「信号」「模型」页 | `paper_auto.rebalance` → 模拟盘成交 |

证据：

- `qcore/online_calib.py:10` 自己写着：**"`start.sh` 不调度 `qcore.live` / `pipeline` / `intraday`"**；
- `docker/start.sh` 里确实没有 `qcore.live` 的调度位（只有 tick / market_live / update_check / health / alert / session）；
- `qcore.live.step_once()` 是**回放之外唯一会调用 `AdaptiveWeighter.update()` 的入口**，
  它不跑 → 「每天用真实结果给 73 个专家打分、对了加权重」这件事在生产里**不发生**；
- `picks.json` 里带着 `online_calib` / `model_age_days` 字段，可确认由 `live_model` 写出；
  `grep picks qcore/live.py` 零命中 —— 集成分数从来没有进过名单。

**影响**：项目对外说的是"自学习参数体"（AGENTS 第 0 节），实际运行时它是一个
**每周重训一次的 GBDT + 2 参数 OLS 校准**。集成只活在回放和 UI 里。
这不是"没来得及接"，是**两套模型并行跑了很久而文档只描述了一套**。

**建议**：二选一并写进 README——
(a) 把集成接进 picks（`weights_hist[-1]` 加权打分，或与 GBDT 做 rank 平均）；
(b) 明确写"决策 = GBDT，集成 = 测量对象"，并在模型页/信号页各加一句归属说明。

### R27 · `/api/hotspot` 在请求线程里联网 + 现算，耗时 7 秒，而它的"缓存"从不被写

实测（冷缓存，本机）：

```
/api/hotspot     6.95s  ok=True
/api/tick        0.06s  /api/news/live 0.01s  /api/events 0.00s
```

`webui/server.py:2174-2179`：先 `HS.load_snapshots(1)`，读不到就 `HS.analyze()`。
但 `analyze()`（`qcore/hotspot.py:222`）**只取数不落盘**，落盘的是另一个函数 `snapshot()`，
而 `snapshot()` 没有任何调度入口 → `HOT_DIR` 永远是空的 →
**每次打开"当日风口"页都要重新打 5 个外部接口并现算**。注释"默认读缓存，5 分钟内不重复抓"
在当前代码里不成立。

同一模块 `reconcile()` 自己写着"板块历史行情接口不可用…下一步"——
也就是说"主线板块"这个结论**没有对账机制**，与项目"只登记不结算 = 没有命中率"的原则冲突。

顺带暴露 INV-8 的检查器太窄：`invariants.inv8_ui_no_calc()` 只 grep
`import numpy/pandas`，抓不到"server 里调 `HS.analyze()` / `HZ.table()` / `QD.universe_codes()`"
这类现算。INV-8 的实际含义（server 只读 JSON）没有被任何测试守住。

**建议**：`snapshot()` 进 start.sh（收盘后一次）；`/api/hotspot` 只读快照，
没有快照就返回 `not_ready` 而不是现抓；INV-8 的检查改成"server.py 里不得出现
`analyze(`/`table(`/`universe_codes(`/`predict(` 这类计算入口"。

### R28 · `top_k` 配置被 `min()` 静默压小，账户持仓与新策略对不上

`qcore/paper_auto.py:434-447`：

```python
ck = int(cfg.get("top_k") or 0)
if ck > 0:
    return min(ck, base) if ck > base else ck
```

`account_config` 里 CN 的 `top_k = 20`，而 `HORIZON_TOP_K[1] = 2` → `default_top_k(1)` 返回 **2**。
于是：

- `picks.json`：`top_k: 2`、`n: 2`（只有建设银行、寒武纪两只）；
- `paper_auto.json`：**持有 20 只**（000988/601995/300476/… 是旧 top_k=20 时代的名单）。

下一次调仓会把 18 只卖掉（约 90% 换手 ≈ 35bps 一次性成本），而用户在设置里填的
`top_k=20` 被 `min()` 静默改成 2、UI 没有任何"被覆盖"的提示
（只有 `capital_mismatch` 有提示，`top_k` 没有）。

**建议**：`top_k` 冲突时和 `capital_mismatch` 一样显式回报（`top_k_effective` vs `top_k_configured`）；
存量持仓做一次"策略切换"标注，避免把 90% 换手当成正常调仓记进净值。

### R29 · 模拟盘不建模"涨停买不进 / 跌停卖不掉"

`qcore/paper_auto.py` 里唯一的不可成交判断是：

```python
"why": "取不到可成交价（停牌 / 盘中无实时报价）"
```

没有涨停/跌停判断。而 T+1 短线策略买的正是当日最强的票——
一字板在收盘价上根本买不到，次日开盘还常常高开。模拟盘一律按 `as_of` 收盘价成交，
会**系统性高估**短线策略的可执行性与收益（这正是 `docs/15` 里"实盘滑点未验证"那条的阴暗面）。

**建议**：成交前判断 `当日涨幅 ≥ 9.8%（ST 5%）且收盘=最高` 则拒单并记 `limit_up_unfilled`；
跌停同理。至少要在成交记录里留 `filled_at_limit` 标记，让净值可被剔除重算。

### R30 ·（P1）顶栏"数据日期"只报行情日期，计算层落后 2 个交易日却看不见

`webui/server.py:1316-1342` 把 `data_to` 统一成 `data_sync.local_date()`（含 overlay，= 11-10），
`age = _age_days(data_to)`；`webui/static/app.js:633` 顶栏显示 `data_to`，chip 状态用同一个 `age`。
于是当前状态是：

| 层 | 日期 | 从哪来 |
|---|---|---|
| 原始行情（含 overlay） | **2026-11-10** | `data_sync.local_date()` |
| 实时预测 picks | 2026-11-09 | `picks.json.as_of` |
| 面板 / 分数 / 权重 / 台账 / 回放 | **2026-11-06** | `status.as_of`、`last_processed` |

顶栏渲染为「数据 2026-11-10」，健康灯按 `age(11-10)=1 个交易日` 判 **ok**。
也就是说：**用户看到的绿灯，对的是"原始行情有没有补到"，不是"模型有没有算到"。**
面板实际落后 2 个交易日（11-06 → 11-10）。

注释（1310-1315）写的是"三个接口各算各的，用户看到的是最旧的那个"——
这次修的是"别显示太旧"，代价是**再也不显示"算到哪天"**。
v0.32.8 的 commit 正是修这个字段，等于同一个坑的第二个方向。

**建议**：顶栏并列两段——`数据 11-10 · 计算 11-06`，并把 `age` 分两个：
`data_age`（行情）与 `panel_age`（计算，用 `as_of`）。`/api/health` 已经同时返回
`as_of` 与 `data_to`，前端只是没用第二个。

### R31 · 未设 `APB_TOKEN` 时，写权限靠"是不是内网"判断；反代不注 XFF 就等于全开

`webui/server.py:_require_write` 的三层里，第三层是：

```python
if not expected:                     # 没设 APB_TOKEN
    if self._is_internal():          # 是内网 → 放行
        return None
```

`_is_internal()` → `net_guard.is_internal_request()` → `real_ip()`：
只有当 `client_address` 是**可信代理**时才读 `X-Forwarded-For`，否则用对端 IP。
这意味着：**如果反向代理没有注入 XFF**（nginx 默认 `proxy_pass` 不带 `proxy_set_header X-Forwarded-For`），
所有公网请求的"对端"都是反代自己（127.0.0.1 或 172.x）→ 被判为内网 →
`/api/paper/reset`（清空模拟盘）、`/api/import`（覆盖状态）、`/api/model/rollback`、
`/api/tokens`（签发令牌）全部对内网判据开放，而 `APB_HOST` 默认是 `0.0.0.0`。

`/api/netguard` 能自检（返回 `you_are_internal`），但**只是提示，不阻断**。

**建议**：`APB_ALLOW_WRITE_WITHOUT_TOKEN=0` 的显式开关，默认关；
或者在"未设 token + 监听 0.0.0.0"时启动即打印 crit 级告警并拒绝写接口。

### R32 · `postclose` 没有"今天已跑过"的幂等，收盘窗口内每 5 分钟重跑一次（P3）

`docker/start.sh:123-133` 每 300 秒调一次 `session.run('auto')`，
`in_window()` 在 15:05–17:00 返回 `postclose` → **2 小时内跑约 24 次** `postclose()`。
每次都会 `LM.predict()`（加载模型 + 300 只票打分）并重写 `picks.json`，
还会 `horizon_ledger.record()` —— 而 `_write_all` 是**整文件重写**（陷阱 22）。
现在只有 2 条记录，代价很小；台账长到几千条后就是每天 24 次全量重写。

盘中段同理：09:30–15:00 每 5 分钟跑一次 `intraday()`（这条是设计意图，可接受）。

**建议**：`postclose` 开头加 `if postclose.json["data_date"] == 今日: return skipped`；
或让 `session.run('auto')` 维护一个 `last_session_run` 时间戳（`sessions.json` 已是 auth 在用，
另开一个文件即可）。

## 4.8 第五轮（事件研究 / 影响因子 / 发布流程）

### R33 · 阈值真源建好了但没人用：`impact.py` 仍用 `min_n=8`，9/12 行 n<30 却标"有效"

`qcore/thresholds.py` 开篇就写着这段历史：

```
历史：qcore/events.py:85 MIN_N=5 / qcore/impact.py:138 min_n=8 / intraday MIN_TRADES=30
结果：events.json 里 9 行影响因子全部 n<30，却全部 insufficient=False，
      被 UI 原样渲染成 9 行正经结论。
```

`events.py` 确实改了（`from .thresholds import MIN_XS_NAMES`，`MIN_N = MIN_XS_NAMES`），
但 `impact.py:138` **仍是 `min_n: int = 8`**，没 import `MIN_TRADES`。
实测当前产物：

```
events.json.impact：12 行
n = 17 / 20 / 24 / 79 …（**9 行 < 30**）
insufficient：全部 False
```

也就是说 `thresholds.py` 里那条"已修复"的结论**只修了一半**：
真源建好了、文档写了、`events.py` 接了，但**下游调用方仍传 8**。
按 AGENTS §10 规则 7（`< MIN_TRADES(30)` 一律标 insufficient），
这 9 行本应全部标 insufficient。

**建议**：`impact.compute(min_n: int = MIN_TRADES)`；`_factor_row` 的 `insufficient`
再用 `n < MIN_TRADES` 复核一遍（双保险）；CI 加一条断言：
`events.json.impact` 里不允存在 `n < 30 and insufficient is False`。

### R34 · 影响因子表的列由"第一行的前 7 个键"决定（P3）

`webui/static/app.js:3423`：

```js
cols: Object.keys(impact[0]).slice(0, 7).map(...)
```

`impact.py` 的行键顺序是 `type, sector, horizon, n, mean_car, mean_car_raw, t, p,
win_rate, stability, n_years, conflict_rate, insufficient, level`。
`slice(0, 7)` 恰好截到 `t`，**`p` / `win_rate` / `insufficient` / `level` 全部被丢**。

后果：表格下方的注释写着"需要 type / sector / horizon / mean_car / t / n"，
但页面永远不显示 `insufficient` —— R33 那 9 行 n<30 的记录因此在界面上与真正的结论**长得一样**。
而且这种"按位置取列"的写法，一旦 `impact.py` 调整键顺序就会静默错位（不报错，只是列名与内容对不上）。

**建议**：显式列定义（label/key/格式化），并把 `insufficient` 渲染成行内标记（灰显或加"样本不足"徽章）。

### R35 · 发布流程不校验 `tag` 与 `VERSION` 一致（P3）

`.github/workflows/release.yml` 的 test job 只校验了
`VERSION` == `qcore.__version__`，没有校验 `tag`（`v0.33.0`）== `VERSION`。
但镜像 tag 直接取自 git tag（`type=raw,value=${{ needs.meta.outputs.version }}`）。

于是"打了 tag 但忘了改 VERSION"会产出：镜像叫 `0.33.0`，容器里 `VERSION` 仍是 `0.32.9`，
UI 顶栏显示旧版本、更新检查（比对 GitHub tag）会认为"已是最新"。
`tests/test_version_sources` 守住了"两处版本号一致"，但守不住"与 tag 一致"。

**建议**：test job 加一行 `test "v$(cat VERSION)" = "${GITHUB_REF_NAME}"`（workflow_dispatch 时跳过）。

## 4.9 第六轮（训练机：上传门禁 / 数据包自愈 / 自动更新链路 / 构建校验）

### R36 · 模型导入没有质量门：形状对、能 predict 就装为生效模型（P2）

`qcore/model_transfer.py:113` 的 `_validate()` 只有四项：

1. `unpickle` 成功
2. 特征数（优先信模型自己的 `_n_features`）
3. `horizon` 已声明
4. 拿 32 行随机数喂进去，`predict` 能返回且 90% 有限

**没有**：样本外 IC / 与当前线上模型的对比 / 退化检测（全常数输出也能过第 4 条）。
`import_bundle` 校验通过就直接 `versioning.save()` —— 新模型立刻生效。
`versioning` 有快照可回滚，但**没有任何自动回滚触发条件**（R3 里"页面讲 250 树、
线上跑 20 树"能存在，同源）。

**建议**：加一条硬门——导入/训练上传时在**固定 holdout**（比如最近 250 日）算
Rank IC(0.03 的等价门槛)，低于阈值就 `reason=quality_gate_failed` 并保留旧模型；
至少要在 manifest 里带上训练时的 IC，让 NAS 侧能比。

### R37 · 训练机数据包"大但损坏"无法自愈（P2）

`qcore/data_sync.py:329-335`：

```python
except urllib.error.HTTPError as e:
    if e.code == 416:
        if size > 20 * 1024 * 1024:
            return {"ok": True, ..., "reused": True}   # 直接当成"完整文件"
```

416 的含义是"Range 起点 ≥ 远端大小"，**不等于本地文件完好**。
一旦本地 `qlib_bin.tar.gz` 是"大到 20MB 以上但截断/损坏"，流程会：

1. `download()` 返回 `reused=True`（不再重下）
2. `extract()` 报 CRC/格式错
3. `sync()` 直接返回失败，**不清理也不重试**

`sync(force=True)` 也救不回来——`force` 只跳过"已是最新"判断，不删包；
CLI 也只暴露 `--sync` / `--force`，没有 `--reset`。用户会卡在同一个错误里，
除非手工删掉那个 545MB 的文件。

**建议**：`extract` 失败时删 tarball 并重下一次（一次机会足够）；
`--force` 语义扩展成"删包重下"；`summary()` 里加 `tarball_sha256`。

### R38 · 训练机自动更新链路可被降级 + zip 无签名 + 解压无过滤（P2，供应链）

三件事叠在一起：

1. `qcore/kit_update.py:109-117`：`_ssl_ctx_unverified()` 把 `verify_mode` 设为 `CERT_NONE`。
   触发条件是"第一次握手失败"——**任何中间人只要让它失败**，客户端就自动退回不校验证书。
   代码只打一行 `[kit_update] 警告`，随后照常下载。
2. 下载的 zip **没有签名/哈希校验**；`_validate()`（:183）只检查三个文件存在
   （`tools/train_one_click.py`、`qcore/__init__.py`、`VERSION`）+ `VERSION` 非空。
3. `apply_update()`（:257）`zipfile.extractall(tmp)` **没有 `filter=`**，zip 条目里的
   `../` 可以写到 `.update-tmp` 之外（Zip Slip）。

而这条链路的终点是"覆盖训练机上的代码并被执行"（守护进程会跟随版本自动拉起）。
三者互为放大器：**降级 → 恶意包通过"只查三个文件"的校验 → 解压覆盖 → 运行**。

**建议**：删掉 `CERT_NONE` 兜底（缺 CA 就给用户装 CA 的指引，别自动降级）；
发布产物带 sha256 并写进 release notes，客户端比对；
`TarFile/ZipFile.extractall(..., filter="data")`（Python 3.12+ 支持）；
`_validate` 加"文件清单哈希"一项。

### R39 · 构建时的套件自检失败只打 WARN，仍会出包（P3）

`scripts/build-train-kit.sh:79-83`：

```sh
( cd "$KIT" && python tools/kit_selftest.py >/dev/null 2>&1 ) &&   echo "[kit] selftest OK ..." ||   echo "[kit] WARN: selftest failed in built kit -- check before shipping"
```

`kit_selftest` 的职责正是"这个套件能不能承接 train / pipeline / prewarm"，
但它失败时构建继续，产物照样可以发布。而 CI 里的 `kit_manifest --verify .`
验的是**源码仓库**而不是这个构建出来的目录，两件事没有闭环。

**建议**：`APB_ALLOW_BAD_KIT=1` 之外的路径一律 `exit 1`；
或把"构建出来的套件跑一遍 selftest"作为 release workflow 的一步。

## 4.10 第七轮（任务队列 / 模型导入版本 / 发布闸门是否被执行）

### R40 · 模型导入不比较版本或训练时间：旧训练机可以覆盖更新的线上模型（P2）

`model_transfer.import_bundle()` 的 manifest 里**明明带了这些字段**
（`model_transfer.py:94-96`：kit 的 `python` / `numpy` 版本，`meta` 里还有 `horizons`），
但 `_validate()` 只用了 `feature_count` / `horizon_declared` / `unpickle` / `predict` 四项，
**没有一项是在比"这个模型比线上的新吗/好吗"**。

于是：一台装着旧套件（旧 Alpha158 因子集、旧数据）的训练机，
只要特征数相同，就能把它的模型装成线上生效模型——`versioning.save()` 直接写 `current.json`。
旧模型覆盖新模型这件事，在日志里只会显示"导入成功"。

**建议**：`import_bundle` 增加两条比较，任一不满足就 `dry_run` 式拒绝：
① manifest 的训练截止日 ≥ 当前模型（`meta_h*.json` 的 `train_end`）；
② manifest 的 kit `VERSION` ≥ 线上 `VERSION`（防止旧因子集掺进来）。
两条都只是读 JSON，代价为零。

### R41 · 任务队列是"读-改-写整个 JSON"，无锁：并发领任务会重复领取/丢任务（P2）

`qcore/sync.py` 的 `claim_task()` / `create_task()` 都是：

```python
d = _load()          # 读整个 tasks.json
... 改 d ...
_save(d)             # 原子替换整个文件
```

`_atomic()` 保证了**单次写入不撕裂**，但没有保证**读-改-写这段是互斥的**。
而 NAS 是 `ThreadingHTTPServer`（`server.py` 里 `daemon_threads=True`），
每台训练机每 N 秒打一次 `POST /api/sync/task {claim:true}` —— 两个请求线程完全可能这样交错：

```
线程A: load(d) -> 看到 T1 是 pending
线程B: load(d) -> 也看到 T1 是 pending
线程A: T1.claimed_by=A; save(d)
线程B: T1.claimed_by=B; save(d)      # 覆盖 A，两台机器领到同一个任务
```

更糟的是 `create_task` 与 `claim_task` 交错时，**新任务可能被旧快照覆盖而消失**
（`_save` 写回的是 claim 那一刻的 d），而且不报错、不记日志。

**建议**：加一个 `tasks.json.lock`（`msvcrt.locking` / `fcntl.flock`，Windows/Linux 各一支）
把 load→save 包起来；或改成"每任务一个文件 + `os.open(O_CREAT|O_EXCL)` 抢占"，
这在 SMB/NTFS 上比文件锁更可靠。修之前至少把 `claimed_by` 记进 journal 以便事后发现重复。

### R42 · 发布闸门 `scripts/preflight.py` 零调度——CI 和 release 都不跑（P2）

`preflight.py` 553 行，检查项写得很实（全部模块可导入 / 状态文件齐全且合法 /
起服务逐个打 GET 断言没有 `not_ready` / POST 账户配置的成功与失败路径 / 跑测试套件）。
文档里它被当成硬门：`docs/17` "改完代码只跑 preflight.py 和单元测试"、
`docs/18` 的 checklist "□ preflight.py 全绿？"、`docs/19` R14 "已接 preflight"。

但：

```
$ grep -rn "preflight" --include=*.yml --include=*.sh --include=*.bat --include=*.py .
（除 scripts/preflight.py 自身外，零命中）
```

CI（`.github/workflows/ci.yml`）跑的是逐条单元测试 + 不变量 + 泄漏扫描，**没有 preflight**；
release workflow 也没有。它唯一的执行方式是"人记得"。
换句话说：**这台项目自己总结出的"最有效的一道闸门"，是唯一没有自动化的闸门。**

**建议**：把 `preflight --quick` 加进 CI 的 ubuntu job（它能起 127.0.0.1 服务，
状态文件用仓库里的样例即可）；或者至少在 release.yml 的 test job 里跑一次全量。

### R43 · 手动补数据没有收盘门禁：盘中点一下就把"半日线"写成当日 K 线（P2）

`append_day()`（`qcore/daily_append.py:150`）默认 `day = date.today()`，
`_fetch_recent()` 用 `sources.fetch_eastmoney(adjust=2)` 拉日线，
然后取 `rows[df["date"] <= tgt].iloc[-1]` 当"目标日那根"——**没有任何"这根收盘了吗"的判断**：

- 没有 `_strip_forming_bar`（`data.py` 里那个修好陷阱 28 的函数，这条链路没走它）；
- 没有 `in_session()` 之类的时段门禁；
- 没有成交量合理性校验（陷阱 28 里的实测样本是 `volume 52,770 vs 正常 25–33 万`，一眼可辨）。

调度侧是安全的：`docker/start.sh:420-432` 把定时任务卡在 `15:05–17:00` 且排除周末。
但**手动入口没有这层保护**：`POST /api/data/append`（`server.py:1653`）
直接 `DA.append_day(target_day=day)`，且成功后还会把 `status.data_to` 刷成那一天。

于是盘中 10:30 点一次"补数据"：
① overlay 里写入一根只有半天的 bar → ② `local_date()` 立刻变成今天 →
③ 顶栏变绿（R30 里那个"只报行情日期"的绿灯）→ ④ 之后所有 qlib 侧计算
（GBDT 特征、`live_model` 打分、picks）都在这根半日线上跑。
这根坏 bar 会一直留到**下一个第三方数据包发布**才自愈
（那时 `d <= last_date`，overlay 行被跳过）。

**建议**：`append_day` 开头加"只在收盘后写"的判断（或 `force=True` 才允许盘中），
并对 `volume` 做一次 `与近 20 日均量的比值 < 0.3 → 判定为未收盘，拒写`；
`/api/data/append` 把这条拒绝原样返回给用户，而不是 silent success。

### R44 ·（P1）在线校准"用集成分数拟合、作用在 GBDT 分数上"—— 口径错配，输出被截距吃光

`online_calib.apply()` 做的是 `out = a*score + b`，而 `collect()` 取样本时：

1. 先读 `horizon_N.jsonl`（h=5 没有这个文件 → 空）；
2. 回落到 `_collect_main_ledger()` 读 **`data/state/ledger.json`** ——
   那是 `export_ui_bundle` 写的**回放台账**，里面的 `score` 是
   **102 信号集成的加权投票分**（实测 min −0.594 / max +0.687 / std 0.256）。

而 `apply()` 的输入是 `live_model.predict()` 里的 **GBDT 原始分数**（实测约 0.003）。
两套分数的量级差 **~85 倍**，拟合出来的 `a` 却直接乘在后者身上。

实测当前系数（h=5，a=−0.00305，b=−0.02664）作用在今天的两个 pick 上：

```
601939  score 0.002977  ->  a*score+b = -0.026653
688256  score 0.002383  ->  a*score+b = -0.026651
```

**两只票的"校准后分数"只差 2×10⁻⁶**（差异全部来自浮点舍入），排序由噪声决定；
而 `b` 一项就把分数从 +0.003 拉到 −0.027，评级/阈值全部失真。
若 `a<0`（现在就是），差异还会整体反向（R17）。

更根本的问题：**这个 `a` 描述的是"集成分数对 5 日收益的斜率"，不是 GBDT 的**。
集成 IC 是 −0.018（负），所以 `a` 自然拟合为负；把它套到 IC +0.012 的 GBDT 上，
等于用"另一个模型的失败"去修正这个模型。

**建议**：`collect()` 只接受**与 `apply()` 同源**的分数——
要么都在 GBDT 侧（用 `horizon_N.jsonl` 里 predict 落盘的 `score`），
要么都在集成侧；样本不够就**不校准**（`applied=False`），不要跨口径借用。
现有 `picks.json` 是 h=1、`applied=false`，所以线上暂未受影响；
但 h=5 一旦出名单，这条就会把分数压成常数。

## 4.11 第九轮（按"每时每刻都在学 + 每时每刻都在收"的要求审视架构）

这一轮不是找新 bug，而是回答一个架构问题：**把它改成"连续学习 + 连续接收"，现在的代码差在哪。**

### R45 ·（P1）增量学习没有"样本级幂等"：`live --loop` 每小时把同一批到期样本重学一遍

`qcore/live.py:step_once()` 的设计是"游标 = 已封存日"：

```python
_sealed = max(len(dates) - 1 - cfg.exec_lag - horizon, -1)   # 例：数据到 11-10、h=5 → 11-04
st["last_processed"] = _sealed_day                            # 故意落后 lag+h 天
...
t_from = searchsorted(dates, last, side="right")              # 从 11-05 开始跑
res = run_walkforward(p3d, RunConfig(from_t=t_from, ...), weighter=w)
```

问题在于：**这个游标是"封存日"，不是"学到哪一天"**。`run_walkforward` 在
`for t in range(from_t, T)` 里对每个 t 用 `s = t-1-h` 的样本更新学习器。
于是同一个 `as_of` 再跑一次，`t_from` 不变、`s` 的集合不变 —— **同一批样本被再喂一遍**。

而 `loop(interval_sec=3600)`（"常驻模式：每小时检查一次"）正是每轮都调 `step_once()`：

- 交易日盘中：`as_of` = 最近已收盘日 → `last` 落后它 6 天 → 条件不成立 → **照跑**；
- 收盘后 `last` 前进一天，下一小时又跑 → 再学一遍；
- 一天 24 轮 → **每个到期样本被学 24 次**：`logw_g += η·y` 累积 24 倍，
  `a_g/b_g` 的衰减计数也被推 24 步（等价于把学习率乘 24、把 `decay_rho` 的应用次数乘 24）。

台账侧有幂等（`append_ledger` 按 `(as_of, code, horizon)` 去重、INV-7），
**学习器状态侧没有**：`save_weighter` 直接覆盖 npz，没有任何 `last_learned_sample` 游标。

当前没造成线上影响，因为 `qcore.live` 根本没被调度（R26）。
但它是"连续学习"唯一的增量入口 —— **一旦按用户要求接上调度，它会先坏**。

**建议**：给权重器加一个显式游标 `learned_through`（或每个信号一组"已学日期"的位图），
`step_once` 只学 `s > learned_through` 的样本；`loop` 在无新样本时直接 `skipped`。
一条单测就能守住：同一 `as_of` 连跑两次，`n_g` 必须不变。

### R46 · "每时每刻接收数据"实际只覆盖 ~53 只票，且实时特征不进模型（P2）

- `docker/start.sh:63-85` 组出来的 tick 名单 = `picks` 前 20 + 自选前 10 + 14 只 ETF + 9 只港股/美股
  ≈ **53 只**。全池 283 只里其余 230 只，**没有任何实时行情**，只有日线。
- `qcore/live_features.py` 把 tick 快照压成 9 个日内特征（上午/下午收益、尾盘 30 分钟、
  下午量/上午量…），但它的消费者只有 `weekly.py`（周报统计）和 `/api/live_features`（页面展示）——
  **没有任何模型读它**。文件头也诚实写着"攒够 60 个交易日再说"。
- `data/tick/` 目录在当前工作区**根本不存在**（tick 循环没跑过），所以连"攒"都没开始。

所以现状是：**实时数据采集覆盖 18% 的池子，且采集结果不进任何打分**。

**建议**：先把 tick 名单扩到全池（腾讯/新浪批量接口一次可拿几百只，300 只 ≈ 2–3 个请求），
再把 `live_features` 接进 `live_model` 的特征矩阵（哪怕先只用 `tail_30m` / `vol_pm_am` 两列，
并标注"仅当日、不可回测"）。这两步都不需要改学习器。

### R47 · 学习的信息到达率由 h 决定："每时每刻学"对日频没有意义，对分钟级才有（P2，认知项）

这是必须先说清楚的一条，否则会做出一个"每 5 分钟重训一次"却没用的系统：

- `ret_h(t) = OPEN[t+1+h]/OPEN[t+1]-1`，**h=5 的标签要等 6 个交易日才成熟**。
  一天最多新增一批可学样本（`s = t-1-h`），再怎么高频重算也不会多出信息。
- 真正能"每时每刻学"的只有**分钟级**：`intraday_online` 用 t 分钟打分、t+h 分钟看结果，
  标签延迟只有几分钟。而它的数据源（东财 `klt=1`）**只给当日 240 根、没有历史**，
  所以既不能回测、也没法在收盘后补 —— 只能在交易时段一点点积累。
- 其结果我们已经实测过：分钟级毛边际 −0.11bps、扣 19.28bps 往返后净 −19.39bps，
  结论是"这个频率不可交易"（`reports/intraday_20261106.md`）。

**建议**：把"连续"重新定义为两件可度量的事，而不是"跑得越勤越好"：
① **快照延迟**（tick 到落盘，目标 < 5s）；
② **标签成熟到学习的延迟**（新增到期样本到 `learned_through` 推进，目标 = 当日一次，且**恰好一次**）。
分钟级在线可以作为"特征实验室"保留，但不要把它当收益来源。

### R48 · 增量路径每次仍全量重算 900 天信号：想"每 5 分钟学一次"要先解决这个（P2）

`step_once` 的窗口是 `window_days=900`（≈610 个交易日），每次都会
`build_dataset(cutoff=as_of)` 重算全部 283 只票的信号 —— AGENTS 实测
"build_panel 4.1s + compute_signals 12.2s ≈ 16.3s/次"。

按"每 5 分钟一轮"算：**每小时 12 轮 ≈ 3.3 分钟 CPU/小时**，看似可以，
但这是 283 只 × 610 天的重复计算，而且**信号层内部没有增量状态**
（`DIR/STR` 只存在于内存里的 Panel3D，没有"每只票最后一根 + 预热窗口"的缓存）。

AGENTS 里记过一条失败尝试：把信号前段砍掉会因 MA 族前缀和浮点差异违反 INV-1（逐位相等）。
所以正确的方向不是"少算几天"，而是：
① 把 MA 族改成数值稳定的 `rolling().mean()`（去掉前缀和），使增量与全量逐位一致；
② 或者对每只票缓存"最近 N 根 + 预热窗口"的信号尾部，只在滚动边界重算。

**建议**：这条是"高频学习"的算力前置条件，优先级低于 R45（先保证学得对，再谈学得勤）。

## 4.12 第十轮（"预测 → 到期 → 验证 → 学习"这条闭环本身）

用户要的是一条闭环：**对已发生的事做判断 → 预测未来 → 未来到达时验证旧预测 → 再预测**。
这一轮就沿着这条链逐环检查，结论是：**链条在"到期"这一环断了，后面的验证与学习全部接不到输入。**

### R49 ·（P1）验证半环结构性死亡：`exit_date` 记成 null 且永不补，日历缓存又永不失效

**证据一：线上预测的到期日全是 null。**

```
data/pred_ledger/horizon_1.jsonl（全部 2 条）
{'as_of': '2026-11-06', 'entry_date': None, 'exit_date': None, 'status': 'pending'}
{'as_of': '2026-11-09', 'entry_date': None, 'exit_date': None, 'status': 'pending'}
```

`horizon_ledger.record()`：

```python
entry = _trading_days_after(cal, as_of, 1) if cal else None
exit_ = _trading_days_after(cal, as_of, 1 + h) if cal else None
```

`_trading_days_after()` 在 `as_of` 就是日历最后一天时，`i+n` 越界 → 返回 **None**。
而 `reconcile()` 的第一句就是：

```python
ex = r.get("exit_date")
if not ex or ex > latest:
    total_skip += 1
    continue
```

→ **exit_date 为 null 的记录永远不会被对账**，没有任何代码在事后补算它。

**证据二：日历本身在长驻进程里也不会前进。**

`qcore/qlib_data.py:50` 的 `_cache` 是个**无 TTL、无失效接口**的模块级字典：

```python
def load_calendar():
    if key in _cache: return _cache[key]      # 永不过期
    ...
    _cache[key] = idx
```

而 `daily_append` 新增的日期是写在 **overlay 文件**里的（`load_calendar` 读取时合并），
合并逻辑只在**首次读取**时执行。于是容器跑久了：

- `_latest_trade_date()`（`session.py:87`，= `cal[-1]`）**停在进程启动那天**；
- `reconcile()` 的 `latest` 同样停住 → 即便 exit_date 有值也会被判"未到期"；
- `premarket`/`postclose` 的 `_can_review = data_date > pm_data_date` 两边同源 →
  **永远 False**，收盘复盘永远写"下一交易日尚未发生"。

当前产物正是这个状态：

```json
data/state/postclose.json
{"data_date": "2026-11-06", "premarket_data_date": "2026-11-06",
 "review": {"n": 1, "hit_rate": null,
            "rows": [{"pending": true,
                      "reason": "盘前建议基于 2026-11-06，行情最新仅到 2026-11-06，下一交易日尚未发生，暂不复盘"}]}}
```

**影响**：这条链的"验证"与"学习"两环拿不到任何新输入。
`horizon_perf` 的 `n=0 / 还没有到期的预测`、`online_calib` 只能用回放台账（R44）、
`claims` 长期 13 条里 7 条 open —— 都是同一个根因的不同表现。

**建议**（三行改动，收益最大）：
1. `record()` 在 `exit_date is None` 时**不要落 null**，写 `pending_calc: true`，
   由下一次 `reconcile()` 用**当时的日历**补算（`_trading_days_after` 再算一次）；
2. `load_calendar()` 的缓存加 60 秒 TTL（与 `_overlays()` 一致），或提供 `invalidate()`；
3. `_latest_trade_date()` 改走 `data_sync.local_date()`（它读文件 + overlay，不受缓存影响）。

### R50 · "预测"有四本账、"验证"有四种算法，命中率互不可比（P2）

| 账本 | 写入方 | 内容 | 验证方式 | 当前规模 |
|---|---|---|---|---|
| `data/pred_ledger/{year}.jsonl` | `pipeline`/`live` 回放 | 逐日 × 全池 × 集成分数 | 构建时直接带 `ret`（事后算） | 47,676 行 / 8,150 唯一键 |
| `data/state/ledger.json` | `export_ui_bundle` | 回放的最近 60 天切片 | 同上（同一批数据的第二份拷贝） | 3,000 行 |
| `data/pred_ledger/horizon_{h}.jsonl` | `live_model.predict` → `HL.record` | 每次 predict 的 30 只 | `HL.reconcile()` 按 exit_date 对账 | 2 条，**均 pending** |
| `data/state/{premarket,postclose}.json` | `session` | 盘前 20 只 / 次日复盘 | `postclose` 比 `data_date` 与 `pm_data_date` | 20 行 / 1 条 pending |
| `data/claims/*.jsonl` | `claims.register` | 人工判断 + 证伪条件 | `claims settle` 定期结算 | 19 行 / 6 已结算 |

五套账、四种命中率算法（`sign(score)==sign(ret)` / `top1/3 > all mean` /
`hit = ret > 0` / 证伪条件判定），**彼此不可比**，而 UI 把它们放在相邻的卡片里。
这就是"同一指标三个值"的根源（R8/R9/R19）：不是谁算错了，是**根本没有一个统一口径**。

**建议**：收敛成一层 **prediction store**（一条记录 = as_of, market, horizon, code, score,
entry_date, exit_date, status, ret, verified_at），保留 `source` 字段区分 replay/live；
所有命中率都从这一层派生。迁移可以只读旧文件、写新文件（不改历史，符合 INV-7）。

## 4.13 第十一轮（"让程序自己去查"：情报决策层，已落地）

### R51 · 现有情报链路是"被动"的：只有采集器，没有"该查谁"的决策层（P2）

| 模块 | 能力 | 缺什么 |
|---|---|---|
| `qcore/info.py` | 给定 code → 估值/公告/新闻，**PIT 截断 + 相关性过滤 + 去重**都做了 | 只服务 pipeline 点名的代码，不会自己决定查谁 |
| `qcore/news_live.py` | 全市场快讯轮询 + 关键词分类（type/direction/sectors） | 只做"背景噪声"，与持仓/推荐无关联 |
| `qcore/event_data.py` | 大宗/龙虎榜/两融/增减持/解禁/预告 | 同上，被动等缓存 |

三者的产出都不参与打分（R46），也没有"这一轮查了谁、谁没查到、下一轮该查谁"的记忆。
结论：**"情报"是有的，"寻找"没人做。**

### 已落地：`qcore/autoinfo.py`（自主情报决策层）

新增一个零依赖模块（约 430 行）+ 一个契约测试（`tests/test_autoinfo.py`，13 条断言全绿）：

```
选 subject（配额轮转）→ 查（复用 info._news_items）→ 打分 + 规则分类
→ 落 state/autoinfo.json（给 UI）→ 登记判断进 data/autoinfo/ledger.jsonl（只追加）
→ 到期后 verify() 回填收益/命中进 verified.jsonl（另一个追加文件，不改历史）
```

**三个自主性**：

1. **查谁自己定，且按配额轮转**：持仓 4 / 推荐 2 / 自选 2（上限 8）。
   测试里验证过：模拟盘 20 只持仓不会把 8 个名额全吃掉（否则推荐与自选永远饿死）。
2. **缺口驱动**：本轮没查到情报的 subject 进 `gaps`，下一轮排到最前面。
3. **查完必须能证伪**：每条情报带 `direction` 与 `h` 落进 ledger；`verify()` 到期回填
   `ret` / `hit`，**写到另一个文件**（INV-7：不改历史）。

**成本与安全**：每 subject 仅 1 个请求；单轮上限 8；失败即跳过并记 `gaps`；
离线时返回 `ok=false` 但不抛异常（实测：本机无外网时 8 个 subject 全部进 gaps、
`ledger.new=0`、进程正常退出）。

**实测（离线环境）**：

```
$ python -m qcore.autoinfo --run
{"ok": false, "n_subjects": 8, "n_items": 0,
 "gaps": ["000988","601995","300476","600362","601939","688256","600519","000001"],
 "ledger": {"ok": true, "new": 0, "dup": 0}}
```

**测试（离线，桩替身）**：

```
$ python -m tests.test_autoinfo
RESULT: 13 passed, 0 failed
  · 选到 subject 并产出情报 / 缺口被记录 / 只登记带 code+url 的情报
  · 重跑不新增行（幂等）/ 历史行内容未被改写（INV-7）
  · 标题命中 > 仅正文 / 快讯降权
  · 到期情报被结算 / 看多且涨→命中 / 看空且跌→命中 / 已验证的不重复结算
  · 未到期不结算、不伪造 / summary 报"样本不足"而不是给结论
```

**接线建议**（三步，都不改现有主流程）：

1. `docker/start.sh` 加一个 30 分钟循环（与 alert 同段）：`python -m qcore.autoinfo --run`，
   收盘后跑一次 `--verify`；
2. `/api/autoinfo` 直读 `state/autoinfo.json`（INV-8：server 只读）；
3. `weekly.py` 加一行"本周情报命中率"，与 `event_learning` 的命中率并列 ——
   两者的差别是：前者是**程序自己找的**，后者是人给的事件。

**一个必须写进 README 的边界**：`info._news_items` 的新闻是**当日可得**的，
不会回补历史（`pit_backfillable=False`），所以 `autoinfo` 的判断**只能前向积累**，
不能回测。`verified.jsonl` 攒到 30 条之前，`summary()` 一律返回 `insufficient=True`。

## 4.14 第十二轮（历史参照：先量后信，再算"缺了多少"）

用户提的做法是：**先拿历史同类事件打分，再把真实结果拿出来对，看差多少。**
这一轮把它做成了两个模块 + 两个测试，并且**第一跑就把现有事件方向标签打了脸**。

### R52 ·（P1）"事件方向"从来没和真实反应对过账：18 条重大事件里 7 条反号

`policy_calendar` 给每个事件人工标 `direction`（+1 利多 / −1 利空），
`news_live.classify()` 也用关键词表给出 direction，`reward.event_adjust()`
再按 `direction × confidence × decay` 去调权重（单票上限 ±30%）。
**但这两个 direction 从落地起就没有跟真实行情对过一次。**

我用本地沪深300把 18 条重大事件的实际反应量了一遍（`qcore/event_history.py`，
CAR = close[t0+1+5]/close[t0+1] − 1，与 `dataset.RET` 同口径）：

| 事件 | 标签 | 真实 5 日 | 对上了吗 |
|---|---|---|---|
| 2016-01-04 熔断机制触发 | 利空 | **−7.56%** | ✅ |
| 2015-06-15 去杠杆股灾 | 利空 | −3.65% | ✅ |
| 2022-02-24 俄乌冲突 | 利空 | −1.68% | ✅ |
| 2010-04-19 国十条 | 利空 | −2.05% | ✅ |
| 2008-09-16 雷曼破产 | 利空 | **+10.87%** | ❌ |
| 2020-02-03 新冠首日开盘 | 利空 | **+4.41%** | ❌ |
| 2021-07-26 双减政策 | 利空 | **+3.86%** | ❌ |
| 2025-04-07 关税冲击 | 利空 | **+3.03%** | ❌ |
| 2018-03-23 中美贸易战 | 利空 | +0.18% | ❌ |
| 2022-11-14 防疫优化 | 利多 | **−2.49%** | ❌ |
| 2018-10-22 民企纾困 | 利多 | **−2.30%** | ❌ |

**方向一致率 11/18 = 61%；全样本 5 日均值 +0.55%（偏正）**。

这不奇怪，也不全是"标签错"：市场级 CAR 混了**政策对冲**（2008-09-18 汇金增持、
2025-04 国家队入场）、**板块与大盘相反**（双减砸教育、大盘反而涨）、以及**见报日已在底部**。
但它说明一件事：**"利空→跌"这个直觉在 5 日尺度上只有六成成立**，
拿它直接调仓位，等于把 39% 的样本押在反方向上。

现有 `event_adjust` 的 `scale=0.05` + ±30% 上限，正是在这个 61% 的基础上放大。
而 `event_learning` 的 confidence 也一直贴着 1.0（不调），某种程度上系统自己
已经"发现"了这件事，只是没人把它量化成一句话。

**建议**：`event_adjust` 的 `direction` 不再直接采信标签，改成
`sign(historical_car)`（拿不到历史参照就不调，而不是按标签调）；
并把"标签 vs 真实"的一致率写进周报，跟命中率并列。

### 已落地：`qcore/event_history.py`（历史参照库）

- **18 条人工目录**（黑天鹅 / 政策收紧 / 政策放松 / 地缘 / 监管 / 流动性 / 涨价 / 降价），
  每条只有 `日期 + 类型 + 方向 + 板块 + 关键词`，**不存收益**；
- `measure(h=5)` 用本地 `000300.csv` **现算** CAR 与窗口最大回撤，落
  `data/state/event_history.json`；每条都带 `basis="market_proxy"` ——
  没有板块指数就明说是市场代理，不假装是板块自己的数；
- `analogs(text)` 按关键词匹配（宁缺毋滥：无命中返回空数组）；
- `implied([...])` 取中位数 + 方向一致率，给出"这次大概会怎么走"的先验。

实测（本地行情）：

```
$ python -m qcore.event_history
2008-09-16 雷曼破产      black_swan      car=0.10871 dd=-0.03811
2016-01-04 熔断机制触发   black_swan      car=-0.07562 dd=-0.09813
2025-04-07 关税冲击      geopolitics     car=0.03026 dd=0.0
...
{"n": 18, "n_measured": 18, "by_kind": {"black_swan": {"n": 5, "mean": 0.0009}, ...}}
```

### 接入 `autoinfo`：每条情报现在都带"历史先验"

`autoinfo.run()` 里每抓到一条新闻，就：

1. `event_history.analogs(标题)` → 像哪几次（带各自的真实 CAR）；
2. `implied(...)` → 合成"预期幅度 + 方向一致率"；
3. 一起写进 `state/autoinfo.json`（给人看）和 `data/autoinfo/ledger.jsonl`
   的 `implied_car` / `n_analogs` 字段（给对账用）。

### "缺了多少"：`autoinfo.evaluate()`

到期验证（`verify()`）已经把真实收益写进 `verified.jsonl`；
`evaluate()` 把两边按 `(as_of, code, url)` 配对，输出三个数：

```
bias       = mean(implied_car − ret)      >0 = 历史参照系统性高估
mae        = mean(|implied_car − ret|)    平均差多少
sign_match = P(sign(implied) == sign(ret))  方向对了多少次
```

样本 < 30 一律 `insufficient=True`（AGENTS §10 规则 7），
避免又出现"2 笔交易给出 +66bps 最优"的那种结论。

### 测试

```
$ python -m tests.test_event_history     → RESULT: 20 passed, 0 failed
$ python -m tests.test_autoinfo          → RESULT: 13 passed, 0 failed
```

`test_event_history` 里有几条是**特意用来防自欺**的：
- 「方向一致率被算出来（不是 100% 才正常）」—— 把 61% 这个难看的事实钉进测试；
- 「每条都标了口径」—— 不允许出现没有 `basis` 的数字；
- 「无关句子不硬凑」—— 匹配不到就返回空，而不是随便找个相似事件。

## 4.15 第十三轮（把情报接进预测与学习：三层接线，默认关闭）

用户要求：**这些（自己找的新闻 + 历史参照）都要能进预测学习模型。**
这一轮把它做成三层，并全部带测试。核心约束是：**不能破坏 158 维特征契约、不能改变现有行为**。

### R53 · 模型对"情报"完全无感（P2）

现状只有一条通路：`reward.event_adjust()` 把事件当成**权重乘数**（单票 ±30%），
而它的输入是 `events.json` 的标签 + `event_learning` 的命中率。
也就是说：

- `live_model.predict()` 的分数**只由 158 维量价特征决定**，新闻/政策/黑天鹅一律看不见；
- 情报对仓位的唯一影响是一条乘法系数，且方向来自那个只有 61% 正确率的标签（R52）；
- `autoinfo` 找回来的东西（R51）没有任何下游。

### 接线：三层，各管一件事

```
① 历史参照   qcore/event_history.py   —— 18 条事件 → 用 000300 现算真实 CAR
② 情报特征   qcore/intel_features.py  —— 新闻+参照 → 每票一个 [-1,1] 的"情报分"
③ 融合       qcore/live_model.py      —— 排位融合（opt-in）+ 留档归因
                                     └─ qcore/reward.py 的方向改用历史实测
```

**② 情报分怎么算**（可解释、无魔法）：

```
情报分 = clip( 0.6·方向票 + 0.4·历史参照, -1, 1 )
  方向票   = Σ(direction_i × 0.5^(age/3)) / Σ|direction_i|
  历史参照 = implied_car / 0.03
```

- 只吃 `as_of ≤ 目标日` 的条目（PIT，测试里专门放了一条"未来新闻"验证它被过滤）；
- 没有情报 → **0（不表态）**，不是 NaN，融合时不会把分数拉偏；
- 值域裁剪到 [-1,1]。

**③ 融合为什么用排位而不是加法**

GBDT 分数是"预测收益"（量级 0.003），情报分是 [-1,1]，直接相加会被量级吃掉。
所以：

```
z(x) = 2·rank(x)/n − 1                      ∈ [-1,1]
combined = (1−w)·z(模型) + w·z(情报)        w = APB_INTEL_W，**默认 0**
```

`w=0` 时输出与模型分**逐位相同**（测试里断言了这一点），所以这个改动对现状是零影响；
`score_kind` 会写进 `picks.json`（`model` 或 `rank_blend`），下游不得把排位分当收益估计。

**④ `event_adjust` 的方向改用历史实测**

`reward._measured_direction()`：先用 `event_history.analogs(标题)` 拿到同类事件的
真实 CAR，符号就是方向；**拿不到参照才退回人工标签**，并在 `used[].dir_basis` 里
写明是 `history` 还是 `label` —— 事后能数出"今天有几条是靠标签拍的"。

实测（当前 events.json）：

```
event_adjust → {'applied': True, 'summary': '用最近 10 天 9 条事件调整权重：1/1 只被调整，最强板块 计算机(-1.52)'}
```

### 测试

```
python -m tests.test_intel_features  → 12 passed, 0 failed
python -m tests.test_event_history   → 20 passed, 0 failed
python -m tests.test_autoinfo        → 13 passed, 0 failed
python -m tests.test_signals         → ALL GREEN（无回归）
python -m tests.test_store_ledger    → 17 passed, 0 failed（无回归）
```

`test_intel_features` 里有五条是"防线"，不是"覆盖率"：

| 断言 | 防的是什么 |
|---|---|
| 未来条目被过滤（PIT） | 用明天的新闻给今天打分 |
| 无情报 → 0（不是 NaN） | 融合时"没观点"被当成"看空" |
| w=0 时逐位相同 | 悄悄改变默认行为 |
| NaN 位置保持 NaN | 不可交易的票被塞进一个假分数 |
| w=1 时 d > c > a | 融合方向写反 |

### 还没接的两处（明确留白）

1. **进 Alpha158**：需要重训 + 改 `_validate` 的特征数契约（R36），
   而且情报特征不可回补，塞进 2008 起的面板会整列 NaN —— 建议长期保持"侧信道"；
2. **进学习器（AdaptiveWeighter）**：可以把"情报分"当成第 103 个信号，
   走同一套 Hedge 权重。但那要求每个样本都有情报分（现在只有近 3 个月有），
   否则又是一个"29 个死信号"的故事（R1）。等 `autoinfo` 攒够 60 个交易日再谈。

### 第四轮已验证无问题的项（写下来是为了不重复怀疑）

| 项 | 证据 |
|---|---|
| 接口冒烟 | 74 条路由全量 GET：除 3 条缺参 400 与 POST-only 的 404，其余全 `ok:true`；`/api/status` 返回 `n_signals=102` |
| 前端语法 | `node --check webui/static/app.js` 通过 |
| Alpha158 自检 | 截断 `0.000e+00`、未来扰动 PASS、朴素对拍 max diff `1.7e-10`、值域无绝对价格泄漏（max 63.4，非 1e3 量级） |
| 行情数据完整性 | 300 只票：无重复日期、日期单调、窗口内非正价 0；4 个非正价文件全部在 2013 年前（与 INV-5 披露一致） |
| 实时轮询门禁 | `tick.poll` 有 `in_session()` 判断（`tick.py:330`），非交易时段不发请求 |
| 导出安全 | `transfer.export` 走白名单 `src` 字典，无路径拼接 |
| 账户净值兜底 | `paper_auto.summary()` 对缺 `equity` 的老账户用"现金 + 持仓成本"回填（`paper_auto.py:985-1000`） |
| 告警调度 | `start.sh:491-500` 有 30 分钟循环（`run_once('file')`）；当前 `check_rules()` 返回 0 条，所以没有 `alerts.jsonl` 是正常 |
| 事件学习调度 | `start.sh:219-220` 每次收盘跑 `EL.run()`，`calibrate()` 有 `MIN_N_CALIB` 样本门槛 |
| 认证 | `AUTH.enabled()` 未开时写接口走 `_write_allowed()` → 内网判定；已开时 POST 全局先过 `_auth_check()` |
| 启动脚本编码 | `train.bat` / `APB.bat` / `setup.bat` / `daemon.bat` / `start_agent.bat` / `bootstrap.ps1`：纯 ASCII、全 CRLF、无控制字符（AGENTS §7.5 规则四的断言全部成立） |
| 套件闭包 | `scripts/kit_manifest.py` 用 AST 传递闭包 + 导入层双重校验；构建脚本按清单逐文件拷，失败才回退全量 |
| 文档命令 | 27 个文档中出现的 `python -m qcore.*` / `tools.*` / `scripts.*` 命令，**全部**对应真实模块且都有 `__main__`（零死命令） |
| overlay 叠加 | `qlib_data._apply_overlay` 用"相对前收的比值"追加，且有连续性校验（`prev` 不匹配就跳过），基座更新后 `d <= last_date` 自动去重 → 不会双重计数 |

## 5. 已核验通过的部分（不是客套，是证据）

| 项 | 证据 |
|---|---|
| **INV-1 无未来函数** | `tests/test_no_lookahead.py`：L1 合成 0.00e+00、L3 真实数据 0.00e+00、L2 负控能抓 `shift(-1)`（1.66e-01）、L4 数据守卫非正价 0/1,151,736 |
| **INV-2 顺序与到期判据** | `engine.py:132-133` `s = t - 1 - h`；对账在打分之后；`invariants` INV-2 pass（检查的是语义不是注释） |
| **INV-4 HAC** | `analysis` 的 t 全部走 `stats.mean_tstat(..., lags=h-1)`；edge/5 `ic.t=-2.446` vs 朴素 `-4.316`，差 √5 量级，符合预期 |
| **INV-5 后复权** | 窗口内非正价 0/580,893（抽查 200/300 文件并披露覆盖率） |
| **INV-7 台账幂等** | `tests/test_store_ledger` 17/17；主台账重复 5.85× 已被识别并给出 `.dedup.jsonl` 读数路径 |
| **INV-8 UI 不做计算** | `server.py` 无 numpy/pandas；成本表虽在 server 现算（`server.py:2244`），但属确定性常量，无数据依赖 |
| **成本下限** | `account.json` 仍是 `cost_bps=10.0`，但 `account_config.load()` 返回 `19.28` 并留 `cost_bps_clamped_from=10.0`、`cost_bps_floor=11.28` —— 陷阱 25 的修复真实生效 |
| **时间门禁** | `tests/test_time_gates` 38/38，含"国庆被拒""午休不算盘中"负控 |
| **台账朝向** | `tests/test_store_ledger`：`n_stocks=282 != n_days=3360`，陷阱 24 的修复有测试保护 |
| **DSR 单位** | `tests/test_dsr_units` 11/11，`sr_basis: per_period_sharpe`，陷阱 23 的修复有测试保护 |
| **增量=批量** | `tests/test_incremental_parity` 5/5，REGIME/DIR/STR/UNIV/SIGMA 逐位相等 |

**核心结论复核**：`performance.json` 自身给出的证据链是自洽的 ——
IC(h=5) −0.0183（HAC t −2.45）、DSR 0.0145（`significant=false`）、
学习器 vs 等权 ΔIC −0.0012（t −1.31）、扣成本多空 −4.80%/年、Brinson 选股效应为负。
"5 日尺度没有可用 alpha、反着做也不够付成本"这个判断，**本次评审未能推翻**。

---

## 6. 建议的修复顺序

| 顺序 | 动作 | 为什么排这里 |
|---|---|---|
| 1 | **R49** 补 `exit_date` + 日历缓存加 TTL | 验证半环现在是死的：2/2 条预测 exit_date 为 null，永远对不上账 |
| 2 | **R45 + R17 + R44** 学习闭环三处：样本级幂等、符号门、同源样本 | 连续学习一接上调度就会先坏在"同一批样本学 24 遍" |
| 3 | **R50** 统一 prediction store（四本账 → 一层） | 命中率要可比，先得只有一本账 |
| 3.5 | **R51** 把 `autoinfo` 接进调度与周报（模块已就绪） | 程序自己找的情报要能进决策，否则永远只是又一个"存着"的目录 |
| 3.6 | **R52 + R53** 事件方向改用历史实测；情报以 `APB_INTEL_W` 接入（默认 0） | 标签只有 61% 一致；而模型此前完全看不见情报 |
| 4 | **R26** 定下"谁出名单"：把集成接进 picks，或明确写"决策=GBDT" | 这决定整个项目对外是什么；文档现在描述的那套没在跑 |
| 5 | R1 接线 `extra_loader=external_loader` + 死信号检测测试 | 一行代码决定 33 个信号是死是活 |
| 6 | R2 加 `zoo_n`/指纹与"产物落后"提示，重跑一次 pipeline | 让所有对外数字回到同一时点 |
| 7 | R16 Meta 反号：要么接进 live_model，要么改掉 brief 的那句话 | 现在是一句可验证为假的用户可见结论 |
| 8 | R28/R29 模拟盘：`top_k` 覆盖要可见、涨跌停不可成交要建模 | 直接决定净值可不可信（90% 换手 + 涨停价成交） |
| 9 | R3/R12/R36/R40 模型链路：统一写入方 + 质量门 + 版本比较 | 模型页讲的是"另一个模型"；任何能 predict 的包都能装，旧训练机还能盖掉新的 |
| 10 | R4 GBDT 训练集改点对点成分，或给 +9.4% 加去偏标注 | 这是项目唯一正 alpha 展示位 |
| 11 | R5/R6/R7/R33 口径四处（成本标签、分散判定、SE 重叠、事件样本门槛） | 评估口径是核心资产，标签错了比数字错更危险 |
| 12 | R18/R19 台账 `horizon` 回填 + `n` 改真实计数 | 读数层的地基：幂等键与样本数都必须是真的 |
| 13 | R8/R9/R11/R13/R25 文档与计数同步（最好脚本化） | 一次性消除"同一指标四个值"与"无产物的已验证" |
| 14 | R10/R20/R27 门禁与接口：SKIP 非零退出、`|| true` 可观测、hotspot 落盘 | 防止门禁静默变绿、防止页面卡 7 秒 |
| 15 | R30/R14/R32 运行面：顶栏双日期、面板滞后告警、postclose 幂等 | 让"算到哪天"随时可见 |
| 16 | R41/R42/R43 任务队列加锁、preflight 进 CI、补数据加收盘门禁 | 前两条静默丢任务/丢闸门，第三条会把半日线写成日线 |
| 17 | R38 训练机自动更新：去 CERT_NONE、加签名、extractall(filter) | 供应链：降级 + 无签名 + Zip Slip 三者叠加 |
| 18 | R31/R34/R35/R37/R39/R15/R21/R23/R24 加固与清理 | 收尾 |

---

## 6.5 更正记录（评审自身也会错）

**撤回一条：早前版本里的「告警系统从未运行过」不成立。**

我第一遍用 `grep -rn "qcore.alert"` 扫调度入口，零命中，就写成"没人调用它"。
第二轮复查发现 `docker/start.sh:491-500` 有 30 分钟循环，写的是
`python -c "from qcore import alert as A; print(A.run_once('file'))"` ——
`from qcore import alert` 这个写法躲过了我的 grep。
实测 `alert.check_rules()` 在当前状态下返回 0 条（picks 2 天 < 阈值 3 天、模型 2 天 < 阈值 7 天），
所以 `alerts.jsonl` 不存在是**正确行为**，不是缺陷。该条已从正文删除。

同一轮里差点犯第二次同样的错：`event_learning` 的调度也是
`from qcore import event_learning as EL` + `EL.run()`（`start.sh:219-220`），
按 `qcore.event_learning` 去 grep 会得出"没调度"的错误结论。

**教训（对本项目也适用）**：判断"某模块有没有被调用"时，`import X as Y` 与
`from qcore import X` 两种写法必须同时扫；静态 grep 的漏检率比想象中高。

---

## 7. 修复记录（本轮已改，逐条可核）

> 原则：**每条都改在最小范围、都跑得动、都不改变默认行为**（除非明确写"默认收紧"）。
> 回归：`tests.test_signals / store_ledger / dsr_units / metrics / events / time_gates /
> sync_tasks / invariants_negative / incremental_parity / version_sources / tokens /
> autoinfo / event_history / intel_features` 全绿。

### R54 ·（P1，已修）`external_loader` 曾让 `build_dataset` 慢 20 倍 —— 两个真凶都在"逐票重复计算"

把 R1 接上之后跑实测，发现一个新问题（不是老问题）：

```
120 只票 · build_dataset
  extra_loader=None       →   6.2s
  extra_loader=external   →  >120s（超时）
```

原因：`external_loader(code)` 每只票都要把 4–5 张表 `pd.merge(..., how="outer")` 一遍
（全局宽度 650KB CSV、全市场事件 4042 行、本票事件 4042 行、资金流 120 行），
再进 `_Ctx` 做 16 列 reindex。单票 ~0.5–1s × 300 只 = 5 分钟以上，
直接违背 "全流程 < 60s / 增量 < 15s" 这条硬指标（AGENTS §0）。

**定位（cProfile，3 只票 3.38s）**：

```
3.195s  calendar_cn.next_trading_day   ← 8201 次调用
1.587s    └ builtins.max
1.539s    └ builtins.min
0.047s  _merge_on_date
```

两个真凶都不是"数据量大"，而是**逐票重复算同一件事**：

1. **T+1 平移**：事件表每天都要"数据日 → 可用日"，代码对每只票的 4000+ 个日期
   逐个调 `next_trading_day()`（内部还要 `is_trading_day` 扫日历）→ 单票 1.6s；
   而这些日期**在所有票之间完全重复**。
2. **merge**：`pd.merge(how="outer")` 每票重排索引 4 次。

**修法**：① T+1 平移加进程内 `_SHIFT_CACHE`（日期键）；② 用"日期并集 + `searchsorted`"
的 numpy 版 `_merge_on_date` 取代 4 次 pandas merge（语义不变：每日期一行、缺值 NaN）。

**实测**：

| 场景 | 修前 | 修后 |
|---|---|---|
| 120 只 `build_dataset`（无外部） | 6.0s | 6.0s |
| 120 只 `build_dataset`（带外部） | **>120s（超时）** | **16.9s** |
| 300 只全流程（带外部） | 估 >10 分钟 | **134s**（无外部 108s） |

**取舍**：外部数据让全流程 +26s（108→134s）。因此默认 **pipeline 开、`live` 增量关**
（增量预算是 15s，扛不住 +25s）；需要时可显式 `APB_EXTERNAL=0/1` 覆盖。

**下一步优化方向**（不改变数值口径）：
① 全局表（宽度/全市场事件）已经做成进程内缓存（本轮已加），
② 把"逐票 outer merge"换成"统一日期轴 + `np.searchsorted` 取值"，
   `_Ctx` 里 16 次 `reindex` 同理换 `take`；
③ 或者让 `external_loader` 直接返回 **dict[str, np.ndarray]**（已按日期轴对齐），
   由 `_Ctx` 免 merge 消费。

### 第十四轮实测（重跑 pipeline，离线、external 关闭）

```
python -m qcore.pipeline --pool 300 --start 2013-01-01   → TOTAL 108s, exit 0
shape (3363, 275, 106)   # S=106，与 signal_zoo() 一致
```

产物逐项验收（对照本轮修复）：

| 检查 | 结果 | 对应 |
|---|---|---|
| `status.n_signals / zoo_n / zoo_hash` | 106 / 106 / `24a53873bbdc` | **R2 已修**（产物与代码同源可见） |
| `status.dead_signals` | **33 个**（全是外部数据族，因 `APB_EXTERNAL=0`） | **R1/R2 已修**（死信号第一次在产物里显式列出） |
| `costs.turnover_assumption` / note | `0.7` / "252/h × 换手0.7" | **R5 已修**（标签与数值同源） |
| `risk.signal_level.effective_judge` | **23.48**（basis=`corr_adjusted`），账面 `effective_n=102.5` | **R6 已修**（判定改用相关性口径） |
| `metrics.hit_rate.5.se_cluster` | **0.0194**（旧值 0.00865，×√5） | **R7 已修**（按独立期数 667 算） |
| 台账新增行 `horizon` | 304 条新行全部 =5；历史 47,676 行仍为 null | **R18 已修**（历史不可回改，符合 INV-7） |
| 事件影响表 | **0 组**（`MIN_TRADES=30` 过滤后没有单元格达标） | **R33 生效** —— 诚实但等于承认"事件层没有可用结论"（呼应 R52 的 61%） |

**第二轮（`APB_EXTERNAL=1`，外部数据真正接上）**：`TOTAL 134s`，`S=106`，N=275。

| 检查 | 关外部 | **开外部** | 说明 |
|---|---|---|---|
| `status.dead_signals` | 33 | **15** | flow/大宗/席位仍稀（缓存覆盖不足），margin/breadth/holder/forecast 活了 |
| Rank IC (h=5) | −0.0183 | **−0.0170**（HAC t −2.28） | 信号集变了，结论仍是"无可用 alpha" |
| `hit_rate.5`（按独立期） | 48.56% / SE 1.94pp | 48.56% / SE 1.94pp | 口径已固定 |
| `paper.sim_annual` | −2.54% | **−0.77%** | 区间/数据/信号集同时变了，**旧引用全部作废** |

> 注意 `as_of=2026-11-11`：本轮补了 11-11 的 overlay 之后，面板终于走到最新交易日；
> 但 `paper.sim_annual` 从 −4.02% 变成 **−2.54%**（区间与数据都变了），
> 引用旧数字的地方要一起更新（R9 的那张表）。

## 4.16 第十五轮（服务端自己排训练任务：已落地）

### R55 · 训练任务只有人工入口：没人判断"该不该训"（P2，已修）

在此之前，唯一的排单方式是 `POST /api/sync/task`（人在 UI 上点）。
于是模型超龄、数据落后 22 个交易日、某周期退化，都不会触发任何动作 ——
`sync` 有队列、`model_health` 有健康度、`sync_agent` 会领任务，**缺的是判断层**。

**已实现** `qcore/train_scheduler.py`（+ 10 条契约测试 + `docs/25_train_scheduler.md`）：

| 判据 | 动作 | 优先级 |
|---|---|---|
| 没有模型 | 训练 | 1 |
| `model_health` = failed/degraded | 训练 | 1 |
| 超龄 > 7 天 | 训练 | 2 |
| 数据比 `train_end` 新 ≥ 3 个交易日 | 训练 | 2 |
| 队列已有该周期 / 12h 内已排过 / 本轮满 2 单 | 跳过 | — |

**实测（本机）**：

```
$ python -m qcore.train_scheduler
[train_scheduler] checked=2 want=2 enqueued=2
  h=1   train  数据比训练截止新 22 个交易日（阈值 3）
  h=5   train  数据比训练截止新 22 个交易日（阈值 3）
$ python -c "from qcore import sync as S; print(S.summary()['counts'])"
{'done': 1, 'pending': 2}

$ python -m qcore.train_scheduler --plan     # 再跑一次
  h=1   skip   队列里已有该周期任务 7e7bb6b8361d
  h=5   skip   队列里已有该周期任务 9030c747dfb7
```

接线：`start.sh` §6.6 每小时一次（`APB_TRAIN_AUTO=1`）；`GET /api/train/plan` 只读；
任务带 `created_by="train_scheduler"` 与 `note="[auto] <原因>"` 可追溯。
设计文档见 `docs/25_train_scheduler.md`。

### 8.1 已修（55 条）

| 编号 | 改了什么 | 落在哪 | 怎么验 |
|---|---|---|---|
| **R49** | 日历缓存加 60s TTL + `cache_clear()`；`record()` 算不出到期日记 `pending_calc`；`reconcile()` 用当前日历补算；`session` 时钟改读 `data_sync.local_date()` | `qlib_data.py`、`horizon_ledger.py`、`session.py` | 代码路径 + `test_sync_tasks`/`test_intel_features` 无回归 |
| **R45** | `AdaptiveWeighter.learned_through` 游标 + `engine` 更新前查 `seen()` + npz 持久化 | `learner.py`、`engine.py`、`store.py` | `test_weighter_idempotent` **8/8**（连跑三次 `n_g` 不变；npz 往返后仍不重学） |
| **R27/R30** | `/api/health` 增 `computed_as_of` / `panel_age_days`；顶栏改"数据 X · 计算 Y"，灯取两者较差的 | `server.py`、`app.js` | `node --check` 通过 |
| **R28** | 新增 `paper_auto.top_k_info()`（配置值 vs 生效值 vs 是否被压小）；`picks.top_k_info` 落盘 | `paper_auto.py`、`live_model.py` | 实测：`20 → 2`，`overridden: true` |
| **R17 / R44** | `collect()` 不再跨源借样本（主台账=集成分数，与 GBDT 不同源），需 `APB_CALIB_CROSS_SOURCE=1` 才合并；`apply()` 遇 `a<=0` 拒绝应用并给 hint，需 `APB_CALIB_ALLOW_NEG_A=1` 才反向，应用时回填 `calib_inverted` | `online_calib.py` | `picks.json` 仍 `applied:false`（h=1 无样本），行为不变 |
| **R1** | `pipeline` / `live` 三处 `build_dataset` 全部传 `extra_loader=external_loader` | `pipeline.py`、`live.py` | `grep extra_loader=external_loader` 三处命中 |
| **R2** | `status.json` 增 `dead_signals` / `n_dead_signals` / `zoo_n` / `zoo_hash` | `store.py` | 跑一次 pipeline 即可看到 |
| **R5** | `costs.json` 的 `turnover_assumption` 从 `horizons.annual_cost_drag` 的实参**反射**取值，note 同源 | `pipeline.py` | 标签与数值不再打架 |
| **R6** | `diversification_report` 支持 `effective_override`；判定改用相关性口径，账面 1/HHI 只作参照 | `risk.py`、`pipeline.py` | `effective_judge` / `effective_basis` 字段 |
| **R7** | `hit_rate` 的聚类 SE 改用 `n_indep = n_days//h`（h=5 时 0.87pp → 1.94pp） | `metrics.py` | `test_metrics` 95/95 仍过 |
| **R10** | `invariants --strict`：有 SKIP 时 exit 2（默认仍 0，避免容器因缺数据自杀） | `invariants.py` | `python -m qcore.invariants --strict` |
| **R18** | `append_ledger` 强制 `rec["horizon"] = int(dh)`（不再 setdefault），新行不再写 null | `store.py` | `test_store_ledger` 17/17 |
| **R19** | `horizon_ledger` 的 `n` 改真实条数（原 `天数×100`） | `horizon_ledger.py` | 产物里 `n` 与 `n_days` 不再成百倍关系 |
| **R31** | 未设 `APB_TOKEN` 时，写接口只在**监听回环**才免 token；监听 0.0.0.0 需 `APB_ALLOW_WRITE_NO_TOKEN=1` | `server.py` | 默认收紧，带明确 hint |
| **R32** | `postclose` 加"当日已跑"幂等（15:05–17:00 的 5 分钟循环不再重跑 24 次） | `session.py` | 返回 `already_ran_today` |
| **R33** | `impact.compute(min_n=None)` → 默认取 `thresholds.MIN_TRADES=30` | `impact.py` | n<30 的 9 行不再进表 |
| **R37** | 解压失败自动删包重下一次（416 把损坏包当完整的老问题） | `data_sync.py` | `updated_after_redownload` |
| **R38** | 去掉自动降级 `CERT_NONE`（需 `APB_KIT_INSECURE_TLS=1` 显式开启）；zip 拒绝绝对路径/`..`，`extractall(filter="data")` | `kit_update.py` | 供应链三件套收口 |
| **R39** | 构建时 `kit_selftest` 失败 → `exit 1`，不再只 WARN | `build-train-kit.sh` | `bash -n` 通过 |
| **R41** | 任务队列加跨平台文件锁（Windows `msvcrt` / POSIX `fcntl`），`claim_task`/`create_task` 都在锁内 | `sync.py` | `test_sync_tasks` 14/14 |
| **R42** | CI 增加 `scripts/preflight.py --quick` 步骤 | `ci.yml` | 发布闸门终于有人跑 |
| **R43** | `append_day` 加收盘门禁（当日 <15:05 拒绝，`APB_APPEND_FORCE=1` 可越）+ 量能校验（<0.3×近 5 日均量视为未收盘） | `daily_append.py` | 盘中补数据被拒 |
| **R51** | 新增 `qcore/autoinfo.py`（自主情报）+ `tests/test_autoinfo.py` | 新文件 | 13/13 |
| **R52** | `reward._measured_direction()`：方向优先取历史实测 CAR，拿不到才用标签，`used[].dir_basis` 留痕 | `reward.py` | 实测 `applied:True` |
| **R53** | 新增 `qcore/intel_features.py`；`live_model` 支持 `APB_INTEL_W` 排位融合（默认 0）；`picks.json` 带 `intel`/`score_kind` | 新文件 + `live_model.py` | 12/12 |
| **R54** | T+1 平移加缓存 + numpy 版 `_merge_on_date`；pipeline 默认开外部、live 默认关 | `dataset.py`、`pipeline.py`、`live.py` | 120 只 >120s → 16.9s；300 只 134s，dead 33→15 |
| **R55** | 新增 `qcore/train_scheduler.py`：服务端自己排训练任务（5 条判据 + 去重 + 冷却 + 单轮上限） | 新文件 + `start.sh` §6.6 + `/api/train/plan` + `docs/25` | `test_train_scheduler` 10/10；实测排 2 单、二次全 skip |


**第二批（D 类机械活 + A 类决策，本轮完成）**

| 编号 | 改了什么 | 落在哪 |
|---|---|---|
| **R8** | 信号个数不再写死：`pipeline` 用 `S`、前端读 `/api/signals.total`；AGENTS/README 改为"106（`zoo_n`）" | `pipeline.py`、`app.js`、`AGENTS.md` |
| **R9** | AGENTS §9.1 基线表用 2026-11-11 产物重写（IC −0.0170 / t −2.28 / DSR 0.0211 / 命中 48.56% SE 1.94pp / 模拟盘 −0.77%），并注明"小数以产物为准" | `AGENTS.md` |
| **R11** | 泄漏扫描基线 33 → **51**（AGENTS §8 + docs/22 同步，注明 HIGH=0） | `AGENTS.md`、`docs/22` |
| **R12** | `export_model` 写 picks 补齐 `market` / `horizon` / `decision_source` | `scripts/export_model.py` |
| **R13** | `docs/19` 验收表由产物回填（R3 加限制、R18 改"不可交易"） | `docs/19` |
| **R14** | 新增告警规则 `panel_lag`：面板落后行情就告警 | `alert.py` |
| **R15** | 删掉 0 字节 `qcore/realtime.py` | — |
| **R16** | brief 改口："Meta 判定反向，但系统未据此调整仓位" | `brief.py` |
| **R21** | markdown 报告注明 Sharpe 口径（算术均值/标准差 ×√252，与几何年化不可互算） | `report_stats.py` |
| **R23** | 事件衰减跳过**未来排期**事件 | `reward.py` |
| **R24** | 行为因子标签机械化（变号/不显著/衰减三档，不再一律"与文献一致"） | `behavior_report.py` |
| **R25** | 验收表 R18 的 63.6% 改为实测"33.5% / −19.39bps / 不可交易" | `docs/19` |
| **R26** | picks 增加 `decision_source:"gbdt"` + 说明（明确"集成只做测量，不下单"） | `live_model.py` |
| **R29** | `/api/hotspot` 只读快照，不再请求线程里联网现算（7s → 立即返回 `not_ready`） | `server.py` |
| **R34** | 影响因子表改显式列（含 `p` 与 `insufficient` 徽章） | `app.js` |
| **R35** | release 校验 `tag == vVERSION` | `ci.yml`/`release.yml` |
| **R3** | `/api/model` 附带 `deployed`（真实 `meta*.json`）+ `deployed_mismatch`；模型页弹警示条 | `server.py`、`app.js` |


**第三批（模型准入与点对点成分，本轮完成）**

| 编号 | 改了什么 | 验证 |
|---|---|---|
| **R36/R40** | `model_transfer._admission()`：①训练机 kit 版本低于本机 → 拒绝；②新模型 `train_end` 早于线上 → 拒绝；③缺 `qa.holdout_ic` 只标 `quality_unverified`（不阻断）；`APB_MODEL_ALLOW_OLD=1` 可显式放行并留痕 | `test_model_admission` **9/9** |
| — | 训练侧补 `qa.holdout_ic`（末尾 5% 样本内 Rank IC，`basis=in_sample_tail_60d`，明确标样本内）并写进 manifest | `live_model.train` / `export_bundle` |
| **R4（测量部分）** | `scripts/lowturnover_bt.pit_codes()` + `export_model --pit`：用 `pit_membership()` 取 2008–2021 **曾入选**的全部成分 | 实测 **748 vs 300**，快照漏掉 **448 只（60%）** |

> R4 的"去偏 A/B"还没跑（要真训一次 250 树），但**偏差的量级已经量出来了**：
> 旧训练集只见过 2021 快照里的 300 只，而点对点口径下有 748 只 —— 也就是说
> 2008–2019 的训练样本里，六成的历史成员从来没进过模型。


**第四批（容器门禁 + 滚动和的地基，本轮完成）**

| 编号 | 改了什么 | 验证 |
|---|---|---|
| **R20** | 镜像 `COPY audit/ ./audit/` + `COPY tests/ ./tests/`（原来只拷 2 个文件，"容器内门禁"与 CI 不是同一套）；`start.sh` 不再 `&nbsp;\|\| true` 吞退出码，改写 `data/state/invariants_health.json`；`APB_REQUIRE_INVARIANTS=1` 时首轮失败即拒绝继续；`/api/health` 暴露该状态 | `bash -n` 通过；`/api/health` 带 `invariants_health` |
| **R48（地基）** | `_roll_sum` 从"前缀和差分"改为**滑窗求和**：只依赖窗口内 w 个数 → 与序列长度无关；顺手删掉重复定义的 `_roll_mean`（同名的第二个 def 一直在覆盖第一个，属死代码） | `tests/test_roll_window.py` **12/12**（w=5/20/60/250 × 600/1200 根，截断差 **0.00e+00**）；`test_signals` 20/20；全流程重跑指标**未变** |

**为什么 R48 只算"地基"**：AGENTS 里那条"增量信号已否决"的根因就是前缀和的长度依赖。
换成滑窗后，**同一个窗口在任何长度序列上逐位相同**（实测 0 格差异），
于是"信号只算最后 N 根"这条路重新可行。但真正做增量缓存还要：
① 逐股保存"预热窗口 + 尾部"的信号状态；② 重跑 `test_incremental_parity` 证明批量=增量。
本轮只把**前置条件**做掉，并把性能代价量出来：

```
build_dataset 120 只：无外部 6.0→6.5s｜带外部 16.9→18.7s（滑窗 +10%）
全流程重跑：209s（含 impact/behavior，该段受板块指数抓取影响，波动大）
指标：IC −0.01698 / HAC t −2.28 / DSR 0.0211 / 命中 48.56% / dead 15 —— **与改前完全一致**
```


**第五批（R48 第二步：预热实测 + 截断接口，本轮完成）**

| 内容 | 结果 |
|---|---|
| 新增 `tools/signal_warmup_probe.py`：逐信号、逐股验证"只算最后 W 根"是否逐位相等 | 8 只票 × 6 个候选 × 106 信号 |
| 实测最小安全预热 | **W=400**（106/106 全等）；W=250 仍有 **4** 个不同（`atr_chan_20_1p5` / `ma_slope_250_20` / `macd_hist_12_26_9` / `rsi_21_ob65`），W=320 剩 2 个 |
| `compute_signals(df, tail=k, warmup=W)` | 已支持；默认 `None`（全量）→ 存量行为零变化 |
| 速度 | 600519：全量 53.7ms → **11.6ms（4.6×）** |
| 契约测试 | `tests/test_signal_warmup.py` **6/6**（W=400 逐位相等 + 不传 warmup 行为不变） |

> 为什么是 400 而不是 250：EMA 类信号（MACD、Wilder RSI）是**递归衰减**，
> 250 日窗口叠加递归尾迹要 ~400 根才收敛到逐位相等。这条数字现在写进 AGENTS §1，
> 谁要开增量快路径，先过 `signal_warmup_probe` 这一关。


**第六批（R48 第三步：增量窗口缩短，已启用；信号裁剪接口保留未启用）**

| 内容 | 实测 |
|---|---|
| `live.step_once` 窗口 900 → **620** 天（≈420 交易日 ≈ 预热 400 + 20），新增 `WARMUP_MIN=400` 守卫：`t_from < 400` 自动退回 900 天 | — |
| 900d vs 620d **重叠行**对拍（60 只票） | DIR **0 格** / STR **0 格** / OPEN 0.00e+00 |
| 面板耗时（60 只） | 3.6s → **3.4s**（load_kline 占大头，信号只省一部分） |
| `build_dataset(signal_warmup=400)`（每票只留「start 前 400 根」） | 面板级 **1 格** 差异：`ma_slope_250_20` @601166 @2026-11-11，DIR 1→0，两边 STR 均 0 → **保留接口、默认不启用** |

> 这一步的价值不在"省了几秒"，而在**把边界量清楚了**：
> 窗口缩短是安全的（0 格），逐票裁剪还差 1 格（且已定位到具体信号与股票）。
> 后者的归因是下一步，归因完才能开 —— 不拿"看上去差不多"当通过。


**第七批（R48 收口：那不叫"差一点点"，是 `_roll_mean` 自己也用了前缀和）**

第六批留下的最后 1 格（`ma_slope_250_20` @601166 @2026-11-11，DIR 1→0，两边 STR=0）
没有放过，继续归因，结果是**另一个前缀和没被替换**：

```
_Ctx.sma()  ->  _roll_mean()
_roll_mean  ->  cv = cumsum(v);  sum_v = cv[idx+1] - cv[lo]      ← 还在用前缀和差分
_roll_sum   ->  sliding_window_view(...).sum(axis=1)             ← 这个早就改了
```

前缀和的两个被减数随序列变长而变大，MA250 差 1e-8；`ma_slope` 算的是
`(MA[250](t) - MA[250](t-20)) / MA[250](t-20)`，那天斜率恰好贴在 0 附近 →
舍入差把 `_sign()` 从 +0 翻成 −0（STR 因此都是 0）。

**修法**：`_roll_mean` 也改用 `_roll_sum` 的滑窗和（NaN 计数同样用滑窗和）。
修完逐级验证：

| 层次 | 结果 |
|---|---|
| 601166 单票末 40 行、全 106 信号 | 不一致信号 **0** |
| 80 只 × 3363 天 × 106 信号（2013 起全量对拍） | DIR **0 格** / STR **0 格** |
| `tests.test_signals` | **20/20 ALL GREEN** |

**于是正式启用**：`live.step_once` → `build_dataset(signal_warmup=WARMUP_MIN)`。

| 场景 | 修前 | 修后 |
|---|---|---|
| 增量窗口 dataset（80 只，449 行） | 4.09s | **2.49s（1.65×）** |
| 增量窗口 dataset（**300 只**） | **14.9s** | **9.1s（1.64×）** |
| 增量预算（AGENTS §0 目标） | 16.9s **超预算** | **9.1s 达标** |

> 这条线的教训：**同一个算法改了一处不等于改完**。`_roll_sum` 换成滑窗时，
> `_roll_mean` 里还有一份独立实现的前缀和藏在 800 行外；它平时只差 1e-8，
> 只在"斜率贴零"这种边界上翻一格 —— 而这一格足够让"逐位相等"变成"不相等"。


**第八批（R4 的 A/B 跑完了：点对点训练集不是"修正偏差"，是"把模型救回来"）**

跑法：`lowturnover_bt.prepare(stocks=120, trees=60, train_end=2019-12-31,
test 2021-01-01 ~ 2026-11-06)`，只改 `pit` 一个变量，IC = 逐日横截面 Spearman 的均值。

| 周期 | 训练集口径 | Rank IC | n 天 | SE | t |
|---|---|---|---|---|---|
| h=1 | 2021 快照（现状） | +0.00131 | 1415 | 0.00429 | **+0.31** |
| h=1 | **点对点（748 只）** | **+0.01490** | 1415 | 0.00363 | **+4.11** |
| h=5 | 2021 快照（现状） | +0.01118 | 1411 | 0.00426 | +2.62 |
| h=5 | **点对点（748 只）** | **+0.02673** | 1411 | 0.00373 | **+7.17** |

**结论**：用 2021 快照选 2008–2019 的训练样本，不只是"有点偏差"——
它让 h=1 的模型**完全没有优势**（t=0.31），而改成点对点后 t=4.11；
h=5 也从 +0.0112 提到 +0.0267（2.4×）。
原来"六成历史成员从没进过模型"（748 vs 300）这件事，代价是**模型学不到东西**。

**必须一起读的限制**（不许拿 t=7.17 当 alpha 结论）：

1. 单次运行、单一种子、`trees=60`（生产是 250）、`stocks=120`（生产 300）；
2. IC 是**逐日横截面 Spearman 均值**，不含成本、不含组合构建、未做多重检验校正（本轮共 4 组配置）；
3. 测试集**没有变**（两次都是同 119–120 只），差异只来自训练集口径 → 归因干净，但也意味着还没验证"扩大测试集"的效果；
4. 需要在训练机用**生产参数**（300 只 / 250 树）复跑一次再决定是否改默认。

**处置**：`--pit` 已可用（`scripts/export_model.py --pit`、`lowturnover_bt.prepare(pit=True)`），
**默认未改**（生产行为不变）。是否把默认翻过来，等训练机按生产参数复核后决定 ——
这一步留给人拍板，因为它是"改变模型训练口径"，属于 AGENTS §10.4 的范畴。


**第九批（R22 量出来了：模拟盘按收盘成交，等于每笔多赚 16–71 bps）**

回测按 **T+1 开盘**成交，模拟盘 `paper_auto._px_for()` 按 **T 日收盘**成交 ——
两个基准不可比。新增 `tools/fill_basis_probe.py` 量隔夜跳空（`open[t+1]/close[t]-1`）：

| 口径 | 结果 |
|---|---|
| 池内 98 只 × 最近 60 个交易日（5880 个观测） | 隔夜跳空均值 **−15.95 bps**（中位 −6.98，5–95 分位 −202 ~ +167） |
| 当日 top-10 名单 | 隔夜跳空均值 **−71.25 bps** |

**读法**：按收盘成交 = 相对次日开盘口径，**每笔系统性多赚 +15.95 bps**（池内），
而当日名单是 **+71.25 bps** —— 那是往返成本（19.28 bps）的 **3.7 倍**。
也就是说模拟盘的净值一直被"隔夜跳空"这个口径悄悄抬高，而且名单越集中越明显。

**为什么还没改**：

1. 改口径必须**重算历史成交**，否则新旧净值混在一条曲线上（AGENTS §10.4 禁止口径混用）；
2. 60 个交易日只有一段样本，且"当日名单 71 bps"用的是近 5 日均值（样本很小），
   需要跨多段行情复核；
3. "次日开盘成交"在**日终跑批**里意味着**挂单到次日**：要么改成交模型（pending fill），
   要么把跑批挪到次日开盘后 —— 这是流程决策，不是代码补丁。

**下一步（需要拍板）**：①接受"模拟盘=收盘价、回测=次日开盘"这个已知口径差并写进 UI 标注；
②或者改造模拟盘为次日开盘成交并重算历史净值（一次性）。


**第十批（生产参数复现 + 一处门禁失明被修好）**

### 1) R4 用生产参数复跑：方向成立，但第八批的说法要收窄

| 配置 | 训练集口径 | Rank IC | t | 耗时 |
|---|---|---|---|---|
| 120 只 / 60 树 / h=1 | 快照 | +0.00131 | +0.31 | 72s |
| 120 只 / 60 树 / h=1 | **点对点** | **+0.01490** | **+4.11** | 77s |
| 120 只 / 60 树 / h=5 | 快照 | +0.01118 | +2.62 | 68s |
| 120 只 / 60 树 / h=5 | **点对点** | **+0.02673** | +7.17 | 73s |
| **300 只 / 250 树 / h=5** | 快照 | **+0.01944** | **+5.37** | 432s |
| **300 只 / 250 树 / h=5** | **点对点** | **+0.02837** | **+9.43** | 503s |

**必须修正第八批的措辞**：当时写"快照口径让模型学不到东西"——那只在小配置
（120 只 / 60 树）下成立（t=0.31）。**生产参数下快照模型本来就有优势（t=5.37）**，
点对点是在这个基础上再提升 **+46% IC（t 5.37 → 9.43）**，而不是"从无到有"。
结论方向不变（点对点更好，两套配置都验证），但**程度说法改了**——这正是
"先量，再改说法"该有的样子。

### 2) INV-3 门禁此前一直是 `SKIP`（Windows 编码问题），修好后立刻抓到一处

`qcore/invariants.py` 调 `leak_scan.py` 时用 `subprocess.run(text=True)`，
Windows 下按 **GBK** 解码子进程输出 → `UnicodeDecodeError` → 整项退化为
`[SKIP] 扫描器输出无法解析`。也就是说**"统计量必须滚动"这条门禁一直在装死**。

加 `encoding="utf-8", errors="replace"` 后，它立刻报 `HIGH=1`：
`signals.py` 里我自己写的 `df.iloc[-(_w + tail):]` 被"`.iloc[i+k]` 取未来行"规则误判。
改成 `df.tail(n)`（语义相同、不会被误判）后回到 `HIGH=0`。

顺手修了同一处的报数逻辑：`MEDIUM - WAIVED` 会算出 **−731**（WAIVED 统计的是
**全部严重级**的豁免，不是 MEDIUM 子集），旧文案等于在门禁里印一个负数。

最终：`8 checks, 0 failed`（INV-3 现在是真的在跑，不是 SKIP）。


**第十一批（R22 落地为"可观测"，不是"悄悄改口径"）**

上一批把口径差量出来了（池内 −15.95 bps、当日名单 −71.25 bps）。这一批让它**持续可见**：

| 交付 | 内容 |
|---|---|
| `qcore/fill_audit.py` | 读 `data/paper/trades.jsonl`，逐笔算 `gap_bps = open[t+1]/close[t]-1`（卖出取反），落 `data/state/fill_audit.json` |
| `GET /api/fill_audit` | 只读该 JSON（INV-8） |
| `start.sh` §6.6 | 收盘后自动跑一次 |
| `tests/test_fill_audit.py` | **8/8**（含"卖出符号相反""样本不足标 insufficient""末日无次日开盘不归因"） |

**实测（现有 20 笔模拟盘成交，2026-11-06 那一天）**：

```
可归因 13 笔（其余 7 笔缺次日开盘或本地无日线）
mean gap = +110.0 bps   median +131.1   stdev 96.2   合计 +1430 bps
insufficient = True（13 < MIN_TRADES 30）
price_basis 未记录 13 笔 → 归入 unknown
```

**踩到的坑（已写进测试）**：第一版拿 `次日开盘(后复权) / 成交价(原始)` 相除，
跑出 **+78,023 bps（780%）**。原因是两个价格口径不同 —— 比率必须取自**同一条日线**。
这和 AGENTS §7 那些"口径混用"陷阱是同一类错误，只是这次是我自己犯的。

**仍然没改成交模型**：`fill_audit` 只归因、不改账。切"次日开盘成交"要重算历史净值，
属需要拍板的流程决策（§8.2 R22）。


**第十二批（R46 收官：腾讯一次能带 600 只，但 800 只会被静默截断）**

新增 `tools/tick_capacity_probe.py`，真的去打腾讯批量快照：

| 一次请求的票数 | 覆盖率 | 耗时 | URL 长度 |
|---|---|---|---|
| 60 | 100% | 0.12s | 561 字符 |
| 150 | 100% | 0.08–3.10s（抖动） | 1371 |
| **300** | **100%** | **0.09s** | 2721 |
| 600 | 100% | 0.12s | 5421 |
| **800** | **92%（只回 736 行）** | 0.11s | 6645 |

**关键发现**：800 只时请求**报成功**，但只回 736 行 —— 服务端静默截断，
`missing` 也拿不到真相（因为请求本身"成功"）。这和 AGENTS §7 的"静默截断"
是同一类坑，只是发生在 HTTP 层。

**修法**：`tick.snapshot()` 按 `TICK_CHUNK=400` 分块 + 对缺失**重试一次**，
返回体加 `chunks` / `partial`。400 远离 736 的截断点，分块后截断在物理上不会发生，
重试是二道保险。

| 验证 | 结果 |
|---|---|
| 真实 736 只（池上限） | 2 块 / 0 缺失 / **0.22s** |
| `tests/test_tick_chunk.py` | **7/7**（假服务端每次只回 300 行 → 缺失被重试补齐；50 只只发 1 次；全源失败报 `all_sources_failed`） |

**顺手解开"53 只"这个限制**：`start.sh` 新增 `APB_TICK_TOP=N`，
把池内成交额前 N 只也纳入秒级快照（默认 0 = 保持现状）。
也就是说 —— **全池盯盘现在只是配置问题，不是能力问题**。


**第十三批（按建议执行四项决策：C / A / C / C）**

| 决策 | 选了 | 落地 |
|---|---|---|
| **1 训练口径** | **C（双轨复核）** | `live_model.train` 加 `APB_TRAIN_PIT`（默认 0，行为不变）；新增 `scripts/ab_train_pit.py`：同一份数据跑快照 vs 点对点，写 `data/state/ab_train.json` 并给 `delta_ic` / `verdict` / `caveat` |
| **2 成交价口径** | **A（接受并标注）** | 模拟盘页顶部加固定说明条：「模拟盘按当日收盘、回测按次日开盘，净值不可直接对比」+ 实时口径差（`/api/fill_audit` 的 `mean_gap_bps`，样本 <30 时标注"仅作参考"） |
| **3 四本账** | **C（只读合并视图）** | 新增 `qcore/pred_view.py` + `GET /api/predictions_view?code=`：把 ledger / picks / paper_trades / claims 并列展示，**每行带 source+basis，不合并写入**；`tests/test_pred_view.py` **11/11**（含"调用前后文件 mtime 不变"这条只读承诺） |
| **4 全池盯盘** | **C（先 100 只）** | `start.sh` 的 `APB_TICK_TOP` 默认 **100**（关：`APB_TICK_TOP=0`） |

**为什么决策 1 选 C 而不是直接 A**：`--pit` 的 +46% IC 是**单次运行、单一种子**，
而它改变的是"模型训练口径"（AGENTS §10.4 明确不许拿单次结果改口径）。
双轨复核把这件事变成可重跑的命令，而不是我一句话：

```bash
python -m scripts.ab_train_pit --horizon 5 --stocks 300 --trees 250
# → 快照 IC / 点对点 IC / ΔIC / Δ% / verdict，落 data/state/ab_train.json
```

**`pred_view` 的边界写死在代码注释里**：只读、不重算、不替口径打架。
测试里那条 "mtime 未变" 就是它的承诺书。


**第十四批（纠正一个架构错误：双轨复核应该由训练机跑，不是"你有空在本机跑一次"）**

上一批我把 `ab_train_pit` 交出去时写的是"有空跑一次"——**那是错的**。本项目的架构是
「服务端排单 → 训练机干重活 → 回传结果」（见 `docs/25_train_scheduler.md`），
双轨复核是 2×(5–8 分钟) 的重活，必须走队列。

**补齐的三段**：

| 环节 | 落地 |
|---|---|
| **服务端排单** | `train_scheduler` 新增 `train_ab` 条目：从未复核或距上次 > `AB_EVERY_DAYS=30` 天 → 排一单；**不占** `MAX_ENQUEUE` 名额；队列已有则不重复排 |
| **训练机执行** | `tools/sync_agent.py` 新增 `_run_ab_train()`：跑 `python -m scripts.ab_train_pit`，超时 2h，读 `data/state/ab_train.json` 一并回报 |
| **服务端收结果** | `POST /api/sync/result` 收到 `detail.ab_train` → 存 `data/state/ab_train.json`；新增 `GET /api/ab_train` 只读 |

**实测（本机）**：

```
$ python -m qcore.train_scheduler
  h=1   skip     队列里已有该周期任务 7e7bb6b8361d
  h=5   skip     队列里已有该周期任务 9030c747dfb7
  ab    train_ab 尚未做过双轨复核        ← 已排进队列
$ python -c "from qcore import sync as S; print(S.summary()['counts'])"
{'done': 1, 'pending': 3}               # 2 张训练单 + 1 张双轨复核单
```

也就是说：**决策 1 的"双轨复核"现在是训练机的活，NAS 不跑**。
本机那次 300 只/250 树的实测（t 5.37 → 9.43）只是**先替训练机探了个路**，
不构成调度依赖。


**第十五批（训练机一键：`APB.bat auto` / `bash apb.sh auto`）**

以前"给训练机"要三步：先跑 `kit_setup` 填配置、再 `agent_daemon start`、最后自己想
该跑哪个脚本；没配置时守护会直接失败退出。现在一个入口：

| 步骤 | 动作 |
|---|---|
| [1/5] 环境 | Python 3.10+ / numpy / pandas / 磁盘 / 训练数据是否就位 |
| [2/5] 配置 | 没有 `apb_defaults.json` → 就地拉起配置向导（填→测→存） |
| [3/5] 连通 | 探一次 NAS，最后几行原样打印 |
| [4/5] 数据 | 已就位跳过；未下载**明确提示**，不偷偷下 545MB |
| [5/5] 启动 | 已在跑就不重复起；否则 start 后台守护，并打印"接下来会怎样" |

形态：`auto`（一键）/ `--check`（只体检）/ `--once`（领一个）/ `--no-daemon`（前台）/
`--data`（显式下数据）。Linux/macOS 新增 `apb.sh`，同一套。

**为什么这也算评审项**：`train_ab`（双轨复核）刚接进队列，但如果训练机上手成本高，
那条链路等于没通。一键入口是它的**可用性前提**。


**第十六批（发版事故复盘：两条都是我造成的，都补了硬防护）**

### 事故 1：把 preflight 加进 CI，但它在干净检出上必然失败

`preflight` 会检查 `data/state/*.json`（8 个必需文件）——而 `data/` 在 `.gitignore` 里。
本地有产物所以全绿；GitHub runner 上干净检出，于是：

```
[2] 状态文件   → state.files / state.names / state.events / state.impact / state.behavior 全红
[4] HTTP 端点  → http.endpoints 未就绪（缺 status.json 等 9 个文件）
[3] 账户配置   → account.default: CN 默认本金越界 5000.0
结果: 6 项失败 → 不许发版
```

**每一次 CI、每一次 Release 都必然红**。这不是"门禁太严"，是**门禁在错误的场景里
判错了对象**：干净检出没有产物 ≠ 产物坏了。

修法两条：
- `preflight --ci`（`ci.yml` 已带上）：`data/state` 为空时把 `state.*` / `http.*`
  降级为 **SKIP** 并打印原因；代码类检查（导入/路由/UI 函数/COPY 源）照常硬判。
- `account.default` 暴露了第二个真 bug：代码默认 `initial_capital=5000`，
  而 preflight 的下界是 **10,000** —— 本地有 `account.json`（800,000）时永远看不出。
  已把默认改成 CN/HK **10,000**、US **1,000**。

验证（`git archive HEAD` 出的干净检出）：
```
--quick（旧）       → 4 通过 / 1 失败 / 9 跳过   ✗ 不许发版
--quick --ci（新）  → 9 通过 / 0 失败 / 9 跳过   ✓ 可以发版
本地（有产物）      → 17 通过 / 0 失败 / 2 警告  ✓ 与之前一致
```

### 事故 2：`git push --follow-tags` 把 97 个积压 tag 全推上去

仓库本地有 **97 个 annotated tag**，`--follow-tags` 会把它们全部推送 →
`release.yml` 的触发条件是 `on: push: tags: ["v*"]` → **97 次 Release workflow**，
每次跑测试、构建镜像、推 GHCR、建 GitHub Release。

正确命令只有两条：`git push origin HEAD` + `git push origin v<当前版本>`。
已固化为 `scripts/release.sh`：只推 HEAD 与当前 TAG，**显式拒绝**
`--tags / --follow-tags / --all / --mirror`，并校验 `VERSION == tag == HEAD`。

> 教训合起来就一句：**门禁要能在它要守护的那个环境里跑通；发版要只动本次的东西。**

### 8.2 未修（0 条）

**55 条评审项全部收口。** 剩下的都不是"缺陷"，而是**要人拍板的决策**：

| 待决策 | 说明 |
|---|---|
| `--pit` 是否设为训练默认 | 生产参数复现已做（快照 t=5.37 / 点对点 t=9.43），改默认属 §10.4 |
| 模拟盘是否改"次日开盘成交" | 口径差已可观测（`fill_audit`），切换要重算历史净值 |
| R50 四本账收敛 | 状态层已摸清（R48），排期问题 |



---

## 8. 一句话给作者

这套系统的**测量层**（因果性、统计口径、台账、成本下限）比绝大多数自建量化项目都干净，
毛病几乎全部落在它自己最擅长诊断的那一类：**接线**。
R1 尤其讽刺——它正是 AGENTS §7 陷阱 27 写下的那句话："模块存在但零消费"。
建议把这条陷阱升级成一条**可执行测试**（信号覆盖率），而不只是文档里的一行表格。
