# 01 项目调研：可自学习/自适应参数的股票分析系统

**结论先行**：所有成熟项目的"自适应"都发生在**权重层**（给历史样本、给策略、给信号重新分配权重），不在模型层。Qlib DDG-DA 的 `TimeReweighter`、FreqAI 的 `weight_factor`、Hedge 的乘性更新，本质是同一件事。我们只实现这一层，~300 行 numpy，天然可解释、可回放。

约束：A股日线 / 纯 pandas+numpy / 无重型依赖 / 可解释 / 必须走样外验证。

## 1. 逐项目核实

### 1.1 Qlib（rolling / DDG-DA / online_srv / Alpha158）
机制：`RollingGen(step=20, trunc_days=horizon+1)` 切滚动窗口重训；DDG-DA 在其上学一个元模型，对历史每个样本段输出权重（`TimeReweighter`），使重训样本分布贴近当前市场。
**抄**：① `trunc_days=horizon+1`——源码注释原文 "the last two days should be truncated to avoid information leakage"，这就是 purge 的最小实现；② 权重落地形态 `Reweighter.reweight()` 返回与样本对齐的 weight Series 喂给 `sample_weight`，我们的信号权重应同构（date×signal 的 DataFrame）；③ 元模型超参起点 `clip_method="tanh", clip_weight=2.0, lr=0.001, max_epoch=30, hist_step_n=30, fea_imp_n=30`；④ Alpha158 的命名与规范化约定：**所有价量因子除以 `$close` 做无量纲化**（`KMID/KUP/ROC5/MA20/BETA20/RSV20/CORD20` 等，见 loader.py）。
**坑**：DDG-DA 需 torch+LightGBM+MLflow，元模型 30 epoch，对日线样本量是杀鸡用牛刀且权重不可读；`online_srv` 四态状态机源码自注 "not implemented in the proper way!!!"；qlib 的 .bin 数据格式不兼容。

### 1.2 FreqAI（freqtrade）
机制：后台线程按固定频率重训，训练样本按时间指数衰减加权。
**抄**：① `W_i = exp(−i/(α·n))`（i 为样本序号，n 总数）——一行 numpy 的"近期更重要"，α 可解释为有效记忆长度，比硬滚动窗平滑；② **DI 外推预警**：`DI_k = d_k / d̄`（d_k 为新样本到训练集最近邻距离，d̄ 为训练集平均两两距离），超阈值即拒绝交易——"我不知道"的可解释表达，阈值 1.0 起步；③ pipeline 顺序 `MinMaxScaler(-1,1)` → `VarianceThreshold`（剔零方差列）。
**坑**：万级特征自动膨胀（timeframes×shifted_candles×corr_pairs）；SVM/DBSCAN 剔离群在日线小样本上会剔掉有用的极端行情；每标的独立模型。

### 1.3 river
机制：`learn_one/predict_one` 流式 API + 漂移检测器族。
**抄**：① ADWIN 参数语义（`delta=0.002`、`clock=32`、`min_window_length=5`、`grace_period=10`，检出后窗口重置为 W1）——我们只需监控**一维滚动 IC 序列**；② `optim.FTRLProximal(alpha,beta,l1,l2)` / `PassiveAggressive` 已在库内，说明 PA/FTRL 在流式场景成熟；③ `ensemble` 的 ADWIN 触发成员替换是"漂移→换模型"标准范式。
**坑**：核心已迁 Rust（`river._river_rust.drift`），装它即引入编译轮子。**不装**，按 100 行 numpy 重写：Page-Hinkley 比 ADWIN 更好写（累计 `(x−μ−δ)` 超 λ 报警）。

### 1.4 FinRL
机制：Gym 环境 + PPO/A2C/DDPG 直接输出每股买卖股数。
**抄（仅一条）**：**turbulence 熔断**——`turbulence_threshold` 超限即强制清仓并禁止买入；turbulence 是收益向量对历史均值/协方差的马氏距离，numpy 十行，是"市场状态"最便宜的代理，可作 regime 输入。
**坑**：`reward = end_total_asset − begin_total_asset`（绝对金额奖励，随资金规模漂移，靠 reward_scaling 手调）；动作取整股、按 argsort 排序成交，撮合不真实；RL 不可解释、样本效率极低，几乎必然过拟合单一行情段。

### 1.5 经典在线学习/集成加权（主战场）
- **Hedge/MWU**：`w_i ← w_i·exp(−η·ℓ_i)` 后归一化，遗憾界 O(√(T ln N))。ℓ_i 取"信号 i 当期亏损"即"打脸降权"。
- **FTRL-Proximal**：`z += g − (1/η)(√n−√n_prev)·w`，`w = −(z−λ₁sgn)/((β+√n)/α+λ₂)`。抗噪更好但需逐特征梯度，我们只有十几个信号，**留作 v2**。
- **Passive-Aggressive**：仅在错时更新，步长 `τ=ℓ/‖x‖²`（+C 截断），适合稀疏事件。
- **Beta-Bernoulli**：命中率后验 `Beta(α₀+hits, β₀+misses)`，均值作权重、方差作置信度——**这是做收缩的工具**：`w^shrunk=(1−λ)w+λw_prior`，λ 由后验方差定（n<30 时 λ≥0.5）。
- **Meta-labeling**（López de Prado）：一级定方向，二级只预测"这注该不该下/下多大"，概率直接当仓位乘数，可解释为"信心系数"。
**取舍**：主更新用 Hedge，幅度用 Beta 收缩，Meta-labeling 做二级闸门；三者都不需要梯度库。

### 1.6 因子评价与回测口径
- **alphalens-reloaded**（原 Quantopian 版停更，现 0.5.x）：`get_clean_factor_and_forward_returns(factor, prices, quantiles=5)` 一行出 IC 表/分位收益/turnover。**口径照抄**：IC=Spearman(factor_t, fwd_ret)、ICIR=mean(IC)/std(IC)、分位单调性、top-bottom spread、分位换手。
- **vectorbt 1.2.0 / backtrader 1.9.78.123**（后者长期停更）：只借其成交假设清单——次日开盘成交、双边成本、涨跌停不可成交、停牌跳过。**都不进依赖树**（numba 编译栈 / 事件驱动慢）。
- **quantstats**：只吃日收益序列算 Sharpe/Sortino/CVaR/回撤，不校验撮合——绩效层与撮合层应解耦。
- **walk-forward 必须配 purge/embargo**（AFML §7）：训练尾切 `label_horizon` 天，测试头留 embargo 缓冲；与 Qlib `trunc_days=horizon+1` 同源。

### 1.7 技术指标库
- **TA-Lib**：C 库+编译/whl 依赖，口径权威但装不上就装不上，不进依赖。
- **pandas-ta**：PyPI 最新仍为 `0.4.71b0`（beta），要求 Python ≥3.12，主仓库维护不稳，**不可作硬依赖**。
- **stockstats**（BSD-3，纯 pandas）：`wrap(df)` 后按列名惰性计算——`df['rsi_6']`、`df['close_20_sma']`、`df['macd']/['macds']/['macdh']`、`df['boll_ub']`、`kdjk/kdjd/kdjj`。**最贴合我们约束的范式**：列名即公式、按需计算、删列即重算。
- 实现注意：EMA 用 `adjust=False` 与首值初始化的差异会让 RSI/MACD 与券商软件差 1-2 点；ATR 用 SMMA 而非 SMA；KDJ 默认 (2/3,1/3) 平滑；**指标必须在前复权价上算**，且窗口内不跨除权日。

### 1.8 "信号权重自学习"的开源实现
- river `bandit`（Exp3/UCB/Thompson）+ `model_selection.BanditClassifier`：把"选哪个策略族"建成 bandit，天然处理 explore/exploit，可避免"最近谁好就一直用谁"的追高。
- Hedge/EG 就是"专家建议组合"，与多信号加权同构；OLPS 系（PAMR/OLMAR）把同一思想用于组合权重，可借鉴其**单纯形投影 + 权重上限 cap**。

## 2. 对比表

| 项目 | 核心机制 | 借鉴点 | 不借鉴点 |
|---|---|---|---|
| Qlib Rolling/DDG-DA | 滚动重训+元模型给样本加权 | `trunc_days=horizon+1` 截断；Reweighter 形态；除以 close 的无量纲化 | torch/LGBM/MLflow 重栈；黑箱权重；online_srv 状态机 |
| FreqAI | 定时重训+时间衰减样本权重 | `exp(−i/(αn))`；DI 外推预警；剔零方差列 | 万级特征膨胀；SVM/DBSCAN 剔离群；每标的独立模型 |
| river | 流式学习+漂移检测 | ADWIN 参数语义；只监控一维 IC；PA/FTRL 可用性 | Rust 核心的编译依赖 |
| FinRL | RL 直接输出下单股数 | turbulence 熔断 | 绝对金额奖励；撮合不真实；不可解释 |
| Hedge/Beta/Meta-labeling | 乘性更新+后验收缩+二级闸门 | 三者组合，无需梯度库 | 逐特征梯度优化（样本不够） |
| alphalens-reloaded | 因子分层与 IC 报表 | IC/ICIR/分位单调/turnover 口径 | 作运行时依赖（只抄口径） |
| vectorbt/backtrader | 向量化/事件驱动回测 | 成交假设清单 | numba 编译栈；停更 |
| stockstats/pandas-ta/TA-Lib | 指标计算 | 列名即公式、惰性计算、纯 pandas | C 依赖；beta 版本与维护风险 |

## 3. 最终架构取舍（6 条原则）

1. **只学权重，不学参数**。可学习对象仅四类：信号权重 w、regime 条件权重、收缩强度 λ、漂移阈值 δ。指标窗口/阈值/因子定义一律写死——在线学这些就是在线过拟合。
2. **乘性更新为主，贝叶斯收缩定幅度**。`w_i ← normalize(w_i^shrunk · exp(η·EWMA_IC_i))`；η≈0.05–0.1，EWMA 半衰期 20–60 日。放弃梯度下降：样本量不支持且不可解释。
3. **漂移检测只调"学习率"，不调"模型"**。对滚动 IC 跑 Page-Hinkley；报警时 η 临时放大 2–3 倍并重置先验，而非重训。响应可解释、可回滚。
4. **要分状态，但用软分箱+最小样本门槛**。状态取 2×2（趋势温度高/低 × 波动分位高/低），每格一套条件权重；格内样本 <60 交易日即退回全局权重。绝不按行业/个股学权重。
5. **信号先横截面标准化再加权**：每个信号先转 rank/z-score 到 [-1,1] 再乘权重求和，单信号贡献 cap ≤35%，避免量纲与极值主导。
6. **验证强制 walk-forward + purge/embargo**：训练尾切 `horizon+1` 天、测试头留 5 日 embargo；只报 OOS 的 IC/RankIC/ICIR/分层单调性/换手，禁止全样本调参；每个自适应动作写 JSONL 权重快照（date, w, regime, η, 触发原因），可回放解释。

## 4. 最小可行参数体

```
core/       纯函数：load_bars(前复权) → 校验（停牌/涨跌停/缺失）
indicators/ 纯函数：df → df，~20 个指标，列名即公式（stockstats 范式）
signals/    纯函数：df → score∈[-1,1]，一信号一函数，无状态
regime/     纯函数：df → (trend_state, vol_state, turbulence)，2×2 软分箱
weights/    ★唯一有状态层：Hedge 更新 + Beta 收缩 + 单纯形投影 + cap
meta/       二级闸门：逻辑回归/GBM 判"该信号是否有效"，输出 0-1 乘数
eval/       purged walk-forward、IC/RankIC/ICIR、分层、换手、漂移监控
store/      JSONL 权重快照 + 版本号，可回放
```

状态总量 ≈ `15 信号权重 + 4×15 条件权重 + 4 个超参` ≈ 80 个数，全部可打印、可人工审阅。

## 参考
- Qlib rolling：https://github.com/microsoft/qlib/blob/main/qlib/contrib/rolling/base.py ｜ DDG-DA：https://github.com/microsoft/qlib/blob/main/qlib/contrib/rolling/ddgda.py ｜ 元模型：https://github.com/microsoft/qlib/blob/main/qlib/contrib/meta/data_selection/model.py ｜ online：https://github.com/microsoft/qlib/blob/main/qlib/workflow/online/manager.py ｜ Alpha158：https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
- FreqAI：https://www.freqtrade.io/en/stable/freqai/ ｜ 特征工程（weight_factor/DI）：https://www.freqtrade.io/en/stable/freqai-feature-engineering/
- river ADWIN：https://github.com/online-ml/river/blob/main/river/drift/adwin.py ｜ FTRL：https://riverml.xyz/latest/api/optim/FTRLProximal/
- FinRL 环境：https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py
- alphalens-reloaded：https://github.com/stefan-jansen/alphalens-reloaded ｜ purging/embargo：https://www.interactivebrokers.com/campus/contributors/cross-validation-in-finance-purging-embargoing-combination/
- stockstats：https://github.com/jealous/stockstats ｜ pandas-ta：https://pypi.org/project/pandas-ta ｜ vectorbt：https://pypi.org/project/vectorbt ｜ backtrader：https://pypi.org/project/backtrader
