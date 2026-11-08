# 04 理论依据：这几条公式为什么长这样

> 本文不是教科书摘抄。每条机制都标注了它在 `qcore/learner.py` / `qcore/engine.py` 里的**落点与默认参数**，
> 可以逐条对着代码核。§1–§5 讲系统在算什么，§6–§8 讲为什么不能相信算出来的东西。

---

## 1. 专家建议框架：Hedge / Multiplicative Weights（乘性权重）

把 N 个「信号族 × 参数」当成 N 位专家。第 t 天专家 i 给出方向 `dir_i`，系统付给它标准化收益 `y_i`
（口径见 SPEC §4.1：`y = clip(dir·r/(σ√h), −C, C)`，其中 σ 是该股 20 日收益率标准差，
`dataset.py:137` 的 `rolling(20).std()`——除以 √h 是为了把 h 日收益还原到"每日波动"的量纲）。
把「打脸」写成损失 `ℓ_i = −y_i`，Hedge 的更新是乘性的：

```
w_i ← w_i · exp(−η·ℓ_i) = w_i · exp(η·y_i)
对数域实现（learner.py:119）:   logw[i] += η · y_i
归一化后即 softmax(η · Σ_t y_i) —— 历史越准，权重指数级越大
```

标准遗憾界（N 位专家、损失 ℓ∈[0,1]、T 轮；Arora–Hazan–Kale 综述的同一形式）：

```
Regret(T) = Σ_t ⟨w_t, ℓ_t⟩ − min_i Σ_t ℓ_i(t)  ≤  (ln N)/η + ηT/8
取 η = sqrt(8 ln N / T)   ⇒   Regret ≤ sqrt(T·ln N / 2)  =  O( sqrt(T ln N) )
```

含义比数值重要：**最优专家的平均损失与「事后诸葛亮」之差随 T 递减到 0**。它保证权重会向表现好的专家集中，
但**不保证任何专家真的赚钱**——遗憾界是相对量，不是绝对收益。

注意这个界要求损失有界：我们的 ℓ = −y、y 被 clip 到 ±3，严格写应是 ηT·C²/8。**`clip_C = 3` 不只是防极值，
它还是这条界成立的前提**——无界损失下乘性更新会被单次暴涨彻底带偏，理论保证直接失效。

本项目对标准 Hedge 的三处刻意偏离：

1. `eta = 0.02` 远小于理论最优（N≈80、T=1000 时 η 应取 ≈0.19）。因为真正的判据是命中率项 `β_R·(2R−1)`，
   Hedge 只承担慢速趋势项；代价是**失去最优性保证**，只保留「方向正确」的定性性质。
2. `logw` 每天乘 `lw_decay = 0.9995`（`_decay_credit`），给累积信用加了半衰期 ≈ 1386 个交易日的泄漏，
   防止早期偶然连胜把权重永久锁死。
3. 权重还要过温度 `tau = 1.5`、探索项 `eps_explore = 0.03`、下限 `w_min = 0.005`（见 §3 末尾的 `weights()` 链）。

## 2. Sleeping experts：没发声不罚

金融信号天然稀疏——RSI 超卖一年只喊几十次，均线金叉几乎天天有。若把「没发声」按 0 收益计分，
稀疏信号会被高频信号淹没，系统退化为「谁话多信谁」。

`update()` 第一行就是这件事（learner.py:109）：

```python
m = np.asarray(active, dtype=bool) & np.isfinite(payoff)
# 只有 m 为真的位置才更新 a / b / E / logw / n
```

`active` 由引擎提供：在**被对账的那个预测日 s**（= t−1−h），该信号全池中方向非零的股票数 > 0 才算发声
（engine.py:127 `active = cnt > 0`）。没发声的专家权重**冻结**、既不奖励也不惩罚。这正是 Freund–Schapire–Singer–Warmuth 的 specialist 模型：遗憾界是相对
「任意子集上的最优专家」，比标准 Hedge 的「全体最优」更强、也更难达到。

## 3. Beta-Bernoulli 收缩：小样本必须向先验低头

命中率不是「命中数 / 总数」，而是后验均值（learner.py:71）：

```
a_i ← ρ·a_i + 1[y_i>0]      b_i ← ρ·b_i + 1[y_i<0]        （带遗忘的伪计数）
R_i = (a_i + κ·p0) / (a_i + b_i + κ)          κ = 8, p0 = 0.5
z_i = β_R·(2R_i − 1) + β_E·E_i + β_H·logw_i    β = (1.0, 0.5, 1.0)
```

`κ` 的读法：**等效先验样本数**（pseudocount）。n=0 时 R=0.5（完全不信）；n=κ=8 时先验与数据各占一半；
n≫κ 时 R → a/(a+b)。这正是 `Beta(κp0, κ(1−p0))` 先验下的后验均值，也是经验贝叶斯收缩的最简形式。
收益均值项 `E ← ρ·E + (1−ρ)·y` 同理：EWMA 的标准误随样本数下降，小样本时被 `β_E` 自然压住。

为什么必须收缩：h=5 的信号一年只有约 50 次不重叠样本，重叠窗口下独立样本还要再打折。
不收缩的比率估计标准误在 0.07 量级，足以让 50% 与 60% 无法区分——此时「谁命中率高」纯属噪声排序。

最终权重链（learner.py:85-99），把上面所有量合成一个数：

```
γ = γ_max·n/(n+γ_n0)                    # γ_max=0.6, γ_n0=60：regime 表随样本增长才敢说话
z = (1−γ)·z_global + γ·z_regime          # 分层收缩：格内样本少 → 退回全局
w = softmax(τ·z);  w ← (1−ε)·w + ε/S;  w ← max(w, w_min);  w ← w/Σw
```

（`S` 是**信号总数**，不是当日发声数——探索项按全体均摊，保证长期不发声的专家也留有一线配额。）

## 4. 非平稳与遗忘：1/(1−ρ) 就是「记忆长度」

`decay_rho = 0.995`。几何衰减计数的稳态有效样本量是

```
n_eff = Σ_{k≥0} ρ^k = 1/(1−ρ) = 200 次更新        半衰期 = ln0.5 / ln ρ ≈ 138
```

引擎每个交易日对每个信号更新一次（engine.py:132，payoff 取被对账日 s 的全池截面均值），所以
**200 次更新 ≈ 200 个交易日 ≈ 10 个月**——这就是系统「记得多久以前的对错」。

为什么金融数据必须遗忘：结构变化（注册制、涨跌停规则、量化占比、行业轮动周期）让「2015 年的动量有效」
对今天没有约束力。用全历史平均等价于假设平稳，而这是最贵的隐含假设。Besbes–Gur–Zeevi 给出了
总变差预算受限的非平稳 bandit 的极小极大遗憾下界 **Θ(T^{2/3})**——在这种环境下，平稳假设的 O(√T) 根本不可达；
Garivier–Moulines 进一步证明指数折扣的 UCB 在该设定下接近最优。所以遗忘不是妥协，而是**代价明码标价的正确选择**，
比「假设平稳然后被现实打脸」便宜。

## 5. 概念漂移检测：Page-Hinkley（对比 ADWIN）

单侧 PH 检验「收益均值是否显著下移」（learner.py:139）：

```
m_t = m_{t−1} + (y_t − m_{t−1}) / n_t          累计均值
S_t = S_{t−1} + (y_t − m_t − δ)                δ = ph_delta = 0.02（容忍度）
M_t = min(M_{t−1}, S_t)
报警条件： (S_t − M_t) > λ                      λ = ph_lambda = 8.0
```

即「累计偏离自身均值超过 λ」。报警动作**不是重训模型，而是给记忆打折**：`a,b,E,logw` 全部乘
`ph_reset = 0.35`，`S/M` 清零，`alarms += 1`。效果是「这位专家过去的好成绩作废，从半信半疑重新开始」——
比「永久相信」安全，比「立刻清零」平滑，且完全可回放解释。

ADWIN（Bifet & Gavaldà）是另一条路：维护可变长窗口，用 Hoeffding 界检验窗口两段的均值差，检出即丢弃旧段。
它给出 δ 置信下的假阳率保证，但要维护多尺度窗口。本项目只有 80 条一维序列、每天一次，PH 的 20 行 numpy
更划算；代价是 λ/δ 需要标定——`audit/sensitivity.py` 的参数 ±50% 网格就是干这个的。

## 6. 市场有效性与可预测性边界

弱式有效（Fama 1970）说：仅凭历史价格不能系统性获取超额收益。它的现代版本更精确——
**不是「没有预测力」，而是「预测力小到扣完成本后不剩」**：

- 技术信号的横截面 IC 通常在 **0.02–0.05** 量级。IC=0.03、池子 80 只时，按基本法则 `IR ≈ IC·√BR`
  算出的理论 IR 只有 0.3 上下；日频换手扣掉双边 10bps，很容易归零。
- 异象会衰减：McLean & Pontiff 测得学术发表后因子收益平均下降约 58%（发表前的样本外已降 26%）；
  A 股量价异象因散户结构与量化拥挤，衰减更快。
- **别期待圣杯。** 本项目的目标不是找圣杯，而是「在 80 个都不怎么样的信号里，动态判断今天该多信谁一点」。
  这是权重层的边际改善，量级是个位数百分点 IR，不是翻倍。

因此 SPEC §6 把「必须跑赢等权集成」写成硬性基线：**跑不赢就说明学习器无效，必须如实报告。**

## 7. 防泄漏：walk-forward、purge/embargo、PIT、生存者偏差

四件事缺一件，全部指标作废：

1. **Walk-forward**：只用 ≤t 的信息决策，t+1 开盘成交，`r_h(t) = open(t+1+h)/open(t+1) − 1`。
2. **Purge / embargo**：训练/更新窗口的尾部必须切掉 `h+1` 天。因为标签 `r_h(t)` 用到 `t+1+h` 的价格，与
   `t+h` 的标签在时间上重叠；不切就是「用未来样本训练当下」。本项目做成硬编码：t 日只用 `s = t−1−h`
   的预测对账（engine.py:117），等价于环形缓冲 `queue[t]` 只弹出已到期项。
3. **PIT（point-in-time，点对点）**：任何统计量只许 `rolling` / `expanding`，禁止全样本 z-score、
   分位数、min/max——那是隐性未来函数。可证伪检验是**截断不变性**：数据截到 T 日重跑，T 日的
   `score/weights/regime` 必须逐元素一致（容差 1e-12），见 `tests/test_no_lookahead.py`。
4. **生存者偏差**：用今天的成分股回测十年前，等于只研究活下来的公司。本项目按**当日**成交额/换手率
   动态取池，不用今天的市值排名反筛历史。

## 8. 多重检验：网格搜索必然过拟合

`signal_zoo` 生成 50–90 个组合，再网格调 `h/ρ/κ/η/τ/ε`，试过的配置数 M 轻松上百。M 次试验里
最大夏普的期望值随 `√(2 ln M)` 增长——**「最好的那条回测曲线」是选择偏差的产物，不是发现**。

- **Deflated Sharpe Ratio**（Bailey & López de Prado）：把观测 SR 对照「M 次试验下的最大期望 SR」标准化，
  输出「多重检验下仍显著的概率」。
- **White's Reality Check**：对「策略集里最好的那个是否优于基准」做 bootstrap 检验，控制 family-wise error。

工程落地是审计而非公式：引擎自带的 `mode="placebo"`（按 sid 平移或翻符号后重跑，OOS 指标必须回落到 ~0，
否则是评估代码泄漏），加上 SPEC §10 规划的 `audit/sensitivity.py` 参数 ±50% 扫描（看是否「孤峰」）。
**孤峰即过拟合**，比任何 p 值都直观。

## 9. Meta-labeling（v2 方向，当前未实现）

一级模型出方向（即本项目的 `score`），二级模型只回答「这一注该不该下、下多大」。标签为 `1[一级方向正确]`，
输出概率 p 直接当仓位乘数——落在 [0,1]、可解释为「信心系数」，且不损失一级模型的召回率。
它解决的痛点：一级模型永远在喊多或喊空，缺少「今天别玩」这种表达。二级闸门用 5–10 个特征
（regime、横截面离散度、近期 IC、turbulence）做逻辑回归即可，样本量要求远低于重训一级模型。

---

## 参考文献

在线学习与专家建议
- Freund & Schapire, *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*, JCSS 55(1), 1997 — https://doi.org/10.1006/jcss.1997.1504
- Arora, Hazan & Kale, *The Multiplicative Weights Update Method: a Meta-Algorithm and Applications*, Theory of Computing 8, 2012 — https://theoryofcomputing.org/articles/v008a006/
- Freund, Schapire, Singer & Warmuth, *Using and Combining Predictors that Specialize*, STOC 1997 — https://www.schapire.net/papers/FreundScSiWa97.pdf

非平稳、遗忘与漂移检测
- Besbes, Gur & Zeevi, *Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards*, NeurIPS 2014 — https://proceedings.neurips.cc/paper_files/paper/2014/file/91ba7292e5388b90b58d0b839a7f19ec-Paper.pdf
- Garivier & Moulines, *On Upper-Confidence Bound Policies for Switching Bandit Problems*, ALT 2011 — https://doi.org/10.1007/978-3-642-24412-4_16
- Kulhavý & Zarrop, *On a General Concept of Forgetting*, International Journal of Control, 1993 — https://doi.org/10.1080/00207179308923034
- Page, *Continuous Inspection Schemes*, Biometrika 41(1-2), 1954 — https://doi.org/10.1093/biomet/41.1-2.100
- Hinkley, *Inference about the Change-Point from Cumulative Sum Tests*, Biometrika 58(3), 1971 — https://doi.org/10.1093/biomet/58.3.509
- Bifet & Gavaldà, *Learning from Time-Changing Data with Adaptive Windowing*（ADWIN）, SDM 2007 — https://epubs.siam.org/doi/10.1137/1.9781611972771.42

收缩估计
- Efron, *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*, Cambridge, 2010 — https://doi.org/10.1017/CBO9780511761362

有效性与可预测性
- Fama, *Efficient Capital Markets: A Review of Theory and Empirical Work*, J. Finance 25(2), 1970 — https://doi.org/10.2307/2325486
- Grinold, *The Fundamental Law of Active Management*, JPM 15(2), 1989 — https://doi.org/10.3905/jpm.1989.409211
- McLean & Pontiff, *Does Academic Research Destroy Stock Return Predictability?*, J. Finance 71(1), 2016 — https://onlinelibrary.wiley.com/doi/10.1111/jofi.12365

防泄漏与多重检验
- Bergmeir & Benítez, *On the Use of Cross-Validation for Time Series Predictor Evaluation*, Information Sciences 191, 2012 — https://doi.org/10.1016/j.ins.2011.12.028
- Elton, Gruber & Blake, *Survivor Bias and Mutual Fund Performance*, RFS 9(4), 1996 — https://doi.org/10.1093/rfs/9.4.1097
- López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018（第 3、7 章：meta-labeling、purge/embargo）— https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- López de Prado, *The 10 Reasons Most Machine Learning Funds Fail*, JPM, 2018 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816
- Bailey & López de Prado, *The Deflated Sharpe Ratio*, JPM 40(5), 2014 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- White, *A Reality Check for Data Snooping*, Econometrica 68(5), 2000 — https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152
- Joubert, *Meta-Labeling: Theory and Framework*, SSRN, 2022 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018
