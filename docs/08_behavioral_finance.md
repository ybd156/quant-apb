# 08 行为金融学文献综述：从人性偏差到可量化因子

> **写法**：每节「理论 → 可量化代理 → A 股适用性」三段。**三条纪律**：① 文献只列能点开核实的；
> ② 方向是「文献在原市场的结论」，不是承诺；③ 不写"这些因子能赚钱"。
> **现实锚点**：本项目 73 个技术信号在 A 股 283 只 / 2013–2026 / h=5 上实测 **Rank IC = −0.0185、
> HAC t = −2.48**（AGENTS.md §9）。任何"理论上该有效"的因子，都得先过这条线。

---

## 1. 前景理论与损失厌恶

**理论**：Kahneman & Tversky (1979) 用价值函数替代期望效用——收益区**凹**（风险厌恶）、损失区**凸**
（风险偏好），且损失更陡。Tversky & Kahneman (1992) 加入概率权重：小概率被高估、中高概率被低估。

```
v(x) = x^α (x≥0)        α≈0.88
v(x) = −λ·(−x)^β (x<0)  β≈0.88, λ≈2.25
```

**代理**：价值函数不可观测，但它给出三条可检验推论——参考点依赖、损失区凸、概率权重扭曲，
分别落到 §2、§4。本项目不拟合 `α/β/λ`（样本不支持），只把 `λ>1` 当作"下跌时风险偏好上升"的结构假设。

**A 股**：参考点在散户中极强（成本价、"回本就走"），但**参考点不可观测**，只能用换手率加权均价近似
（见 §2）。这是全篇最大的测量误差来源，必须承认。

## 2. 处置效应（最可量化的一条）

**理论**：Shefrin & Statman (1985) 把"卖赢守亏"接到前景理论；Odean (1998) 用万个账户证实：
盈利股被卖出的概率比亏损股高约 50%。

**代理**：Grinblatt & Han (2005) 的**资本利得突出量**把个体行为升格为横截面因子：

```
RP_t  = Σ(V·P)/ΣV      # V=换手率，P=成交均价，n=60~250
CGO_t = (P_t − RP_t)/P_t
```

`CGO>0`（整体浮盈）→ 处置抛压 → **预期负**；`CGO<0`（浮亏惜售）→ 供给不足 → 预期正。

**A 股**：**最该优先做**——只需日频 OHLCV + 换手率，`data.py` 全有；散户换手高、参考点更新快；
Feng & Seasholes (2005) 用中国券商账户直接证实处置效应。但涨跌停 + 打板接力可能让符号翻转，必须实测。

## 3. 过度自信与过度交易

**理论**：Odean (1999) 发现交易越频繁、净收益越差；Barber & Odean (2001) 用性别做自然实验：
男性交易频率高 45%、净收益低 0.94pp/年，归因于过度自信。

**代理**：换手率 20 日均值（横截面 z）；**异常换手** `abn = mean(turn,5)/mean(turn,60)`（剥离个股基线）。
预期方向：**负**。

**A 股**：年换手常 400%+，横截面离散度大 → 理论上信噪比更好；但换手率与市值、流动性、涨停高度共线，
**必须行业+市值中性化**，否则测到的是小盘股效应。Liu, Stambaugh & Yuan (2019) 在中国三因子里
把换手率列为解释力最强的投机类指标，这是 A 股最直接的支持证据。

## 4. 彩票偏好与 MAX 效应

**理论**：Barberis & Huang (2008) 把累积前景理论装进均衡模型——概率权重高估小概率 → 投资者系统性
高估"小概率暴涨"股 → 过度定价 → 未来收益低。Bali, Cakici & Whitelaw (2011) 给出极简代理 MAX。

**代理**：`MAX20 = max(r, 过去20日)`（或最大 5 日均值）；`SKEW60 = skew(r, 60日)`。预期方向：**负**。

**A 股**：涨跌停（±10%/±5%/±20%）制造大量截断右尾，散户结构强化打板博弈，是彩票偏好最理想的实验室。
Nartea, Kong & Wu (2017) 用中国样本报告了显著 MAX 效应。**但** A 股 MAX 常由涨停构成，而涨停有次日惯性，
5 日尺度可能"先正后负"——`h` 的选取会直接改变符号，这正是必须多 `h` 实测的原因。

## 5. 特质波动率之谜

**理论**：Ang, Hodrick, Xing & Zhang (2006) 发现 IVOL 最高的组合收益最低，且规模/价值/动量都解释不了。

**代理**：60 日滚动市场回归 `r_i = α + β·r_m + ε`，`IVOL = std(ε)`。预期方向：**负**。

**A 股**：IVOL 与 MAX、换手率都指向"投机性"，**三者同时入模会互相吃掉**。建议先测横截面相关，
若 ρ>0.7 只保留一个，看**增量 IC** 而不是各自单独 IC。

## 6. 投资者情绪

**理论**：Baker & Wurgler (2006) 用 6 个代理（封闭式基金折价、换手率、IPO 数量与首日收益、股权融资占比、
股息溢价）的第一主成分构造**市场级**情绪；结论是情绪高涨期，难套利难估值的股票后续收益低。
Baker & Wurgler (2007) 是通俗综述。

**代理（A 股版 BW 指数）**：全市场换手率（log 后 z）、**融资买入占成交额比**、涨停/跌停家数、
新增开户（月度）、封闭式基金折价（标的少，可选）。

**A 股**：情绪是**市场级时序**变量，决定"整体该不该减仓"，不是选股因子。`REGIME` 四档已是粗糙代理。
**关键限制**：开户数、IPO 数是低频且事后修订的，**PIT 可得性差**，回测只能用当日可观测的换手率/两融/涨跌停。

## 7. 注意力与显著性

**理论**：Barber & Odean (2008) 提出散户是"注意力买家"——面对海量股票只买**进入视野**的（新闻、异常量、
极端涨跌），且买入侧远强于卖出侧。Bordalo, Gennaioli & Shleifer (2012, 2013) 给出机制：注意力分配给
"与参照值差异最大"的维度，导致小概率尾部被系统性高估。

**代理**：20 日均额排名 − 60 日均额排名（注意力跃升）；过去 20 日涨停次数、`|r|>7%` 天数；换手突变。
预期方向：**短期（1–5 日）正、随后转负**——全篇唯一明确"随 `h` 翻转"的因子族。

**A 股**：涨停板 + 龙虎榜 + 股吧让注意力极度可观测，信噪比高于美股。但龙虎榜、股吧热度**历史上不可回溯**
（当年无存档），属"不可 PIT 回填"，**只能实盘监控、不能进回测**（对应 `info.py` 的 `pit_backfillable=False`）。

## 8. 羊群行为与信息瀑布

**理论**：Banerjee (1992)、Bikhchandani, Hirshleifer & Welch (1992) 证明：观测前人行为并忽略私有信息，
会形成**信息瀑布**——价格脱离基本面且极脆弱。

**代理**：Chang, Cheng & Khorana (2000) 的两个**市场级**指标：

```
CSSD = sqrt( Σ(r_i − r̄)² / (N−1) )
CSAD = (1/N)·Σ abs(r_i − r_m)
回归：CSAD = α + γ1·abs(r_m) + γ2·r_m²      γ2 < 0（显著）⇒ 存在羊群
```

**A 股**：这是**择时/风险**指标而非选股因子——CSSD 骤降 + 指数大涨 = 追涨羊群，后续回撤概率升。
Yao, Ma & He (2014) 用沪深两市报告了显著羊群，且**下跌行情中更强**。可作 `REGIME` 第五维候选或降杠杆触发器。

## 9. 锚定与 52 周高点

**理论**：George & Hwang (2004) 发现"距 52 周高点的距离"比传统动量预测力更强，机制是**锚定**：
投资者把 52 周高点当参考点，接近高点时不愿卖（处置效应的镜像）→ 对好消息反应不足。
注意方向与 §2 相反——**越接近高点，未来收益越高**。

**代理**：`P52 = max(P, 252日)`，`D52 = (P − P52)/P52 ∈ [−1,0]`。预期方向：**正**。

**A 股**：计算零成本，但 2015 与 2021 两轮牛熊让 52 周高点在不同年份含义完全不同，
**必须滚动标准化**（用 D52 的滚动分位而非原值），否则等于在赌"当前处于牛熊哪一段"。

## 10. 套利限制（为什么这些偏差不会被套利掉）

**理论**：Shleifer & Vishny (1997)：套利需要杠杆 + 外部资金，价格偏离恶化时出资人撤资，
套利者被迫在最差时点平仓 → **错误定价可以长期存在**，理性套利者的容量有限。
这一节是全篇的"存在性证明"：若套利无摩擦，§2–§9 的因子几天内就会被抹平。

**A 股**：套利限制比美股更强，是双刃剑——更强则错误定价更持久、因子可能更显著；
但也意味着**做空几乎不可行**（券源少、成本高、T+1 与涨跌停）。所有"多空价差"结论在 A 股只能实现多头端，
**空头端 alpha 拿不到**。这是回测与实盘差距最大处，必须在报告里明写。

## 11. A 股特殊性

1. **散户主导**：个人投资者贡献 60%+ 成交额，行为偏差的总量效应应强于机构主导市场。
   Carpenter, Lu & Whitelaw (2021) 评估了 A 股价格发现功能，指出信息效率在改善但仍低于成熟市场。
2. **换手率全球最高**：年换手 400%+。含义：横截面区分度大，但**成本吃掉更多**——本项目实测
   A 股 5 日调仓 70% 换手成本拖累约 **6.8%/年**（AGENTS.md §9），这是任何因子先要跨的门槛。
3. **涨跌停制度**：制造截断分布 → MAX/偏度/涨停次数被**制度性扭曲**；涨停当日**无法买入**，
   回测须按"不可成交"处理。Chen, Gao, He, Jiang & Xiong (2019) 证明涨跌停诱发"磁吸效应"，
   是**制度本身制造的行为偏差**，不是纯粹心理。
4. **T+1**：当日买入不可卖 → 日内反转不可行；也意味着注意力驱动的当日追高**无法当日纠错**，
   放大次日开盘抛压。
5. **壳价值**：Liu, Stambaugh & Yuan (2019) 指出 A 股小市值的"规模溢价"很大程度是**借壳期权**，
   必须剔除后才干净——这直接影响所有行为因子的市值中性化做法。

---

## 12. 可量化因子清单

> 下表同时是 `qcore/behavioral.py` 的**实现契约**（模块在写；`bf_*` 为建议命名，若实现不符以代码为准）。
> 「方向」是**文献在原市场的结论**，A 股是否成立须用 `stats.mean_tstat(lags=h-1)` + Deflated Sharpe 实测。

| # | 因子 | 定义（公式） | 文献方向 | 代表文献 | 数据需求 | A 股可行性 |
|---|---|---|---|---|---|---|
| 1 | `bf_cgo` 资本利得突出量 | `(P−RP)/P`，`RP=Σ(V·P)/ΣV` | 负 | Grinblatt & Han (2005) | OHLCV+换手率 | ★★★ 优先做 |
| 2 | `bf_abn_turn` 异常换手 | `mean(turn,5)/mean(turn,60)` | 负 | Odean (1999); Barber & Odean (2001) | 换手率 | ★★★ |
| 3 | `bf_max20` 彩票 MAX | 过去 20 日 `max(r)` | 负 | Bali et al. (2011) | 日收益 | ★★★ |
| 4 | `bf_skew60` 收益偏度 | 过去 60 日 `skew(r)` | 负 | Barberis & Huang (2008) | 日收益 | ★★★ |
| 5 | `bf_ivol` 特质波动 | 60 日市场回归残差 `std(ε)` | 负 | Ang et al. (2006) | 日收益+指数 | ★★★ |
| 6 | `bf_d52` 距 52 周高 | `(P−max(P,252))/max(P,252)` | 正 | George & Hwang (2004) | 日收盘 | ★★★ 须滚动标准化 |
| 7 | `bf_attn` 注意力跃升 | 20 日均额排名 − 60 日均额排名 | 短正长负 | Barber & Odean (2008) | 成交额 | ★★★ |
| 8 | `bf_limit_cnt` 涨停次数 | 20 日内 `pct_chg≥9.8%` 天数 | 短正长负 | Bordalo et al. (2013); Chen et al. (2019) | 涨跌幅 | ★★★ 制度性 |
| 9 | `bf_cssd` 横截面离散 | `sqrt(Σ(r_i−r̄)²/(N−1))` | 择时/风险 | Chang et al. (2000) | 全池日收益 | ★★★ 非个股 |
| 10 | `bf_csad` 羊群回归 | `CSAD=α+γ1·abs(r_m)+γ2·r_m²` | 择时/风险 | Chang et al. (2000); Yao et al. (2014) | 全池+指数 | ★★★ 非个股 |
| 11 | `bf_sent` 情绪指数 | 换手+融资占比+涨停家数 的第一主成分 | 市场级 | Baker & Wurgler (2006, 2007) | 换手+两融+涨跌停 | ★★☆ 两融 2010+ |
| 12 | `bf_margin` 融资买入占比 | 融资买入额/成交额（20 日均） | 负 | Baker & Wurgler (2006) | 两融明细 | ★★☆ 仅两融标的 |
| 13 | `bf_disp` 分歧代理 | 20 日振幅均值/换手率 | 负 | —— 代理，无直接文献 | OHLCV | ★★☆ 是代理非原变量 |
| 14 | `bf_guba` 股吧/龙虎榜 | 讨论数、上榜次数 | 短正长负 | Barber & Odean (2008) | 另接数据源 | ☆ 不可 PIT，仅实盘 |

**共 14 条，其中 #1–#10 只用现有日频 OHLCV 就能算**，是优先实现集。

## 13. 为什么行为因子可能比技术因子更持久

McLean & Pontiff (2016) 测了 97 个已发表异象：**样本外收益比样本内低 26%，论文发表后再降 58%**——
公开化 → 套利资金涌入 → 抹平。技术因子（均线、MACD、RSI）是**最公开的一类**，写进了每本教材与每个
行情软件的默认指标，被套利得最彻底。这也解释了本项目实测技术信号 IC 为负：**不是参数没调好，
而是这批信号的信息早已被定价**。

行为因子有两点不同：① **来源是效用函数，不是价格模式**——`λ≈2.25`、概率权重扭曲是**偏好参数**，
不随"被发现"而消失，除非人的心理变了；② **衰减的是"可套利部分"**，在 Shleifer-Vishny 框架下，
A 股的做空限制、T+1、涨跌停让套利端结构性受限，偏差的价格后果更难被消除
（反过来说，你也同样拿不到空头端收益）。

**必须泼的冷水**：以上只是"衰减更慢"的论证，**不是"一定有效"的保证**。行为因子同样会因拥挤失效
（2015 年后 A 股量化规模暴增，小市值+反转明显钝化），且行为代理的测量误差远大于价格类因子。
**结论应写成**："文献在美股/部分 A 股样本上报告了方向 X 的显著结果；本项目实测见 `reports/`，
两者不一致时以本项目实测为准。"

---

## 参考文献

**前景理论**
- Kahneman, D. & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica* 47(2), 263–291. https://doi.org/10.2307/1914185
- Tversky, A. & Kahneman, D. (1992). Advances in Prospect Theory: Cumulative Representation of Uncertainty. *Journal of Risk and Uncertainty* 5(4), 297–323. https://doi.org/10.1007/BF00122574

**处置效应与动量**
- Shefrin, H. & Statman, M. (1985). The Disposition to Sell Winners Too Early and Ride Losers Too Long: Theory and Evidence. *Journal of Finance* 40(3), 777–790. https://doi.org/10.1111/j.1540-6261.1985.tb05002.x
- Odean, T. (1998). Are Investors Reluctant to Realize Their Losses? *Journal of Finance* 53(5), 1775–1798. https://doi.org/10.1111/0022-1082.00072
- Grinblatt, M. & Han, B. (2005). Prospect Theory, Mental Accounting, and Momentum. *Journal of Financial Economics* 78(2), 311–339. https://doi.org/10.1016/j.jfineco.2004.10.006

**过度自信与过度交易**
- Odean, T. (1999). Do Investors Trade Too Much? *American Economic Review* 89(5), 1279–1298. https://doi.org/10.1257/aer.89.5.1279
- Barber, B. M. & Odean, T. (2001). Boys Will Be Boys: Gender, Overconfidence, and Common Stock Investment. *Quarterly Journal of Economics* 116(1), 261–292. https://doi.org/10.1162/003355301556400

**彩票偏好与 MAX**
- Bali, T. G., Cakici, N. & Whitelaw, R. F. (2011). Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns. *Journal of Financial Economics* 99(2), 427–446. https://doi.org/10.1016/j.jfineco.2010.08.014
- Barberis, N. & Huang, M. (2008). Stocks as Lotteries: The Implications of Probability Weighting for Security Prices. *American Economic Review* 98(5), 2066–2100. https://doi.org/10.1257/aer.98.5.2066

**特质波动率**
- Ang, A., Hodrick, R. J., Xing, Y. & Zhang, X. (2006). The Cross-Section of Volatility and Expected Returns. *Journal of Finance* 61(1), 259–299. https://doi.org/10.1111/j.1540-6261.2006.00836.x

**投资者情绪**
- Baker, M. & Wurgler, J. (2006). Investor Sentiment and the Cross-Section of Stock Returns. *Journal of Finance* 61(4), 1645–1680. https://doi.org/10.1111/j.1540-6261.2006.00885.x
- Baker, M. & Wurgler, J. (2007). Investor Sentiment in the Stock Market. *Journal of Economic Perspectives* 21(2), 129–151. https://doi.org/10.1257/jep.21.2.129

**注意力与显著性**
- Barber, B. M. & Odean, T. (2008). All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors. *Review of Financial Studies* 21(2), 785–818. https://doi.org/10.1093/rfs/hhm079
- Bordalo, P., Gennaioli, N. & Shleifer, A. (2012). Salience Theory of Choice Under Risk. *Quarterly Journal of Economics* 127(3), 1243–1285. https://doi.org/10.1093/qje/qjs018
- Bordalo, P., Gennaioli, N. & Shleifer, A. (2013). Salience and Asset Prices. *American Economic Review P&P* 103(3), 623–628. https://doi.org/10.1257/aer.103.3.623

**羊群行为**
- Banerjee, A. V. (1992). A Simple Model of Herd Behavior. *Quarterly Journal of Economics* 107(3), 797–817. https://doi.org/10.2307/2118364
- Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992). A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades. *Journal of Political Economy* 100(5), 992–1026. https://doi.org/10.1086/261849
- Chang, E. C., Cheng, J. W. & Khorana, A. (2000). An Examination of Herd Behavior in Equity Markets: An International Perspective. *Journal of Banking & Finance* 24(10), 1651–1679. https://doi.org/10.1016/S0378-4266(99)00096-5

**锚定与 52 周高点**
- George, T. J. & Hwang, C.-Y. (2004). The 52-Week High and Momentum Investing. *Journal of Finance* 59(5), 2145–2176. https://doi.org/10.1111/j.1540-6261.2004.00695.x

**套利限制与异象衰减**
- Shleifer, A. & Vishny, R. W. (1997). The Limits of Arbitrage. *Journal of Finance* 52(1), 35–55. https://doi.org/10.1111/j.1540-6261.1997.tb03807.x
- McLean, R. D. & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance* 71(1), 5–32. https://doi.org/10.1111/jofi.12365

**中国市场证据**
- Liu, J., Stambaugh, R. F. & Yuan, Y. (2019). Size and Value in China. *Journal of Financial Economics* 134(1), 48–69. https://doi.org/10.1016/j.jfineco.2019.03.008
- Carpenter, J. N., Lu, F. & Whitelaw, R. F. (2021). The Real Value of China's Stock Market. *Journal of Financial Economics* 139(3), 679–696. https://doi.org/10.1016/j.jfineco.2020.08.012
- Feng, L. & Seasholes, M. S. (2005). Do Investor Sophistication and Trading Experience Eliminate Behavioral Biases in Financial Markets? *Review of Finance* 9(3), 305–351. https://doi.org/10.1007/s10679-005-2262-0
- Nartea, G. V., Kong, D. & Wu, J. (2017). Do Extreme Returns Matter in Emerging Markets? Evidence from the Chinese Stock Market. *Journal of Banking & Finance* 76, 189–197. https://doi.org/10.1016/j.jbankfin.2016.12.008
- Yao, J., Ma, C. & He, W. P. (2014). Investor Herding Behaviour of Chinese Stock Market. *International Review of Economics & Finance* 29, 12–29. https://doi.org/10.1016/j.iref.2013.03.002
- Chen, T., Gao, Z., He, J., Jiang, W. & Xiong, W. (2019). Daily Price Limits and Destructive Market Behavior. *Journal of Econometrics* 208(1), 249–264. https://doi.org/10.1016/j.jeconom.2018.09.014

---

## 附：未采用的候选文献（诚实说明）

- **中文期刊（《金融研究》《管理世界》等）**：找不到可点开核实的原文链接，无法核实卷期与结论，故未引用；
  改用使用中国样本的英文论文（Feng & Seasholes 2005、Nartea et al. 2017、Liu et al. 2019）替代。
- **CSMAR / CNRDS 数据说明书**：非学术结论，不列入文献。
- **分析师预期分散度**：理论文献充分，但本项目无分析师数据，故降级为 `bf_disp` 代理并标注"代理非原变量"。
