# 量化预知 · APB（Adaptive Parameter Body）

> **一句话定位**：把 80 个技术信号当成 80 位会说话的专家，每个交易日用真实结果给它们打分，
> **让权重随市场状态自适应**——纯 Python + numpy/pandas，约 300 行核心逻辑，全流程可回放、可解释。

---

## 它解决什么问题

**不是「预测涨跌」。** 单看任何一个技术指标，胜率都在抛硬币附近；指望找到一个"神指标"是这条路上最常见的死法。

它解决的是另一个问题：**你手上有一堆都不怎么样的信号，怎么决定今天该多信谁一点。**

- 动量在趋势市有效、在震荡市连续打脸；RSI 超卖在急跌里是接飞刀，在横盘里是好买点。
- 手工给每个信号配一个固定权重，等于假设"市场结构永远不变"——这个假设每年都在被证伪。
- 于是本项目把**参数本身变成学出来的**：对了加权重、错了减权重、失效了快速减配、样本少时不敢乱信。

学到的状态总量只有约 `S + R×S + 超参` ≈ 400 个数，全部可打印、可人工审阅、可逐日回放。

---

## 架构

```
        ┌────────────────────────────────────────────────────────────────────────┐
        │  qcore/data.py        行情层：东财直连 → 本地 CSV 缓存                 │
        │                       后复权 fqt=2（乘法复权，价格恒正）               │
        │                       股票池 = 当日成交额/换手率动态取池（非今日市值） │
        └───────────────────────────────┬────────────────────────────────────────┘
                                        ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │  qcore/signals.py     信号库：50–90 个 (family × params)               │
        │                       ma_cross / macd / rsi / kdj / boll / vol_break   │
        │                       mom / gap / pattern / rs / dist_ma / atr_stop    │
        │                       输出 DIR∈{−1,0,+1}、STR∈[0,1]，严格因果          │
        └───────────────────────────────┬────────────────────────────────────────┘
                                        ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │  qcore/dataset.py     3D 面板 (N股票 × T日 × S信号)                    │
        │                       + REGIME(4档，只用≤t信息) + RET{h} + VALID       │
        └───────────────────────────────┬────────────────────────────────────────┘
                                        ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │  qcore/engine.py      逐日重放（时间机器）                             │
        │   ① 用 ≤t 的信息算综合分 score                                         │
        │   ② 预测写进台账并封存（只追加，历史不得回改）                         │
        │   ③ 对账：t 日揭晓 t−1−h 日预测的结果（天然 h+1 天 purge）             │
        │   ④ 学习器加深/减弱权重 → 进入 t+1                                     │
        └───────────────────────────────┬────────────────────────────────────────┘
                                        ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │  qcore/learner.py     ★ 唯一有状态层（本项目的"参数体"）               │
        │    Hedge 乘性更新  +  Beta-Bernoulli 收缩  +  指数遗忘                 │
        │    +  Page-Hinkley 漂移检测  +  regime 分层收缩                        │
        └───────────────────────────────┬────────────────────────────────────────┘
                                        ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │  qcore/metrics.py / backtest.py / report.py                            │
        │    走样外：方向命中率、Rank IC/ICIR、5 分层、多空组合、基线对比        │
        │    输出：reports/*.md（含"缺陷与局限"一节，不许省略）                  │
        └────────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux 用 .venv/bin/pip
python -m qcore.live --once                                            # 首次会拉全量历史，耗时较长
python -m webui.server --port 8765                                     # 浏览器开 http://127.0.0.1:8765
```

纯离线跑一遍回测（不需要网页，也不落台账）：

```python
from qcore.data import get_universe
from qcore.dataset import build_dataset
from qcore.engine import run_walkforward, RunConfig
from qcore.metrics import evaluate_all
from qcore.backtest import run_long_short, run_baselines
from qcore.report import render

uni  = get_universe(top_n=80)
p3d  = build_dataset(list(uni["code"]), start="2013-01-01", horizon=5)
res  = run_walkforward(p3d, RunConfig(horizon=5))
eq   = run_walkforward(p3d, RunConfig(horizon=5, mode="equal"))   # 必须有的基线
m    = evaluate_all(res, horizons=(1, 5, 20))
bt   = {"long_short": run_long_short(res, 5, 5, 10),
        "baselines": run_baselines(res, eq, 5, 10)}
render(res, m, bt, "reports/report.md")
```

> **硬性基线**：若 `learned` 跑不赢 `equal`（等权集成），说明学习器无效——必须如实写进报告，
> 不许换指标掩盖。`mode="placebo"` 会把信号方向打乱重跑，OOS 指标必须回落到 ~0，否则是评估代码泄漏。

---

## 目录结构

```
qcore/
  data.py      行情获取与缓存（东财直连，后复权，动态取池）
  signals.py   参数化信号库（列名即公式，严格因果，预热期输出 0）
  dataset.py   3D 面板 + regime 分档 + 前向收益
  learner.py   ★ 自学习权重器（Hedge + Beta 收缩 + 遗忘 + PH 漂移）
  engine.py    逐日重放引擎 + 预测台账
  metrics.py   走样外指标（命中率 / Rank IC / ICIR / 分层 / 权重集中度）
  backtest.py  多空组合与基线对比（T+1 开盘成交，双边成本）
  report.py    Markdown 报告渲染（无 matplotlib 依赖）
  live.py      每日增量入口（`--once`，定时任务调它）
webui/         本地网页服务（默认只监听 127.0.0.1）
tests/         截断不变性、学习器性质、指标口径
audit/         静态泄漏扫描；参数敏感性、安慰剂检验按 SPEC §10 落地
docs/          01 调研 · 02 SPEC · 04 理论 · 05 部署
configs/       参数配置（魔法数字必须写这里，不许散落在代码里）
data/          落盘数据（raw/cache 可再生；pred_ledger、state 必须备份）
reports/       生成的报告
```

---

## 核心公式摘要

```
① 标准化收益   y   = clip(dir · r_h / (σ·√h), −3, +3)          σ = 该股 20 日收益率标准差
② 带遗忘计数   a,b ← ρ·a + 1[y>0], ρ·b + 1[y<0]   E ← ρ·E + (1−ρ)·y     ρ=0.995 ⇒ 记忆 1/(1−ρ)=200 次
③ 收缩命中率   R   = (a + κ·p₀) / (a + b + κ)                   κ=8 等效先验样本数，p₀=0.5
④ Hedge 更新   logw ← logw + η·y                                η=0.02；没发声的信号不更新
⑤ 复合与归一   z = β_R(2R−1)+β_E·E+β_H·logw → w = (1−ε)·softmax(τ·z)+ε/S → clip[w_min,1] → 归一
   漂移报警     (PH_t − min PH) > λ ⇒ 记忆 ×0.35 并清零累计量    Page-Hinkley, λ=8
```

三件事分开管：**Hedge 管累积对错，Beta 收缩管"样本少别乱信"，Page-Hinkley 管"失效了要快减"**。
完整推导、参数含义与文献见 [`docs/04_theory.md`](docs/04_theory.md)；系统契约见 [`docs/02_SPEC.md`](docs/02_SPEC.md)。

---

## 与现有项目的关系

| 项目 | 借鉴了什么 | 明确没借鉴 |
|---|---|---|
| **Qlib**（Rolling / DDG-DA） | `trunc_days = horizon+1` 的截断思想（purge 的最小实现）；权重以 `date × signal` 的 DataFrame 落地；价量因子除以 close 做无量纲化 | torch + LightGBM + MLflow 的重栈；元模型输出的黑箱权重；`online_srv` 的状态机（源码自注未正确实现）；`.bin` 数据格式 |
| **FreqAI**（freqtrade） | 时间指数衰减的样本权重 `exp(−i/(αn))`（本项目用等价的几何遗忘 ρ）；DI 外推预警的"我不知道"表达思路 | 万级特征自动膨胀；SVM/DBSCAN 在日线小样本上剔离群；每标的一个独立模型 |
| **river** | ADWIN 的参数语义与"只监控一维序列"的做法；漂移触发"换模型/换权重"的范式 | Rust 编译核心（`river._river_rust`）——本项目用 20 行 numpy 重写 Page-Hinkley，零编译依赖 |
| **FinRL** | turbulence 熔断作为"市场状态"最便宜的代理 | 绝对金额奖励、不真实撮合、不可解释的 RL 策略 |
| **alphalens-reloaded** | IC = Spearman(factor, fwd_ret)、ICIR、分位单调性、换手率的报表口径 | 作为运行时依赖（只抄口径，自己实现 200 行） |
| **TA-Lib / pandas-ta / stockstats** | stockstats 的"列名即公式、惰性计算、纯 pandas"范式 | C 库编译依赖；beta 版本与维护风险（pandas-ta 仍是 0.4.71b0） |
| **vectorbt / backtrader** | 成交假设清单：T+1 开盘成交、双边成本、涨跌停不可成交、停牌跳过 | numba 编译栈与事件驱动的性能开销 |

一句话总结取舍：**所有成熟项目的"自适应"都发生在权重层，不在模型层。我们只实现这一层，因为它是唯一在小样本下还能诚实评估的那一层。**

---

## ⚠️ 风险声明（请先读这一段）

1. **本项目是研究工具，不是投资建议，也不构成任何形式的荐股。** 代码里没有任何"买入信号"API，
   输出的 `score` 只是历史口径下的相对强弱打分。
2. **历史回测不代表未来表现。** 回测能告诉你的只有"这套逻辑在过去那段数据上没有明显崩坏"，
   它对明天的约束力接近零。任何把回测曲线当预期的用法都是误用。
3. **技术信号的边际预测力极小。** 横截面 IC 通常在 0.02–0.05 量级，扣掉双边成本与冲击成本后
   很容易归零甚至为负。**不要期待圣杯**，也不要把本项目的输出当成"胜率"来宣传。
4. **过拟合是默认结局。** 网格搜索必然挑出历史上最漂亮的参数组合。项目内置了截断不变性检验、
   安慰剂检验、参数敏感性扫描与"必须跑赢等权基线"的硬性要求，但它们只能降低风险，不能消除。
5. **A 股特有风险**：涨跌停不可成交、停牌、T+1、注册制与退市规则变化、政策市导致的 regime 突变，
   都可能让任何历史规律在一夜之间失效。本项目用后复权价与动态取池部分缓解，但没有消除。
6. **数据源是公开接口**，可能延迟、缺失、口径变更。生产使用前必须自己校验。
7. **请自行承担全部风险。** 若你把任何输出用于真实交易，亏损与合规责任完全由你承担。

---

## License

MIT —— 见 [`LICENSE`](LICENSE)。Copyright (c) 2026 ybd156.
