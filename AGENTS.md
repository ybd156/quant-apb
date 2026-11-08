# AGENTS.md — 给 AI 协作者的工程契约

> 这份文件是**机器可读的项目说明书**。人类读者请看 `README.md`。
> 目标是：任何 AI 读完本文件，无需再读源码即可安全地修改、扩展、诊断本项目。
> 所有数值均为**实测值**，不是设计目标。

---

## 0. 一句话定义

本项目是一个**自学习参数体**：把 73 个「信号族 × 参数」当成一群专家，
每天用真实结果给专家打分，对的加权重、错的减权重，并按市场状态分层记忆。

**它不是**：涨跌预测器、印钞机、LLM 选股工具。
**它是**：一台测量仪 —— 回答"现在有没有优势、有多大、还能不能信、何时该停手"。

---

## 1. 30 秒上手

```bash
python -m qcore.pipeline --pool 300 --start 2013-01-01   # 全流程，约 45 秒
python -m qcore.live --once                              # 每日增量（秒级）
python -m webui.server --port 8765                       # 零依赖 Web UI
python -m qcore.claims score                             # 查看判断命中率
python tests/test_no_lookahead.py                        # 泄漏检验（必须 PASS）
```

依赖：**只有 numpy + pandas**。Python ≥ 3.10。禁止引入 scipy / sklearn / talib / requests。

---

## 2. 模块职责表

| 文件 | 职责 | 是否有状态 |
|---|---|---|
| `qcore/data.py` | A 股行情（东财，后复权）、股票池、缓存 | 磁盘缓存 |
| `qcore/sources.py` | 多源接入（东财/腾讯/新浪）+ 交叉验证 + 健康度 | `data/state/source_health.json` |
| `qcore/us_data.py` | 美股/港股行情、SPY/HSI 基准 | 磁盘缓存 |
| `qcore/info.py` | 多信息源快照（估值/基本面/板块/宽度/公告/新闻） | `data/info/` |
| `qcore/events.py` | 政策与大事件影响因子（事件研究） | `data/events/` |
| `qcore/signals.py` | 73 个参数化技术信号 | 无 |
| `qcore/dataset.py` | 3D 面板 `(T,N,S)` + regime + 点对点股票池 | 无 |
| `qcore/learner.py` | **自学习权重器（唯一有状态的核心）** | 可序列化 |
| `qcore/engine.py` | 逐日重放 + 预测台账 | 无 |
| `qcore/meta.py` | 二级学习：横截面 IC 显著性 → 决定是否反向 | 无 |
| `qcore/confidence.py` | 可信度校准（Brier / ECE / 可靠性曲线） | 无 |
| `qcore/stats.py` | HAC / Bootstrap / Deflated Sharpe / 漂移监控 | 无 |
| `qcore/analysis.py` | 优势体检（IC 显著性、多空、分年度） | 无 |
| `qcore/costs.py` | 分市场交易成本（A股/美股/港股） | 无 |
| `qcore/risk.py` | 风险分摊（ERC / 上限 / 组合配权） | 无 |
| `qcore/forecast.py` | 前景与空间测算（Grinold 基本法则） | 无 |
| `qcore/claims.py` | 判断台账（先登记后验证） | `data/claims/` |
| `qcore/paper.py` | 模拟盘 vs 实盘对账 | `data/state/live_trades.jsonl` |
| `qcore/context.py` | 当时形势 / 历史类比 | 无 |
| `qcore/store.py` | 状态落盘（原子写 + NaN→null） | `data/state/` |
| `qcore/pipeline.py` | 全流程编排 | 无 |
| `qcore/live.py` | 每日增量 / 常驻循环 | 读 weighter 状态 |
| `qcore/intraday.py` | 分钟级前向验证（无历史，不可回测） | `data/intraday/` |
| `webui/server.py` | 零依赖 HTTP API（**只做 IO，不做计算**） | 只读 |

---

## 3. 核心不变量（违反即 bug，不可协商）

```
INV-1 无未来函数
      在 t 日产生的任何分数/权重，只允许使用 ≤ t 日的信息。
      检验方式：build_dataset(cutoff=T) 重跑，T 日输出必须与全量运行逐位相同。
      已有测试：tests/test_no_lookahead.py（当前 worst diff = 0.000e+00）

INV-2 先预测后对账
      引擎循环顺序必须是：① 打分并封存 ② 用「已到期」样本对账 ③ 更新权重。
      到期判据：决策日 s 的样本，在 t ≥ s + 1 + h 时才可用（h = horizon）。

INV-3 统计量必须滚动
      禁止全样本 mean/std/quantile/min/max 参与信号或标准化。
      一律 rolling(w) 或 expanding()。检测：audit/leak_scan.py

INV-4 重叠收益必须用 HAC
      h 日收益重叠，朴素 t 值虚高约 √h。
      所有报告的 t 值必须用 stats.mean_tstat(x, lags=h-1)。

INV-5 后复权，且不是零风险
      数据源必须用 fqt=2（后复权）。fqt=1 是减法复权，会产生负价
      （实测 601088 最低 −9.55，异常 bar 639 个）。
      但后复权仍有约 0.03% 的非正价 bar（全在 2013 年前），
      每扩一次股票池必须重跑数据守卫。

INV-6 股票池点对点
      不能用"今日市值排名"筛历史股票（生存者偏差实测约 +22pp/年）。
      正确做法：Panel3D.UNIV 掩码 = 过去 60 日均成交额当日横截面排名前 n_univ。
      注意：当 N < n_univ 时该掩码恒真，等于没生效。

INV-7 台账只追加
      data/pred_ledger/*.jsonl 与 data/claims/*.jsonl 不允许修改历史记录，
      只允许追加结果字段。

INV-8 UI 不做计算
      webui/ 只读 data/state/*.json 并展示。禁止在 server.py 里算指标。
```

---

## 4. 数据结构契约

```python
Panel3D:
  dates   : np.ndarray (T,) datetime64
  codes   : list[str]   (N,)
  sids    : list[str]   (S,)               # 信号 ID，如 "rsi_14_th30"
  DIR     : np.ndarray (T,N,S) int8        # ∈ {-1,0,1}
  STR     : np.ndarray (T,N,S) uint8       # ∈ [0,255]，p3d.str_float(t) 还原为 [0,1]
  OPEN    : np.ndarray (T,N) float32
  CLOSE   : np.ndarray (T,N) float32
  RET     : dict[int, np.ndarray (T,N)]    # r_h(t)=OPEN[t+1+h]/OPEN[t+1]-1，停牌缺口处为 NaN
  SIGMA   : np.ndarray (T,N) float32       # 20 日收益率标准差（相对量，非 ATR 绝对值）
  REGIME  : np.ndarray (T,) int8           # 0=跌低波 1=跌高波 2=涨低波 3=涨高波
  VALID   : np.ndarray (T,N) bool          # 可交易 = traded & UNIV
  UNIV    : np.ndarray (T,N) bool          # 点对点股票池

engine.run_walkforward() 返回（对外统一 (N,T) 朝向）:
  scores        : (N,T) float32            # 综合分 ∈ [-1,1]
  scores_TN     : (T,N)                    # 引擎内部朝向，供 calibrate 使用
  concord       : (N,T) float32            # 信号一致度 [0,1]
  quality       : (N,T) float32            # 数据完整度 [0,1]
  weights_hist  : (T,S) float32            # 每日全局权重
  RET / OPEN / CLOSE / SIGMA / VALID : (N,T)
  ledger        : DataFrame                # 预测台账
```

**朝向陷阱**：`scores` 对外是 `(N,T)`，内部是 `(T,N)`。
凡是拿 `p3d.RET[h]`（`(T,N)`）与 `res["scores"]`（`(N,T)`）做运算，必须先转置。
这是本项目历史上最高频的 bug（已出现 3 次）。

---

## 5. 关键公式

```
标准化收益     y = clip(dir·r_h/(σ·√h), ±3)          # clip_C=3 是 Hedge 界成立的前提
命中计数       a ← ρ·a + 1[y>0],  b ← ρ·b + 1[y<0]   # 仅发声时更新（sleeping experts）
贝叶斯命中率   R = (a + κ·p0)/(a + b + κ)             # κ=8 等效先验样本数
收益均值       E ← ρ·E + (1-ρ)·y
乘性权重       logw += η·y                            # Hedge/MWU
合成得分       z = β_R(2R-1) + β_E·E + β_H·logw
regime 混合    γ = γ_max·n/(n+γ_n0);  z = (1-γ)z_g + γz_r
最终权重       w = (1-ε)softmax(τz) + ε/S  →  max(w,w_min)  →  归一
漂移检测       PH: S += (y - m - δ);  (S - minS) > λ ⇒ 记忆 ×0.35

有效性口径     IR ≈ IC·√BR
有效广度       BR_eff = (252/h)·N/(1+(N-1)ρ)          # ρ 为持仓间平均相关，默认 0.30
有效独立数     N_eff = (Σw)²/(wᵀΣw)                    # Σ 为相关矩阵；比 1/HHI 诚实
```

**默认参数**：`h=5, ρ=0.995, κ=8, η=0.02, τ=1.5, ε=0.03, w_min=0.005, γ_max=0.6, γ_n0=60, clip_C=3`

---

## 6. 如何扩展

### 加一个信号
1. 在 `qcore/signals.py` 的 `signal_zoo()` 里追加 `SignalSpec(sid, family, kind, params, note)`。
2. 在 `compute_signals` 里实现，**必须**：
   - 严格因果（第 t 行只用 ≤t 行）
   - 预热期输出 0，不输出 NaN
   - 只允许相对量/滚动量，**禁止绝对价格阈值**（后复权价格与现价差 10 倍以上）
   - 用 `qcore/signals.py` 已有的前缀累加工具，不要 `apply/iterrows`
3. 跑 `python -m tests.test_signals`（含截断一致性断言，必须逐位相等）。

### 加一个市场
1. 在 `qcore/us_data.py` 模式上实现 `load_*_kline`，列名必须与 `qcore/data.py` 完全一致。
2. 在 `qcore/costs.py` 的 `MARKETS` 里加 `CostSpec`，填 `verified_on`。
3. 基准指数要能取到（作为主日历）。
4. 注意：不同市场的复权口径、涨跌停制度、停牌规则都不同。

### 加一个信息源
1. 在 `qcore/info.py` 加采集函数，返回 `InfoItem(as_of, source, kind, payload, reliability, pit_backfillable)`。
2. `reliability` 不许一律 1.0；新闻类必须 `pit_backfillable=False`（不能回测）。
3. 单源失败只能进 `missing`，不许抛异常中断主流程。

---

## 7. 已知陷阱（按踩坑频率排序）

| # | 陷阱 | 症状 | 检测方法 |
|---|---|---|---|
| 1 | `(N,T)` vs `(T,N)` 朝向混用 | `broadcast` 报错，或静默算出垃圾 IC | 看 shape 是否匹配；`analysis._align` 会转置 |
| 2 | 前复权产生负价 | 收益率 ±10000% | 检查 `close.min() > 0`；用 fqt=2 |
| 3 | 重叠收益朴素 t 检验 | t 值虚高 √h 倍 | 必须 `lags=h-1` |
| 4 | 生存者偏差伪装成 alpha | 净值漂亮但 IC≈0 | 相对"当日池等权"算超额，不是相对沪深300 |
| 5 | 停牌复牌跳变 | 单日 +394%/+622% 被 clip 成满格喂进权重 | `RET` 已在停牌缺口处置 NaN |
| 6 | 东财 clist 单页上限 100 | 股票池退化成超级大盘股 | `fetch_universe` 已分页 |
| 7 | 东财后复权与腾讯/新浪有 13bp 系统性差异 | 多源交叉验证报 basis_mismatch | `sources.compare_returns` 三档判定 |
| 8 | 港股双边 0.1% 印花税 | 5 日调仓年化拖累 13.5% | `costs.annual_cost_drag` |
| 9 | 信号同质性 | "73 个专家"实际只有 33 个独立方向 | `risk.effective_n_corr` |
| 10 | NaN 写进 JSON | 浏览器 `JSON.parse` 整页崩 | `store.write_json` 已全局转 null |

---

## 8. 诊断路径

```
症状：回测很好看但 IC≈0
  → 查生存者偏差（相对池等权 vs 相对沪深300）
  → 查是否用了重叠样本朴素 t 值
  → 查 decile 单调性是否来自 beta 而非 alpha（做多空、看市场中性后是否还在）

症状：增量运行与批量运行结果不一致
  → 查信号是否在完整历史上计算（build_panel clip=False）
  → 查 weighter 状态是否被重复更新（from_t 起点）

症状：Web UI 打不开或空白
  → 查 data/state/*.json 是否存在（缺文件会显示"未就绪"而非崩）
  → 查是否有裸 NaN（server 已兜底，但根因在写出端）

症状：多源报 inconsistent
  → 先看 suspected_adjustment_days 是否落在分红除权日（那是口径差异不是错误）
  → 再看 median_abs_diff 是否 < 2e-3（是则降级为 basis_mismatch 仍可用）
```

---

## 9. 当前实测基线（2026-11，283 只 / 2013-2026 / h=5）

| 指标 | 数值 | 含义 |
|---|---|---|
| 横截面 Rank IC | **−0.0185** | 方向系统性做反 |
| HAC t | **−2.48** | 95% CI 全在 0 以下 |
| 原始命中率 | 48.4% | 低于抛硬币 |
| Meta 反向命中率 | 50.7% | 纠错后略高于 50%，但 t 未达显著 |
| 多空年化（扣 19.3bps） | +4.5%，t=0.59 | 不显著 |
| Deflated Sharpe | 0.008 | 未通过多重检验 |
| 学习器 vs 等权 | 无显著差异 | **学习器当前无效** |
| 有效独立信号（相关性调整） | 12.1 / 73 | 信号高度同质 |
| 成本拖累（A股/5日/70%换手） | 6.8%/年 | 港股同口径 13.5%/年 |

**结论**：当前信号集在 A 股大盘股 5 日尺度上**没有可用 alpha**，且略呈系统性反向。
任何声称"改进"的提交，必须给出 HAC t 值与 Deflated Sharpe 的前后对比，否则不予采信。

---

## 10. 修改本项目时的硬性要求

1. 改完必须跑：`tests/test_no_lookahead.py`、`tests/test_signals.py`、`audit/leak_scan.py`。
2. 任何性能改动必须给出实测数字（本项目目标：N100 级小主机，全流程 < 60s，增量 < 15s）。
3. 任何"提升效果"的改动必须同时给出**样本外**证据，且说明是否通过多重检验。
4. 不允许为了让数字好看而调整评估口径。评估口径是本项目的核心资产。
5. 新增判断（结论）必须通过 `qcore.claims.register` 登记，写明证伪条件与复核日期。
