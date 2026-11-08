# SPEC — 自学习参数体（Adaptive Parameter Body, APB）

> 版本 v1.0 · 目标：一套**纯 Python + numpy/pandas**、可解释、严格走样外验证的 A 股日线自学习系统。
> 核心机制：把「信号 × 参数」当成一群专家，每天用真实结果给专家打分，**对的加权重、错的减权重**，
> 并让权重随市场状态（regime）分层，从而参数本身是"学出来的"而不是"拍出来的"。

---

## 0. 铁律（所有模块必须遵守）

### 0.0 第一铁律：假装未来没发生（Point-in-Time Replay）

> 系统**不是**用全样本数据反推历史，而是像真的活在当天一样：
> 站在 t 日，只拿得到 t 日及以前的数据 → 做分析 → **先把预测写进台账并"封存"** →
> 时间推进到 t+h，真实结果揭晓 → 用揭晓的结果判定对错 → 才对权重做加深/减弱。

具体到工程，必须做到：

1. **先出预测，后看结果**。引擎逐日推进，t 日的预测记录在任何 t 日之后的数据被读取之前写入
   `data/pred_ledger/YYYY.parquet`（字段见 §9）。台账是**只追加**的，历史记录不得回改。
2. **一切统计量必须滚动/扩张计算**。禁止对全样本做 z-score、分位数、min/max、标准化、
   regime 阈值标定——这类"全样本口径"是隐性未来函数。统一改为
   `expanding()` / `rolling(window).quantile()`，窗口只向过去看。
3. **股票池必须点对点**。今天的市值排名不能拿来筛十年前的可交易股票（生存者偏差）。
   改为：在候选大池内，按**当日**成交额/换手率阈值动态取池（当日可观测）。
4. **信号预热期不许用未来数据补齐**；不足则输出 0（中性），不参与当日投票。
5. **可证伪检验：截断不变性（Truncation Invariance）**
   —— 把数据集截断到 T 日之后重跑全流程，T 日输出的 `score / weights / regime`
   必须与全量运行**逐元素完全一致**（容差 1e-12）。任何差异即证明存在泄漏，构建失败。
   此检验必须做成 `tests/test_no_lookahead.py`，并纳入 CI 与最终报告。

### 0.1 其余铁律


1. **无未来函数**：任何在 t 日产生的信号/权重，只允许使用 ≤ t 日的信息。
2. **执行滞后**：t 日收盘后得到信号 → **t+1 开盘成交**。收益口径 `r_h(t) = open(t+1+h)/open(t+1) - 1`。
3. **权重更新只用已兑现结果**：更新 t 日权重时，只能使用**已到期**（exit_date ≤ t-1）的样本，
   即 embargo/purge 窗口 = h 个交易日。禁止用未到期样本。
4. **参数不手调**：任何"魔法数字"必须写进 `CONFIG` 并在报告里列明，且做敏感性分析。
5. **可复现**：固定随机种子；数据落盘缓存；报告附数据区间与股票池清单。

---

## 1. 数据契约（`qcore/data.py`，已完成）

```python
from qcore.data import get_universe, build_panel, load_index

uni   = get_universe(top_n=80)          # DataFrame: code,name,mktcap,amount,industry
panel = build_panel(codes, start="2010-01-01")   # dict[str, DataFrame]
ix    = load_index("000300")            # 基准指数日线
```

每只股票的 DataFrame 列（升序、无重复日期、已剔除停牌与零成交）：

| 列 | 含义 | 备注 |
|---|---|---|
| date | 交易日 | datetime64 |
| open/high/low/close | 前复权价 | float |
| volume | 成交量（手） | float |
| amount | 成交额（元） | float |
| amplitude | 振幅 % | 接口直给 |
| pct_chg | 涨跌幅 % | 接口直给 |
| change | 涨跌额 | |
| turnover | 换手率 % | |
| code | 代码 | str |

---

## 2. 信号契约（`qcore/signals.py`）

### 2.1 数据结构

```python
@dataclass(frozen=True)
class SignalSpec:
    sid:    str    # 全局唯一，格式 "<family>_<paramhash>"，如 "rsi_14_th30"
    family: str    # rsi / macd / boll / ma / kdj / vol / atr / pattern / cross / mom ...
    kind:   str    # trend|reversal|momentum|volume|volatility|pattern|cross
    params: dict   # 该信号的全部参数（用于报告可解释）
    note:   str    # 一句话中文说明（"RSI14 跌破 30 视为超卖反转"）

def signal_zoo(cfg: dict | None = None) -> list[SignalSpec]:
    """返回全部 (family, params) 组合，默认 50-90 个。参数网格即"可学习的参数空间"。"""

def compute_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    输入单只股票日线（含 date），输出：
      DIR: DataFrame  index=date  columns=sid  取值 {-1,0,+1}（float，NaN 视为 0）
      STR: DataFrame  index=date  columns=sid  取值 [0,1]   信号强度
    约束：
      - 严格因果：第 t 行只依赖第 ≤t 行；
      - 指标预热期（如 MA250）不足时输出 0（不是 NaN 传播）；
      - 单只股票 3000 根 K 线 × 80 信号计算时间 < 1.5s；
      - 不得抛异常（数据异常时降级为 0）。
    """
```

### 2.2 信号族最低清单（每族至少 2 组参数变体）

| family | 说明 | 方向语义 |
|---|---|---|
| ma_cross | 双均线金叉/死叉（多组快慢参数） | +1 金叉 |
| ma_slope | 均线斜率/多头排列 | +1 向上 |
| macd | DIF/DEA 金叉死叉、零轴穿越、柱状放大 | +1 金叉 |
| rsi | RSI 超卖反转 / 超买回落 / 50 中轴突破 | +1 超卖 |
| kdj | K/D 金叉、J 值极值 | +1 金叉 |
| boll | 下轨反弹、上轨突破、带宽收口 | 见 note |
| vol_break | 放量突破 N 日高点（量比阈值） | +1 |
| vol_shrink | 缩量回调到支撑（趋势内） | +1 |
| volume_anom | 量能异动（量比/换手分位） | 按价格方向 |
| atr_stop | ATR 通道突破 / 波动压缩 | +1 向上突破 |
| mom | N 日动量 / 反转（1/5/20/60 日） | +1 强动量（或反转，注明） |
| gap | 跳空缺口方向 | +1 向上跳空 |
| pattern | 锤子线、吞没、十字星、三连阴后大阳 | +1 看多形态 |
| rs | 相对指数强弱（超额动量） | +1 强于指数 |
| dist_ma | 偏离均线程度（乖离率极值） | 均值回复语义 |

### 2.3 强度 STR 定义

统一口径，便于跨信号比较：
- 连续型：`STR = clip(|z|/2, 0, 1)`，z 为可比的标准化强度（如 RSI 距 50 的距离 / 20）；
- 布尔型：`STR = 1.0`（信号成立）或按量能/幅度分级 0.5/1.0。

---

## 3. 数据集契约（`qcore/dataset.py`）

```python
@dataclass
class Panel3D:
    dates:  np.ndarray      # (T,) datetime64，主日历（基准指数交易日）
    codes:  list[str]       # (N,)
    sids:   list[str]       # (S,)
    DIR:    np.ndarray      # (N,T,S) float32  {-1,0,1}
    STR:    np.ndarray      # (N,T,S) float32  [0,1]
    OPEN:   np.ndarray      # (N,T)   float32  开盘价
    CLOSE:  np.ndarray      # (N,T)
    RET:    dict[int,np.ndarray]  # h -> (N,T) float32，r_h(t)=open(t+1+h)/open(t+1)-1
    REGIME: np.ndarray      # (T,) int8，市场状态 0..3
    VALID:  np.ndarray      # (N,T) bool，该股当日是否可交易（有价格、非停牌）
```

`REGIME` 由基准指数计算（只用 ≤t 信息）：
- 用 60 日均线方向 + 20 日已实现波动分位 + 指数 60 日动量，切成 4 档：
  `0=下跌低波, 1=下跌高波, 2=上涨低波, 3=上涨高波`（也允许 3 档，配置可选）。

---

## 4. 学习器契约（`qcore/learner.py`）

```python
class AdaptiveWeighter:
    def __init__(self, n_signals: int, n_regimes: int = 4, cfg: WeightConfig = ...): ...
    # 状态
    #   logw:      (R, S) float64   每 regime 的专家权重（对数域）
    #   logw_glob: (S,)   float64   全局权重
    #   hits/miss: (R, S) float64   衰减后的命中/未命中计数
    #   payoff:    (R, S) float64   衰减后的标准化收益累计
    #   n_seen:    (R, S) int       有效更新次数

    def weights(self, regime: int) -> np.ndarray:
        """返回 (S,) 权重，已混合 regime 表与全局表并归一化，含下限/探索项。"""

    def update(self, regime: int, active: np.ndarray, payoff: np.ndarray) -> None:
        """active: (S,) bool 当日该信号是否发声；payoff: (S,) 标准化收益（未发声处忽略）。"""
```

### 4.1 更新公式（必须按此实现，参数进 CONFIG）

对每个**发声**的信号 s：

```
1) 标准化收益   y_s = clip(dir_s * r / (sigma_s * sqrt(h)), -C, +C)      # sigma_s=该股 20 日收益率标准差（相对量，非 ATR 绝对值）
2) 命中计数（带遗忘）  a_s ← ρ·a_s + 1[y_s>0] ,  b_s ← ρ·b_s + 1[y_s<0]
3) 贝叶斯收缩命中率     R_s = (a_s + κ·p0) / (a_s + b_s + κ)             # p0=0.5
4) 收益均值（带遗忘）   E_s ← ρ·E_s + (1-ρ)·y_s
5) 乘性权重（Hedge）    logw_s ← logw_s + η · y_s
6) 复合得分             z_s = β_R·(2R_s-1) + β_E·E_s + β_H·logw_s
```

最终权重：

```
w_s = exp(τ · z_s) · 1[z_s 有效]
w_s ← (1-ε)·softmax(w_s) + ε/S          # ε=探索率，防止饿死；S=信号总数（不是当日发声数）
w_s ← clip(w_s, w_min, 1) 后重新归一化
regime 混合： z = (1-γ)·z_global + γ·z_regime    # γ 随 n_seen 增长：γ = γ_max·n/(n+n0)
```

### 4.2 更新时机（关键，防泄漏）

在 t 日更新时，只喂入 **exit_date ≤ t-1** 的样本。实现上用环形缓冲区：
每日把新样本压入 `queue[t+h]`，到 t 日只弹出 `queue[t]`（其 exit 已过）。

### 4.3 必测性质（`tests/test_learner.py`）

- 合成数据：信号 A 有 65% 真实胜率、信号 B 50%（噪声）→ 300 次更新后 `w_A > 3·w_B`；
- 信号 A 前 200 次有效、之后完全失效 → 权重必须在 ~100 次内衰减到噪声水平（遗忘生效）；
- 权重恒在 (0,1]、Σw=1、无 NaN；
- 未发声信号权重不被惩罚（sleeping expert）；
- 用同一随机种子两次运行结果完全一致。

---

## 5. 引擎契约（`qcore/engine.py`）

```python
def run_walkforward(p3d: Panel3D, cfg: RunConfig) -> dict:
    """
    逐日循环 t = warmup .. T-1-h：
      1. w = weighter.weights(REGIME[t])
      2. score[n] = Σ_s w_s·DIR[n,t,s]·STR[n,t,s] / Σ_s w_s·1[DIR≠0]
      3. 记录 score 与次日开盘可执行仓位
      4. 用 t 日到期样本更新权重
    返回：{'scores': (N,T), 'weights_hist': (T,S), 'weights_regime': (T,R,S), 'meta': {...}}
    """
```

输出必须包含：
- `scores` 每日每股综合分（[-1,1]）；
- `weights_hist` 每日全局权重矩阵（用于画"谁在变强/变弱"）；
- 分 regime 权重快照；
- 每条信号的首末有效日期与累计贡献（用于可解释性）。

---

## 6. 评估契约（`qcore/metrics.py` / `qcore/backtest.py`）

必须报告的指标（**全部为走样外**）：

| 指标 | 口径 |
|---|---|
| 方向命中率 | sign(score) == sign(r_h) 的比例，按 h=1/5/20 分别统计 |
| Rank IC | 每日横截面 Spearman(score, r_h)，报告 mean/std/ICIR/胜率 |
| 多空组合 | 每日按 score 分 5 组等权，T+1 开盘建仓，持有 h 日，双边成本 bps 可配 |
| 超额收益 | 多空组合 vs 等权基准、vs 买入持有 |
| 权重集中度 | Herfindahl 指数、Top5 权重占比随时间 |
| 稳定性 | 分年度 IC / 命中率；股票池内外（in-sample vs OOS 股票）对比 |

**基线（必须有）**：等权集成（所有信号权重相同）、单一最优信号（事后选，作为上限参照）、
随机打分、全市场买入持有。**若集成跑不赢等权集成，则说明学习器无效**——必须如实报告。

---

## 7. 目录结构

```
qcore/
  data.py        # 数据层（多源、后复权、缓存）
  signals.py     # 信号库（73 个参数化变体）
  dataset.py     # 3D 面板 + regime
  learner.py     # 自学习权重器
  engine.py      # 走样外引擎
  metrics.py     # 指标（IC/命中率/分层收益）
  backtest.py    # 组合回测
  report.py      # 报告生成
tests/           # 单测（各自负责）
docs/            # 调研与设计文档
configs/         # 参数配置
data/            # 缓存（不入版本）
```

---

## 9. 预测台账（Prediction Ledger）—— "先封存，后对账"

引擎每交易日追加一行/一批记录，字段：

| 字段 | 含义 |
|---|---|
| `as_of` | 决策日 t（只用 ≤t 数据） |
| `code` | 股票 |
| `score` | 综合分 [-1,1] |
| `action` | 多/空/观望（阈值化后） |
| `entry_date` | 实际建仓日 = t+1 |
| `entry_price` | t+1 开盘价（**决策时不可知，事后回填**） |
| `horizon` | 预测周期 h |
| `regime` | 当日市场状态 |
| `weights_hash` | 当日权重向量的哈希（证明权重是当时学的，非事后拟合） |
| `status` | `pending` / `scored` |
| `exit_date`, `exit_price`, `ret`, `hit` | 到期后回填 |
| `contrib_top5` | 当日贡献最大的 5 个信号（可解释性） |

对账流程：`status=pending` 且 `t+h` 已到 → 回填 `ret/hit` → 触发一次权重更新。
**台账不允许删除或修改已 scored 记录的预测字段**，只允许回填结果字段。

---

## 10. 审计与反过拟合（`audit/`，独立第三人负责）

1. `tests/test_no_lookahead.py` —— 截断不变性（见 0.0.5），对 3 个随机日期 × 3 只股票验证。
2. `tests/test_learner.py` —— 学习器性质（见 4.3）。
3. `audit/leak_scan.py` —— 静态扫描：全样本 `mean()/std()/quantile()/min()/max()` 未带滚动窗口、
   `shift(-n)`、`iloc[i+k]`、`bfill`、`fillna(method='bfill')`、未来日期 join 等模式，输出可疑行。
4. `audit/sensitivity.py` —— 关键参数（h, ρ, κ, η, τ, ε）±50% 网格，看指标是否悬崖式崩塌；
   若某组参数是"孤峰"，判定为过拟合。
5. `audit/placebo.py` —— 安慰剂检验：把信号方向随机打乱后重跑，OOS 指标必须回落到 ~0；
   若不回落，说明评估代码本身有泄漏。
6. 最终报告必须含"缺陷与局限"一节，如实写明：生存者偏差程度、交易成本假设、
   信号同质性、样本期覆盖的牛熊结构。



```yaml
horizon: 5              # 主预测周期（交易日）
exec_lag: 1             # T+1 开盘成交
decay_rho: 0.995        # 遗忘因子（半衰期 ~138 次更新）
kappa: 8.0              # 贝叶斯收缩强度（等效先验样本数）
eta: 0.02               # Hedge 学习率
beta_R: 1.0             # 命中率项系数
beta_E: 0.5             # 收益均值项系数
beta_H: 1.0             # Hedge 累计项系数
tau: 1.5                # 温度（越大越集中）
eps_explore: 0.03       # 探索率
w_min: 0.005            # 权重下限
gamma_max: 0.6          # regime 表最大权重
gamma_n0: 60            # regime 收缩强度
clip_C: 3.0             # 单次收益标准化截断
cost_bps: 10            # 双边成本
```
