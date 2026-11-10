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
python -m qcore.pipeline --pool 300 --start 2013-01-01   # 全流程，本机实测 81.9–103.7s
python -m qcore.live --once                              # 每日增量（dataset 段已 16.9s，超预算）
python -m webui.server --port 8765                       # 零依赖 Web UI
python -m qcore.claims score                             # 查看判断命中率
python tests/test_no_lookahead.py                        # 泄漏检验（必须 PASS）
python -m tests.test_signals                             # 必须带 -m，否则 ModuleNotFoundError
python -m qcore.invariants                               # 8 条不变量自检
```

依赖：**运行时只有 numpy + pandas**（`pillow` 仅用于 `chart.py`/`brief.py` 出图）。
Python ≥ 3.10。禁止引入 scipy / sklearn / talib / requests。

> 耗时口径（2026-11-10 本机 Zen3/16 线程实测，**N100 上未测**）：
> 端到端 81.9–103.7s（两次同命令，含 impact/behavior/报告/导出）｜核心 18.2s｜
> 增量 dataset 段 16.9s（已超 15s 预算）｜信号层 300 只 ≈12–17s。
> **「全流程 <60s」当前未达标**，别再把 45s 当实测值引用。
>
> 增量耗时拆解（实测）：`build_panel` **4.1s** + `compute_signals` **12.2s**（43ms/只 × 283）
> ≈ 16.3s —— **信号层占 75%**。曾尝试「信号前只算最后 N 根」砍掉 3/4 计算，
> 但即使 warmup 给到 620 根，仍有 15/513k 单元格与全量计算不同（MA 族用前缀和，
> 序列变短后浮点舍入改变、在阈值边界翻转 DIR）→ **违反 INV-1 逐位相等，已否决**。
> 真要提速得先把 MA 族改成数值稳定的 `rolling().mean()`，或做逐股增量信号缓存。

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
| ~~`qcore/forecast.py`~~ | **已从核心表移除：实测零引用**（见 `docs/21_unwired_modules.md`） | — |
| `qcore/claims.py` | 判断台账（先登记后验证） | `data/claims/` |
| `qcore/paper.py` | 模拟盘 vs 实盘对账 | `data/state/live_trades.jsonl` |
| `qcore/slippage.py` | 滑点追踪：回测假设价 vs 实盘成交价，验证回测可信度 | `data/slippage/*.jsonl` |
| `qcore/alert.py` | 告警规则 + 1h 去重 + file/webhook 分发 | `data/state/alerts.jsonl`、`alert_state.json` |
| `qcore/context.py` | 当时形势 / 历史类比 | 无 |
| `qcore/store.py` | 状态落盘（原子写 + NaN→null）+ 台账幂等追加 | `data/state/`、`data/pred_ledger/` |
| `qcore/thresholds.py` | **样本量门槛唯一真源**（`MIN_TRADES=30` / `MIN_XS_NAMES=5`） | 无 |
| `qcore/invariants.py` | 8 条不变量的可执行自检（含负控） | `data/state/invariants.json` |
| `qcore/pipeline.py` | 全流程编排 | 无 |
| `qcore/live.py` | 每日增量 / 常驻循环 | 读 weighter 状态 |
| `qcore/intraday.py` | 分钟级前向验证（无历史，不可回测） | `data/intraday/` |
| `webui/server.py` | 零依赖 HTTP API（**只做 IO，不做计算**） | 只读 |
| `audit/leak_scan.py` | 静态泄漏扫描（HIGH/MEDIUM/INFO + 行内与文件级豁免） | 无 |
| `audit/gen_waivers.py` | 两层分流 MEDIUM：链路闭包 + 闭包内逐文件人判 | `audit/waivers.json` |
| `audit/basis_probe.py` | 复权方式逐股分类 + 压缩因子 + 秩重排 | `data/state/basis.json` |
| `audit/basis_ic.py` | 同一份 scores 换三种收益口径重算 Rank IC | `data/state/basis_ic.json` |
| `audit/pool_probe.py` | 生存者偏差：池快照 / UNIV 周转 / 掩码排除率 | 无 |

> **本表只覆盖核心链路。** `qcore/` 实际有 **120 个 .py**（约 4 万行）。
> 引用某个模块前先 `ls qcore/`，并对照 `docs/21_unwired_modules.md`：
>
> | 类别 | 模块数 | 行数 |
> |---|---|---|
> | 被 import（主链路） | 91 | 32,643 |
> | CLI 入口（有 `__main__`，不进自动流程） | 10 | 3,016 |
> | **零引用且无入口（死代码候选）** | **19** | **4,033** |
>
> 「不在表里」不等于「不存在」，也**不等于「已接线」**。

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

INV-7 台账只追加（且必须幂等）
      data/pred_ledger/*.jsonl 与 data/claims/*.jsonl 不允许修改历史记录，
      只允许追加结果字段。
      「只追加」不等于「随便追加」：append_ledger 以 (as_of, code, horizon) 为幂等键，
      重复写入直接跳过。实测教训：rebuild 每次重追加最近 90 天，47,676 行里
      只有 8,150 个唯一键（5.85×重复），同一决策日同一票挂着 3–9 个不同分数，
      台账因此无法回答「s 日对 x 股的预测是什么」。
      历史重复不可回改 → 读数用 store.compact_ledger() 生成的 `{year}.dedup.jsonl`。

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
| 11 | **共享状态无归属标识** | 美股账户用 A 股 picks 下单，买 1-3 股 A 股代码 | picks 必须带 `market`；`paper_auto.rebalance` 校验 |
| 12 | **多写入方覆盖同一文件** | 重算 impact 后 `live` 段消失，页面 2026 年变空白 | 写入前读-改-写，保留他方字段（`to_state_json` 已修） |
| 13 | **多实例共用文件名** | 训练 h=5 覆盖 h=1 的 `gbdt.pkl` | 文件名带实例标识：`gbdt_h5.pkl` / `paper_auto_cn_h5.json` |
| 14 | **脚本生成时转义字母** | `\t`/`\b` 变成控制字符，`can't find '__main__'` | 生成 bat/ps1 后断言：纯 ASCII + CRLF + 无 0x00-0x08 |
| 15 | **嵌套 JSON 只遍历顶层** | 数据源表显示 `as_of`/`sources` 而不是真实源 | UI 必须下钻到实际数据层 |
| 16 | **静默截断列表** | 31 个交易日只显示 1 天 | 不分页就明示总量；`slice(0,N)` 必须配提示 |
| 17 | **样本量不足就当结论** | 2 笔交易给出 +66bps，"最优阈值" | 任何"最优X"必须同时报样本数；< 30 标观察 |
| 18 | **只改写入方或只改读取方** | 接口改了但 UI 还读旧字段 | 共享字段必须两边同时改，见 §7.5 |
| 19 | **验证工具自己失灵** | 冒烟测试对报错页面报「全过」 | Windows 下子进程必须 `encoding="utf-8"`；先用已知失败样本验证检测器 |
| 20 | **const 先用后定义（TDZ）** | `Cannot access 'x' before initialization` | 语法合法、运行时炸。长函数里 const 集中到开头 |
| 21 | **插入代码不读上下文** | `r is not defined`（回调参数其实是 e） | 插入前读目标位置往上 5 行，确认变量名和作用域 |
| 22 | **`open(tmp,'w')` + `os.replace` 写 JSONL = 全量重写，不是追加** | `horizon_ledger._write_all` 在读失败时把整本台账写成 1 条 | 写入侧 `_read_strict` 读不全就**拒绝写**；`tests/test_store_ledger.py` |
| 23 | **把「收益均值」当「Sharpe」传给 PSR/DSR** | DSR 恒等于 `Φ(−E[max SR])`≈0.0078，h=1/5/20 三个策略给同一个数 | 传 `mean/std`（每期 Sharpe），n=期数不乘 h；`tests/test_dsr_units.py` |
| 24 | **`scores.shape[1]` 当股票数** | `status.json` 的 `n_stocks` 恒等于 `n_days`，UI 显示「2633 只」真值 112 | 用 `len(res["codes"])`；断言 `n_stocks != n_days` |
| 25 | **成本假设低于法定下限** | `account.json` 写 `cost_bps=10.0`，回测 19.28 / 模拟盘 10.0，同一次输出两个基准 | `account_config.COST_FLOOR_BPS`（CN 11.28 / HK 11.0）自动回落并留痕 |
| 26 | **豁免清单把自己的证据洗掉** | 生成器沿用上次的 waivers 扫描 → 已豁免文件扫出 0 条 → 新清单一路归零（246→207→0） | 扫描前先 `_FILE_WAIVERS = {}`；`gen_waivers --check` 进 CI |

---

## 7.5 共享状态契约（本版血的教训）

**本版 90% 的 bug 不是算法错，是接线错。** 算法部分有测试保护，
模块之间的约定没有 —— 只能靠这张表。

### 规则一：共享数据必须带归属标识

```
picks_hN.json     必须带 market 字段
paper_auto 账户   文件名带 market + horizon：paper_auto_{mkt}_h{h}.json
模型文件          带 horizon：gbdt_h{h}.pkl
成交记录          每条带 market + horizon
```

**接收方必须校验**，不能假设"给我的就是对的"：

```python
# paper_auto.rebalance 里的实际代码
_pm = str((picks or {}).get("market") or "").upper()
if _pm and _pm != market:
    return {"ok": False, "reason": "market_mismatch", ...}
# 没写 market 时用代码形态反查
if not _pm and rows[0]["code"].isdigit() and market != "CN":
    return {"ok": False, "reason": "picks_market_mismatch", ...}
```

### 规则二：同一文件多写入方 → 读-改-写

`data/state/events.json` 有两个写入方：

| 写入方 | 写什么 |
|---|---|
| `policy_calendar.to_state_json()` | `events` / `rows` / `impact` |
| `news_live.merge_into_events()` | `live` / `live_meta` |

两边都必须**先读旧文件再写**，不能整体覆盖：

```python
# to_state_json 里的实际代码
if _os.path.isfile(path):
    old = _json.load(f)
for k in ("live", "live_meta"):
    if old.get(k):
        payload[k] = old[k]
```

### 规则三：多实例的东西，文件名带实例标识

| 对象 | 命名 |
|---|---|
| 模型 | `gbdt.pkl`+`meta.json`+`current.json`(h1) / `gbdt_h5.pkl`+`meta_h5.json`+`current_h5.json`(h5) |
| 模型快照 | `versions/gbdt_h{h}_{ver}.pkl`（带周期，h1/h5 快照不再互相覆盖） |
| **权重器状态** | **`weighter_{mkt}_h{h}.npz`**（旧 `weighter.npz` 只读兼容；读时校验 cfg.horizon，不符即拒） |
| 模拟盘账户 | `paper_auto.json`(CN/h1) / `paper_auto_cn_h5.json` / `paper_auto_us_h1.json` |
| 预测 | `picks.json`(h1) / `picks_h5.json` |
| 主台账 | `data/pred_ledger/{year}.jsonl`（幂等键 `(as_of, code, horizon)`）+ 去重视图 `{year}.dedup.jsonl` |

### 规则四：生成脚本后的三条断言

```python
raw = open('train.bat','rb').read()
assert all(b < 0x80 for b in raw), "必须纯 ASCII"
assert b'\r\n' in raw, "必须 CRLF"
assert not any(b in raw for b in range(0x09)), "不能有控制字符"
```

`\t` 会变成 0x09、`\b` 变成 0x08 —— 这个坑已经踩了 3 次。

### 规则五：UI 展示的字段，必须有代码写它

`/api/events` 的 `impact` 字段 UI 读 5 次，但 `impact.compute()`
只在完整 pipeline 里跑，NAS 上从没跑过 → 表格永远空。

**新加 UI 字段时，同时确认「谁写它、谁读它、结构对不对」。**

### 规则六：报告"最优 X"必须带样本数

```python
MIN_TRADES = 30   # 单组最少交易次数，低于此标 insufficient
```

实测教训：阈值扫描里 t=0.65 的单边做多只有 **2 笔**样本，
给出 +66.89bps 的"最优"结论 —— 纯噪音。

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

症状：怀疑复权口径有问题（收益率幅度不对 / IC 能不能信）
  → python -m audit.basis_probe --codes 40
  → 看 adjust_kind_counts：multiplicative 为主 = fqt=2 是乘法复权（正常）
  → 再看 `python -m audit.basis_ic --codes 60`：同一份 scores 换收益口径重算 IC
  → 实测 ΔIC +1.0%（修正加法复权股票）/ +1.5%（改用 fqt=0）
    → 口径影响 ≤1.5%，**IC 的方向与量级都可引用**；不必做全池 A/B
  → 产物落在 data/state/basis.json 与 basis_ic.json

症状：怀疑生存者偏差 / 池子没在换人
  → python -m audit.pool_probe --codes 300
  → 看 exclusion_rate（掩码排除率，<2% 说明 N≈n_univ 点对点无效）
  → 看 jaccard_first_last（≈1 且未全选 = 池成员恒定）
  → 池构造层（今日市值 top-800 快照、ST/退仅个位数）本探针只能证明、不能消除

症状：leak_scan 报一大堆 MEDIUM，不知道该不该管
  → python -m audit.gen_waivers --check（清单是否与代码一致）
  → 规则是机械的：**不在打分/建模链路闭包内**的文件，全样本统计只影响事后指标
    → 文件级豁免（写进 audit/waivers.json，可重复生成、可复核）
  → 闭包内的文件**不豁免**，那部分才是需要逐条看的人工量
  → 当前分流：**MEDIUM 33（5 个核心/训练文件，故意保留）/ 文件级豁免 685**
  → 判定表见 docs/22_leak_waivers.md，机器可读版 audit/waivers.json

症状：多源报 inconsistent
  → 先看 suspected_adjustment_days 是否落在分红除权日（那是口径差异不是错误）
  → 再看 median_abs_diff 是否 < 2e-3（是则降级为 basis_mismatch 仍可用）
```

---

## 9. 当前实测基线

> **本节不再硬编码"唯一真相"。** 仓库里同时存在多套配置（283 只/2013 起 vs
> 112 只/2016 起），过去谁引用都不标配置，导致同一指标出现 3–4 个数字。
> **数字以产物为准，产物必须能指出是哪条命令、哪个配置、什么时间生成的。**

### 9.1 基线 A：283 只 / 2013-01-01 起 / h=5（研究口径，**默认引用这个**）

生成命令：`python -m qcore.pipeline --pool 300 --start 2013-01-01`
产物：`data/state/performance.json`（`generated_at` 字段即生成时间）
最近一次实测：**2026-11-10，端到端 81.9s / 103.7s（两次，机器负载不同）**
（T=3360 N=283 S=73；同机同命令，差 22s 说明「<60s」这条线对负载敏感）

| 指标 | 数值 | 说明 |
|---|---|---|
| 横截面 Rank IC (h=5) | **−0.01847** | HAC t = **−2.48**（lags=4，n=3335） |
| h=1 / h=20 的 IC | −0.00680 (t=−1.61) / −0.02680 (t=−1.93) | 三个周期同向 |
| 原始命中率 (h=5) | **48.49%**，n=328,559，n_eff=3,335 | 聚类 SE ±0.87pp，距 50% 仅 1.7 个 SE |
| Meta 反向命中率 | 50.72% | 同一产物 `meta.compare`，非"IC>0 占比" |
| 多空年化（**毛**、算术 `mean·252/h`） | +4.18%（h=5）/ +13.27%（h=1） | 未扣成本，**不可当净值读** |
| Deflated Sharpe（每期 Sharpe 口径） | **0.0313**（h=5）/ 0.2411（h=1）/ 0.0308（h=20） | 全部 `significant=false` |
| 学习器 vs 等权 | 无显著差异（配对 Δ HAC t ≈ −1.0） | **学习器当前无效** |
| 有效独立信号（相关性 `effective_n_corr`） | **12.13 / 73** | 权重口径 `1/HHI`=69 是另一个数，永不混用（部署态 112 只池为 10.40） |
| 成本拖累（A股/5日/70%换手） | 6.80%/年（19.28bps 往返，verified_on 2026-11） | 港股同口径 13.47%/年；`horizons` 原先用 250/0.8 算出 7.71%，已统一为 252/0.7 → `data/state/costs.json` |
| 核心耗时（build_dataset + walkforward） | 18.2s | 不含 impact/behavior/报告/导出 |
| 模拟盘年化（CN/h=5，修正后） | **−3.49%/年**（修正前 +4.33%） | 成本口径从 10.0bps 改回 19.28bps 的净效果：**7.8pp/年** |

**结论**：当前信号集在 A 股大盘股 5 日尺度上**没有可用 alpha**；IC 显著为负但
幅度极小（折算 IR≈−0.06），扣成本后为负 —— **"反着做"也救不了**（需要 IC≥0.021）。

### 9.2 基线 B：112 只 / 2016 起（部署口径，UI 与 `/api/performance` 展示的就是它）

同一份代码、不同池与区间，数字显著不同（IC −0.0050、t −0.57、Meta 46.15%）。
**引用任何数字前先确认是 A 还是 B**，否则会出现"同一指标三个值"的假矛盾。

### 9.3 三条已被证伪的旧说法（勿再引用）

| 旧说法 | 实测 | 结论 |
|---|---|---|
| Deflated Sharpe = 0.008 | 该值是"把收益均值当 Sharpe 传参"的产物；修正后按周期为 0.03–0.24 | **0.008 作废**（见 §7 陷阱 23） |
| 命中率 48.4%"低于抛硬币" | 朴素二项 SE 把它说成 5.5 个标准差；按决策日聚类只有 0.18 个标准差 | **不成立**，只能说"与 50% 不可区分" |
| 多空 +4.5% 是"扣费后" | +4.18% 是**毛**算术年化；`analysis.long_short_series` 不含成本 | **标签错误**，净值须另算 |
| 模拟盘「+4.3%/年」 | 那个数是按 10.0bps 往返成本算的，低于 A 股法定下限 11.28bps | **改用 19.28bps 后为 −3.49%/年**（见 §7 陷阱 25） |
| 「fqt=2 是加法复权，收益率被压缩到 0.58–0.83 倍」 | 同源 fqt=0 对照：压缩因子中位 **1.0000**，乘法复权 34/36；换口径后 Rank IC 只动 **1.0%** | **不成立**（那是跨源差异）；见 §9.4 |

### 9.4 未测项（不许估算冒充）

- **N100 上的耗时**：从未实测。已知本机（Zen3/16 线程）端到端 81.9s、核心 18.2s、
  增量 dataset 段 16.93s（已超 15s 预算）。
- **端到端 <60s 目标**：本机 81.9–103.7s **未达标**（含 impact/behavior/report/导出）。
- **fqt=2 收益率尺度（已闭环，结论与初判相反）**：
  ① `audit/basis_probe.py`（36 只）：**乘法复权 34 / 加法复权 2**，压缩因子
  **中位 1.0000**（p05 0.862）→「收益率被压缩到 0.58–0.83 倍」**不成立**，
  那个数是跨源（东财 vs 腾讯）比出来的，不是 fqt=2 自身的误差。
  ② `audit/basis_ic.py`（56 只 × 2633 天，同一份 scores 换三种收益口径）：
  ```
  h2                IC −0.01104  t −1.27
  修正加法复权股票   IC −0.01093  t −1.26   ΔIC +0.00011 (+1.0%)
  fqt=0 原始价       IC −0.01088  t −1.25   ΔIC +0.00017 (+1.5%)
  ```
  → **复权口径对 Rank IC 的影响 ≤1.5% 相对值，方向与量级都可引用**。
  数学原因：乘法复权下 `p_adj = p_raw·k(t)`，`r_true = p_adj[t+1]/p_adj[t] − 1 = r_h2`
  恒等 —— fqt=2 本身就是总收益序列。真正的风险只剩少数加法复权股票（实测 2/56）。
  产物：`data/state/basis.json`、`data/state/basis_ic.json`。
  **不需要再做全池 300 只 A/B**（此前以为需要）。

---

## 10. 修改本项目时的硬性要求

1. 改完必须跑：`python tests/test_no_lookahead.py`（worst diff 必须 `0.000e+00`）、
   `python -m tests.test_signals`（19 passed，截断一致性逐位相等）、`python audit/leak_scan.py`（HIGH=0）。
   改了 IO / 台账 / 统计口径，还要跑 `python -m tests.test_store_ledger`、
   `python -m tests.test_dsr_units`、`python -m tests.test_invariants_negative`、
   `python -m tests.test_incremental_parity`、`python -m qcore.invariants`、
   `python -m qcore.claims settle`（到期判断必须结算，否则 CI 红）、
   `python -m audit.gen_waivers --check`（豁免清单必须与代码一致，否则 CI 红）。
2. 任何性能改动必须给出实测数字（本项目目标：N100 级小主机，全流程 < 60s，增量 < 15s）。
3. 任何"提升效果"的改动必须同时给出**样本外**证据，且说明是否通过多重检验。
4. 不允许为了让数字好看而调整评估口径。评估口径是本项目的核心资产。
5. 新增判断（结论）必须通过 `qcore.claims.register` 登记，写明证伪条件与复核日期。
6. **任何与规模/流动性/波动相关的因子，不做行业+市值中性化不许上报 alpha。**
   实测教训：amihud 原始价差 +0.566%(t=5.94)，中性化后 +0.031%(t=0.40) —— 全是小市值效应。
7. **任何「最优 X / 有提升」必须报样本数**，`< MIN_TRADES(30)` 一律标 insufficient；
   门槛只认 `qcore/thresholds.py` 一个真源（历史上散落成 5 / 8 / 30 三套）。
8. **判断必须能自动结算**：`claims.register` 强制 `falsifier` 与 `review_by`；
   `python -m qcore.claims settle` 对有产物口径的判断自动判定，对无口径的列
   `needs_manual` 并非零退出。只登记不结算 = 没有命中率。
9. **UI 上每个数字必须有写入方**。硬编码一个有利数字（曾经把
   「Alpha158 IC +0.0260 样本外验证通过」写死在页面上，而产物是 0.0116）
   比不显示更糟 —— 它让测量仪替用户下结论。
10. **不利结论不许折叠**：73 个手写信号是测量本体，不许降级成 `<details>` 归档区。
11. **台账必须有人读**：`/api/ledger?source=file` 直读主台账（分页 + 总数 + 去重数）；
    只写不读的文件，重复写入 5.85× 也没人会发现。
