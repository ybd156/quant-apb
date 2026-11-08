# APB Web UI — 零依赖控制台

自学习参数体（Adaptive Parameter Body）的只读看板 + 实盘成交录入。
**纯 Python 标准库**（`http.server`）+ **原生 JS/CSS**，无框架、无 CDN、无 npm、完全离线可用。
目标机：Intel N100 小主机 / 飞牛 fnOS（Linux）/ Windows。

```
webui/
├─ server.py            # ThreadingHTTPServer + 路由（只读 IO，不做任何指标计算）
├─ README.md
└─ static/
   ├─ index.html        # 4 个页签骨架
   ├─ app.js            # 渲染 + 手绘 SVG 图表（无图表库）
   └─ style.css         # 深色主题 / 中文优先字体 / 响应式
```

---

## 1. 启动

在**项目根目录**（`webui/` 的上一级）执行：

```bash
# 默认：127.0.0.1:8765，只允许本机访问
python -m webui.server

# 换端口
python -m webui.server --port 8766

# 内网访问（飞牛/N100 上让手机也能看）—— 务必同时设置 token，见 §3
APB_TOKEN='一串足够长的随机字符' python -m webui.server --host 0.0.0.0 --port 8765
```

Windows PowerShell：

```powershell
$env:APB_TOKEN = "一串足够长的随机字符"
python -m webui.server --host 0.0.0.0
```

启动后终端会打印监听地址、状态目录、token 状态；浏览器打开 `http://127.0.0.1:8765/`。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 绑定地址。`0.0.0.0` = 内网可访问（有安全风险，见 §3） |
| `--port` | `8765` | 端口 |
| `--state-dir` | `<项目根>/data/state` | 状态目录，便于测试/多实例 |
| `--run-timeout` | `900` | `POST /api/run` 子进程超时（秒） |
| `--no-run` | 关 | 彻底禁用 `POST /api/run` |

依赖：Python 3.8+（实测 3.12）。**不 import numpy/pandas**，所以 30 MB 的运行时也不会拖慢启动。

---

## 2. 数据契约（UI 只读，全部由 qcore 落盘）

| 文件 | 用途 | 关键字段 |
|---|---|---|
| `data/state/status.json` | 引擎状态卡 | `as_of, data_from, data_to, n_stocks, n_signals, n_days, mode, horizon, updated_at` |
| `data/state/weights.json` | 信号权重 Top/Bottom | `sid, family, weight, hit_rate, n_updates, mean_payoff, alarms`（按 `weight` 降序） |
| `data/state/regime.json` | 市场状态色带 | `date, regime`（最近 500 天，0=下跌低波 1=下跌高波 2=上涨低波 3=上涨高波） |
| `data/state/ledger.json` | 预测台账 | `as_of, code, score, regime, entry_price, exit_price, ret, hit, status` |
| `data/state/performance.json` | 命中率 / IC / 多空 / 基线 | `metrics.hit_rate{h:{hit_rate}}`、`metrics.rank_ic{h:{mean,icir,positive_ratio}}`、`backtest.long_short{ann_return,sharpe,max_drawdown,...}`、`backtest.baselines[]` |
| `data/state/paper.json` | 模拟盘 vs 实盘 | `dates[], sim_nav[], live_nav[], compare{sim_final,live_final,sim_annual,tracking_error_ann,pnl_gap,exec_slippage_bps,...}` |
| `data/state/live_trades.jsonl` | 实盘成交（**UI 唯一写入的文件**） | 每行 `{date, code, side, price, qty, note}` |

> 前端同时兼容 SPEC 的简写形态（`performance.hit_rate.h1` / `ic.mean` / `long_short.curve`、
> `paper.sim.nav` / `paper.diff`），两套字段名都在 `app.js` 的 `adaptPerformance()` /
> `adaptPaper()` 里归一，**不改动任何数值**。

**UI 绝不做计算**：命中率、IC、多空净值、权重全部来自上面这些文件；
`server.py` 只做「读 JSON → 取 Top N / 最近 N 天 → 原样转发」，不派生任何指标。

---

## 3. 安全说明（重要）

1. **默认只绑回环** `127.0.0.1`：本机以外访问不到，这是默认的安全姿态。
2. `--host 0.0.0.0` 之后任何人都能打开页面并**读取你的持仓台账**。此时请务必：
   - 设置 `APB_TOKEN`（否则 `/api/run` 直接拒绝），
   - 只在内网/家庭网使用，不要把端口映射到公网，
   - 需要远程访问时优先用 SSH 隧道：`ssh -L 8765:127.0.0.1:8765 你的小主机`，然后本地开 `http://127.0.0.1:8765`。
3. `POST /api/run` 三重保护：`APB_TOKEN` 未设置 → 一律拒绝；`X-Token` 头与 `APB_TOKEN` 不匹配 → 403
   （用 `hmac.compare_digest` 定长比较）；同一时刻只允许一次运行（第二次返回 409 busy）。
4. `POST /api/paper/trade` **没有鉴权**（只写本地 JSONL、不触发交易），因为它就是给内网手工记账用的。
   若你把 UI 暴露到不可信网络，请在前面加一层反向代理 + Basic Auth，或用 `--no-run` 并只绑回环。
5. 静态文件服务做了路径穿越防护（`resolve()` + `relative_to()` 校验），`/static/../server.py` 返回 403。
6. 请求体上限 64 KB；成交记录做了完整校验（日期格式、方向枚举、正数价格、正整数数量）。
7. 服务端**绝不返回 500/堆栈**：所有异常都被捕获成 `{"ok": false, "error": "..."}`。

---

## 4. HTTP 接口

统一响应封套：

```jsonc
// 成功
{ "ok": true, "data": { ... } }
// 失败（HTTP 200 也可能是失败，前端按 ok 判断）
{ "ok": false, "error": "not_ready", "missing": "performance.json", "hint": "..." }
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` `/index.html` `/static/*` | 静态文件 |
| GET | `/api/status` | 引擎状态 |
| GET | `/api/signals?top=20` | `{top:[], bottom:[], total, top_n}`（`top` 上限 200） |
| GET | `/api/regime` | `{rows:[], n}` |
| GET | `/api/ledger?days=30` | `{rows:[], n, days}`（取最近 N 个不同 `as_of`，上限 500 天） |
| GET | `/api/performance` | 原样转发 |
| GET | `/api/paper` | 原样转发 |
| GET | `/api/paper/trades?limit=50` | 实盘成交明细（最新在前） |
| GET | `/api/run/status` | 最近一次 `/api/run` 的状态 |
| GET | `/api/health` | 状态目录/静态目录/是否允许 run（排查用） |
| POST | `/api/paper/trade` | 追加一行到 `live_trades.jsonl`；需 `Content-Type: application/json` |
| POST | `/api/run` | 触发 `python -m qcore.live --once`；需 `X-Token` |

错误码：`not_ready`（数据文件缺失，**HTTP 200**）、`bad_json`、`read_error`、`bad_schema`、
`bad_date` / `bad_code` / `bad_side` / `bad_price` / `bad_qty`（400）、`body_too_large`（413）、
`forbidden`（403）、`not_found` / `unknown_endpoint`（404）、`unauthorized` / `token_not_configured`（403）、
`busy`（409）、`serialize_error` / `internal_error`（200，兜底不崩）。

`POST /api/paper/trade` 示例：

```bash
curl -X POST http://127.0.0.1:8765/api/paper/trade \
  -H 'Content-Type: application/json' \
  -d '{"date":"2026-11-06","code":"600519","side":"buy","price":1520.5,"qty":100,"note":"手工补录"}'
```

写入的 JSONL 与 `qcore/paper.py::load_live_trades()` 的字段完全一致，
所以**下一次引擎运行会自动把它算进实盘净值**。

---

## 5. 数据未就绪时怎么表现（优雅降级）

页面不会白屏、不会崩、不会报 500。缺哪个文件，只有对应的卡片变成占位块：

| 缺失文件 | 表现 |
|---|---|
| `status.json` | 顶栏「数据日 —」，总览状态卡显示 *未就绪：等待引擎首次运行* |
| `weights.json` | 信号页两侧条形区显示同一占位提示 |
| `regime.json` | 色带位置显示占位块（容器自动退回普通块，不会被压成 26px 高） |
| `ledger.json` | 台账表体显示占位行 |
| `performance.json` | 命中率 / IC / 多空三块各自显示占位；基线卡自动隐藏 |
| `paper.json` | 模拟盘/实盘卡、净值双线、差异三块各自显示占位 |
| `live_trades.jsonl` | 「最近成交」显示 *还没有录入成交*（不算错误） |

占位文案会**明确指出缺哪个文件**以及下一步命令：`python -m qcore.live --once`。
刷新按钮（或 90 秒自动刷新）即可让页面自己恢复，不用重启服务。

其它容错：文件正在被写入（读到半截 JSON）→ `bad_json` + 提示稍后重试；
服务进程挂了 → 前端显示 *无法连接服务*；数据里出现 `NaN`/`Infinity` → 服务端统一转成 `null`
（`NaN` 不是合法 JSON，浏览器 `JSON.parse` 会直接抛错），前端显示 `—`。

---

## 6. 前端说明

- **4 个页签**：总览 / 信号 / 台账 / 模拟实盘。切页签才拉数据，避免首屏打满。
- **图表全部手绘 SVG**：多空净值曲线、模拟盘 vs 实盘双线（`chartSVG()` 约 90 行，无任何图表库）。
  窄屏（<700px）自动切换 viewBox 宽度，避免手机上被压扁。
- **信号条形图**：纯 CSS 宽度百分比，点任意条看该信号的命中率/更新次数/平均收益/发声次数。
- **配色约定**（两个口径分开，别当成 bug）：
  - 台账「命中」= 绿、「未中」= 红、「pending」= 灰（对/错语义）；
  - 数值列（收益、净值、IC）用 A 股口径 **红涨绿跌**。
- **字体**：只用系统已装字体（PingFang SC / 微软雅黑 / Noto Sans CJK SC / 思源黑体 / 文泉驿），
  不下载任何网络字体，飞牛上没装中文字体时会回落到 `sans-serif`。
- **性能护栏**：台账一次最多渲染 500 行（真实数据 30 天约 3000 行），超出时提示缩小天数；
  这不是统计，只是渲染上限。
- 响应式：`@media` 三档（900 / 620px），手机上单列、表格横向滚动、按钮不换行溢出。
  尊重 `prefers-reduced-motion`。

---

## 7. 自测记录

```bash
# 1) 空状态目录（全部文件缺失）—— 验证不崩
python -m webui.server --port 8766 --state-dir tmp/empty

# 2) 真实 data/state —— 验证正常路径
python -m webui.server --port 8767
```

用 `urllib` 逐个打端点，并另写了一个最小 DOM shim 在 Node 里跑 `app.js`，
把真实 `/api` 响应喂给渲染函数，断言不出现 `undefined`/`NaN`/`[object Object]`：

- 8 个 `/api` 端点在两种状态下均返回 **HTTP 200**，且响应体通过浏览器口径的 `JSON.parse`；
- 4 个页签的 20+ 个渲染点在真实数据下全部 PASS；
- 空目录下全部渲染点显示占位块，PASS；
- 路径穿越 `/static/../server.py` → 403；未知端点 → 404 JSON；
- `POST /api/paper/trade` 合法 → 200 写入；`bad_side`/`bad_date`/`bad_json` → 400 且不写盘；
- `POST /api/run` 无 token → 403，错 token → 403，对 token → 启动子进程并可用 `/api/run/status` 轮询。

---

## 8. 已知限制

1. **多空净值曲线**：`performance.json` 的 `backtest.long_short` 目前不含 `equity` 序列
   （引擎导出时把 `np.ndarray` 过滤掉了），所以总览的曲线位置显示占位，
   但年化 / Sharpe / 最大回撤照常显示。若引擎补一个 `equity: [...]`（可下采样到 ~250 点），
   前端会立刻画出来，无需改代码。
2. `POST /api/run` 依赖能创建子进程。在受限沙箱里 `subprocess` 会被系统拒绝
   （Windows 上是 `WinError 5`），此时接口返回
   `{"ok":false,"error":"spawn_denied",...}` 而不会崩；在正常主机上无此问题。
3. 响应体未压缩：真实 `performance.json` 约 640 KB、`ledger?days=30` 约 470 KB。
   局域网/本机够用；若要更省，可在前面加一层 nginx 开 gzip，或调小 `days`。
4. 没有登录/多用户，定位是「单机自用看板」，不要直接暴露到公网。
5. 台账 `hit` 由引擎判定（多空方向），前端只做展示；`status=pending` 表示 h 日未到、尚未对账。
