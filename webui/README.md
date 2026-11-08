# APB 控制台 — 零依赖 Web UI

> 版本号唯一真源是 `webui/VERSION`，由发布流程维护；页面顶栏只显示 `/api/version` 读到的值，前端不硬编码。

自学习参数体（Adaptive Parameter Body）的只读看板 + 实盘成交录入。

**纯 Python 标准库**（`http.server`）+ **原生 JS/CSS**：无框架、无 CDN、无 npm、无构建步骤、无图表库，
所有图表都是手绘 SVG。目标机：Intel N100 小主机 / 飞牛 fnOS（Linux）/ Windows。

```
webui/
├─ VERSION              # 版本号唯一真源，前端只显示 /api/version 读到的值
├─ server.py            # ThreadingHTTPServer + 路由（只读 IO，不做任何指标计算）
├─ README.md
└─ static/
   ├─ index.html        # 应用外壳：侧边栏 + 顶栏 + 图标精灵（内联 SVG symbol）
   ├─ app.js            # 路由 + 8 个页面 + 组件 + 手绘 SVG 图表
   └─ style.css         # 设计令牌 / 深色+亮色双主题 / 响应式
```

单页总下载 **117.7 KB（gzip 33.2 KB）**，**3 个请求、0 个外部请求**。

---

## 1. 启动

在**项目根目录**（`webui/` 的上一级）执行：

```bash
python -m webui.server                       # 默认 127.0.0.1:8765（只允许本机）
python -m webui.server --port 8766           # 换端口
APB_TOKEN='随机长字符串' python -m webui.server --host 0.0.0.0   # 内网访问（务必配 token）
```

Windows PowerShell：

```powershell
$env:APB_TOKEN = "随机长字符串"
python -m webui.server --host 0.0.0.0
```

浏览器打开 `http://127.0.0.1:8765/`。URL 带 `#/页面` 可直接进入某页（如 `#/risk`），
刷新后会记住上次页面。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 绑定地址；`0.0.0.0` = 内网可访问（见 §4 安全） |
| `--port` | `8765` | 端口 |
| `--state-dir` | `<项目根>/data/state` | 状态目录（测试/多实例） |
| `--run-timeout` | `900` | `POST /api/run` 子进程超时（秒） |
| `--no-run` | 关 | 彻底禁用 `POST /api/run` |

依赖：Python 3.8+（实测 3.12）。**不 import numpy/pandas**，启动即秒开。

---

## 2. 8 个页面

| 页面 | 内容 | 数据来源 |
|---|---|---|
| **总览** | 4 个 KPI（市场状态 / 可信度 Brier / Rank IC / 成本拖累）+ 净值曲线 + regime 色带 + 结论横幅 + 分年度 IC | `performance` `regime` `paper` `status` |
| **市场** | 9 个指数卡（点位/涨跌幅/涨跌点数）+ 情绪温度条（0–100，分档配色）+ 市场宽度堆叠条 + 涨停/成交额 KPI + 60 秒自动刷新 | `market.json` |
| **信号** | 权重 Top20 / Bottom20 条形图 + 全表（权重/命中率/更新次数/平均收益/发声 可排序）+ 族筛选 + 搜索 | `weights.json` |
| **预测台账** | 最近预测表（命中绿 / 未中红 / pending 灰）+ 代码搜索 + 日期区间 + 结果筛选 | `ledger.json` |
| **模拟实盘** | 模拟 vs 实盘双线净值 + 差异指标 + 成交录入表单 + 最近成交 | `paper.json` `live_trades.jsonl` |
| **行为因子** | 10 个行为因子 IC / 多空价差双向条形 + 明细表（含「中性化后失效」标注） | `performance.behavior`（缺失时显示因子定义） |
| **政策事件** | 政策日历时间线（类型/方向筛选）+ 影响因子表 | `events.json` → 回退 `data/events/policy_calendar.jsonl` |
| **风险体检** | 8 条不变量检查 + 数据源健康度 + 成本明细 + 集中度 + 可信度校准 + 漂移监控 | `performance.risk/costs/confidence` `source_health.json` `invariants.json` |
| **设置** | 版本/构建时间/Python/平台、**版本与更新（检查更新 / 立即更新）**、数据文件清单、触发引擎更新（需 token）、导出数据、主题 | `/api/version` `/api/health` `/api/update` |

### 布局与交互

- **左侧固定侧边栏**（232px，可折叠为 64px 图标条，状态存 localStorage）+ 右侧内容区。
- **顶栏**：页面标题 + 副标题、数据时间戳、版本号、**连接状态灯**（绿=正常 / 黄=数据陈旧 >1 天 / 红=缺文件或服务异常）、刷新。
- 断点：**>1200px 三列**，**768–1200px 两列**，**<768px 单列 + 底部导航**（侧栏变抽屉 + 遮罩）。
- 主题：深色默认 / 亮色，`data-theme` + localStorage，**首屏前用内联脚本应用，无闪白**。

---

## 3. 数据契约（UI 只读）

| 文件 | 用途 | 关键字段 |
|---|---|---|
| `data/state/status.json` | 状态卡 | `as_of, data_from, data_to, n_stocks, n_signals, n_days, mode, horizon, updated_at` |
| `data/state/weights.json` | 信号权重 | `sid, family, weight, hit_rate, n_updates, mean_payoff, alarms`（按 weight 降序） |
| `data/state/regime.json` | regime 色带 | `date, regime`（0 跌低波 / 1 跌高波 / 2 涨低波 / 3 涨高波） |
| `data/state/ledger.json` | 预测台账 | `as_of, code, score, regime, entry_price, exit_price, ret, hit, status` |
| `data/state/performance.json` | 总览/行为/体检 | `metrics.hit_rate{h:{hit_rate}}`、`metrics.rank_ic{h:{mean,icir,positive_ratio}}`、`backtest.long_short`、`edge{h:{ic:{mean,t,ci95},subperiods,long_short,deflated_sharpe}}`、`confidence`、`risk`、`costs`、`learned_vs_equal`、`meta`、`behavior`（待导出） |
| `data/state/paper.json` | 模拟 vs 实盘 | `dates[], sim_nav[], live_nav[], compare{...}` |
| `data/state/source_health.json` | 数据源健康度 | `{源: {ok, fail, latency_ms, last_error}}` |
| `data/state/live_trades.jsonl` | 实盘成交（**UI 唯一写入**） | 每行 `{date, code, side, price, qty, note}` |
| `data/state/events.json`（可选） | 政策日历 | `{events:[{date,type,subtype,title,direction,sectors[],evidence_level,note,lag}], impact:[...]}`；缺失时回退 `data/events/policy_calendar.jsonl` |
| `data/state/invariants.json`（可选） | 不变量机读结果 | `[{id:"INV-1", name, status:"pass"/"fail", detail}]` |
| `data/state/market.json` | 市场页 | `as_of, indices[{code,name,price,chg_pct,change}], breadth{up,down,flat,up_ratio,by_market}, limit_up, amount_yi, sentiment{ok,temperature,level,reason,parts}, missing[], elapsed_ms` |
| `data/state/update.json`（可选） | 版本检查 | `{latest_version, checked_at}`（`/api/update` 用它和 `VERSION` 比对） |

**UI 绝不做计算**：所有数值来自上面这些文件；`server.py` 只做「读 JSON → 取 Top N / 最近 N 天 → 原样转发」，
前端只做取字段、排序、筛选、排版。图表坐标是像素映射，不改变数值含义。

### 给引擎的两个可选导出（有则页面自动点亮，无需改前端）

```python
# 1) 行为因子（10 个）写进 performance.json
perf["behavior"] = [{"factor": sid, "ic": ..., "t": ..., "spread": ...,
                     "ic_neutral": ..., "note": "中性化后失效"}, ...]

# 2) 多空净值曲线写进 performance.json（前端优先用它画总览净值曲线）
perf["backtest"]["long_short"]["equity"] = [float(x) for x in equity[::5]]   # 下采样即可
```

---

## 4. 安全说明（重要）

1. **默认只绑回环** `127.0.0.1`，本机以外访问不到。
2. `--host 0.0.0.0` 后任何人都能读到你的持仓台账。此时务必设置 `APB_TOKEN`，只在内网使用，
   远程访问优先走 SSH 隧道：`ssh -L 8765:127.0.0.1:8765 你的小主机`。
3. `POST /api/run`：`APB_TOKEN` 未设置 → 一律拒绝；`X-Token` 不匹配 → 403（`hmac.compare_digest`）；
   同一时刻只允许一次运行（第二次 409 busy）。前端把 token 只存在**当前标签页的 sessionStorage**。
4. `POST /api/paper/trade` **无鉴权**（只写本地 JSONL、不触发交易）。暴露到不可信网络时请前置反向代理 + Basic Auth。
5. 静态文件做了路径穿越防护（`resolve()` + `relative_to()`），`/static/../server.py` → 403。
6. 请求体上限 64 KB；成交记录完整校验（日期格式、方向枚举、正数价格、正整数数量）。
7. **绝不返回 500/堆栈**：所有异常都转成 `{"ok": false, "error": "..."}`。

---

## 4.1 应用内更新（流程与错误码）

**入口**：顶栏 `v{current}` 右侧的蓝色徽章「有新版本 vX.Y.Z」（带呼吸光圈，只在**远端严格更新**时出现）。

1. 点徽章 → 弹窗显示「当前版本 / 目标版本 / 最近检查」+ token 输入框（预填 `localStorage['apb.token']`）。
2. 点「确认更新」→ `POST /api/update/apply`（`X-Token`）。
3. 成功后弹窗切换成「更新中」：转圈 + 「每 10 秒自动检查一次」，顶栏徽章变成不可点的「更新中…」。
4. 前端每 10 秒轮询 `/api/version`，一旦 `version` 与更新前不同 → toast「已更新到 vX」→ 2.5 秒后自动刷新页面；5 分钟未变则提示手动刷新。
5. 失败按后端错误码分类提示，不把裸错误码丢给用户：

| 后端 error | 前端提示 |
|---|---|
| `token_not_configured` | 服务端未设置 APB_TOKEN，无法在应用内更新（并清空本地 token） |
| `unauthorized` | X-Token 与 APB_TOKEN 不匹配，请重新输入（并清空本地 token） |
| `docker_socket_unavailable` | 容器未挂载 docker.sock，无法应用内更新 + 给出 `docker compose pull && docker compose up -d` |
| `already_latest` | 已是最新版本 |
| `import_error` / `helper_create_failed` / `helper_start_failed` | 对应失败原因 + 改用命令行 |
| `network_error` | 无法连接服务（可能正在重启），稍等 1–3 分钟刷新 |

> **前端有一层语义化版本兜底**：后端 `has_update` 用的是 `latest != current`，若远端版本比本地**旧**（例如本地 0.0.6、远端 0.0.5）也会为 true。
> 前端用 `cmpVersion()` 再判一次，只有 `latest > current` 才显示徽章；否则设置页显示「远端更旧，已忽略」，**不会把降级当成升级**。

token 只存在本机 `localStorage['apb.token']`（跨标签页只填一次），换设备需重填。

---

## 5. HTTP 接口

统一封套：成功 `{"ok":true,"data":{...}}`；失败 `{"ok":false,"error":"...","hint":"..."}`
（**数据文件缺失时 HTTP 200 + `error:"not_ready"`**，不是 5xx）。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` `/index.html` `/static/*` | 静态文件 |
| GET | `/api/version` | `{version, build_time, python, platform, webui_dir}`（版本读 `webui/VERSION`） |
| GET | `/api/status` | 引擎状态 |
| GET | `/api/signals?top=20` | `{top:[], bottom:[], total, top_n}`（`top` 上限 200） |
| GET | `/api/regime` | `{rows:[], n}` |
| GET | `/api/ledger?days=30` | `{rows:[], n, days}`（最近 N 个不同 `as_of`，上限 500 天） |
| GET | `/api/performance` | 原样转发 |
| GET | `/api/behavior` | `{generated_at, dataset, edge, behavior}`（原样转发 performance.json 的段） |
| GET | `/api/events?limit=500` | `{rows:[], n, source, impact?}`；`events.json` → 回退 `policy_calendar.jsonl` |
| GET | `/api/paper` | 原样转发 |
| GET | `/api/paper/trades?limit=50` | 实盘成交明细（最新在前） |
| GET | `/api/health` | `{ok, as_of, data_age_days, missing_files[], files{}, optional_files{}, source_health{}, invariants, run_enabled, ...}` |
| GET | `/api/market` | 实时行情与情绪（原样转发 `market.json`） |
| GET | `/api/update` | `{current, latest, has_update, checked_at, command}` |
| POST | `/api/update/apply` | 应用内一键更新（需 `X-Token`）；错误码见 §4.1 |
| GET | `/api/export?what=all\|ledger\|signals\|performance` | 导出 JSON 附件（原样拼接，不加工） |
| GET | `/api/run/status` | 最近一次 `/api/run` 的状态 |
| POST | `/api/paper/trade` | 追加一行到 `live_trades.jsonl`（需 `Content-Type: application/json`） |
| POST | `/api/run` | 触发 `python -m qcore.live --once`（需 `X-Token`） |

错误码：`not_ready`、`bad_json`、`read_error`、`bad_schema`、`bad_date`/`bad_code`/`bad_side`/`bad_price`/`bad_qty`（400）、
`body_too_large`（413）、`forbidden`（403）、`not_found`/`unknown_endpoint`（404）、
`unauthorized`/`token_not_configured`（403）、`busy`（409）、`bad_param`（400）、
`serialize_error`/`internal_error`（200，兜底不崩）。

> `data_age_days` 是「今天 − `status.as_of`」的**文件新鲜度**，只驱动顶栏状态灯，不是任何分析指标。

---

## 6. 数据未就绪时怎么表现（优雅降级）

页面不白屏、不崩、不报 5xx。缺哪个文件，只有对应卡片变占位块：

| 缺失 | 表现 |
|---|---|
| `status.json` | 顶栏「数据 —」+ 状态灯红；总览状态卡占位 |
| `weights.json` | 信号页两侧条形区占位，表格空态 |
| `regime.json` | 色带位置占位（容器退回普通块，不会被压成 24px） |
| `ledger.json` | 台账表体占位行 |
| `performance.json` | 总览 KPI/曲线/明细、行为因子、风险体检各自占位；**行为因子页仍展示 10 个因子定义** |
| `paper.json` | 模拟实盘净值/差异占位；**成交录入表单照常可用** |
| `events.json` + JSONL 都缺 | 政策事件页显示占位 + 生成命令提示 |
| `invariants.json` | 不变量表 8 行全部「未提供」并给出写入格式 |
| `source_health.json` | 数据源健康度占位 |
| `live_trades.jsonl` | 「还没有录入成交」（不算错误） |
| `market.json` | 市场页显示「行情采集尚未启动」占位 + 期望字段说明，不空白、不崩 |
| `update.json` | 更新徽章隐藏；设置页「远端最新 = 未知 / 状态 = 尚未检查」 |

其它容错：读到半截 JSON → `bad_json` 提示重试；服务挂了 → 「无法连接服务」+ 红灯；
数据里出现 `NaN`/`Infinity` → 服务端统一转 `null`（`NaN` 不是合法 JSON，浏览器 `JSON.parse` 会崩），前端显示 `—`。
右上角「刷新」清缓存重取，无需重启服务。

---

## 7. 自测记录

```bash
python -m webui.server --port 8777                                   # A 真实 data/state
python -m webui.server --port 8766 --state-dir tmp/uitest_empty      # B 空目录
python -m webui.server --port 8768 --state-dir tmp/state_update      # C 带 update.json（latest 0.0.7）

node tmp/ui_dom_test.mjs http://127.0.0.1:8777 full      # A 全量
node tmp/ui_dom_test.mjs http://127.0.0.1:8766 empty     # B 降级
node tmp/ui_dom_test.mjs http://127.0.0.1:8768 update    # C 更新徽章 + 弹窗 + 错误分类
```

`tmp/ui_dom_test.mjs` 是一个最小 DOM shim（约 200 行），在 Node 里真正执行 `app.js`，
把真实 `/api` 响应喂给渲染管线，断言：

- 12 个端点全部 **HTTP 200** 且通过浏览器口径 `JSON.parse`（含 `NaN` 检测）；
- 8 个页面在「有数据」「空目录」两种模式下都渲染成功，无 `undefined` / `NaN` / `[object Object]`；
- 顶栏版本号来自 `/api/version`（正则 `^v\d+\.\d+\.\d+$`），且 **HTML 里不含硬编码版本号**；
- 状态灯在空目录下为 `down`、有数据时为 `ok`/`stale`；
- 成交表单已绑定 submit；
- 市场页：9 个指数卡、宽度堆叠条、温度 `ok=false` 时显示 `—`（不是 0）且带「暂不计算」标注；
- 更新徽章满足不变量「可见 ⟺ 远端严格更新」，点徽章开弹窗、错 token 走分类提示、点关闭能收起；
- 设置页含「版本与更新」卡、检查更新按钮、命令行等价操作。

---

## 8. 已知限制

1. **无法在本环境截图验证视觉**：沙箱禁止创建子进程（`WinError 5`），Chrome headless 无法 fork，
   所以视觉效果靠「DOM 结构 + 渲染断言」验证，像素级排版请在真机浏览器确认。
   `POST /api/update/apply` 的**成功分支**同样无法在本机跑通（没有 docker.sock，实测返回 503
   `docker_socket_unavailable`），已覆盖其错误分支；成功分支的「更新中 → 轮询 → 自动刷新」逻辑按契约实现。
2. `performance.json` 的 `backtest.long_short` 目前**不含 `equity` 序列**，总览的净值曲线回退到
   `paper.sim_nav/live_nav`（页面上有标注）。引擎补 `equity` 后自动切换，无需改前端。
3. `behavior` 段尚未导出 → 行为因子页显示「定义 + 未就绪」而不是编造数字。
4. `POST /api/run` 依赖能创建子进程；受限沙箱里会返回 `{"ok":false,"error":"spawn_denied"}` 而不崩。
5. 响应体未压缩：`performance.json` 约 640 KB、`ledger?days=30` 约 460 KB。
   局域网够用；要更省可在前面加 nginx 开 gzip，或调小 `days`。台账一次最多渲染 500 行（性能护栏）。
6. 没有登录/多用户，定位是「单机自用看板」，不要直接暴露公网。
7. 配色口径：**命中/未中 = 绿/红**（对错语义）；**数值列 = 红涨绿跌**（A 股习惯）。两者分开是刻意的。
