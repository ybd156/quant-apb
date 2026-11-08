# 05 部署手册（Windows / 飞牛 fnOS）

> 目标：一台低功耗机器（N100 级，8GB 内存）上每天自动跑一次增量，本地网页看结果，**不暴露公网**。
> 本文所有命令都假设项目根目录为 `%APB_HOME%`（Windows）或 `/vol1/1000/apb`（fnOS），请按实际路径替换。

---

## 0. 前置条件与入口检查

| 项 | 要求 |
|---|---|
| Python | ≥ 3.10（3.11/3.12 均可），仅需 `numpy`、`pandas` |
| 内存 | ≥ 4GB（8GB 舒适） |
| 磁盘 | ≥ 2GB 可用 |
| 网络 | 能访问东方财富行情接口（数据层直连，无第三方 key） |

系统的两个入口（**先确认存在再谈部署**）：

```bash
python -c "import qcore.live; print('live ok')"       # 每日增量入口：--once
python -c "import webui.server; print('webui ok')"    # 本地网页入口：--port
```

> 若上述 import 失败，说明该入口尚未落地（`qcore/live.py`、`webui/server.py`），
> 此时只能跑离线回测（用 README「快速开始」里的 Python 片段），**不要**继续配置计划任务——
> 定时任务指向一个不存在的模块只会每天静默失败。

安装依赖（无编译、无 wheel 平台问题）：

```powershell
# Windows
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
```

```bash
# Linux / fnOS
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

---

## 1. 目录规划

```
apb/
├── qcore/                 # 代码（只读运行）
├── webui/                 # 本地网页服务
├── configs/               # ★ 参数配置（必须备份）
├── data/                  # ★ 全部落盘数据
│   ├── raw/kline/         #   行情 CSV 缓存（可再生，可删）
│   ├── raw/index/         #   基准指数（可再生）
│   ├── raw/meta/          #   股票池快照（可再生）
│   ├── cache/             #   中间产物（可再生）
│   ├── pred_ledger/       # ★ 预测台账 YYYY.parquet（只追加，不可再生）
│   └── state/             # ★ 学习器权重状态（不可再生，丢了等于失忆）
├── logs/                  # 日志（按日/按大小切分）
└── reports/               # 生成的 Markdown 报告
```

`★` 三项是**唯一必须备份**的东西：台账是"先封存后对账"的证据链，`state/` 是学习到的权重，
两者丢了就得从零重学（且历史对账无法重建）。`data/raw/` 与 `data/cache/` 删掉重下即可。

---

## 2. Windows 部署

### 2.1 本地网页服务

```powershell
cd C:\apb
py -3 -m webui.server --port 8765
```

- 默认**只监听 127.0.0.1**（见 §7 安全），浏览器打开 <http://127.0.0.1:8765>。
- 首次启动若 `data/` 为空，先手动跑一次 `py -3 -m qcore.live --once`（会拉全量历史）再开网页，否则页面是空的。

### 2.2 每日增量：任务计划程序（15:30 触发）

A 股 15:00 收盘，行情落地通常 15:10 之后。**15:30 是稳妥起点**，脚本内部应带重试与"数据未齐则跳过并告警"。

先写一个批处理包装（比在 `schtasks /TR` 里塞重定向可靠得多）：

`scripts\daily.bat`（`%DATE%` 的格式随系统区域设置变化；若你的日期不是 `yyyy-MM-dd` 形式，
把 `%DATE:~0,4%…` 换成固定文件名，或用 `for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set D=%%i`）

```bat
@echo off
setlocal
cd /d C:\apb
set LOG=logs\live_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%.log
echo [%DATE% %TIME%] start >> "%LOG%"
.venv\Scripts\python.exe -m qcore.live --once >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [%DATE% %TIME%] exit=%RC% >> "%LOG%"
exit /b %RC%
```

注册任务（管理员 PowerShell / CMD）：

```bat
schtasks /Create /TN "APB-Daily" /TR "C:\apb\scripts\daily.bat" ^
  /SC DAILY /ST 15:30 /RL HIGHEST /F
```

再进"任务计划程序 → APB-Daily → 属性"勾选三项（这三项比命令行参数重要）：

1. **常规**：勾"不管用户是否登录都要运行"（否则锁屏/注销就不跑）。
2. **条件**：取消"只有在计算机使用交流电源时才启动"；勾"如果错过计划开始时间，请尽快启动任务"
   （断电/休眠后补跑，`qcore.live --once` 必须幂等——同一天跑两次不能重复入账）。
3. **设置**：勾"如果任务失败，按以下频率重新启动：每 10 分钟，最多 3 次"。

验证：`schtasks /Run /TN "APB-Daily"`，然后看 `logs\live_YYYYMMDD.log` 最后一行是否为 `exit=0`。

### 2.3 开机自启（网页服务）

两种做法，二选一：

- **任务计划程序**（简单）：新建任务 → 触发器"登录时" → 操作指向
  `.venv\Scripts\pythonw.exe -m webui.server --port 8765`，勾"隐藏"。
- **NSSM 注册为服务**（无需登录、可崩溃重启）：

```bat
nssm install APB-Web "C:\apb\.venv\Scripts\python.exe" "-m webui.server --port 8765"
nssm set APB-Web AppDirectory "C:\apb"
nssm set APB-Web AppStdout "C:\apb\logs\web.log"
nssm set APB-Web AppStderr "C:\apb\logs\web.err"
nssm set APB-Web AppRotateFiles 1
nssm set APB-Web AppRotateBytes 10485760
nssm start APB-Web
```

> 注意 `pythonw.exe` 无控制台、看不到报错；调试期用 `python.exe`，稳定后再换。

---

## 3. 飞牛 fnOS 部署

fnOS 是 Debian 系，两条路都能走：**Docker（推荐，隔离干净）** 或 **systemd 直跑（推荐，省内存）**。

### 3.1 方案 A：Docker

`Dockerfile`（放在项目根）：

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 依赖单独一层，代码改动不触发重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 非 root 运行；data/ 与 logs/ 由卷挂载覆盖
RUN useradd -m -u 1000 apb \
 && mkdir -p /app/data /app/logs \
 && chown -R apb:apb /app
USER apb

EXPOSE 8765
CMD ["python", "-m", "webui.server", "--host", "0.0.0.0", "--port", "8765"]
```

构建与运行：

```bash
cd /vol1/1000/apb

docker build -t apb:latest .

docker run -d --name apb \
  --restart unless-stopped \
  -p 127.0.0.1:8765:8765 \
  -v /vol1/1000/apb/data:/app/data \
  -v /vol1/1000/apb/logs:/app/logs \
  -v /vol1/1000/apb/configs:/app/configs \
  -e TZ=Asia/Shanghai \
  -e APB_API_TOKEN='请改成随机串' \
  --memory 512m --memory-swap 768m \
  apb:latest
```

每日增量（宿主机 cron，调容器里的 CLI，共享同一个 `data` 卷）：

```cron
30 15 * * 1-5  docker exec apb python -m qcore.live --once >> /vol1/1000/apb/logs/live.log 2>&1
```

> **容器内 `--host 0.0.0.0` 是必须的**（否则容器外访问不到），但 `-p 127.0.0.1:8765:8765`
> 已经把监听锁在宿主机回环，公网访问不到。不要写成 `-p 8765:8765`。
> 若 `webui.server` 尚不支持 `--host` 参数，改用 `-e APB_HOST=0.0.0.0` 或让入口在检测到
> `/.dockerenv` 时自动绑 `0.0.0.0`；无论哪种，宿主机侧的 `-p 127.0.0.1:...` 都不能省。
> `--memory 512m` 是硬性护栏：超过就 OOM 重启，比拖垮整机好。

### 3.2 方案 B：systemd（直跑，无 Docker 开销）

先建运行用户与虚拟环境（两个 unit 里都写了 `User=apb`，用户不存在会启动失败）：

```bash
sudo useradd -r -s /usr/sbin/nologin -d /vol1/1000/apb apb
sudo chown -R apb:apb /vol1/1000/apb

cd /vol1/1000/apb
sudo -u apb python3 -m venv .venv
sudo -u apb .venv/bin/pip install -r requirements.txt
```

**（1）网页服务** `/etc/systemd/system/apb-web.service`

```ini
[Unit]
Description=APB local web UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=apb
Group=apb
WorkingDirectory=/vol1/1000/apb
Environment=PYTHONUNBUFFERED=1
Environment=TZ=Asia/Shanghai
ExecStart=/vol1/1000/apb/.venv/bin/python -m webui.server --port 8765
Restart=always
RestartSec=5
StandardOutput=append:/vol1/1000/apb/logs/web.log
StandardError=append:/vol1/1000/apb/logs/web.err

# 基本加固：只写自己的目录
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/vol1/1000/apb/data /vol1/1000/apb/logs /vol1/1000/apb/reports
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

**（2）每日增量** `/etc/systemd/system/apb-daily.service`

```ini
[Unit]
Description=APB daily incremental run
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=apb
Group=apb
WorkingDirectory=/vol1/1000/apb
Environment=PYTHONUNBUFFERED=1
Environment=TZ=Asia/Shanghai
ExecStart=/vol1/1000/apb/.venv/bin/python -m qcore.live --once
StandardOutput=append:/vol1/1000/apb/logs/live.log
StandardError=append:/vol1/1000/apb/logs/live.err
TimeoutStartSec=900
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/vol1/1000/apb/data /vol1/1000/apb/logs /vol1/1000/apb/reports
```

**（3）定时器** `/etc/systemd/system/apb-daily.timer`

```ini
[Unit]
Description=Run APB incremental every trading day 15:30

[Timer]
OnCalendar=Mon..Fri 15:30:00
Persistent=true
RandomizedDelaySec=120

[Install]
WantedBy=timers.target
```

`Persistent=true` 等价于 Windows 的"错过则补跑"；`RandomizedDelaySec` 避免与其它任务同时抢 IO。
节假日由 `qcore.live --once` 内部判断（非交易日直接退出 0，不写台账）。

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now apb-web.service
sudo systemctl enable --now apb-daily.timer
systemctl list-timers apb-daily.timer
journalctl -u apb-daily.service -n 50 --no-pager
```

---

## 4. 端口

| 端口 | 用途 | 监听 |
|---|---|---|
| 8765 | 本地网页 UI | 仅 `127.0.0.1` |
| — | 每日增量 | 无端口（CLI 一次性任务） |

若 8765 被占用：`--port 8766` 即可，无需改代码。检查占用：
Windows `netstat -ano | findstr :8765`；Linux `ss -ltnp | grep 8765`。

---

## 5. 日志轮转

日志只增不减，一年能吃掉几个 GB。三种做法任选：

**（1）应用内（最省事）**——若 webui 使用标准 `logging`，配置：

```python
from logging.handlers import RotatingFileHandler
h = RotatingFileHandler("logs/web.log", maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8")
```

**（2）Linux logrotate** `/etc/logrotate.d/apb`

```
/vol1/1000/apb/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

**（3）Windows 定期清理**（任务计划每周日 03:00）

```powershell
Get-ChildItem C:\apb\logs\*.log |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
  Remove-Item -Force
```

---

## 6. 数据备份

按"丢了会不会心疼"分级，别整目录无脑同步（`data/raw/` 上百 MB 且可再生）：

| 路径 | 可再生 | 备份策略 |
|---|---|---|
| `configs/` | 否 | 每次改动后同步 |
| `data/pred_ledger/` | 否（只追加证据链） | 每日增量 |
| `data/state/` | 否（学习器记忆） | 每日增量 |
| `data/raw/`、`data/cache/` | 是 | 不备份 |

**Windows（robocopy，镜像到移动硬盘或 NAS）**

```bat
robocopy "C:\apb\configs"          "E:\apb-bak\configs"          /MIR /R:2 /W:5 /LOG+:E:\apb-bak\bak.log
robocopy "C:\apb\data\pred_ledger" "E:\apb-bak\pred_ledger"     /MIR /R:2 /W:5 /LOG+:E:\apb-bak\bak.log
robocopy "C:\apb\data\state"       "E:\apb-bak\state"            /MIR /R:2 /W:5 /LOG+:E:\apb-bak\bak.log
```

**fnOS（rsync 到另一块盘 / 另一台机器）**

```bash
rsync -a --delete /vol1/1000/apb/configs/          /vol2/backup/apb/configs/
rsync -a --delete /vol1/1000/apb/data/pred_ledger/ /vol2/backup/apb/pred_ledger/
rsync -a --delete /vol1/1000/apb/data/state/       /vol2/backup/apb/state/
```

要点：

- **备份前先停写**（或至少确认当日任务已结束），台账是只追加的 parquet，边写边拷可能拷到半个文件。
- 每月做一次**恢复演练**：把 `state/` 拷回空环境，跑 `--once` 看权重是否延续——没演练过的备份等于没有备份。
- 台账建议再存一份纯文本 `csv` 快照（年度一次），防 parquet 版本不兼容。

---

## 7. 安全

默认姿态：**这是内网研究工具，不是对外服务。**

1. **只监听回环**：`--port 8765` 默认绑 `127.0.0.1`；Docker 用 `-p 127.0.0.1:8765:8765`。
   真正决定"暴不暴露"的是**宿主机端口绑定**：写成 `-p 8765:8765` 就等于对全网开放，容器内 `--host 0.0.0.0`
   本身没问题（那是容器内的视角）。路由器上也不要做 8765 的端口映射。
2. **要远程访问就加反代**（Caddy 示例，自带 HTTPS）：

```
apb.example.lan {
    reverse_proxy 127.0.0.1:8765
    basicauth {
        ybd $2a$14$替换成caddy hash-password生成的哈希
    }
}
```

   nginx 等价物：`auth_basic` + `allow 192.168.0.0/16; deny all;`。
3. **API token**：若 webui 提供写操作接口（触发增量、改配置），用环境变量
   `APB_API_TOKEN` 注入，请求头 `Authorization: Bearer <token>` 校验；token 只存 `.env`（已在 `.gitignore`）。
4. **不要开目录穿越**：静态文件服务必须限定在 `reports/` 内，禁止 `..`。
5. **依赖面极小是最大优势**：只有 numpy + pandas，没有 web 框架之外的攻击面；
   但 numpy/pandas 也要跟版本升级（`pip list --outdated`）。
6. **数据不含隐私**，但含你的研究结论——别把 `reports/` 与 `pred_ledger/` 传到公开仓库。

---

## 8. 资源预算（N100 级）

| 指标 | 预算 | 说明 |
|---|---|---|
| 每日增量耗时 | **< 5 秒**（不含网络） | 80 只 × 1 根新 K 线；网络请求是主要成本，限速下 80 次请求约 10–20 秒 |
| 峰值内存 | **< 500MB** | 80 只 × 3000 根面板 float32 ≈ 几十 MB；pandas 开销是主项 |
| 磁盘占用 | **< 2GB** | 行情 CSV ~60MB、parquet 面板 ~40MB、台账每年 ~10 万行、venv ~200MB |
| 空闲 CPU | ~0 | 无守护进程轮询；网页服务空闲时 < 1% |

实测方法：

```bash
# Linux
/usr/bin/time -v .venv/bin/python -m qcore.live --once     # 看 Maximum resident set size
du -sh data/ logs/ reports/
```

```powershell
# Windows
Measure-Command { .venv\Scripts\python.exe -m qcore.live --once }
Get-Process python | Select-Object WorkingSet64
```

超预算时的处理顺序：① 减少股票池（`top_n` 80 → 50）；② 面板用 `float32`/`uint8`（已默认）；
③ 行情缓存改按年分片；④ 别加 Redis/数据库——这个体量用 parquet 就够。

---

## 9. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| 定时任务"成功"但台账没新行 | 非交易日 / 数据未落地 / 入口模块不存在 | 看 `logs/live_*.log` 最后一行；非交易日退出 0 是正常的 |
| 网页打不开 | 只绑了 127.0.0.1 而你在另一台机器访问 | 用反代，或 SSH 端口转发 `ssh -L 8765:127.0.0.1:8765` |
| 权重每天都不变 | 当日无到期样本（h 未满）/ `active` 全 False | 正常现象；连续多日不变才是问题 |
| 内存持续上涨 | 网页服务里缓存了全量面板 | 重启服务；检查是否有全局缓存没设上限 |
| Docker 里时区错、日期差一天 | 缺 `TZ` | 加 `-e TZ=Asia/Shanghai` |
| 同日跑两次导致台账重复 | `--once` 不幂等 | 按 `as_of` 去重后重写；并给入口加"当日已入账则跳过" |

---

## 10. 升级流程

```bash
# 1. 备份不可再生数据（§6）
# 2. 拉新代码
git pull
# 3. 更新依赖（只在 requirements.txt 变化时）
.venv/bin/pip install -r requirements.txt
# 4. 跑测试（截断不变性 + 学习器性质；pytest 是开发依赖，不在 requirements.txt 里）
.venv/bin/pip install pytest && .venv/bin/python -m pytest tests/ -q
# 5. 重启服务
sudo systemctl restart apb-web.service
```

> `data/state/` 里的权重向量带 `S`（信号数）与 `R`（regime 数）维度。若升级改变了信号库规模，
> 旧状态不可直接加载——必须显式迁移或重学，否则维度不匹配会静默产生错位权重。
