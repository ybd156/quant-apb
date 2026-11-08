# 飞牛 fnOS 部署指南

> ⚠️ **构建全部在 GitHub Actions 完成，本地不构建、不产出任何二进制。**
> 所有安装包（`.fpk` 与 Docker 镜像）都从 [Releases](https://github.com/ybd156/quant-apb/releases) 下载。
> 本地只需要 `git clone` 读源码，不需要装 Docker、不需要 Python 打包工具。

---

## 方式一：fpk 原生安装包（最省事）

1. 打开 [Releases](https://github.com/ybd156/quant-apb/releases) 页面
2. 按设备架构下载：
   - x86_64（绝大多数 N100/NAS）→ `apb_<version>_x86.fpk`
   - ARM（部分 ARM NAS）→ `apb_<version>_arm.fpk`
3. fnOS → **应用中心 → 手动安装** → 选择该 `.fpk`
4. 安装完成后自动拉取镜像并启动，浏览器打开 `http://<NAS-IP>:8765`

fpk 内部结构（源码在 `deploy/fnos/fpk/`，构建脚本 `scripts/build-fpk.sh`）：

```
manifest            元数据（appname/version/platform/checksum）
app.tgz             应用载荷（qcore + webui + VERSION）
cmd/service-setup   服务控制（Docker 后端，宿主机无需 Python）
config/privilege    运行权限
config/resource     端口映射 + 数据共享目录
ui/config           桌面入口
ui/images/256.png   图标
APB.sc              端口转发
```

> fpk 采用 **Docker 后端**：安装包很小（约 200 KB），真正运行的是 GHCR 上的镜像。
> 好处是宿主机不需要 Python/numpy/pandas，也不会因为系统 Python 版本不同而崩。

---

## 方式二：Docker Compose

> fnOS（飞牛私有云）基于 Debian，自带 Docker 应用中心。
> 本项目以**容器**方式交付，amd64 / arm64 都有镜像。

---

## 三、准备（以下为 Docker 方式）

| 项 | 要求 |
|---|---|
| 系统 | fnOS（任意版本，需已启用 Docker） |
| 架构 | x86_64 或 arm64 均可 |
| 内存 | ≥ 1 GB 可用（实测运行约 300 MB） |
| 磁盘 | ≥ 2 GB（行情缓存约 300 MB，随股票池扩大而增长） |
| 网络 | 需能访问 `push2his.eastmoney.com` 等行情接口 |

---

### 3.1 安装（Docker Compose）

### 1. 建目录

fnOS 的存储卷通常在 `/vol1`、`/vol2`。选一个空间大的：

```bash
mkdir -p /vol1/docker/apb
cd /vol1/docker/apb
```

### 2. 放编排文件

从 Release 页下载 `docker-compose.yml`，或直接创建：

```yaml
services:
  apb:
    image: ghcr.io/ybd156/quant-apb:latest
    container_name: apb
    restart: unless-stopped
    ports:
      - "8765:8765"
    environment:
      TZ: Asia/Shanghai
      APB_TOKEN: "换成你自己的随机串"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

### 3. 启动

```bash
docker compose up -d
docker compose logs -f --tail=50
```

浏览器打开 `http://<NAS的IP>:8765`。

### 4. 首次初始化

首次启动时**没有数据**，界面会显示"未就绪"。在容器内跑一次全量构建：

```bash
docker exec -it apb python -m qcore.pipeline --pool 300 --start 2013-01-01
```

约 1-2 分钟（N100 上约 3-4 分钟）。完成后刷新页面即可看到净值、信号、台账。

---

### 3.2 安装（图形界面）

1. fnOS → **应用中心** → **Docker** → **Compose**
2. 新建项目，粘贴上面的 `docker-compose.yml`
3. 工作目录选 `/vol1/docker/apb`
4. 启动

---

## 四、每日自动更新

fnOS 的「任务计划」里加一条，每天 **15:40**（A股 15:00 收盘后）执行：

```bash
docker exec apb python -m qcore.live --once
```

若你的 NAS 时钟不是东八区，先把容器 `TZ` 设成 `Asia/Shanghai`，否则"今天"会算错。

> 增量更新实测 **3-15 秒**（视股票池大小），不会拖慢 NAS。

---

## 五、数据与备份

**必须备份**（不可再生）：

| 路径 | 内容 | 丢了会怎样 |
|---|---|---|
| `data/state/` | 权重状态、UI 产物 | 学习记忆丢失，需重跑全量 |
| `data/pred_ledger/` | 预测台账 | 历史对账无法重建 |
| `data/claims/` | 判断台账 | 我的命中率统计归零 |

**可删可重建**：`data/raw/`（行情缓存）、`reports/`（报告）。

备份命令（fnOS 计划任务）：

```bash
tar czf /vol1/backup/apb-$(date +%F).tar.gz \
  -C /vol1/docker/apb data/state data/pred_ledger data/claims
```

---

## 六、升级

```bash
cd /vol1/docker/apb
docker compose pull
docker compose up -d
```

数据在 volume 里，升级不丢。**升级前建议先备份**（见上）。

---

## 七、安全

- 默认只映射到 NAS 内网端口。**不要直接把 8765 暴露到公网**。
- 需要外网访问时，用 fnOS 自带的**反向代理 + HTTPS**，并在反代上加访问控制。
- `APB_TOKEN` 不设时，`/api/run`（触发更新）会**一律拒绝** —— 这是安全的默认。
- 容器以非 root 用户（uid 10001）运行。

---

## 八、排障

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面显示"未就绪" | 没跑过 pipeline | `docker exec apb python -m qcore.pipeline --pool 300` |
| 数据一直是旧的 | 任务计划没生效 | 手动跑 `--once` 看日志 |
| 容器反复重启 | 端口冲突 / 权限 | `docker logs apb`；确认 8765 未被占用 |
| 中文乱码 | 缺字体 | 镜像已内置回退，图表中文可能显示为方块；装 `fonts-wqy-microhei` 可解 |
| 内存占用过高 | 股票池太大 | 把 `--pool` 从 300 降到 150 |

---

## 九、资源实测

| 场景 | 耗时 | 内存 |
|---|---|---|
| 全量构建（300 只 / 13 年） | 45 秒（本机）/ 约 3-4 分钟（N100） | 峰值约 600 MB |
| 每日增量 | 3-15 秒 | 约 300 MB |
| Web UI 常驻 | — | 约 60 MB |
| 磁盘（300 只行情 + 状态） | 约 350 MB | — |
