# 反向代理与域名部署

> 本文解决：把 APB 放到 `https://apb.example.com` 后面，并且**不把写接口暴露给全网**。

---

## 0. 先说最危险的那件事

本项目的写接口（改账户、跑流水线、导入模型、排训练任务）策略是：

> 设了 `APB_TOKEN` 就校验 token；**没设就只允许内网**。

判断"是不是内网"原来靠 `self.client_address`。**加了反向代理之后，这个值会变成代理的 IP。**

如果 nginx 和容器在同一台机器上，那就是 `127.0.0.1` → `is_private()` 恒为真。

> **后果：任何能访问域名的人，都能改你的账户、触发训练、导入模型。**

这不是理论问题，是必然发生。所以必须做下面三件事。

---

## 1. 必须设置的环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `APB_TOKEN` | **是** | 长随机串，写接口的凭据 |
| `APB_ALLOWED_HOSTS` | **是** | 允许的 Host 头，防 DNS rebinding |
| `APB_TRUSTED_PROXIES` | 建议 | 可信代理的 CIDR，只有它们传来的 `X-Forwarded-For` 才被采信 |
| `APB_CORS_ORIGINS` | 按需 | 前后端不同域时才需要 |
| `APB_PUBLIC_URL` | 可选 | 只用于界面展示 |

### 生成 token

```bash
openssl rand -hex 24          # 或
python -c "import secrets; print(secrets.token_hex(24))"
```

### 示例（docker run）

```bash
docker run -d --name apb \
  -p 127.0.0.1:8765:8765 \
  -e APB_TOKEN='你生成的token' \
  -e APB_ALLOWED_HOSTS='apb.example.com,192.168.250.3:8765' \
  -e APB_TRUSTED_PROXIES='127.0.0.1/32,172.17.0.0/16' \
  -e APB_PUBLIC_URL='https://apb.example.com' \
  -v /usr/local/apps/@appdata/apb/data:/app/data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/ybd156/quant-apb:latest
```

注意 `-p 127.0.0.1:8765:8765` —— **只监听回环**，让代理是唯一的入口。
这样即使有人绕过域名直接打 8765，也打不进来。

---

## 2. nginx 配置

```nginx
server {
    listen 443 ssl http2;
    server_name apb.example.com;

    ssl_certificate     /etc/letsencrypt/live/apb.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/apb.example.com/privkey.pem;

    # 模型导入是 base64 JSON，体积可能几十 MB
    client_max_body_size 64m;

    # 训练/预测接口可能跑几分钟
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:8765;

        # 这三行是「真实 IP 还原」的关键，缺一不可
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket（当前没用到，留着不影响）
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 健康检查不打日志
    location = /api/healthz {
        proxy_pass http://127.0.0.1:8765/api/healthz;
        access_log off;
    }
}

# HTTP 跳 HTTPS
server {
    listen 80;
    server_name apb.example.com;
    return 301 https://$host$request_uri;
}
```

**`proxy_set_header Host $host;` 必须写。** 不写的话 nginx 默认传 `Host: 127.0.0.1:8765`，
会被 `APB_ALLOWED_HOSTS` 拒掉（这正是它该有的行为）。

---

## 3. Caddy 配置（更简单，自动 HTTPS）

```caddyfile
apb.example.com {
    encode gzip

    reverse_proxy 127.0.0.1:8765 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }

    request_body {
        max_size 64MB
    }

    @health path /api/healthz
    log_skip @health
}
```

Caddy 默认就会传 `X-Forwarded-For`，但 **`header_up Host {host}` 建议显式写**。

---

## 4. 验证部署是否正确

### 4.1 界面自检

打开 `设置` 页，或者直接调：

```bash
curl -s https://apb.example.com/api/netguard | python -m json.tool
```

关注这几个字段：

| 字段 | 期望 | 说明 |
|---|---|---|
| `your_ip` | 你的**公网** IP | 如果显示 `127.0.0.1` 或内网 IP，说明 XFF 没传对 |
| `you_are_internal` | **false** | 如果为 `true`，说明反代被当成了内网 → **有安全风险** |
| `host_header` | `apb.example.com` | 应该是域名，不是 `127.0.0.1:8765` |
| `ready_for_public` | **true** | 需要 token + hosts 都配了 |
| `warnings` | 空数组 | 有内容就按提示改 |

### 4.2 写接口必须被拒

```bash
# 不带 token 调写接口，应该返回 403
curl -s -X POST https://apb.example.com/api/account/config \
     -H 'Content-Type: application/json' \
     -d '{"initial_capital":1000000}'
# 期望: {"ok":false,"error":"unauthorized"或"forbidden",...}

# 带正确 token 才通
curl -s -X POST https://apb.example.com/api/account/config \
     -H 'Content-Type: application/json' \
     -H "X-Token: 你的token" \
     -d '{"initial_capital":1000000}'
```

**第一条如果返回 `ok:true`，说明配置有问题，立刻停下来修。**

### 4.3 Host 头防护

```bash
# 伪造 Host，应该 403
curl -s -H 'Host: evil.com' https://你的服务器IP/api/version
# 期望: {"ok":false,"error":"host_not_allowed"}
```

---

## 5. 常见问题

### Q: 页面能打开，但所有写操作都 403

检查 `APB_ALLOWED_HOSTS` 是否包含你的域名。用 `/api/netguard` 看 `host_header` 的实际值。

### Q: `/api/netguard` 里 `your_ip` 是 127.0.0.1

代理没传 `X-Forwarded-For`，或者传了但 `APB_TRUSTED_PROXIES` 没包含代理的 IP。

如果你用 Docker 里的 nginx，代理 IP 通常是 `172.17.0.1`，要加进 `APB_TRUSTED_PROXIES`。

### Q: 上传模型返回 413

`client_max_body_size` 太小。模型 zip 约 70KB，base64 后约 95KB，但如果是蒸馏前的完整模型可能几 MB。设 64m 足够。

### Q: 训练/预测接口超时

`proxy_read_timeout` 默认 60s 不够。训练一个周期约 6 分钟，建议 600s。

### Q: 我不想暴露到公网，只想局域网用

那就**不要设** `APB_ALLOWED_HOSTS`，也不要做反代。直接 `http://192.168.x.x:8765` 就行，
行为和老版本完全一样（只允许内网）。

---

## 6. 安全清单（上线前逐条打勾）

- [ ] `APB_TOKEN` 已设置，且长度 ≥ 24 字符
- [ ] 容器只监听 `127.0.0.1`（`-p 127.0.0.1:8765:8765`）
- [ ] `APB_ALLOWED_HOSTS` 包含实际域名
- [ ] `APB_TRUSTED_PROXIES` 包含代理 IP
- [ ] `/api/netguard` 的 `you_are_internal` 为 `false`
- [ ] 不带 token 的写请求返回 403
- [ ] 伪造 `Host` 头返回 403
- [ ] HTTPS 证书有效（Caddy 自动，nginx 用 certbot）
- [ ] `APB_CORS_ORIGINS` 没有 `*`（如果设了）

---

## 7. 相关代码

| 文件 | 职责 |
|---|---|
| `qcore/net_guard.py` | 真实 IP 还原、Host 校验、CORS 白名单 |
| `webui/server.py` → `_is_internal()` | 写接口免 token 的唯一判断入口 |
| `webui/server.py` → `_guard_request()` | 每个请求的前置检查 |
| `GET /api/netguard` | 配置自检 |
| `GET /api/healthz` | 代理用的健康检查 |
