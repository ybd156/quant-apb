"""
反代与域名支持 —— 把服务安全地暴露到域名后面。

## 为什么不能"直接挂个 nginx 就行"

本项目有一批写接口（改账户、跑流水线、导入模型、排训练任务）采取
「**设了 APB_TOKEN 就校验 token，没设就只允许内网**」的策略。

一旦加了反向代理，`self.client_address[0]` 变成代理的 IP ——
如果代理和容器在同一台机器上，那就是 `127.0.0.1`，`is_private()` 恒为真。

**结果：任何能访问域名的人，都能改你的账户、触发训练、导入模型。**

这是真实的安全洞，不是理论问题。所以必须：

1. **从 `X-Forwarded-For` 还原真实客户端 IP**，但**只信任已知代理**
   （否则攻击者自己伪造 `X-Forwarded-For: 127.0.0.1` 就能绕过）
2. **校验 Host 头**，防 DNS rebinding（攻击者把自己的域名解析到你的内网 IP）
3. **显式的 CORS**，不用 `*`

## 配置（环境变量）

| 变量 | 作用 | 示例 |
|---|---|---|
| `APB_TRUSTED_PROXIES` | 可信代理的 CIDR，逗号分隔 | `127.0.0.1/32,172.17.0.0/16` |
| `APB_ALLOWED_HOSTS` | 允许的 Host 头 | `apb.example.com,192.168.250.3:8765` |
| `APB_CORS_ORIGINS` | 允许的跨域来源 | `https://apb.example.com` |
| `APB_PUBLIC_URL` | 对外地址（只用于展示） | `https://apb.example.com` |

**都不设时行为不变**（只允许内网），所以老部署不会因为升级而失效。

## 判断顺序（关键）

    real_ip = 
      client_address 是可信代理？ → 取 X-Forwarded-For 最右边那个「不是可信代理」的 IP
      否则                        → 直接用 client_address

取最右边而不是最左边，是因为左侧可以被客户端随意伪造。
"""
from __future__ import annotations

import ipaddress
import os
from typing import Dict, List, Optional, Tuple


def _split_env(name: str) -> List[str]:
    raw = os.environ.get(name, "") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]


def _nets(name: str) -> List:
    out = []
    for s in _split_env(name):
        try:
            out.append(ipaddress.ip_network(s, strict=False))
        except Exception:
            continue
    return out


def is_private(ip: str) -> bool:
    """内网/回环/链路本地。"""
    try:
        a = ipaddress.ip_address(ip)
        return a.is_private or a.is_loopback or a.is_link_local
    except Exception:
        return False


def is_trusted_proxy(ip: str) -> bool:
    """
    这个 IP 是不是我们信任的反向代理。
    **只有来自可信代理的 X-Forwarded-For 才会被采信。**
    """
    nets = _nets("APB_TRUSTED_PROXIES")
    if not nets:
        # 没配置时**只信回环**（同机 nginx 的常见情况）。
        # 不能退化成「信任所有内网」—— 局域网里任何人都能伪造 X-Forwarded-For，
        # 那样内网判断就形同虚设。
        try:
            a = ipaddress.ip_address(ip)
            return a.is_loopback
        except Exception:
            return False
    try:
        a = ipaddress.ip_address(ip)
        return any(a in n for n in nets)
    except Exception:
        return False


def real_ip(client_address: str, headers: Dict[str, str]) -> str:
    """
    还原真实客户端 IP。

    优先级：
      1. CF-Connecting-IP  —— 走 Cloudflare 时这是权威值（XFF 里会混进 CF 节点 IP）
      2. X-Forwarded-For   —— 从右往左找第一个不属于可信代理的地址
                              （左侧可被客户端伪造，只有右侧才可信）
      3. client_address    —— 直连时的对端

    前提都是：client_address 必须是可信代理。
    """
    if not client_address or not is_trusted_proxy(client_address):
        return client_address or ""

    # Cloudflare 会给一个权威头，比 XFF 链更可靠（XFF 里会混进 CF 的节点 IP）
    cf = headers.get("CF-Connecting-IP") or headers.get("cf-connecting-ip") or ""
    if cf.strip():
        return cf.strip()

    xff = ""
    for k in ("X-Forwarded-For", "X-Real-IP"):
        v = headers.get(k) or headers.get(k.lower()) or ""
        if v:
            xff = v
            break
    if not xff:
        return client_address

    parts = [p.strip() for p in xff.split(",") if p.strip()]
    for p in reversed(parts):
        if not is_trusted_proxy(p):
            return p
    return parts[0] if parts else client_address


def is_internal_request(client_address: str, headers: Dict[str, str]) -> bool:
    """
    「是否内网请求」—— 写接口的免 token 判断必须用这个，不能用 client_address。
    反代之后 client_address 是 127.0.0.1，直接用会把外网请求当成内网。
    """
    ip = real_ip(client_address, headers)
    return is_private(ip)


def host_allowed(host_header: str) -> Tuple[bool, str]:
    """
    校验 Host 头，防 DNS rebinding。
    没配置 APB_ALLOWED_HOSTS 时放行（保持向后兼容），但会返回提示。
    """
    allowed = _split_env("APB_ALLOWED_HOSTS")
    if not allowed:
        return True, "unconfigured"
    h = (host_header or "").strip().lower()
    if not h:
        return False, "empty_host"
    host_only = h.split(":")[0]
    for a in allowed:
        a = a.lower()
        if h == a or host_only == a.split(":")[0]:
            return True, "matched"
    return False, f"host_not_allowed: {h}"


def cors_headers(origin: str) -> Dict[str, str]:
    """
    返回该 Origin 的 CORS 响应头。不在白名单就不返回 CORS 头（浏览器会自己拦）。
    """
    allowed = _split_env("APB_CORS_ORIGINS")
    if not allowed or not origin:
        return {}
    o = origin.strip().rstrip("/")
    for a in allowed:
        a2 = a.strip().rstrip("/")
        if a2 == "*" or a2.lower() == o.lower():
            return {
                "Access-Control-Allow-Origin": o if a2 != "*" else "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, X-APB-Token, X-Token",
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            }
    return {}


def status() -> Dict:
    """给设置页显示：当前反代配置是否合理。"""
    trusted = _split_env("APB_TRUSTED_PROXIES")
    hosts = _split_env("APB_ALLOWED_HOSTS")
    cors = _split_env("APB_CORS_ORIGINS")
    token_set = bool(os.environ.get("APB_TOKEN", ""))
    warnings = []

    if not token_set:
        warnings.append("未设置 APB_TOKEN —— 暴露到公网前**必须**设置，"
                        "否则写接口只靠内网判断，反代后形同虚设")
    if token_set and len(os.environ.get("APB_TOKEN", "")) < 16:
        warnings.append("APB_TOKEN 短于 16 位，建议用随机长串")
    if not hosts:
        warnings.append("未设 APB_ALLOWED_HOSTS —— 无法防 DNS rebinding，"
                        "建议填你的域名")
    if not trusted:
        warnings.append("未设 APB_TRUSTED_PROXIES —— 默认只信回环代理；"
                        "如果 nginx 在别的容器/机器上，要显式配置")
    if cors and any(c.strip() == "*" for c in cors):
        warnings.append("APB_CORS_ORIGINS 含 * —— 允许任何网站跨域调用你的接口，"
                        "建议改成具体域名")

    return {
        "ok": True,
        "token_set": token_set,
        "trusted_proxies": trusted,
        "allowed_hosts": hosts,
        "cors_origins": cors,
        "public_url": os.environ.get("APB_PUBLIC_URL", ""),
        "warnings": warnings,
        "ready_for_public": bool(token_set and hosts),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(status(), ensure_ascii=False, indent=1))
    print()
    # 自测：模拟反代场景
    cases = [
        ("无代理(公网)", "8.8.8.8", {}),
        ("同机反代", "127.0.0.1", {"X-Forwarded-For": "8.8.8.8"}),
        ("伪造 XFF", "8.8.8.8", {"X-Forwarded-For": "127.0.0.1"}),
        ("双层代理", "127.0.0.1", {"X-Forwarded-For": "8.8.8.8, 10.0.0.5"}),
        ("内网直连", "192.168.1.50", {}),
        ("同机直连", "127.0.0.1", {}),
    ]
    print(f"{'场景':<14}{'client_address':<16}{'还原真实IP':<16}{'算内网?'}")
    for name, ca, hd in cases:
        rip = real_ip(ca, hd)
        print(f"{name:<14}{ca:<16}{rip:<16}{is_internal_request(ca, hd)}")
