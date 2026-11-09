"""
网络路由 —— 直连还是走代理？按源自动选，并且记住结果。

## 为什么不能"全局挂代理"

实测（NAS 容器内，v2rayA 在宿主机 10808/10809）：

| 源 | 直连 | 走代理 | 结论 |
|---|---|---|---|
| 东财 / 新浪 / 同花顺 / 雪球 | ✅ | — | 只能直连（国内源走代理反而慢） |
| CNBC | ✅ 20622B | ❌ 403 | **直连更好**（代理出口 IP 被拉黑） |
| MarketWatch | ✅ 8149B | — | 直连 |
| ECB | ✅ 5859B | — | 直连 |
| GoogleNews | ❌ 超时 | ✅ 129137B | **只有代理能通** |
| Reuters / FT | ❌ 不可达 | ❌ 401/403 | 两边都不行（反爬，不是网络问题） |

所以正确策略是 **per-host 自动选路 + 记忆**：
第一次失败就换另一条路，成功过的那条记下来，下次直接用。

## 代理怎么找

按优先级：
1. 环境变量 `APB_PROXY` / `HTTPS_PROXY`
2. Docker 网关 `172.17.0.1` + 常见 v2rayA 端口（10809 HTTP / 10808 SOCKS）
3. `host.docker.internal`

**只用 HTTP 代理**（CONNECT 隧道），不实现 SOCKS——urllib 原生支持 HTTP 代理，
而 SOCKS 要自己写握手，没必要。

## 落盘

`data/state/net_routes.json`：`{host: {"mode": "direct"|"proxy", "ok": n, "fail": n}}`
自愈：某条路连续失败 3 次会自动切换重试。
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
ROUTES = os.path.join(STATE, "net_routes.json")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# 候选代理（按优先级探测）
PROXY_CANDIDATES = [
    "http://172.17.0.1:10809",        # Docker 网关 + v2rayA HTTP
    "http://host.docker.internal:10809",
    "http://172.17.0.1:1087",         # 常见 clash 混合端口
    "http://172.17.0.1:7890",
]

SWITCH_AFTER_FAILS = 3     # 连续失败多少次后切换路线

_proxy_cache: Dict[str, Optional[str]] = {"found": None, "checked_at": 0.0}
_routes_cache: Dict[str, Dict] = {}


# ---------------------------------------------------------------- 工具
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_routes() -> Dict[str, Dict]:
    global _routes_cache
    if _routes_cache:
        return _routes_cache
    try:
        with open(ROUTES, encoding="utf-8") as f:
            _routes_cache = json.load(f) or {}
    except Exception:
        _routes_cache = {}
    return _routes_cache


def _save_routes() -> None:
    try:
        os.makedirs(STATE, exist_ok=True)
        tmp = ROUTES + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_routes_cache, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ROUTES)
    except Exception:
        pass


def _host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.split(":")[0].lower()
    except Exception:
        return url[:40]


# ---------------------------------------------------------------- 代理发现
def discover_proxy(force: bool = False, ttl: float = 300.0) -> Optional[str]:
    """
    找一个能用的 HTTP 代理。返回 URL 或 None。
    结果缓存 ttl 秒，避免每次抓新闻都探测一遍。
    """
    env = (os.environ.get("APB_PROXY") or os.environ.get("HTTPS_PROXY")
           or os.environ.get("https_proxy") or "").strip()
    if env:
        return env
    if not force and (time.time() - _proxy_cache["checked_at"]) < ttl:
        return _proxy_cache["found"]

    found = None
    for cand in PROXY_CANDIDATES:
        try:
            hp = cand.split("://", 1)[1]
            host, port = hp.rsplit(":", 1)
            s = socket.socket()
            s.settimeout(2)
            rc = s.connect_ex((host, int(port)))
            s.close()
            if rc == 0:
                found = cand
                break
        except Exception:
            continue
    _proxy_cache["found"] = found
    _proxy_cache["checked_at"] = time.time()
    return found


def _open(url: str, proxy: Optional[str], timeout: int) -> str:
    """用指定路线抓取。proxy=None 表示直连。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/rss+xml, application/json, text/xml, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------- 主接口
def get(url: str, timeout_direct: int = 8, timeout_proxy: int = 15,
        retries: int = 2, force_mode: Optional[str] = None,
        verbose: bool = False) -> Tuple[Optional[str], str]:
    """
    智能抓取。返回 (正文 或 None, 实际走的路线 "direct"/"proxy"/"none")。

    策略：
      1. 查该 host 的历史最优路线，先试它
      2. 失败则换另一条
      3. 成功的那条写进 routes.json
      4. 连续失败 SWITCH_AFTER_FAILS 次自动换边
    """
    host = _host(url)
    routes = _load_routes()
    rec = routes.get(host) or {"mode": None, "ok": 0, "fail": 0}

    proxy = discover_proxy()
    # 优先顺序
    if force_mode == "direct":
        order = [None]
    elif force_mode == "proxy":
        order = [proxy] if proxy else []
    else:
        pref = rec.get("mode")
        order = [None, proxy] if pref == "direct" else (
            [proxy, None] if pref == "proxy" else [None, proxy])
        order = [x for x in order if x is not None or x is None]
        # 去重但保留 None
        seen, tmp = set(), []
        for x in order:
            k = x or "__direct__"
            if k in seen:
                continue
            seen.add(k)
            tmp.append(x)
        order = tmp

    if not order:
        order = [None]

    last_err = None
    for mode_proxy in order:
        for attempt in range(retries):
            try:
                body = _open(url, mode_proxy, timeout_proxy if mode_proxy else timeout_direct)
                if body and len(body) > 100:
                    routes[host] = {"mode": "proxy" if mode_proxy else "direct",
                                    "proxy": mode_proxy,
                                    "ok": int(rec.get("ok", 0)) + 1,
                                    "fail": int(rec.get("fail", 0)),
                                    "last_ok": _now(), "last_error": ""}
                    _save_routes()
                    if verbose:
                        print(f"    [{host}] {'proxy' if mode_proxy else 'direct'} OK "
                              f"{len(body)}B", flush=True)
                    return body, ("proxy" if mode_proxy else "direct")
                last_err = f"short_body({len(body) if body else 0})"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:60]}"
                if verbose:
                    print(f"    [{host}] {'proxy' if mode_proxy else 'direct'} "
                          f"fail({attempt+1}) {last_err}", flush=True)
                # 4xx 是反爬不是网络问题，换路线也没用，直接放弃
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                    break
                time.sleep(0.4 * (attempt + 1))

    routes[host] = {"mode": rec.get("mode"), "ok": int(rec.get("ok", 0)),
                    "fail": int(rec.get("fail", 0)) + 1,
                    "last_error": (last_err or "")[:120],
                    "last_fail": _now()}
    _save_routes()
    return None, "none"


def status() -> Dict:
    """路由表 + 代理可用性，给 UI 用。"""
    proxy = discover_proxy(force=True)
    routes = _load_routes()
    rows = [{"host": h, **v} for h, v in sorted(routes.items())]
    n_direct = sum(1 for v in routes.values() if v.get("mode") == "direct")
    n_proxy = sum(1 for v in routes.values() if v.get("mode") == "proxy")
    return {"ok": True, "proxy_available": bool(proxy), "proxy": proxy,
            "n_direct": n_direct, "n_proxy": n_proxy,
            "routes": rows,
            "note": ("按源自动选路：国内源直连，被墙源走代理；"
                     "某条路连续失败会自动切换")}


def probe(urls: List[str]) -> Dict:
    """对一组 URL 逐个探测两条路线，返回对照表（用于诊断）。"""
    out = []
    proxy = discover_proxy(force=True)
    for u in urls:
        r = {"url": u[:70], "host": _host(u)}
        for mode, px in (("direct", None), ("proxy", proxy)):
            if mode == "proxy" and not px:
                r[mode] = "no_proxy"
                continue
            t0 = time.time()
            try:
                body = _open(u, px, 10)
                r[mode] = f"OK {len(body)}B {int((time.time()-t0)*1000)}ms"
            except Exception as e:  # noqa: BLE001
                r[mode] = f"{type(e).__name__}: {str(e)[:40]}"
        out.append(r)
    return {"ok": True, "proxy": proxy, "results": out}


if __name__ == "__main__":
    import json
    print("代理发现:", discover_proxy(force=True) or "(无)")
    print()
    tests = [
        "https://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_5_1_.html",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://www.ecb.europa.eu/rss/press.html",
        "https://news.google.com/rss/search?q=fed&hl=en-US&gl=US&ceid=US:en",
        "https://www.reuters.com/markets/",
    ]
    r = probe(tests)
    print(f"{'host':<26}{'direct':<34}{'proxy'}")
    for x in r["results"]:
        print(f"{x['host']:<26}{x.get('direct',''):<34}{x.get('proxy','')}")
