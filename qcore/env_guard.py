"""更新流程的环境变量保护 —— 防止重建容器时丢掉鉴权配置。

## 踩过的坑

用户截图警告「任何人都能看到持仓和策略」，查下来是真的：
容器里没有 APB_PASSWORD，服务完全公开。

根因链：
    1. 某次重建容器时环境变量没带全
    2. 更新流程从旧容器 inspect 读 env 再传给新容器
       —— 旧容器本身就缺，重建后依然缺
    3. **一个丢失会被永久继承**，而且没有任何提示

这是最危险的一类 bug：不报错、不告警，服务照常跑，
只是突然变成公开的。

## 这个模块做什么

1. `capture(container)` —— 重建前把关键环境变量存盘
   （存 data/state/env_snapshot.json，跨容器重建可读）
2. `restore(container_cfg)` —— 重建时合并：快照里的关键变量 + 旧容器现有变量
3. `verify(container)` —— 重建后校验，缺了关键变量就返回告警

关键变量（丢了会出安全问题或功能异常）：
    APB_PASSWORD     登录密码（丢了 = 完全公开）
    APB_TOKEN        机器 token
    APB_ALLOWED_HOSTS  域名白名单
    APB_TRUSTED_PROXIES 反代信任
    APB_PUBLIC_URL   对外地址
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.path.join(ROOT, "data", "state", "env_snapshot.json")

CRITICAL = (
    "APB_PASSWORD", "APB_TOKEN", "APB_ALLOWED_HOSTS",
    "APB_TRUSTED_PROXIES", "APB_PUBLIC_URL",
)

# 其它需要保留的（不含密钥，但影响行为）
OPTIONAL = (
    "APB_HOST", "APB_PORT", "APB_MARKET", "APB_STATE_DIR",
    "APB_CONTAINER_NAME", "APB_IMAGE", "APB_UPDATE_INTERVAL",
    "APB_MARKET_INTERVAL", "APB_TICK_INTERVAL", "APB_LIVE_MAX_AGE",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def capture(env: List[str]) -> Dict:
    """把当前容器的环境变量存一份快照。"""
    d = {}
    for item in (env or []):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        d[k] = v
    keep = {k: v for k, v in d.items() if k in CRITICAL or k in OPTIONAL}
    snap = {"captured_at": _now(), "env": keep,
            "n_total": len(d), "n_kept": len(keep)}
    try:
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        tmp = SNAPSHOT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SNAPSHOT)
    except Exception:
        pass
    return snap


def load_snapshot() -> Dict:
    try:
        with open(SNAPSHOT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def merge(old_env: List[str]) -> List[str]:
    """
    合并环境变量：快照里的关键变量优先补上，旧容器已有的保持。

    返回可直接传给 docker create 的 "K=V" 列表。
    同时会把合并后的结果写回快照（保持最新）。
    """
    current = {}
    for item in (old_env or []):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        current[k] = v

    snap = load_snapshot().get("env") or {}
    # 旧的优先（更可信，是容器实际在用的），快照补缺
    merged = dict(snap)
    merged.update(current)

    # 环境里还有、但快照没有的关键变量（比如刚手动加的），也记下来
    for k, v in current.items():
        if k in CRITICAL:
            merged[k] = v

    capture([f"{k}={v}" for k, v in merged.items()])
    return [f"{k}={v}" for k, v in merged.items()]


def verify(env: List[str]) -> Dict:
    """
    重建后校验关键变量是否还在。

    返回 {"ok": bool, "missing": [...], "warn": str}
    """
    have = set()
    for item in (env or []):
        if "=" in item:
            have.add(item.split("=", 1)[0])

    missing = [k for k in CRITICAL if k not in have]
    # APB_PASSWORD 和 APB_TOKEN 至少要有其一，否则服务完全裸奔
    has_login = ("APB_PASSWORD" in have) or ("APB_TOKEN" in have)
    warn = ""
    if not has_login:
        warn = ("严重：[!] 重建后的容器既没有 APB_PASSWORD 也没有 APB_TOKEN "
                "—— 服务对外完全公开，任何人都能看到持仓和策略。"
                "请用 docker run 显式带上 -e APB_PASSWORD=... 重建。")
    elif missing:
        warn = f"提示：以下变量没带过来 {missing}（不影响登录，但可能影响反代/白名单）"
    return {"ok": has_login, "missing": missing, "warn": warn}
