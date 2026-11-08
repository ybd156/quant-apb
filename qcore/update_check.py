"""
自动更新检查 —— 查远端最新版本，写 `data/state/update.json` 供 UI 显示。

## 为什么要单独一个模块

容器内**没有 docker 权限**（安全设计），所以不能自己更新自己。
本模块只做"查有没有新版"，真正的更新由宿主机脚本 `cmd/update` 执行。

## 查哪里

国内网络经常解析不了 `ghcr.io`（实测），所以按顺序探测：

    1. ghcr.nju.edu.cn（南大镜像，国内可达）
    2. ghcr.io（官方，海外可达）

GHCR 的 registry API 即使公开包也要先换 token：

    GET  /token?scope=repository:<repo>:pull   -> {"token": ...}
    GET  /v2/<repo>/tags/list                  (Authorization: Bearer <token>)

## 版本比较

远端 tag 形如 `0.0.6` / `latest`。取所有**语义化版本**的 tag，比大小，取最大。
`latest` 不参与比较（它只是个别名，可能是旧的）。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
REPO = os.environ.get("APB_IMAGE_REPO", "ybd156/quant-apb")
REGISTRIES = [h.strip() for h in
              os.environ.get("APB_REGISTRIES", "ghcr.nju.edu.cn,ghcr.io").split(",") if h.strip()]
UA = {"User-Agent": "apb-updater"}

_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def local_version() -> str:
    try:
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _get_json(url: str, headers: Optional[Dict] = None, timeout: int = 12):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _tags_from(registry: str) -> Tuple[bool, List[str], str]:
    """从一个 registry 取 tag 列表。返回 (ok, tags, reason)。"""
    token = None
    # 镜像源的 token 服务名可能是自己，也可能是 ghcr.io —— 两种都试
    for svc in (registry, "ghcr.io"):
        try:
            tok = _get_json(f"https://{registry}/token?scope=repository:{REPO}:pull"
                            f"&service={svc}")
            token = tok.get("token") or tok.get("access_token")
            if token:
                break
        except Exception:
            continue
    try:
        if not token:
            return False, [], "no_token"
        d = _get_json(f"https://{registry}/v2/{REPO}/tags/list",
                      headers={"Authorization": f"Bearer {token}"})
        return True, list(d.get("tags") or []), ""
    except Exception as e:  # noqa: BLE001
        return False, [], f"{type(e).__name__}: {e}"[:80]


def _ver_key(tag: str):
    m = _SEMVER.match(tag)
    return tuple(int(x) for x in m.groups()) if m else None


def check(write: bool = True) -> Dict:
    cur = local_version()
    attempts = []
    best_tag, best_key, used = None, None, None

    for reg in REGISTRIES:
        ok, tags, reason = _tags_from(reg)
        attempts.append({"registry": reg, "ok": ok, "n_tags": len(tags), "reason": reason})
        if not ok:
            continue
        for t in tags:
            k = _ver_key(t)
            if k and (best_key is None or k > best_key):
                best_key, best_tag, used = k, t.lstrip("v"), reg
        if best_tag:
            break        # 拿到就不必再试下一个源

    cur_key = _ver_key(cur)
    has_update = bool(best_key and cur_key and best_key > cur_key)

    out = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current": cur,
        "latest_version": best_tag,
        "registry": used,
        "has_update": has_update,
        "attempts": attempts,
        "command": "docker compose pull && docker compose up -d",
        "note": ("检测到新版本，可在宿主机执行上面的命令升级"
                 if has_update else "已是最新版本"),
    }
    if write:
        os.makedirs(STATE_DIR, exist_ok=True)
        path = os.path.join(STATE_DIR, "update.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    return out


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=1))
