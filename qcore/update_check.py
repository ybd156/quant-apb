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
import urllib.parse
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
        # **必须分页**：GHCR 的 /tags/list 默认只给 100 个（按字典序）。
        # 版本数超过 100 后，最新的 0.28.x 会落在第 2 页 ——
        # 不分页时 check() 算出的「最新版本」停在 0.27.0，
        # NAS 因此永远显示「已是最新」，训练机也就永远不更新。
        # （实测：119 个 tag，前 100 个最高只到 0.27.0，0.28.0~0.28.2 在第 2 页。）
        tags: List[str] = []
        last = None
        for _page in range(12):                 # 上限 1200 个 tag，够用很久
            url = f"https://{registry}/v2/{REPO}/tags/list?n=100"
            if last:
                url += f"&last={urllib.parse.quote(str(last), safe='')}"
            d = _get_json(url, headers={"Authorization": f"Bearer {token}"})
            page_tags = list(d.get("tags") or [])
            if not page_tags:
                break
            tags.extend(page_tags)
            if len(page_tags) < 100:
                break
            last = page_tags[-1]
        return True, tags, ""
    except Exception as e:  # noqa: BLE001
        return False, [], f"{type(e).__name__}: {e}"[:80]


def compare_versions(a: str, b: str) -> int:
    """语义化比较：a>b 返回 1，相等 0，a<b 返回 -1。无法解析按字符串比。"""
    ka, kb = _ver_key(a or ""), _ver_key(b or "")
    if ka and kb:
        return (ka > kb) - (ka < kb)
    return (a > b) - (a < b)


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
        # 不在第一个源就 break：镜像源（如 ghcr.nju.edu.cn）可能滞后，
        # 拿它当唯一真相会把「镜像没同步」误报成「没有新版本」。
        # 继续问下一个源，最后取所有源里的最大 semver。

    cur_key = _ver_key(cur)
    has_update = bool(best_key and cur_key and best_key > cur_key)
    # 远端比本地旧时明确标记，避免 UI 把降级当升级推销
    behind = bool(best_key and cur_key and best_key < cur_key)

    out = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current": cur,
        "latest_version": best_tag,
        "registry": used,
        "has_update": has_update,
        "behind_remote": behind,
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
