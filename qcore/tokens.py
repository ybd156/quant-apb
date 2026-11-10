"""
访问令牌管理：给每台训练机 / 每个外部集成单独发一个 token。

## 为什么需要它

原来只有**一个** `APB_TOKEN`（容器环境变量），所有问题都堆在这一个值上：

  · 部署到别处（别人家的 NAS、云主机）时，容器环境变量不好改
  · 多台训练机共用一个 token，无法单独吊销某一台
  · token 泄露只能整个换掉，所有机器一起改
  · 训练机没配 token 就 `login_required`，模型训好了传不回来（实测踩过）

所以做成**可管理的多令牌**：

  · 每个令牌有名字（"训练机-A"）、前缀（便于识别）、创建时间、最后使用时间
  · 只在创建时返回一次明文，库里只存 **SHA-256 哈希**
  · 可单独吊销，不影响其它机器
  · 与老的 `APB_TOKEN` 环境变量**并存**（向后兼容，老机器不用改）

## 存储

    data/state/tokens.json
    {"tokens": [{"id","name","prefix","hash","created_at","last_used_at",
                 "revoked","note","scopes"}]}

## 安全边界（如实说明）

  · 哈希是裸 SHA-256，不是 bcrypt —— token 本身是 32 字节随机串，
    熵足够高，不需要慢哈希；但**库里拿到哈希无法反推明文**这一点成立。
  · 令牌不设过期时间（训练机可能几个月不联网）。要失效就吊销。
  · `last_used_at` 每次请求更新，但写盘有节流（默认 60 秒），
    避免高频请求把磁盘写满。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
PATH = os.path.join(STATE, "tokens.json")

#: 明文前缀，便于在日志/配置里一眼认出是 APB 令牌
TOKEN_PREFIX = "apb_"
#: 随机部分字节数（32 字节 -> 43 字符 base64url，熵 256 bit）
TOKEN_BYTES = 32
#: last_used_at 写盘节流（秒）
TOUCH_THROTTLE = 60.0

_last_touch_write = 0.0


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _hash(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def _load() -> Dict:
    try:
        with open(PATH, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("tokens"), list):
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"tokens": []}


def _save(d: Dict) -> None:
    os.makedirs(STATE, exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PATH)


def create(name: str = "", note: str = "", scopes: Optional[List[str]] = None) -> Dict:
    """新建令牌。**明文只在这里返回一次**，之后无法再取回。"""
    d = _load()
    plain = TOKEN_PREFIX + secrets.token_urlsafe(TOKEN_BYTES)
    rec = {
        "id": "tk_" + secrets.token_hex(6),
        "name": (str(name).strip() or "未命名")[:60],
        "prefix": plain[:12],
        "hash": _hash(plain),
        "created_at": _now(),
        "last_used_at": None,
        "revoked": False,
        "note": str(note)[:200],
        "scopes": list(scopes or ["sync", "write"]),
    }
    d.setdefault("tokens", []).append(rec)
    _save(d)
    out = dict(rec)
    out.pop("hash", None)
    out["token"] = plain           # 仅此一次
    out["warning"] = "明文只显示这一次，请立刻复制保存"
    return out


def list_tokens(include_revoked: bool = True) -> List[Dict]:
    d = _load()
    out = []
    for t in d.get("tokens", []):
        if not include_revoked and t.get("revoked"):
            continue
        rec = {k: v for k, v in t.items() if k != "hash"}
        out.append(rec)
    return out


def revoke(tid: str) -> bool:
    d = _load()
    hit = False
    for t in d.get("tokens", []):
        if t.get("id") == tid and not t.get("revoked"):
            t["revoked"] = True
            t["revoked_at"] = _now()
            hit = True
    if hit:
        _save(d)
    return hit


def delete(tid: str) -> bool:
    d = _load()
    before = len(d.get("tokens", []))
    d["tokens"] = [t for t in d.get("tokens", []) if t.get("id") != tid]
    if len(d["tokens"]) != before:
        _save(d)
        return True
    return False


def verify(token: str, scope: str = "") -> Tuple[bool, Optional[Dict]]:
    """校验令牌。返回 (是否有效, 记录)。吊销/不存在都返回 False。

    `scope` 非空时要求该令牌含此 scope。
    """
    if not token:
        return False, None
    h = _hash(token)
    d = _load()
    for t in d.get("tokens", []):
        if t.get("revoked"):
            continue
        if not hmac.compare_digest(str(t.get("hash") or ""), h):
            continue
        if scope and scope not in (t.get("scopes") or []):
            return False, t
        _touch(t.get("id"))
        return True, t
    return False, None


def _touch(tid: Optional[str]) -> None:
    """更新 last_used_at（节流写盘）。"""
    global _last_touch_write
    if not tid:
        return
    now = time.time()
    if now - _last_touch_write < TOUCH_THROTTLE:
        return
    _last_touch_write = now
    try:
        d = _load()
        for t in d.get("tokens", []):
            if t.get("id") == tid:
                t["last_used_at"] = _now()
                break
        _save(d)
    except Exception:  # noqa: BLE001
        pass


def summary() -> Dict:
    toks = list_tokens()
    live = [t for t in toks if not t.get("revoked")]
    return {"ok": True, "total": len(toks), "active": len(live),
            "revoked": len(toks) - len(live),
            "tokens": toks,
            "note": "明文只在创建时显示一次；这里只存 SHA-256 哈希，无法反推"}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if "--create" in a:
        i = a.index("--create")
        nm = a[i + 1] if len(a) > i + 1 else "cli"
        print(json.dumps(create(name=nm), ensure_ascii=False, indent=1))
    elif "--revoke" in a:
        i = a.index("--revoke")
        print("revoked" if revoke(a[i + 1]) else "not found")
    elif "--delete" in a:
        i = a.index("--delete")
        print("deleted" if delete(a[i + 1]) else "not found")
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
