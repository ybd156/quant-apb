"""
授权配对（授权链接 + 授权码）—— 训练机上一步完成配置。

## 解决什么问题

现在训练机要手工填两样东西：NAS 地址 + 一个令牌。用户得：

1. 在 NAS「设置 → 访问令牌」新建一个（明文只显示一次）；
2. 抄下地址和长串令牌；
3. 到训练机上双击 `APB.bat auto` 再粘进去。

地址抄错、令牌少一位、令牌过期忘了 —— 都是"配不上"但看不出原因。

## 这个模块做什么

NAS 侧生成一对 **授权链接 + 授权码**（短期、一次性）：

    https://apb.example.com/#pair=K7M2-QP4R          ← 授权链接
    K7M2-QP4R                                        ← 授权码（可手抄）

训练机侧：

    python tools/train_machine_auto.py --pair "https://...#pair=K7M2-QP4R"
    python tools/train_machine_auto.py --pair K7M2-QP4R

一次兑换后：**配置写好 + 令牌自动签发 + 守护启动**。码用一次即作废。

## 安全边界（写清楚，别当它是"密码"）

- 码是**一次性**的：`redeem()` 成功即标记 `used_at`，第二次直接拒绝；
- 有 **TTL**（默认 30 分钟），过期即不可用；
- 兑换成功才签发令牌，令牌仍是 32 字节随机串，可以单独吊销；
- 码本身**不含任何密钥**（它是索引，不是凭据）——但要按凭据对待：
  谁拿到未过期的码，谁就能换到一个 `sync+write` 令牌。所以别发到群里；
- 兑换接口有失败次数限制（同一 IP 每分钟 10 次失败即拒），防暴力猜码；
- 码用 `secrets` 生成、去掉了易混字符（0/O、1/I/L），长度 8 → 30^8 ≈ 6.5e11 组合，
  配合 TTL + 单次使用 + 限速，猜中概率可忽略。
"""
from __future__ import annotations

import json
import os
import secrets
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
PAIR_FILE = os.path.join(STATE, "pair_codes.json")

#: 去掉易混字符的字母表（0/O/1/I/L 全部剔除）
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LEN = 8
DEFAULT_TTL_MIN = 30
#: 同一 IP 每分钟最多失败次数
MAX_FAILS_PER_MIN = 10


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _load() -> Dict:
    try:
        with open(PAIR_FILE, encoding="utf-8") as f:
            d = json.load(f) or {}
    except Exception:  # noqa: BLE001
        d = {}
    d.setdefault("codes", [])
    return d


def _save(d: Dict) -> None:
    os.makedirs(STATE, exist_ok=True)
    tmp = PAIR_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, PAIR_FILE)


def _new_code() -> str:
    raw = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LEN))
    return f"{raw[:4]}-{raw[4:]}"          # 便于手抄/口述


def normalize(code: str) -> str:
    """把用户给的东西归一成 `XXXX-XXXX`（容忍小写、空格、缺横线）。"""
    s = str(code or "").strip().upper().replace(" ", "")
    s = "".join(ch for ch in s if ch in ALPHABET + "-")
    s = s.replace("-", "")
    if len(s) != CODE_LEN:
        return ""
    return f"{s[:4]}-{s[4:]}"


def parse_link(text: str) -> Dict:
    """从授权链接或裸码里取出 nas + code。

    支持的形态：
        https://nas.example.com/#pair=K7M2-QP4R
        https://nas.example.com/?pair=K7M2-QP4R
        apb://pair?nas=https://nas.example.com&code=K7M2-QP4R
        K7M2-QP4R
    """
    t = str(text or "").strip()
    if not t:
        return {"ok": False, "reason": "empty"}

    nas, code = "", ""

    # apb://pair?nas=..&code=..
    if t.lower().startswith("apb://"):
        try:
            from urllib.parse import parse_qs, urlparse
            u = urlparse(t)
            q = parse_qs(u.query or "")
            nas = (q.get("nas") or [""])[0]
            code = (q.get("code") or [""])[0]
            if not nas and u.path.strip("/"):
                nas = (q.get("nas") or [""])[0]
        except Exception:  # noqa: BLE001
            pass

    # https://host/#pair=CODE 或 ?pair=CODE
    if not code and ("://" in t):
        try:
            from urllib.parse import parse_qs, urlparse
            u = urlparse(t)
            q = parse_qs(u.query or "")
            code = (q.get("pair") or q.get("code") or [""])[0]
            frag = parse_qs((u.fragment or "").replace("#", ""))
            code = code or (frag.get("pair") or frag.get("code") or [""])[0]
            if code:
                nas = f"{u.scheme}://{u.netloc}"
        except Exception:  # noqa: BLE001
            pass

    if not code:
        code = t
    code = normalize(code)
    if not code:
        return {"ok": False, "reason": "bad_code", "hint": "授权码应为 8 位（可带横线）"}
    nas = (nas or "").rstrip("/")
    return {"ok": True, "nas": nas, "code": code}


def create(nas: str = "", ttl_min: int = DEFAULT_TTL_MIN,
           label: str = "", scopes: Optional[List[str]] = None) -> Dict:
    """生成一个授权码 + 授权链接。**只返回这一次**，之后只能看到状态。"""
    d = _load()
    now = time.time()
    # 顺手清理过期且未使用的（保留最近 50 条历史）
    d["codes"] = [c for c in d["codes"]
                  if not (c.get("used_at") is None
                          and float(c.get("expires_ts") or 0) < now)][-50:]

    code = _new_code()
    while any(c.get("code") == code for c in d["codes"]):
        code = _new_code()

    rec = {"code": code, "created_at": _now(), "expires_ts": now + int(ttl_min) * 60,
           "expires_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
               now + int(ttl_min) * 60)),
           "label": str(label)[:60], "scopes": list(scopes or ["sync", "write"]),
           "used_at": None, "token_id": None, "nas": str(nas or "").rstrip("/")}
    d["codes"].append(rec)
    _save(d)

    nas_ = rec["nas"]
    bare = code.replace("-", "")
    links = {"fragment": f"{nas_}/#pair={code}" if nas_ else "",
             "query": f"{nas_}/?pair={code}" if nas_ else "",
             "apb": (f"apb://pair?nas={nas_}&code={code}" if nas_ else f"apb://pair?code={code}")}
    return {"ok": True, "code": code, "code_bare": bare,
            "expires_at": rec["expires_at"], "ttl_min": int(ttl_min),
            "links": links,
            "primary_link": links["fragment"] or ("apb://pair?code=" + code),
            "cli": f'python tools/train_machine_auto.py --pair "{links["fragment"] or code}"',
            "note": "一次性：兑换后立即作废；到期未用自动清理。别发到公开频道。"}


def _fails() -> Dict[str, List[float]]:
    return {}


_FAILS: Dict[str, List[float]] = {}


def _rate_ok(ip: str) -> bool:
    now = time.time()
    lst = [t for t in _FAILS.get(ip, []) if now - t < 60]
    _FAILS[ip] = lst
    return len(lst) < MAX_FAILS_PER_MIN


def _rate_fail(ip: str) -> None:
    _FAILS.setdefault(ip, []).append(time.time())


def redeem(code: str, ip: str = "", nas: str = "") -> Dict:
    """兑换授权码 → 签发令牌。**一次性**：成功即标记 used。"""
    if not _rate_ok(ip or "?"):
        return {"ok": False, "reason": "rate_limited",
                "hint": f"同一地址每分钟最多失败 {MAX_FAILS_PER_MIN} 次"}

    c = normalize(code)
    if not c:
        _rate_fail(ip or "?")
        return {"ok": False, "reason": "bad_code",
                "hint": "授权码应为 8 位（形如 K7M2-QP4R）"}

    d = _load()
    rec = next((x for x in d["codes"] if x.get("code") == c), None)
    if rec is None:
        _rate_fail(ip or "?")
        return {"ok": False, "reason": "unknown_code",
                "hint": "码不存在（拼写错误，或已被清理）"}
    if rec.get("used_at"):
        return {"ok": False, "reason": "already_used",
                "used_at": rec["used_at"],
                "hint": "这个码已经用过了 —— 一次性。请在 NAS 上重新生成一个。"}
    if float(rec.get("expires_ts") or 0) < time.time():
        return {"ok": False, "reason": "expired", "expires_at": rec.get("expires_at"),
                "hint": "码已过期。请在 NAS 上重新生成一个。"}

    try:
        from . import tokens as TK
        t = TK.create(name=f"训练机（配对 {c}）",
                      note=f"由授权码 {c} 于 {_now()} 兑换；来源 {ip or '?'}",
                      scopes=rec.get("scopes") or ["sync", "write"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "token_create_failed",
                "detail": f"{type(e).__name__}: {str(e)[:80]}"}

    rec["used_at"] = _now()
    rec["token_id"] = t.get("id")
    rec["used_from"] = str(ip or "")[:64]
    _save(d)
    return {"ok": True, "nas": (nas or rec.get("nas") or "").rstrip("/"),
            "token": t.get("token"), "token_id": t.get("id"),
            "token_prefix": t.get("prefix"),
            "warning": "令牌明文只返回这一次；训练机应立刻写进 apb_defaults.json",
            "redeemed_at": rec["used_at"]}


def list_codes(include_used: bool = False, limit: int = 20) -> Dict:
    d = _load()
    rows = []
    now = time.time()
    for x in reversed(d["codes"]):
        if not include_used and x.get("used_at"):
            continue
        expired = (not x.get("used_at")) and float(x.get("expires_ts") or 0) < now
        rows.append({"code": x.get("code"), "label": x.get("label"),
                     "created_at": x.get("created_at"), "expires_at": x.get("expires_at"),
                     "used_at": x.get("used_at"), "expired": expired,
                     "state": ("used" if x.get("used_at") else
                               "expired" if expired else "active")})
        if len(rows) >= limit:
            break
    return {"ok": True, "n": len(rows), "rows": rows,
            "active": sum(1 for r in rows if r["state"] == "active")}


def summary() -> Dict:
    d = _load()
    now = time.time()
    active = [c for c in d["codes"]
              if not c.get("used_at") and float(c.get("expires_ts") or 0) >= now]
    return {"ok": True, "n_total": len(d["codes"]), "n_active": len(active),
            "ttl_default_min": DEFAULT_TTL_MIN,
            "note": "授权码一次性、有 TTL；兑换即签发 sync+write 令牌"}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a and a[0] == "--new":
        nas = ""
        for x in a[1:]:
            if x.startswith("--nas="):
                nas = x.split("=", 1)[1]
        print(json.dumps(create(nas=nas), ensure_ascii=False, indent=1))
    elif a and a[0] == "--list":
        print(json.dumps(list_codes(include_used=True), ensure_ascii=False, indent=1))
    elif a and a[0] == "--redeem":
        print(json.dumps(redeem(a[1] if len(a) > 1 else ""), ensure_ascii=False, indent=1))
    elif a and a[0] == "--parse":
        print(json.dumps(parse_link(a[1] if len(a) > 1 else ""), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
