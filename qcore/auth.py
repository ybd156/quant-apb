"""
登录与会话 —— 给整个站点加一道门。

## 为什么必须做服务端会话

之前只保护了写接口（改账户、跑流水线、导入模型），**读接口全开放**。
任何人拿到域名就能看到：

    /api/paper       你的模拟盘净值、持仓、成交
    /api/ledger      全部预测台账
    /api/model       模型推荐了什么票
    /api/performance 策略表现

这不是"随便看看"，是把你的策略和仓位摊开给人看。

**而且纯前端登录框是假安全** —— 在浏览器里挡一下，直接 curl 打 API 就绕过了。
所以门必须设在服务端：每个请求都验会话。

## 设计

    POST /api/login   {password} -> Set-Cookie: apb_session=...
    POST /api/logout              清除会话
    GET  /login                   登录页

会话存内存（进程内字典）+ 可选落盘（重启后免重登）。
密码用 PBKDF2-HMAC-SHA256 加盐哈希，不存明文。

## 配置

| 变量 | 说明 |
|---|---|
| `APB_PASSWORD` | 登录密码。**不设 = 不启用登录**（向后兼容，但会警告） |
| `APB_SESSION_DAYS` | 会话有效期，默认 7 天（滑动续期） |
| `APB_LOGIN_MAX_FAIL` | 5 分钟内最多失败次数，默认 5 |

## 哪些路径不需要登录

    /login              登录页本身
    /api/login          登录接口
    /api/logout         登出
    /api/healthz        给反向代理/容器编排用的健康检查
    /api/version        版本号（前端要显示）
    /static/*           静态资源（CSS/JS，不含数据）

其余全部要会话。**机器访问**（训练机同步代理）走 `X-APB-Token`，
它和登录会话是两套独立凭据，互不替代。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
SESSIONS_FILE = os.path.join(STATE, "sessions.json")

SESSION_DAYS = float(os.environ.get("APB_SESSION_DAYS", "7") or 7)
MAX_FAIL = int(os.environ.get("APB_LOGIN_MAX_FAIL", "5") or 5)
FAIL_WINDOW = 300.0          # 5 分钟

_lock = threading.Lock()
_sessions: Dict[str, Dict] = {}      # sid -> {exp, ip, ua, created}
_fails: Dict[str, List[float]] = {}  # ip -> [timestamps]

# 不需要登录的路径（前缀匹配）
PUBLIC_PATHS = (
    "/login",
    "/api/login",
    "/api/logout",
    "/api/healthz",
    "/api/version",
    # 登录页要显示「是否启用密码」。这个端点只返回布尔和会话列表，
    # 不含持仓/策略信息，可以公开。
    "/api/auth",
    # 行情数据包：训练机首次运行要下载，此时它还没有任何凭据。
    # 内容是公开的 A 股日线数据（不含持仓/预测/策略），可以公开。
    # 不加会导致训练机拿 401，请求立刻失败、没有任何下载流量。
    "/api/data/qlib",
)
PUBLIC_PREFIXES = ("/static/", "/favicon", "/apple-touch")


def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or _now()).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 密码
def enabled() -> bool:
    """是否启用登录。没设 APB_PASSWORD 就是关的。"""
    return bool(os.environ.get("APB_PASSWORD", ""))


def _hash(password: str, salt: bytes, iters: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)


def check_password(password: str) -> bool:
    """恒定时间比较，避免时序侧信道。"""
    expected = os.environ.get("APB_PASSWORD", "")
    if not expected:
        return True
    if not password:
        return False
    # 用哈希后再比，长度一致，hmac.compare_digest 才不会因长度泄漏信息
    a = _hash(password, b"apb-login-v1")
    b = _hash(expected, b"apb-login-v1")
    return hmac.compare_digest(a, b)


# ---------------------------------------------------------------- 限流
def rate_limited(ip: str) -> Tuple[bool, int]:
    """返回 (是否被限, 还需等待秒数)。"""
    with _lock:
        now = _now()
        arr = [t for t in _fails.get(ip, []) if now - t < FAIL_WINDOW]
        _fails[ip] = arr
        if len(arr) >= MAX_FAIL:
            wait = int(FAIL_WINDOW - (now - arr[0])) + 1
            return True, max(wait, 1)
        return False, 0


def record_fail(ip: str) -> None:
    with _lock:
        _fails.setdefault(ip, []).append(_now())


def clear_fails(ip: str) -> None:
    with _lock:
        _fails.pop(ip, None)


# ---------------------------------------------------------------- 会话
def _load_sessions() -> None:
    global _sessions
    try:
        if not os.path.isfile(SESSIONS_FILE):
            return
        with open(SESSIONS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        now = _now()
        _sessions = {k: v for k, v in (d or {}).items()
                     if isinstance(v, dict) and float(v.get("exp", 0)) > now}
    except Exception:
        _sessions = {}


def _save_sessions() -> None:
    try:
        os.makedirs(STATE, exist_ok=True)
        tmp = SESSIONS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_sessions, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SESSIONS_FILE)
        try:
            os.chmod(SESSIONS_FILE, 0o600)
        except Exception:
            pass
    except Exception:
        pass


def create_session(ip: str = "", ua: str = "") -> str:
    """新建会话，返回 sid。"""
    sid = secrets.token_urlsafe(32)
    with _lock:
        _sessions[sid] = {
            "exp": _now() + SESSION_DAYS * 86400,
            "created": _iso(),
            "ip": ip[:60], "ua": (ua or "")[:120],
        }
        _save_sessions()
    return sid


def verify_session(sid: str, touch: bool = True) -> bool:
    """校验会话；有效则滑动续期。"""
    if not sid:
        return False
    with _lock:
        s = _sessions.get(sid)
        if not s:
            return False
        if float(s.get("exp", 0)) <= _now():
            _sessions.pop(sid, None)
            _save_sessions()
            return False
        if touch:
            s["exp"] = _now() + SESSION_DAYS * 86400
            s["last"] = _iso()
        return True


def destroy_session(sid: str) -> None:
    with _lock:
        if sid in _sessions:
            _sessions.pop(sid, None)
            _save_sessions()


def destroy_all() -> int:
    with _lock:
        n = len(_sessions)
        _sessions.clear()
        _save_sessions()
    return n


def list_sessions() -> List[Dict]:
    with _lock:
        return [{"sid": k[:8] + "...", "exp": _iso(v.get("exp")),
                 "created": v.get("created"), "ip": v.get("ip")}
                for k, v in sorted(_sessions.items(),
                                   key=lambda x: -float(x[1].get("exp", 0)))]


# ---------------------------------------------------------------- 路径判断
def is_public_path(path: str) -> bool:
    p = (path or "/").split("?")[0]
    if p in PUBLIC_PATHS:
        return True
    return any(p.startswith(x) for x in PUBLIC_PREFIXES)


def status() -> Dict:
    return {
        "ok": True,
        "enabled": enabled(),
        "session_days": SESSION_DAYS,
        "max_fail": MAX_FAIL,
        "active_sessions": len(_sessions),
        "sessions": list_sessions()[:5],
        "warnings": ([] if enabled() else
                     ["未设 APB_PASSWORD —— 任何人都能打开这个页面看到你的持仓和策略。"
                      "暴露到公网前必须设置。"]),
    }


_load_sessions()


if __name__ == "__main__":
    import json
    print(json.dumps(status(), ensure_ascii=False, indent=1))
    os.environ["APB_PASSWORD"] = "test123"
    print("\n密码校验 test123:", check_password("test123"))
    print("密码校验 wrong  :", check_password("wrong"))
    sid = create_session("127.0.0.1", "test")
    print("会话校验:", verify_session(sid), "| 伪造:", verify_session("fake"))
    print("公开路径 /api/paper:", is_public_path("/api/paper"))
    print("公开路径 /login    :", is_public_path("/login"))
    destroy_session(sid)
    print("登出后:", verify_session(sid))
