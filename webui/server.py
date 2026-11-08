#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APB Web UI —— 零依赖服务端（仅 Python 标准库）

设计铁律
--------
1) 零依赖：只用 http.server / json / pathlib / argparse / subprocess / threading。
   不引入 flask / fastapi / django / uvicorn —— 目标机是 Intel N100 小主机与飞牛 fnOS。
2) UI 只做 IO，不做计算：本文件**不派生任何指标**。命中率、IC、多空净值、权重全部由
   qcore 算好并落盘为 data/state/*.json，这里只做「读文件 -> 切片/筛选 -> 原样转发」。
   切片（取 Top N、取最近 N 天）属于选择行，不产生新的数值。
3) 绝不 500：所有异常被捕获并转成 {"ok": false, "error": "..."}，进程不崩、栈不外泄。
4) 数据缺失优雅降级：返回 {"ok": false, "error": "not_ready"} + HTTP 200。

用法
----
    python -m webui.server                    # 127.0.0.1:8765
    python -m webui.server --port 8766        # 换端口
    python -m webui.server --host 0.0.0.0     # 内网访问（务必配 APB_TOKEN）
    python -m webui.server --state-dir tmp/s  # 指向别的 state 目录（测试用）
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hmac
import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# --------------------------------------------------------------------------- #
# 路径与全局配置
# --------------------------------------------------------------------------- #

WEBUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBUI_DIR.parent
STATIC_DIR = WEBUI_DIR / "static"
VERSION_FILE = WEBUI_DIR / "VERSION"

# 这些由 main() 在启动时赋值（单进程，无需下发到 handler 实例）
STATE_DIR: Path = PROJECT_ROOT / "data" / "state"
RUN_TIMEOUT: int = 900
ALLOW_RUN: bool = True

# 数据文件 -> 用途。全部由 qcore 写入，本服务只读。
STATE_FILES = {
    "status": "status.json",
    "signals": "weights.json",
    "regime": "regime.json",
    "ledger": "ledger.json",
    "performance": "performance.json",
    "paper": "paper.json",
    "source_health": "source_health.json",
    "market": "market.json",
}
LIVE_TRADES_FILE = "live_trades.jsonl"

# 可选文件（缺失不算错误，只是对应页面显示“未就绪”）
OPTIONAL_FILES = {
    "events": "events.json",            # 政策日历（qcore.policy_calendar 的导出）
    "invariants": "invariants.json",    # 不变量检查结果（审计脚本导出）
}
# 政策日历的备用来源：qcore.policy_calendar.to_jsonl() 的真实落盘位置
EVENTS_JSONL_FALLBACK = PROJECT_ROOT / "data" / "events" / "policy_calendar.jsonl"

# 模型（Alpha158+GBDT）产物：优先专用文件，其次合并进 model.json 的段
MODEL_FILES = {
    "model": ("model.json", "model_meta.json"),
    "picks": ("picks.json", "model_picks.json"),
    "importance": ("factor_importance.json", "importance.json"),
}
MODEL_BUNDLE = "model.json"

DEFAULT_VERSION = "0.0.0"

MAX_BODY_BYTES = 64 * 1024
MAX_TOP = 200
MAX_DAYS = 500

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
}

_TRADE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_RUN: dict = {
    "state": "idle",        # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "duration_s": None,
    "tail": "",
    "error": None,
}


# --------------------------------------------------------------------------- #
# 读盘工具（唯一的“业务逻辑”：读 JSON / 选择行）
# --------------------------------------------------------------------------- #

def _read_json(filename: str):
    """返回 (data, error)。error ∈ {None, 'not_ready', 'read_error', 'bad_json'}"""
    path = STATE_DIR / filename
    try:
        if not path.is_file():
            return None, "not_ready"
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, "read_error"
    if not raw.strip():
        return None, "not_ready"
    try:
        return json.loads(raw), None
    except (json.JSONDecodeError, ValueError):
        return None, "bad_json"


def _rows_of(data):
    """容忍 list 或 {rows|items|data: [...]} 两种落盘形态。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "items", "data", "records", "events"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def _is_private_ip(ip: str) -> bool:
    """判断是否内网/回环地址（用于更新接口的来源限制）。"""
    try:
        import ipaddress
        a = ipaddress.ip_address(ip)
        return a.is_private or a.is_loopback or a.is_link_local
    except Exception:
        return False


def _read_version() -> str:
    """版本号唯一真源：webui/VERSION。前端不硬编码，只显示这里读到的值。"""
    try:
        text = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_VERSION
    return text or DEFAULT_VERSION


def _build_time() -> str | None:
    """构建时间 = 前端静态资源 + VERSION 里最新的 mtime（文件元数据，非计算结果）。"""
    newest = 0.0
    for path in (VERSION_FILE, STATIC_DIR / "index.html",
                 STATIC_DIR / "app.js", STATIC_DIR / "style.css"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    if not newest:
        return None
    return _dt.datetime.fromtimestamp(newest).astimezone().isoformat(timespec="seconds")


def _age_days(as_of) -> float | None:
    """数据新鲜度（天）= 今天 - as_of。

    这是**文件元数据**层面的新鲜度，用于顶栏连接指示灯，不是任何指标，
    也不参与展示给用户的分析口径（UI 不做计算的铁律针对的是指标）。
    """
    if not as_of:
        return None
    try:
        stamp = _dt.date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return None
    return round((_dt.date.today() - stamp).days, 1)


def _read_jsonl(path: Path):
    """读 JSONL（每行一个 JSON 对象），坏行跳过。"""
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        return None
    return rows


def _read_first(names):
    """按顺序尝试多个文件名，返回 (data, 命中文件名, error)。

    error: None / 'not_ready' / 'read_error' / 'bad_json'（取第一个非 not_ready 的错误）
    """
    first_err = None
    for name in names:
        data, err = _read_json(name)
        if err is None:
            return data, name, None
        if err != "not_ready" and first_err is None:
            first_err = (name, err)
    if first_err:
        return None, first_err[0], first_err[1]
    return None, names[0], "not_ready"


def _rows_from_importance(obj):
    """把「factor -> importance」字典整形成行；已经是列表就原样返回。

    只做结构整形，不计算任何指标（数值原样保留）。
    """
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        rows = []
        for k, v in obj.items():
            if isinstance(v, dict):
                rows.append(dict({"factor": k}, **v))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                rows.append({"factor": k, "importance": float(v)})
        return rows
    return []


def _file_meta(path: Path) -> dict:
    try:
        st = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "size": int(st.st_size),
        "mtime": _dt.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
    }


def _clamp_int(raw, default: int, lo: int, hi: int) -> int:
    try:
        value = int(str(raw[0]))
    except (TypeError, ValueError, IndexError):
        return default
    return max(lo, min(hi, value))


def _slice_signals(items, top_n: int):
    """按 weight 降序取 Top N 与 Bottom N（文件本身已排序，这里只做选择）。"""
    valid = [x for x in items
             if isinstance(x, dict) and isinstance(x.get("weight"), (int, float))
             and not isinstance(x.get("weight"), bool)]
    valid.sort(key=lambda x: x["weight"], reverse=True)
    n = min(top_n, len(valid))
    return valid[:n], valid[-n:][::-1] if n else [], len(valid)


def _slice_recent_days(rows, days: int):
    """取最近 `days` 个不同 as_of 的全部行（选择行，不做聚合）。"""
    stamps = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        stamp = row.get("as_of")
        if isinstance(stamp, str) and stamp not in seen:
            seen.add(stamp)
            stamps.append(stamp)
    if not stamps:
        return [r for r in rows if isinstance(r, dict)]
    stamps.sort(reverse=True)
    keep = set(stamps[:days])
    out = [r for r in rows if isinstance(r, dict) and r.get("as_of") in keep]
    out.sort(key=lambda r: (str(r.get("as_of")), str(r.get("code"))), reverse=True)
    return out


def _json_safe(obj):
    """把 NaN / Infinity 换成 null。

    NaN 不是合法 JSON：Python 的 json.dumps 默认会写出裸 `NaN`，浏览器 JSON.parse
    直接抛错、整页挂掉。引擎侧若漏了 sanitize（如 paper.json 的 compare），这里兜底。
    只做序列化清洗，不改变任何数值语义。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

class ApiHandler(BaseHTTPRequestHandler):
    server_version = "APBWebUI/1.0"
    protocol_version = "HTTP/1.1"

    # ---------------- 底层响应 ----------------

    def _send_json(self, status: int, payload: dict):
        try:
            body = json.dumps(_json_safe(payload), ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:      # 兜底：序列化失败也不许崩
            body = json.dumps({"ok": False, "error": "serialize_error",
                               "detail": f"{type(exc).__name__}: {exc}"},
                              ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        return status, len(body)

    def _ok(self, data, **meta):
        payload = {"ok": True, "data": data}
        payload.update(meta)
        return self._send_json(200, payload)

    def _fail(self, error: str, status: int = 200, **extra):
        payload = {"ok": False, "error": error}
        payload.update(extra)
        return self._send_json(status, payload)

    def _not_ready(self, filename: str, hint: str = ""):
        # 关键：文件缺失 -> HTTP 200 + ok:false，前端显示“未就绪”，不崩、不 5xx
        return self._fail(
            "not_ready", 200,
            missing=filename,
            hint=hint or "等待引擎首次运行（python -m qcore.live --once）生成该文件",
        )

    def _load_or_not_ready(self, key: str):
        """返回 (data, sent)：sent 非 None 表示已发出错误响应，调用方直接 return sent。"""
        data, err = _read_json(STATE_FILES[key])
        if err is None:
            return data, None
        if err == "not_ready":
            return None, self._not_ready(STATE_FILES[key])
        if err == "bad_json":
            return None, self._fail(
                "bad_json", 200, file=STATE_FILES[key],
                hint="文件存在但无法解析，可能正被写入，稍后重试")
        return None, self._fail(
            "read_error", 200, file=STATE_FILES[key],
            hint="文件存在但读取失败（权限或占用）")

    # ---------------- 静态文件 ----------------

    def _serve_static(self, rel: str):
        rel = rel.split("?", 1)[0].split("#", 1)[0]
        if not rel or rel.endswith("/"):
            rel += "index.html"
        candidate = (STATIC_DIR / rel).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._fail("forbidden", 403)
        if not candidate.is_file():
            return self._fail("not_found", 404, path=rel)
        try:
            blob = candidate.read_bytes()
        except OSError:
            return self._fail("read_error", 200, path=rel)
        ctype = MIME.get(candidate.suffix.lower(), "application/octet-stream")
        try:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(blob)
        except (BrokenPipeError, ConnectionResetError):
            pass
        return 200, len(blob)

    # ---------------- 路由：GET ----------------

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        t0 = time.perf_counter()
        status, size = 0, 0
        try:
            status, size = self._route_get()
        except (BrokenPipeError, ConnectionResetError):
            status = 499
        except Exception as exc:                       # noqa: BLE001 —— 铁律：绝不 500
            status, size = self._fail("internal_error", 200,
                                      detail=f"{type(exc).__name__}: {exc}")
        finally:
            self._access_log(status, size, t0)

    def _route_get(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])

        if path == "/api/status":
            return self._api_status()
        if path == "/api/signals":
            return self._api_signals(query)
        if path == "/api/regime":
            return self._api_regime()
        if path == "/api/ledger":
            return self._api_ledger(query)
        if path == "/api/performance":
            return self._api_passthrough("performance")
        if path == "/api/paper":
            return self._api_passthrough("paper")
        if path == "/api/paper/trades":
            return self._api_paper_trades(query)
        if path == "/api/run/status":
            return self._api_run_status()
        if path == "/api/version":
            return self._api_version()
        if path == "/api/behavior":
            return self._api_behavior()
        if path == "/api/events":
            return self._api_events(query)
        if path == "/api/export":
            return self._api_export(query)
        if path == "/api/tick":
            return self._api_tick(query)
        if path == "/api/watch":
            return self._api_watch()
        if path == "/api/names":
            return self._api_names()
        if path == "/api/account/config":
            return self._api_account_config()
        if path == "/api/market":
            return self._api_market()
        if path == "/api/update":
            return self._api_update()
        if path == "/api/model":
            return self._api_model()
        if path == "/api/picks":
            return self._api_picks(query)
        if path == "/api/factors/importance":
            return self._api_factor_importance(query)
        if path == "/api/health":
            return self._api_health()
        if path.startswith("/api/"):
            return self._fail("unknown_endpoint", 404, path=path)
        return self._fail("not_found", 404, path=path)

    def _api_status(self):
        data, sent = self._load_or_not_ready("status")
        if sent is not None:
            return sent
        if not isinstance(data, dict):
            return self._fail("bad_schema", 200, file=STATE_FILES["status"],
                              hint="期望 JSON 对象")
        return self._ok(data)

    def _api_signals(self, query):
        top_n = _clamp_int(query.get("top"), 20, 1, MAX_TOP)
        data, sent = self._load_or_not_ready("signals")
        if sent is not None:
            return sent
        top, bottom, total = _slice_signals(_rows_of(data), top_n)
        return self._ok({"top": top, "bottom": bottom, "total": total, "top_n": top_n})

    def _api_regime(self):
        data, sent = self._load_or_not_ready("regime")
        if sent is not None:
            return sent
        rows = [r for r in _rows_of(data) if isinstance(r, dict)]
        return self._ok({"rows": rows, "n": len(rows)})

    def _api_ledger(self, query):
        days = _clamp_int(query.get("days"), 30, 1, MAX_DAYS)
        data, sent = self._load_or_not_ready("ledger")
        if sent is not None:
            return sent
        rows = _slice_recent_days(_rows_of(data), days)
        return self._ok({"rows": rows, "n": len(rows), "days": days})

    def _api_passthrough(self, key: str):
        data, sent = self._load_or_not_ready(key)
        if sent is not None:
            return sent
        return self._ok(data)

    def _api_paper_trades(self, query):
        """只读：回显实盘成交明细（JSONL 原样，不聚合）。"""
        limit = _clamp_int(query.get("limit"), 50, 1, 500)
        path = STATE_DIR / LIVE_TRADES_FILE
        if not path.is_file():
            return self._ok({"rows": [], "n": 0})
        rows = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            return self._fail("read_error", 200, file=LIVE_TRADES_FILE)
        rows.reverse()
        return self._ok({"rows": rows[:limit], "n": len(rows)})

    def _api_run_status(self):
        with _RUN_LOCK:
            return self._ok(dict(_RUN))

    # ---------------- 新增：版本 / 行为因子 / 政策事件 / 健康 / 导出 ----------------

    def _api_version(self):
        import platform as _platform
        return self._ok({
            "version": _read_version(),
            "build_time": _build_time(),
            "python": _platform.python_version(),
            "platform": _platform.platform(),
            "webui_dir": str(WEBUI_DIR),
        })

    def _api_tick(self, query):
        """秒级快照：现取现返回（不落盘，避免每次请求都写文件）。"""
        codes = (query.get("codes") or ["600519"])[0].split(",")
        try:
            from qcore import tick
            return self._ok(tick.snapshot([c.strip() for c in codes if c.strip()][:30]))
        except Exception as e:  # noqa: BLE001
            return self._fail("tick_failed", 200, detail=str(e)[:160])

    def _api_watch(self):
        """自选跟踪：列表 + 汇总。"""
        try:
            from qcore import watch
            cur = watch.load()
            return self._ok({"items": cur.get("items") or [],
                             "updated_at": cur.get("updated_at"),
                             "summary": watch.summary()})
        except Exception as e:  # noqa: BLE001
            return self._fail("watch_failed", 200, detail=str(e)[:160])

    def _post_watch(self):
        """添加/移除自选。body: {"add":["600519"],"remove":[]}"""
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        if not isinstance(body, dict):
            return self._fail("bad_body", 400)
        try:
            from qcore import watch
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        out = {}
        if body.get("add"):
            out["add"] = watch.add(body["add"])
        if body.get("remove"):
            out["remove"] = watch.remove(body["remove"])
        if not out:
            return self._fail("nothing_to_do", 400, hint="需要 add 或 remove")
        return self._ok(out)

    def _api_names(self):
        """代码 -> 名称映射（UI 展示用）。文件缺失时返回空表，不报错。"""
        data, _err = _read_json("names.json")
        if not isinstance(data, dict):
            return self._ok({"names": {}, "n": 0, "hint": "尚未导出，跑一次引擎即可"})
        return self._ok(data)

    def _api_account_config(self):
        """模拟账户配置（初始资金等）。"""
        try:
            from qcore import account_config as AC
            return self._ok(AC.load())
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])

    def _post_account_config(self):
        """更新模拟账户配置：只允许内网或带 X-Token。"""
        expected = os.environ.get("APB_TOKEN", "")
        given = self.headers.get("X-Token") or ""
        if expected:
            if not hmac.compare_digest(given, expected):
                return self._fail("unauthorized", 403, hint="X-Token 不匹配")
        else:
            client = (self.client_address or ("", 0))[0]
            if not _is_private_ip(client):
                return self._fail("forbidden", 403, hint="未设 APB_TOKEN 时只允许内网修改")
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        try:
            from qcore import account_config as AC
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        new, verr = AC.save(body if isinstance(body, dict) else {})
        if verr:
            return self._fail("invalid_config", 400, detail=verr)
        return self._ok(new)

    def _api_market(self):
        """实时行情与市场情绪 —— 只读 data/state/market.json，本层不做任何计算。"""
        data, sent = self._load_or_not_ready("market")
        if sent is not None:
            return sent
        return self._ok(data)

    def _api_update(self):
        """版本检查 —— 只比对本地 VERSION 与 update.json 记录的远端版本。"""
        cur = _read_version()
        info, _err = _read_json("update.json")
        info = info if isinstance(info, dict) else {}
        latest = info.get("latest_version") if isinstance(info, dict) else None
        has_update = False
        try:
            from qcore.update_check import compare_versions
            has_update = bool(latest and cur and compare_versions(latest, cur) > 0)
        except Exception:
            has_update = bool(latest and cur and latest != cur)
        return self._ok({
            "current": cur, "latest": latest,
            "has_update": has_update,
            "checked_at": (info or {}).get("checked_at"),
            "command": "docker compose pull && docker compose up -d",
        })

    # ---------------- 模型（Alpha158+GBDT）----------------

    def _api_model(self):
        """模型元信息：model.json（或 model_meta.json）原样转发。"""
        data, name, err = _read_first(MODEL_FILES["model"])
        if err == "not_ready":
            return self._not_ready(name, "模型尚未训练，请先运行回测脚本生成 data/state/model.json")
        if err:
            return self._fail(err, 200, file=name)
        if not isinstance(data, dict):
            return self._fail("bad_schema", 200, file=name, hint="期望 JSON 对象")
        # 合并形态：{"model": {...}, "picks": [...], "factors": [...]}
        inner = data.get("model")
        payload = inner if isinstance(inner, dict) else data
        return self._ok(payload)

    def _api_picks(self, query):
        """当前推荐持仓：picks.json，其次 model.json 的 picks 段。"""
        limit = _clamp_int(query.get("limit"), 500, 1, 2000)
        data, name, err = _read_first(MODEL_FILES["picks"])
        rows, meta = None, {}
        if err is None:
            rows = _rows_of(data)
            meta = data if isinstance(data, dict) else {}
        else:
            bundle, bname, berr = _read_first((MODEL_BUNDLE,))
            if berr is None and isinstance(bundle, dict):
                cand = bundle.get("picks")
                if isinstance(cand, dict):
                    rows = _rows_of(cand)
                    meta = cand
                elif isinstance(cand, list):
                    rows = [r for r in cand if isinstance(r, dict)]
            if rows is None:
                if err == "not_ready" and berr in (None, "not_ready"):
                    return self._not_ready(
                        name, "推荐列表尚未生成，请先运行模型打分（data/state/picks.json）")
                return self._fail(err if err != "not_ready" else (berr or "not_ready"),
                                  200, file=name)
        payload = {"rows": rows[:limit], "n": len(rows)}
        for key in ("as_of", "top_k", "hold_days", "note", "generated_at"):
            if isinstance(meta, dict) and key in meta:
                payload[key] = meta[key]
        return self._ok(payload)

    def _api_factor_importance(self, query):
        """GBDT 因子重要性：factor_importance.json，其次 model.json 的 factors/importance 段。"""
        limit = _clamp_int(query.get("top"), 500, 1, 2000)
        data, name, err = _read_first(MODEL_FILES["importance"])
        rows = None
        if err is None:
            if isinstance(data, dict) and isinstance(data.get("rows"), list):
                rows = [r for r in data["rows"] if isinstance(r, dict)]
            else:
                rows = _rows_from_importance(data)
        else:
            bundle, bname, berr = _read_first((MODEL_BUNDLE,))
            if berr is None and isinstance(bundle, dict):
                cand = bundle.get("factors", bundle.get("importance"))
                if cand is not None:
                    rows = _rows_from_importance(cand)
            if rows is None:
                if err == "not_ready" and berr in (None, "not_ready"):
                    return self._not_ready(
                        name, "因子重要性尚未导出（data/state/factor_importance.json）")
                return self._fail(err if err != "not_ready" else (berr or "not_ready"),
                                  200, file=name)
        return self._ok({"rows": rows[:limit], "n": len(rows)})

    def _api_behavior(self):
        """行为因子：原样转发 performance.json 的 edge / behavior 段（不做任何计算）。"""
        data, sent = self._load_or_not_ready("performance")
        if sent is not None:
            return sent
        if not isinstance(data, dict):
            return self._fail("bad_schema", 200, file=STATE_FILES["performance"],
                              hint="期望 JSON 对象")
        behavior = data.get("behavior")
        return self._ok({
            "generated_at": data.get("generated_at"),
            "dataset": data.get("dataset"),
            "edge": data.get("edge"),
            "behavior": behavior if behavior is not None else None,
        })

    def _api_events(self, query):
        """政策事件：优先 data/state/events.json，缺失时回退 qcore 真实落盘的 JSONL。"""
        limit = _clamp_int(query.get("limit"), 500, 1, 2000)
        primary = STATE_DIR / OPTIONAL_FILES["events"]
        rows, source, impact = [], None, None

        if primary.is_file():
            data, err = _read_json(OPTIONAL_FILES["events"])
            if err == "bad_json":
                return self._fail("bad_json", 200, file=OPTIONAL_FILES["events"],
                                  hint="文件存在但无法解析，稍后重试")
            if err == "read_error":
                return self._fail("read_error", 200, file=OPTIONAL_FILES["events"])
            rows = _rows_of(data)
            if isinstance(data, dict):
                raw_impact = data.get("impact")
                if isinstance(raw_impact, list):
                    impact = raw_impact
                elif isinstance(raw_impact, dict):
                    impact = _rows_of(raw_impact)
            source = str(primary)
        elif EVENTS_JSONL_FALLBACK.is_file():
            parsed = _read_jsonl(EVENTS_JSONL_FALLBACK)
            if parsed is None:
                return self._fail("read_error", 200, file=str(EVENTS_JSONL_FALLBACK))
            rows = parsed
            source = str(EVENTS_JSONL_FALLBACK)

        if not rows:
            return self._not_ready(
                OPTIONAL_FILES["events"],
                "政策日历尚未导出。运行 python -m qcore.policy_calendar --to-json "
                "或让引擎写出 data/state/events.json 后刷新")

        rows = [r for r in rows if isinstance(r, dict)]
        # 按日期升序（文件里本来就是时间序，这里只做稳定排序，不改数值）
        rows.sort(key=lambda r: str(r.get("date") or r.get("as_of") or ""))
        payload = {"rows": rows[:limit], "n": len(rows), "source": source}
        if impact:
            payload["impact"] = [r for r in impact if isinstance(r, dict)]
        return self._ok(payload)

    def _api_health(self):
        """连接指示灯 + 风险体检用：文件齐备度、数据新鲜度、数据源健康度。

        data_age_days 是「今天 - status.as_of」的文件元数据新鲜度，不是指标。
        """
        status, _err = _read_json(STATE_FILES["status"])
        as_of = status.get("as_of") if isinstance(status, dict) else None

        missing = []
        files = {}
        for key, name in STATE_FILES.items():
            meta = _file_meta(STATE_DIR / name)
            files[name] = meta
            if not meta["exists"]:
                missing.append(name)
        optional = {}
        for key, name in OPTIONAL_FILES.items():
            meta = _file_meta(STATE_DIR / name)
            optional[name] = meta
        optional_events_fallback = _file_meta(EVENTS_JSONL_FALLBACK)

        source_health, _ = _read_json(STATE_FILES["source_health"])
        invariants, _ = _read_json(OPTIONAL_FILES["invariants"])

        age = _age_days(as_of)
        return self._ok({
            "ok": not missing,
            "as_of": as_of,
            "data_age_days": age,
            "missing_files": missing,
            "files": files,
            "optional_files": optional,
            "events_fallback": optional_events_fallback,
            "source_health": source_health if isinstance(source_health, dict) else None,
            "invariants": invariants,
            "state_dir": str(STATE_DIR),
            "static_dir": str(STATIC_DIR),
            "run_enabled": ALLOW_RUN,
            "server_time": _now_iso(),
        })

    def _api_export(self, query):
        """导出数据：把已存在的 state 文件打包成一个 JSON 附件（纯拼接，不改数值）。"""
        what = (query.get("what") or ["all"])[0]
        bundle = {"ok": True, "_meta": {
            "exported_at": _now_iso(),
            "version": _read_version(),
            "state_dir": str(STATE_DIR),
        }}
        keys = list(STATE_FILES) if what == "all" else [k for k in STATE_FILES if k == what]
        if not keys:
            return self._fail("bad_param", 400, hint="what 取 all|status|signals|regime|ledger|performance|paper")
        got = 0
        for key in keys:
            data, err = _read_json(STATE_FILES[key])
            if err is None:
                bundle[key] = data
                got += 1
        if not got:
            return self._not_ready("status.json", "没有任何可导出的状态文件")
        body = json.dumps(_json_safe(bundle), ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="apb-state-{stamp}.json"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        return 200, len(body)

    # ---------------- 路由：POST ----------------

    def do_POST(self):
        t0 = time.perf_counter()
        status, size = 0, 0
        try:
            status, size = self._route_post()
        except (BrokenPipeError, ConnectionResetError):
            status = 499
        except Exception as exc:                       # noqa: BLE001
            status, size = self._fail("internal_error", 200,
                                      detail=f"{type(exc).__name__}: {exc}")
        finally:
            self._access_log(status, size, t0)

    def _route_post(self):
        path = urlparse(self.path).path
        if path == "/api/paper/trade":
            return self._post_trade()
        if path == "/api/run":
            return self._post_run()
        if path == "/api/watch":
            return self._post_watch()
        if path == "/api/account/config":
            return self._post_account_config()
        if path == "/api/update/apply":
            return self._post_update_apply()
        if path.startswith("/api/"):
            return self._fail("unknown_endpoint", 404, path=path)
        return self._fail("not_found", 404, path=path)

    def _read_body(self):
        """返回 (obj, error)。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return None, "bad_length"
        if length <= 0:
            return None, "empty_body"
        if length > MAX_BODY_BYTES:
            return None, "body_too_large"
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None, "bad_json"
        if not isinstance(obj, dict):
            return None, "bad_schema"
        return obj, None

    def _post_trade(self):
        obj, err = self._read_body()
        if err == "body_too_large":
            return self._fail(err, 413, hint=f"请求体上限 {MAX_BODY_BYTES} 字节")
        if err == "bad_json":
            return self._fail(err, 400, hint="请求体必须是 JSON 对象")
        if err:
            return self._fail(err, 400)

        # 只做校验与落盘，不做任何估值/盈亏计算
        date = str(obj.get("date") or "").strip()
        code = str(obj.get("code") or "").strip().upper()
        side = str(obj.get("side") or "").strip().lower()
        note = str(obj.get("note") or "").strip()[:200]

        try:
            _dt.date.fromisoformat(date)
        except ValueError:
            return self._fail("bad_date", 400, hint="date 需为 YYYY-MM-DD")
        if not code or len(code) > 24:
            return self._fail("bad_code", 400, hint="code 不能为空且不超过 24 字符")
        if side not in ("buy", "sell"):
            return self._fail("bad_side", 400, hint="side 只能是 buy 或 sell")
        try:
            price = float(obj.get("price"))
        except (TypeError, ValueError):
            return self._fail("bad_price", 400, hint="price 必须是数字")
        if not (price > 0) or price != price or price in (float("inf"),):
            return self._fail("bad_price", 400, hint="price 必须为正数")
        try:
            qty_f = float(obj.get("qty"))
        except (TypeError, ValueError):
            return self._fail("bad_qty", 400, hint="qty 必须是数字")
        if not qty_f.is_integer() or qty_f <= 0:
            return self._fail("bad_qty", 400, hint="qty 必须是正整数")

        record = {"date": date, "code": code, "side": side,
                  "price": price, "qty": int(qty_f), "note": note}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        target = STATE_DIR / LIVE_TRADES_FILE
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with _TRADE_LOCK:
                with open(target, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        except OSError as exc:
            return self._fail("write_error", 200, detail=f"{type(exc).__name__}: {exc}")
        return self._ok({"appended": record, "file": LIVE_TRADES_FILE})

    def _post_run(self):
        if not ALLOW_RUN:
            return self._fail("run_disabled", 403, hint="服务以 --no-run 启动")

        # 与 /api/update/apply 一致：设了 token 就必须匹配，没设则只放行内网
        expected = os.environ.get("APB_TOKEN", "")
        given = self.headers.get("X-Token") or ""
        if expected:
            if not hmac.compare_digest(given, expected):
                return self._fail("unauthorized", 403, hint="X-Token 与 APB_TOKEN 不匹配")
        else:
            client = (self.client_address or ("", 0))[0]
            if not _is_private_ip(client):
                return self._fail("token_not_configured", 403,
                                  hint=f"未设 APB_TOKEN 时只允许内网触发（来源 {client}）")

        with _RUN_LOCK:
            if _RUN["state"] == "running":
                return self._fail("busy", 409, hint="已有一次运行在进行中")
            _RUN.update({"state": "running", "started_at": _now_iso(),
                         "finished_at": None, "returncode": None,
                         "duration_s": None, "tail": "", "error": None})
        threading.Thread(target=_run_engine, name="apb-run", daemon=True).start()
        return self._ok({"started": True, "cmd": _run_cmd()})

    def _post_update_apply(self):
        """
        应用内一键更新：启动一个**辅助容器**执行自我替换（主容器不能自己停自己）。
        必须带 X-Token（与 APB_TOKEN 一致）。
        """
        # 鉴权策略：
        #   - 设了 APB_TOKEN  -> 必须带匹配的 X-Token（对外暴露时用）
        #   - 没设 APB_TOKEN  -> 只允许内网来源（默认，开箱即用）
        expected = os.environ.get("APB_TOKEN", "")
        given = self.headers.get("X-Token") or ""
        if expected:
            if not hmac.compare_digest(given, expected):
                return self._fail("unauthorized", 403, hint="X-Token 与 APB_TOKEN 不匹配")
        else:
            client = (self.client_address or ("", 0))[0]
            if not _is_private_ip(client):
                return self._fail("token_not_configured", 403,
                                  hint=f"未设 APB_TOKEN 时只允许内网更新（来源 {client}）；"
                                       "如需公网访问请设置 APB_TOKEN")

        try:
            from qcore import docker_api as D
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:200])

        if not D.available():
            return self._fail("docker_socket_unavailable", 503,
                              hint="容器未挂载 /var/run/docker.sock，无法应用内更新；"
                                   "请在宿主机执行 docker compose pull && docker compose up -d")

        info, _e = _read_json("update.json")
        info = info if isinstance(info, dict) else {}
        latest = info.get("latest_version")
        cur = _read_version()
        if not latest or latest == cur:
            return self._fail("already_latest", 409, current=cur, latest=latest)

        target = os.environ.get("APB_CONTAINER_NAME", "apb")
        image = os.environ.get("APB_IMAGE", "ghcr.io/ybd156/quant-apb")
        repo = image.rsplit(":", 1)[0] if ":" in image.rsplit("/", 1)[-1] else image

        helper = f"{target}-updater"
        old = D.find_container(helper)
        if old:
            D.remove(old["Id"])
        created = D.create(
            helper, image,
            env=[f"APB_TARGET_NAME={target}", f"APB_IMAGE={repo}", f"APB_TAG={latest}"],
            binds=["/var/run/docker.sock:/var/run/docker.sock"],
            restart="no",
            entrypoint=["python", "-m", "qcore.update_apply"])
        if not created.get("ok"):
            return self._fail("helper_create_failed", 500, error=created.get("error"))
        st, _ = D.request("POST", f"/containers/{created['id']}/start")
        if st not in (204, 304):
            return self._fail("helper_start_failed", 500, status=st)
        return self._ok({"started": True, "target": target, "to": latest,
                         "helper": helper,
                         "note": "更新在后台进行，约 1-3 分钟后刷新页面"})

    # ---------------- 访问日志（精简一行） ----------------

    def _access_log(self, status: int, size: int, t0: float):
        elapsed = (time.perf_counter() - t0) * 1000.0
        stamp = time.strftime("%H:%M:%S")
        client = self.client_address[0] if self.client_address else "-"
        print(f"{stamp} {client} {self.command} {self.path} -> {status} "
              f"{size}B {elapsed:.0f}ms", flush=True)

    def log_message(self, fmt, *args):      # 屏蔽默认的多行日志
        return

    def log_error(self, fmt, *args):
        return


# --------------------------------------------------------------------------- #
# 引擎触发
# --------------------------------------------------------------------------- #

def _run_cmd():
    return [sys.executable, "-m", "qcore.live", "--once"]


def _run_engine():
    started = time.perf_counter()
    result = {"returncode": None, "tail": "", "error": None}
    try:
        proc = subprocess.run(
            _run_cmd(), cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=RUN_TIMEOUT,
            env=os.environ.copy(),
        )
        result["returncode"] = proc.returncode
        blob = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result["tail"] = blob.strip()[-4000:]
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout: 超过 {RUN_TIMEOUT}s 未结束"
    except FileNotFoundError as exc:
        result["error"] = f"python_not_found: {exc}"
    except PermissionError as exc:
        result["error"] = ("spawn_denied: 系统/沙箱拒绝启动子进程，"
                           f"请在宿主机终端手动执行 python -m qcore.live --once（{exc}）")
    except Exception as exc:                # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _RUN_LOCK:
            _RUN.update(result)
            _RUN["duration_s"] = round(time.perf_counter() - started, 2)
            _RUN["finished_at"] = _now_iso()
            _RUN["state"] = "done" if (result["returncode"] == 0 and not result["error"]) else "error"


def _now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #

def _build_parser():
    ap = argparse.ArgumentParser(
        description="APB 零依赖 Web UI（stdlib http.server）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="绑定地址，默认 127.0.0.1；内网访问用 0.0.0.0")
    ap.add_argument("--port", type=int, default=8765, help="端口，默认 8765")
    ap.add_argument("--state-dir", default=str(PROJECT_ROOT / "data" / "state"),
                    help="data/state 目录，默认 <项目根>/data/state")
    ap.add_argument("--run-timeout", type=int, default=900,
                    help="POST /api/run 的子进程超时秒数，默认 900")
    ap.add_argument("--no-run", action="store_true", help="禁用 POST /api/run")
    return ap


def main(argv=None):
    global STATE_DIR, RUN_TIMEOUT, ALLOW_RUN
    # 中文日志在 LANG=C / cp936 / UTF-8 各种终端下都不能炸
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    args = _build_parser().parse_args(argv)

    STATE_DIR = Path(args.state_dir).expanduser().resolve()
    RUN_TIMEOUT = max(10, int(args.run_timeout))
    ALLOW_RUN = not args.no_run

    if not STATIC_DIR.is_dir():
        print(f"[fatal] 静态目录不存在：{STATIC_DIR}", file=sys.stderr)
        return 2
    if not (STATIC_DIR / "index.html").is_file():
        print(f"[fatal] 缺少 {STATIC_DIR / 'index.html'}", file=sys.stderr)
        return 2

    token_state = "已设置" if os.environ.get("APB_TOKEN") else "未设置（/api/run 将被拒绝）"

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    except OSError as exc:
        print(f"[fatal] 无法绑定 {args.host}:{args.port} -> {exc}", file=sys.stderr)
        return 3
    httpd.daemon_threads = True

    print("=" * 62, flush=True)
    print("  APB 自学习参数体 · Web UI（零依赖 stdlib）", flush=True)
    print(f"  监听     http://{args.host}:{args.port}", flush=True)
    print(f"  状态目录 {STATE_DIR}"
          f"{'' if STATE_DIR.is_dir() else '  [尚不存在，页面显示未就绪]'}", flush=True)
    print(f"  静态目录 {STATIC_DIR}", flush=True)
    print(f"  APB_TOKEN {token_state}", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("  ⚠ 绑定到非回环地址：请确认只在内网使用，并务必设置 APB_TOKEN", flush=True)
    print("=" * 62, flush=True)

    try:
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n[info] 收到 Ctrl-C，正在关闭…", flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        print("[info] 已停止", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
