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

# 请求体上限。原来 64KB —— 但模型 zip 有 67KB，base64 后 89KB，
# 导致 /api/model/import 和 /api/sync/push **从来没成功过**（400）。
# 16MB 足够放一个完整模型包（250 树约 260KB，base64 后 350KB，余量很大）。
MAX_BODY_BYTES = 16 * 1024 * 1024
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
    """
    数据新鲜度 —— 按**交易日**算，不是自然日。

    为什么必须这样：11-06 是周五，11-09 是周一，自然日差 3 天，
    但只隔 **1 个交易日**。按自然日算会误报「陈旧 3 天」，
    让人以为数据出问题了，其实完全正常（模型本来就用 T-1 收盘价）。

    返回：距今多少个**交易日**（含节假日判断）。取不到日历则退化为自然日。
    """
    if not as_of:
        return None
    try:
        stamp = _dt.date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return None
    today = _dt.date.today()
    if stamp >= today:
        return 0.0
    try:
        from qcore import calendar_cn as CAL
        n, cur = 0, stamp
        while cur < today and n < 400:
            cur = cur + _dt.timedelta(days=1)
            if CAL.is_trading_day_bool(cur):
                n += 1
        return float(n)
    except Exception:
        return round((today - stamp).days, 1)


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
            self._security_headers()
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
        # 静态资源版本号：每次发版换 URL，彻底绕开浏览器缓存。
        # 之前遇到过「代码部署了但页面还是旧的」，就是缓存导致的。
        if rel.endswith("index.html"):
            blob = blob.replace(b"__APB_VER__", _read_version().encode("utf-8"))
        ctype = MIME.get(candidate.suffix.lower(), "application/octet-stream")
        try:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-cache")
            self._security_headers()
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
        _a = self._auth_check()
        if _a is not None:
            return
        _g = self._guard_request()
        if _g is not None:
            return
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
            return self._api_paper(query)
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
        if path == "/api/train/kit/download":
            return self._api_train_kit_download()
        if path == "/api/fees":
            return self._api_fees(query)
        if path == "/api/paper/period":
            return self._api_paper_period(query)
        if path == "/api/weekly":
            return self._api_weekly(query)
        if path == "/api/paper/auto":
            return self._api_paper_auto(query)
        if path == "/api/live/features":
            return self._api_live_features()
        if path == "/api/train/kit":
            return self._api_train_kit()
        if path == "/api/hotspot":
            return self._api_hotspot(query)
        if path == "/api/horizons":
            return self._api_horizons(query)
        if path == "/api/event/learning":
            return self._api_event_learning(query)
        if path == "/api/news/live":
            return self._api_news_live(query)
        if path == "/api/data/quality":
            return self._api_data_quality()
        if path == "/api/calendar":
            return self._api_calendar(query)
        if path == "/api/lot":
            return self._api_lot(query)
        if path == "/api/auth":
            return self._api_auth()
        if path == "/api/kline":
            return self._api_kline(query)
        if path == "/api/healthz":
            return self._api_healthz()
        if path == "/api/netguard":
            return self._api_netguard()
        if path == "/api/sync":
            return self._api_sync_summary()
        if path == "/api/sync/task":
            return self._api_sync_task(query)
        if path == "/api/model/local":
            return self._api_model_local()
        if path == "/api/model/health":
            return self._api_model_health()
        if path == "/api/model/versions":
            return self._api_model_versions()
        if path == "/api/session":
            return self._api_session(query)
        if path == "/api/transfer":
            return self._api_transfer()
        if path == "/api/tick":
            return self._api_tick(query)
        if path == "/api/watch":
            return self._api_watch()
        if path == "/api/names":
            return self._api_names()
        if path == "/api/account/config":
            return self._api_account_config(query)
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

    def _api_paper(self, query):
        """模拟盘 vs 实盘对账（按市场隔离）。

        ?market=CN|US|HK
          - 优先读 paper_{MKT}.json；
          - 没有专属文件时，只有 CN 会回落到共享的 paper.json（它本来就是 A 股模拟盘）；
          - US / HK 一律返回 no_market_data，绝不让 A 股数据冒名顶替。
        不带 market 参数时保持老行为：原样返回 paper.json。
        """
        mkt = ((query or {}).get("market") or [None])[0]
        mkt = str(mkt).strip().upper() if mkt else None
        if mkt and mkt not in ("CN", "US", "HK"):
            return self._fail("invalid_market", 400,
                              detail=f"market 必须是 CN, US, HK 之一，收到 {mkt}")

        if mkt:
            data, _name, err = _read_first((f"paper_{mkt}.json",))
            if err is None:
                payload = dict(data) if isinstance(data, dict) else {"rows": data}
                payload.setdefault("market", mkt)
                return self._ok(payload)
            if mkt != "CN":
                return self._fail(
                    "no_market_data", 200, market=mkt,
                    hint=f"该市场尚无模拟盘数据；运行 python -m qcore.pipeline --market {mkt} 生成")

        data, sent = self._load_or_not_ready("paper")
        if sent is not None:
            return sent
        if not isinstance(data, dict):
            return self._fail("bad_schema", 200, file=STATE_FILES["paper"], hint="期望 JSON 对象")
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
                             "groups": watch.groups().get("groups") or [],
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
        grp = str(body.get("group") or "")[:20]
        if body.get("add"):
            out["add"] = watch.add(body["add"], group=grp)
        if body.get("remove"):
            out["remove"] = watch.remove(body["remove"])
        if body.get("set_group"):
            out["set_group"] = watch.set_group(body["set_group"], grp)
        if not out:
            return self._fail("nothing_to_do", 400,
                              hint="需要 add / remove / set_group 之一")
        return self._ok(out)

    def _api_names(self):
        """代码 -> 名称映射（UI 展示用）。文件缺失时返回空表，不报错。"""
        data, _err = _read_json("names.json")
        if not isinstance(data, dict):
            return self._ok({"names": {}, "n": 0, "hint": "尚未导出，跑一次引擎即可"})
        return self._ok(data)

    def _api_account_config(self, query=None):
        """
        模拟账户配置。query:
          market=CN|US|HK  -> 只返回该市场
          省略            -> 返回三市场总览 + default_market
        """
        try:
            from qcore import account_config as AC
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        m = ((query or {}).get("market") or [None])[0]
        try:
            if m:
                return self._ok(AC.load(m))
            return self._ok(AC.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("account_failed", 200, detail=str(e)[:160])

    def _post_account_config(self):
        """更新模拟账户配置：只允许内网或带 X-Token。"""
        expected = os.environ.get("APB_TOKEN", "")
        given = self.headers.get("X-Token") or ""
        if expected:
            if not hmac.compare_digest(given, expected):
                return self._fail("unauthorized", 403, hint="X-Token 不匹配")
        else:
            client = (self.client_address or ("", 0))[0]
            if not self._is_internal():
                return self._fail("forbidden", 403, hint="未设 APB_TOKEN 时只允许内网修改")
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        try:
            from qcore import account_config as AC
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        market = (body.get("market") if isinstance(body, dict) else None) or "CN"
        new, verr = AC.save(body if isinstance(body, dict) else {}, market)
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
        """
        当前推荐持仓。支持 ?horizon=N 取该周期的 picks_hN.json。
        """
        limit = _clamp_int(query.get("limit"), 500, 1, 2000)
        try:
            _h = int(((query or {}).get("horizon") or [1])[0])
        except (TypeError, ValueError):
            _h = 1
        if _h != 1:
            fn = f"picks_h{_h}.json"
            d, err = _read_json(fn)
            if err:
                return self._not_ready(fn,
                                       f"该周期还没有预测。训练模型后跑一次 "
                                       f"python -m qcore.live_model --predict --horizon={_h}")
            return self._ok(d)
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
        # 实时快讯（qcore.news_live 写入的 live 段）：与人工整理的日历分开返回，
        # 前端要能区分"人工整理（reliability 1.0）"和"规则分类快讯（0.5）"
        if isinstance(data, dict):
            live = data.get("live")
            if isinstance(live, list) and live:
                lv = [r for r in live if isinstance(r, dict)]
                lv.sort(key=lambda r: str(r.get("time") or ""), reverse=True)
                payload["live"] = lv[:limit]
                payload["live_n"] = len(lv)
                payload["live_meta"] = data.get("live_meta")
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

    def _api_session(self, query):
        """
        三段式时段数据。query: which=premarket|intraday|postclose|auto
        auto 按当前时间判断；不在任何窗口内返回 outside_session。
        """
        which = str(((query or {}).get("which") or ["auto"])[0]).lower()
        try:
            from qcore import session as S
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        if which == "auto":
            cur = S.in_window()
            if cur == "closed":
                return self._ok({"session": "closed",
                                 "note": "不在任何交易时段窗口内",
                                 "windows": {"premarket": "08:30-09:30",
                                             "intraday": "09:30-15:00",
                                             "postclose": "15:05-17:00"}})
            which = cur
        if which == "noon":
            try:
                from qcore import noon as NOON
                return self._ok(NOON.predict(verbose=False))
            except Exception as e:  # noqa: BLE001
                return self._fail("noon_failed", 200, detail=str(e)[:160])
        if which not in ("premarket", "intraday", "postclose"):
            return self._fail("bad_param", 400,
                              hint="which 取 premarket|intraday|postclose|auto")
        fn = {"premarket": S.premarket, "intraday": S.intraday,
              "postclose": S.postclose}[which]
        try:
            return self._ok(fn(verbose=False))
        except Exception as e:  # noqa: BLE001
            return self._fail("session_failed", 200, detail=str(e)[:160])

    def _api_train_kit_download(self):
        """
        把训练机套件发给客户端。本地缓存没有就从 GitHub 拉一次并缓存。
        这样训练机只需要能连 NAS —— 不用能访问 GitHub。
        """
        ver = _read_version()
        name = f"apb-train-kit_{ver}.zip"
        cache_dir = STATE_DIR / "kit"
        cache = cache_dir / name
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if not cache.is_file():
            # 本地缓存没有：先看项目目录有没有构建产物，再不行从 GitHub 拉
            import glob as _g
            local = _g.glob(os.path.join(str(PROJECT_ROOT), "apb-train-kit_*.zip"))
            if local:
                try:
                    import shutil as _sh
                    _sh.copy2(sorted(local)[-1], cache)
                except Exception:
                    pass
            if not cache.is_file():
                repo = os.environ.get("APB_REPO", "ybd156/quant-apb")
                url = (f"https://github.com/{repo}/releases/download/"
                       f"v{ver}/{name}")
                try:
                    from qcore import net as _net
                    body, _mode = _net.get(url, timeout_direct=90, timeout_proxy=150)
                    if body:
                        # net.get 返回解码后的文本，二进制会坏 —— 用裸下载
                        import urllib.request as _ur
                        req = _ur.Request(url, headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                        with _ur.urlopen(req, timeout=180) as r, open(cache, "wb") as f:
                            while True:
                                chunk = r.read(1 << 20)
                                if not chunk:
                                    break
                                f.write(chunk)
                except Exception as e:  # noqa: BLE001
                    return self._fail("kit_unavailable", 200,
                                      detail=f"拉取失败: {type(e).__name__}: {str(e)[:80]}",
                                      hint="该版本的训练机套件还没构建完成，稍后再试")

        if not cache.is_file():
            return self._fail("kit_not_built", 200,
                              hint=f"{name} 尚未生成（CI 构建完成后可用）")
        try:
            size = cache.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(cache, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return True
        except Exception as e:  # noqa: BLE001
            return self._fail("send_failed", 200, detail=str(e)[:100])

    def _api_fees(self, query):
        """各市场费用明细 + 对照。query: market=CN 只返回该市场。"""
        try:
            from qcore import fees as FEES
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        m = str(((query or {}).get("market") or [""])[0]).upper()
        try:
            if m:
                return self._ok({"market": m, "fee": FEES.get(m),
                                 "sources": FEES.SOURCES.get(m, []),
                                 "sample": FEES.trade_fees(m, "buy", 100000, 10000, 10)})
            return self._ok(FEES.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("fees_failed", 200, detail=str(e)[:160])

    def _post_fees_save(self):
        """改某个市场的费率。body: {"market":"CN","patch":{"commission_bps":1.5}}"""
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        body = body if isinstance(body, dict) else {}
        m = str(body.get("market") or "").upper()
        if not m:
            return self._fail("bad_param", 400, hint="需要 market")
        patch = body.get("patch") or body.get("fees") or {}
        try:
            from qcore import fees as FEES
            r = FEES.save(m, patch)
        except Exception as e:  # noqa: BLE001
            return self._fail("save_failed", 200, detail=str(e)[:160])
        return self._ok(r) if r.get("ok") else self._fail(r.get("reason", "failed"), 200)

    def _api_paper_period(self, query):
        """模拟盘分时段成交汇总。query: market=CN&bucket=day|week|month"""
        q = query or {}
        m = str((q.get("market") or ["CN"])[0]).upper()
        b = str((q.get("bucket") or ["day"])[0]).lower()
        if b not in ("day", "week", "month"):
            b = "day"
        try:
            from qcore import paper_auto as PA
            return self._ok({"market": m,
                             "period": PA.by_period(m, b),
                             "fees": PA.fee_summary(m),
                             "account": PA.summary(m)})
        except Exception as e:  # noqa: BLE001
            return self._fail("period_failed", 200, detail=str(e)[:160])

    def _api_weekly(self, query):
        """周报：模拟盘 / 各周期台账 / 归因 / 实时特征 / 模型健康。"""
        q = query or {}
        try:
            days = _clamp_int(q.get("days"), 7, 1, 90)
        except Exception:
            days = 7
        force = str((q.get("refresh") or ["0"])[0]) in ("1", "true", "yes")
        try:
            from qcore import weekly as WK
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        try:
            d = WK.build(days=days, verbose=False) if force else WK.summary()
            if not d or not d.get("summary_lines"):
                d = WK.build(days=days, verbose=False)
            return self._ok(d)
        except Exception as e:  # noqa: BLE001
            return self._fail("weekly_failed", 200, detail=str(e)[:160])

    def _api_paper_auto(self, query=None):
        """
        模拟盘自动交易。query: market=CN|US|HK，all=1 取三个市场。
        三个市场各自独立的本金、成本、币种、成交，绝不混在一起。
        """
        q = query or {}
        want_all = str((q.get("all") or ["0"])[0]) in ("1", "true", "yes")
        m = str((q.get("market") or ["CN"])[0]).upper()
        try:
            from qcore import paper_auto as PA
            if want_all:
                return self._ok(PA.summary_all())
            return self._ok(PA.summary(m))
        except Exception as e:  # noqa: BLE001
            return self._fail("paper_auto_failed", 200, detail=str(e)[:160])

    def _post_paper_rebalance(self):
        """
        按当前 picks 自动调仓。body: {"dry_run": true, "market": "CN", "top_k": 20}
        """
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        body = body if isinstance(body, dict) else {}
        try:
            from qcore import paper_auto as PA
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        try:
            r = PA.rebalance(market=str(body.get("market") or "CN"),
                             top_k=(int(body["top_k"]) if body.get("top_k") else None),
                             dry_run=bool(body.get("dry_run")),
                             verbose=False)
        except Exception as e:  # noqa: BLE001
            return self._fail("rebalance_failed", 200, detail=str(e)[:160])
        if not r.get("ok"):
            return self._fail(r.get("reason", "rebalance_failed"), 200,
                              hint=r.get("hint"))
        return self._ok(r)

    def _api_live_features(self):
        """实时特征（从 tick 提取）：积累天数 + 是否够训练。"""
        try:
            from qcore import live_features as LF
            return self._ok(LF.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("live_features_failed", 200, detail=str(e)[:160])

    def _api_train_kit(self):
        """
        训练机套件的版本清单 —— 服务端下发的更新源。
        训练机轮询这个接口，版本比自己新就自动下载覆盖。
        """
        ver = _read_version()
        repo = os.environ.get("APB_REPO", "ybd156/quant-apb")
        base = f"https://github.com/{repo}/releases/download/v{ver}"
        name = f"apb-train-kit_{ver}.zip"
        payload = {
            "version": ver,
            "name": name,
            "url": f"{base}/{name}",
            "url_mirror": f"https://ghproxy.net/{base}/{name}",
            "released_at": None,
            "note": ("训练机执行 python tools/sync_agent.py --auto-update "
                     "会自动比对版本并覆盖更新"),
            "public_url": os.environ.get("APB_PUBLIC_URL", ""),
        }
        # 永远给 local_url —— NAS 自己当镜像，训练机只连 NAS 就行，
        # 不需要能访问 GitHub（国内很多网络到不了）。
        payload["local_url"] = "/api/train/kit/download"
        payload["local_file"] = name
        return self._ok(payload)

    def _api_hotspot(self, query):
        """
        当日风口：涨停梯队 / 跌停 / 板块强度 / 主力资金。
        query: refresh=1 强制重抓（默认读缓存，5 分钟内不重复抓）
        """
        q = query or {}
        force = str((q.get("refresh") or ["0"])[0]) in ("1", "true", "yes")
        try:
            from qcore import hotspot as HS
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        try:
            if not force:
                d = HS.load_snapshots(1)
                if d and d[0].get("date"):
                    return self._ok(d[0])
            return self._ok(HS.analyze(verbose=False))
        except Exception as e:  # noqa: BLE001
            return self._fail("hotspot_failed", 200, detail=str(e)[:160])

    def _api_horizons(self, query):
        """
        多周期：列出各 horizon 的模型状态 + 成本分析。
        query: horizon=N 时附带该周期的 picks。
        """
        try:
            from qcore import horizons as HZ
            from qcore import live_model as LM
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        market = str(((query or {}).get("market") or ["CN"])[0]).upper()
        rows = []
        for h in HZ.all_horizons():
            mp, _ = LM._paths(h)
            exists = _os.path.exists(mp) if "_os" in dir() else __import__("os").path.exists(mp)
            meta = {}
            if exists:
                try:
                    _, meta = LM.load_model(h)
                except Exception:
                    meta = {}
            rows.append({
                "horizon": h, "label": HZ.label_for(h),
                "note": (HZ.HORIZONS.get(h) or {}).get("note"),
                "supported_markets": (HZ.HORIZONS.get(h) or {}).get("markets"),
                "market_supported": HZ.supports(h, market),
                "model_exists": exists,
                "trained_at": meta.get("trained_at"),
                "n_trees": meta.get("n_trees"),
                "cost_drag_annual": round(HZ.annual_cost_drag(max(h, 1), market) * 100, 2),
            })
        # 各周期的真实战绩（台账对账出来的）
        perf = {}
        try:
            from qcore import horizon_ledger as HL
            for st in (HL.summary().get("horizons") or []):
                perf[int(st["horizon"])] = st
        except Exception:
            pass
        for row in rows:
            st = perf.get(int(row["horizon"])) or {}
            row["perf"] = {
                "n": st.get("n", 0),
                "hit_rate": st.get("hit_rate"),
                "top_mean": st.get("top_mean"),
                "excess_mean": st.get("excess_mean"),
                "recent90_top": st.get("recent90_top"),
                "pending": st.get("pending", 0),
            }

        # 该市场支持哪些周期、扣成本后能不能做
        table = HZ.table({h: 0.03 for h in HZ.all_horizons()}, market)
        want = ((query or {}).get("horizon") or [None])[0]
        out = {"market": market, "default": HZ.DEFAULT_HORIZON,
               "horizons": rows, "cost_table_at_ic_003": table,
               "note": ("成本随周期剧烈变化：h=1 年换手 250 次，A 股光成本就 38.6%/年；"
                        "h=15 只有 2.6%。成本表用 IC=0.03 假设做对照，不是收益预测。")}
        if want:
            try:
                hh = int(want)
                fn = "picks.json" if hh == 1 else f"picks_h{hh}.json"
                d, err = _read_json(fn)
                out["picks"] = (d if not err else None)
                out["picks_file"] = fn
                out["picks_error"] = err
            except Exception as e:  # noqa: BLE001
                out["picks_error"] = str(e)[:80]
        return self._ok(out)

    def _api_event_learning(self, query):
        """
        事件学习：预测方向 vs 实际反应。query: run=1 跑一轮（登记+对账+校准）。
        返回命中率（按 type 和 type×sector）与校准倍数。
        """
        try:
            from qcore import event_learning as EL
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        force = str(((query or {}).get("run") or ["0"])[0]) in ("1", "true", "yes")
        if force:
            try:
                return self._ok(EL.run(verbose=False))
            except Exception as e:  # noqa: BLE001
                return self._fail("run_failed", 200, detail=str(e)[:160])
        d = EL.summary()
        if not d.get("hit_rate"):
            return self._not_ready("event_learning.json",
                                   "还没有事件学习数据，跑一次 /api/event/learning?run=1")
        return self._ok(d)

    def _api_news_live(self, query):
        """
        实时政策/事件快讯。query: refresh=1 强制刷新。
        返回规则分类结果（type/direction/sectors/level），reliability=0.5。
        """
        try:
            from qcore import news_live as NL
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        force = str(((query or {}).get("refresh") or ["0"])[0]) in ("1", "true", "yes")
        if force:
            r = NL.refresh(verbose=False)
            if not r.get("ok"):
                return self._fail(r.get("reason", "refresh_failed"), 200)
            NL.merge_into_events(verbose=False)
            return self._ok(r)
        d = NL.load()
        if not d.get("rows"):
            return self._not_ready("events_live.json",
                                   "还没有实时事件，跑一次 /api/news/live?refresh=1")
        return self._ok(d)

    def _api_data_quality(self):
        """最近一次数据清洗的汇总（data_quality.json）。"""
        try:
            from qcore import cleaning
            d = cleaning.summary()
            return self._ok(d) if d.get("n_input") is not None else self._fail(
                "not_ready", 200, hint=d.get("hint", "先跑 qcore.cleaning"))
        except Exception as e:  # noqa: BLE001
            return self._fail("quality_failed", 200, detail=str(e)[:160])

    def _api_calendar(self, query):
        """交易日历：今天是不是交易日 / 下一个交易日。query: date=YYYY-MM-DD"""
        try:
            from qcore import calendar_cn as CAL
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        d = ((query or {}).get("date") or [None])[0]
        try:
            return self._ok(CAL.summary(d))
        except Exception as e:  # noqa: BLE001
            return self._fail("calendar_failed", 200, detail=str(e)[:160])

    def _api_lot(self, query):
        """
        整手约束下的可成交性。query:
          capital=100000      本金
          market=CN|US|HK
          mode=plan|scenarios|small   （默认 plan）
        """
        try:
            from qcore import lot as L
            import json as _j
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        try:
            cap = float(((query or {}).get("capital") or ["100000"])[0])
        except Exception:
            return self._fail("bad_param", 400, hint="capital 必须是数字")
        mkt = str(((query or {}).get("market") or ["CN"])[0]).upper()
        mode = str(((query or {}).get("mode") or ["plan"])[0]).lower()

        picks, _e = _read_json("picks.json")
        rows = (picks or {}).get("rows") or []
        if not rows:
            return self._not_ready("picks.json", "还没有推荐，先生成 picks")
        scores = {str(r.get("code", "")).zfill(6): float(r.get("score") or 0) for r in rows}
        prices = {}
        try:
            from qcore import tick
            snap = tick.snapshot(list(scores.keys()))
            for x in (snap.get("rows") or []):
                c = str(x.get("code", ""))
                prices[c[2:] if c[:2] in ("sh", "sz", "bj") else c] = x.get("price")
        except Exception:
            for r in rows:
                if r.get("close"):
                    prices[str(r.get("code", "")).zfill(6)] = r.get("close")
        if not prices:
            return self._fail("no_price", 200, hint="取不到现价，无法计算整手约束")

        if mode == "scenarios":
            return self._ok(L.scenarios(scores, prices, mkt))
        if mode == "small":
            return self._ok(L.small_capital(cap, scores, prices, mkt))
        n = len(scores) or 1
        w = {c: 1.0 / n for c in scores}
        return self._ok(L.plan(cap, w, prices, mkt))

    def _sync_auth(self):
        """同步端点的鉴权：设了 APB_TOKEN 就要 X-APB-Token（或 X-Token）。"""
        expected = os.environ.get("APB_TOKEN", "")
        given = (self.headers.get("X-APB-Token") or self.headers.get("X-Token") or "")
        if expected:
            if not hmac.compare_digest(given, expected):
                return self._fail("unauthorized", 403,
                                  hint="X-APB-Token 与 APB_TOKEN 不匹配")
        else:
            client = (self.client_address or ("", 0))[0]
            if not self._is_internal():
                return self._fail("forbidden", 403,
                                  hint="未设 APB_TOKEN 时只允许内网")
        return None

    # ============================== 登录 ==============================

    LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>登录 · APB</title>
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:#0d1117;color:#d5dae3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
  "PingFang SC","Microsoft YaHei",sans-serif}
.card{width:min(380px,calc(100vw - 40px));padding:32px 28px;border-radius:16px;
  background:#151b23;border:1px solid rgba(120,140,180,.18);
  box-shadow:0 20px 60px rgba(0,0,0,.45)}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#4a7dff,#6aa6ff);
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#fff}
h1{font-size:17px;margin:0;font-weight:600}
.sub{font-size:12.5px;color:#7d8794;margin:0 0 22px}
label{display:block;font-size:12.5px;color:#9aa4b2;margin-bottom:7px}
input{width:100%;padding:11px 13px;border-radius:9px;font-size:14px;
  border:1px solid rgba(120,140,180,.28);background:rgba(120,140,180,.06);color:#d5dae3}
input:focus{outline:none;border-color:#4a7dff;background:rgba(74,125,255,.08)}
button{width:100%;margin-top:16px;padding:11px;border-radius:9px;border:0;cursor:pointer;
  background:linear-gradient(135deg,#4a7dff,#6aa6ff);color:#fff;font-size:14px;font-weight:500}
button:hover{filter:brightness(1.08)}
button:disabled{opacity:.6;cursor:not-allowed}
.err{color:#ff6b6b;font-size:12.5px;margin-top:12px;min-height:18px}
.foot{margin-top:18px;font-size:11.5px;color:#5b6470;text-align:center}
</style></head><body>
<form class="card" id="f">
  <div class="brand"><div class="logo">APB</div><h1>自学习参数体</h1></div>
  <p class="sub">请输入访问密码</p>
  <label for="pw">密码</label>
  <input id="pw" type="password" autocomplete="current-password" autofocus required>
  <button type="submit" id="btn">进入</button>
  <div class="err" id="err"></div>
  <div class="foot">连续输错会被临时锁定</div>
</form>
<script>
const f=document.getElementById('f'),pw=document.getElementById('pw'),
      btn=document.getElementById('btn'),err=document.getElementById('err');
f.addEventListener('submit',async(e)=>{
  e.preventDefault();
  btn.disabled=true; err.textContent='';
  try{
    const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({password:pw.value})});
    const d=await r.json().catch(()=>({}));
    if(d && d.ok){ location.replace('/'); return; }
    err.textContent=(d && (d.hint||d.error))||'登录失败';
  }catch(ex){ err.textContent='网络错误'; }
  btn.disabled=false; pw.select();
});
</script></body></html>"""

    def _auth_check(self):
        """
        全局门禁。返回 None 放行；否则已写好响应。
        会话 cookie 或 X-APB-Token（机器访问）任一有效即通过。
        """
        try:
            from qcore import auth as AUTH
        except Exception:
            return None
        if not AUTH.enabled():
            return None
        path = (self.path or "/").split("?")[0]

        # /login 本身：已登录就直接回首页，未登录给登录页
        if path == "/login":
            sid0 = self._cookie("apb_session")
            if sid0 and AUTH.verify_session(sid0, touch=False):
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True
            body = self.LOGIN_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._security_headers()
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return True

        if AUTH.is_public_path(path):
            return None

        # 机器访问：X-APB-Token（训练机同步代理用）
        expected = os.environ.get("APB_TOKEN", "")
        given = self.headers.get("X-APB-Token") or ""
        if expected and given and hmac.compare_digest(given, expected):
            return None

        sid = self._cookie("apb_session")
        if sid and AUTH.verify_session(sid):
            return None

        if path.startswith("/api/"):
            return self._fail("login_required", 401,
                              hint="请先登录（浏览器打开首页输入密码）")
        # 非 API：给登录页
        body = self.LOGIN_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
        return True

    def _cookie(self, name: str) -> str:
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v
        return ""

    def _set_cookie(self, name: str, value: str, max_age: int, secure: bool) -> None:
        parts = [f"{name}={value}", "Path=/", f"Max-Age={int(max_age)}",
                 "HttpOnly", "SameSite=Lax"]
        if secure:
            parts.append("Secure")
        self.send_header("Set-Cookie", "; ".join(parts))

    def _api_login(self):
        try:
            from qcore import auth as AUTH
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        if not AUTH.enabled():
            return self._fail("login_disabled", 200,
                              hint="服务端未设 APB_PASSWORD，登录未启用")
        ip = self._real_ip()
        limited, wait = AUTH.rate_limited(ip)
        if limited:
            return self._fail("too_many_attempts", 429,
                              hint=f"失败次数过多，请 {wait} 秒后再试")
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        pw = str((body or {}).get("password") or "")
        if not AUTH.check_password(pw):
            AUTH.record_fail(ip)
            return self._fail("bad_password", 401, hint="密码不对")
        AUTH.clear_fails(ip)
        sid = AUTH.create_session(ip, self.headers.get("User-Agent") or "")
        secure = self._is_https()
        payload = {"ok": True, "data": {"ok": True, "expires_days": AUTH.SESSION_DAYS}}
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self._set_cookie("apb_session", sid, int(AUTH.SESSION_DAYS * 86400), secure)
        self.end_headers()
        try:
            self.wfile.write(raw)
        except Exception:
            pass
        return True

    def _api_logout(self):
        try:
            from qcore import auth as AUTH
            sid = self._cookie("apb_session")
            if sid:
                AUTH.destroy_session(sid)
        except Exception:
            pass
        payload = {"ok": True, "data": {"ok": True}}
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self._set_cookie("apb_session", "", 0, self._is_https())
        self.end_headers()
        try:
            self.wfile.write(raw)
        except Exception:
            pass
        return True

    def _real_ip(self) -> str:
        try:
            from qcore import net_guard as NG
            return NG.real_ip((self.client_address or ("", 0))[0],
                              {k: v for k, v in (self.headers or {}).items()})
        except Exception:
            return (self.client_address or ("", 0))[0]

    def _is_https(self) -> bool:
        """判断是否 HTTPS（决定 cookie 要不要 Secure）。"""
        xf = (self.headers.get("X-Forwarded-Proto") or "").lower()
        if xf:
            return xf == "https"
        cf = (self.headers.get("CF-Visitor") or "").lower()
        if "https" in cf:
            return True
        return False

    def _api_auth(self):
        try:
            from qcore import auth as AUTH
            st = AUTH.status()
            st["logged_in"] = bool(self._cookie("apb_session")
                                   and AUTH.verify_session(self._cookie("apb_session"), touch=False))
            st["you_are"] = self._real_ip()
            return self._ok(st)
        except Exception as e:  # noqa: BLE001
            return self._fail("auth_status_failed", 200, detail=str(e)[:160])

    def _guard_request(self):
        """
        统一前置检查：Host 头（防 DNS rebinding）+ CORS 预检。
        返回 None 表示放行；否则返回已经写好的响应。
        """
        try:
            from qcore import net_guard as NG
        except Exception:
            return None
        host = self.headers.get("Host") or ""
        ok, why = NG.host_allowed(host)
        if not ok:
            return self._fail("host_not_allowed", 403, detail=why,
                              hint="把域名加进 APB_ALLOWED_HOSTS")
        return None

    def _security_headers(self):
        """
        统一的安全响应头。缺这些会被浏览器当成"不设防的站点"：
          X-Frame-Options      防点击劫持（别人用 iframe 套你的页面骗点击）
          X-Content-Type-Options 防 MIME 嗅探（把 JSON 当脚本执行）
          Referrer-Policy      不要把带 token 的 URL 泄漏给第三方
          CSP                  限制可执行脚本来源，缓解 XSS
          HSTS                 仅在 HTTPS 下加，强制后续走 HTTPS
        """
        try:
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'")
            if self._is_https():
                self.send_header("Strict-Transport-Security",
                                 "max-age=31536000; includeSubDomains")
        except Exception:
            pass

    def _cors_apply(self):
        """把 CORS 头写进响应（只有白名单里的 Origin 才会拿到）。"""
        try:
            from qcore import net_guard as NG
            origin = self.headers.get("Origin") or ""
            for k, v in (NG.cors_headers(origin) or {}).items():
                self.send_header(k, v)
        except Exception:
            pass

    def _is_internal(self) -> bool:
        """
        「是否内网请求」—— 写接口免 token 的唯一判断依据。
        **必须走 net_guard**：反代之后 client_address 是 127.0.0.1，
        直接用它判断会把外网请求当内网（真实安全洞）。
        """
        try:
            from qcore import net_guard as NG
            client = (self.client_address or ("", 0))[0]
            hdrs = {k: v for k, v in (self.headers or {}).items()}
            return NG.is_internal_request(client, hdrs)
        except Exception:
            client = (self.client_address or ("", 0))[0]
            return _is_private_ip(client)

    def _api_kline(self, query):
        """
        个股 K 线 + 均线 + 量能 + 区间统计。
        query: code=600519 & period=daily|weekly|monthly & count=180 & adjust=0
        """
        q = query or {}
        code = str((q.get("code") or [""])[0]).strip()
        if not code:
            return self._fail("bad_param", 400, hint="需要 code，如 600519 / 00700.HK / AAPL")
        period = str((q.get("period") or ["daily"])[0]).lower()
        count = _clamp_int(q.get("count"), 180, 30, 1000)
        try:
            adjust = int((q.get("adjust") or ["0"])[0])
        except (TypeError, ValueError):
            adjust = 0
        if adjust not in (0, 1, 2):
            adjust = 0
        try:
            from qcore import stock_chart as SC
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        try:
            r = SC.build(code, period=period, count=count, adjust=adjust)
        except Exception as e:  # noqa: BLE001
            return self._fail("kline_failed", 200, detail=str(e)[:160])
        if not r.get("ok"):
            return self._fail(r.get("reason", "kline_failed"), 200,
                              detail=r.get("detail"),
                              supported=r.get("supported"))
        return self._ok(r)

    def _api_healthz(self):
        """给反向代理/容器编排用的极简健康检查（不读任何状态文件）。"""
        return self._ok({"status": "ok", "version": _read_version()})

    def _api_netguard(self):
        """反代配置自检：哪些没配好、能不能暴露公网。"""
        try:
            from qcore import net_guard as NG
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        st = NG.status()
        client = (self.client_address or ("", 0))[0]
        hdrs = {k: v for k, v in (self.headers or {}).items()}
        st["your_ip"] = NG.real_ip(client, hdrs)
        st["you_are_internal"] = NG.is_internal_request(client, hdrs)
        st["host_header"] = self.headers.get("Host") or ""
        return self._ok(st)

    def _api_sync_summary(self):
        try:
            from qcore import sync as SY
            return self._ok(SY.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("sync_failed", 200, detail=str(e)[:160])

    def _api_sync_task(self, query):
        """GET /api/sync/task —— 查看任务队列。"""
        try:
            from qcore import sync as SY
            limit = _clamp_int((query or {}).get("limit"), 30, 1, 200)
            return self._ok(SY.list_tasks(limit=limit))
        except Exception as e:  # noqa: BLE001
            return self._fail("task_failed", 200, detail=str(e)[:160])

    def _post_sync_task(self):
        """
        POST /api/sync/task
          {"horizons":[1,5],"trees":250,"stocks":300}  排新任务
          {"claim":true,"worker":"xxx"}                领一个任务
        """
        denied = self._sync_auth()
        if denied:
            return denied
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        body = body if isinstance(body, dict) else {}
        try:
            from qcore import sync as SY
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        if body.get("claim"):
            return self._ok(SY.claim_task(worker=str(body.get("worker") or "unknown")))
        hs = body.get("horizons") or [1]
        try:
            hs = [int(x) for x in hs]
        except Exception:
            return self._fail("bad_param", 400, hint="horizons 必须是整数数组")
        return self._ok(SY.create_task(hs,
                                       trees=int(body.get("trees") or 250),
                                       stocks=int(body.get("stocks") or 300),
                                       note=str(body.get("note") or "")))

    def _post_sync_result(self):
        denied = self._sync_auth()
        if denied:
            return denied
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        body = body if isinstance(body, dict) else {}
        tid = body.get("id")
        if not tid:
            return self._fail("bad_param", 400, hint="需要 id")
        try:
            from qcore import sync as SY
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        r = SY.report_task(str(tid), bool(body.get("ok")),
                           detail=body.get("detail") or {},
                           worker=str(body.get("worker") or ""))
        return self._ok(r) if r.get("ok") else self._fail(r.get("reason", "failed"), 200)

    def _post_sync_push(self):
        """POST /api/sync/push —— 训练机直接上传模型（base64 zip）。"""
        denied = self._sync_auth()
        if denied:
            return denied
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        if not isinstance(body, dict):
            return self._fail("bad_param", 400, hint="需要 JSON 对象")
        data = body.get("data") or body.get("base64")
        if not data:
            return self._fail("bad_param", 400, hint="需要 data（zip 的 base64）")
        h = body.get("horizon")
        try:
            from qcore import model_transfer as MT
            r = MT.import_bundle(str(data), horizon=(int(h) if h else None))
        except Exception as e:  # noqa: BLE001
            return self._fail("push_failed", 200, detail=str(e)[:160])
        if not r.get("ok"):
            return self._fail(r.get("reason", "validation_failed"), 200,
                              detail=r.get("detail"), checks=r.get("checks"),
                              hint="校验未通过，未安装；线上模型保持不变")
        return self._ok(r)

    def _api_model_local(self):
        """本机各 horizon 的模型状态（训练可以在别的机器做，导入进来）。"""
        try:
            from qcore import model_transfer as MT
            return self._ok(MT.list_local())
        except Exception as e:  # noqa: BLE001
            return self._fail("local_failed", 200, detail=str(e)[:160])

    def _post_model_import(self):
        """
        导入别处训练的模型。body: {"data": "<zip 的 base64>", "horizon": 5, "dry_run": false}
        校验不过不安装，线上模型保持不变。
        """
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        if not isinstance(body, dict):
            return self._fail("bad_param", 400, hint="需要 JSON 对象")
        data = body.get("data") or body.get("base64") or body.get("zip")
        if not data:
            return self._fail("bad_param", 400, hint="需要 data（zip 的 base64）")
        h = body.get("horizon")
        try:
            from qcore import model_transfer as MT
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        try:
            r = MT.import_bundle(str(data), horizon=(int(h) if h else None),
                                 dry_run=bool(body.get("dry_run")))
        except Exception as e:  # noqa: BLE001
            return self._fail("import_failed", 200, detail=str(e)[:160])
        if not r.get("ok"):
            return self._fail(r.get("reason", "validation_failed"), 200,
                              detail=r.get("detail"), checks=r.get("checks"),
                              hint="校验未通过，未安装；线上模型保持不变")
        return self._ok(r)

    def _post_model_export(self):
        """导出某个 horizon 的模型为 zip（base64 返回）。body: {"horizon": 5}"""
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        h = (body or {}).get("horizon", 1) if isinstance(body, dict) else 1
        try:
            from qcore import model_transfer as MT
            r = MT.export_bundle(int(h))
        except Exception as e:  # noqa: BLE001
            return self._fail("export_failed", 200, detail=str(e)[:160])
        if not r.get("ok"):
            return self._fail(r.get("reason", "export_failed"), 200, path=r.get("path"))
        try:
            import base64 as _b64
            with open(r["path"], "rb") as f:
                r["data"] = _b64.b64encode(f.read()).decode("ascii")
        except Exception:
            pass
        return self._ok(r)

    def _api_model_health(self):
        """模型失效判据：healthy / watch / degraded / failed。"""
        try:
            from qcore import model_health as MH
            return self._ok(MH.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("health_failed", 200, detail=str(e)[:160])

    def _api_model_versions(self):
        """模型版本列表 + 备份列表。"""
        try:
            from qcore import versioning as V
            return self._ok(V.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("versions_failed", 200, detail=str(e)[:160])

    def _post_model_rollback(self):
        """回滚到指定模型版本。body: {"version":"20261109-081500"}"""
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        ver = (body or {}).get("version") if isinstance(body, dict) else None
        if not ver:
            return self._fail("bad_param", 400, hint="需要 version")
        try:
            from qcore import versioning as V
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        r = V.rollback(str(ver))
        if not r.get("ok"):
            return self._fail(r.get("reason", "rollback_failed"), 400,
                              available=r.get("available"))
        return self._ok(r)

    def _api_transfer(self):
        """导出/导入能力清单。"""
        try:
            from qcore import transfer
            return self._ok(transfer.summary())
        except Exception as e:  # noqa: BLE001
            return self._fail("transfer_failed", 200, detail=str(e)[:160])

    def _api_export(self, query):
        """
        导出数据。query:
          what=all|picks|ledger|watch|trades|factors|account|status|signals|
               regime|performance|paper|model|names|events|market
          format=auto|csv|json   （表格类默认 CSV，带 BOM 方便 Excel）
        """
        what = (query.get("what") or ["all"])[0]
        fmt = (query.get("format") or ["auto"])[0]
        try:
            from qcore import transfer
            r = transfer.export(what, fmt)
        except Exception as e:  # noqa: BLE001
            return self._fail("export_failed", 200, detail=str(e)[:160])
        if not r.get("ok"):
            return self._fail(r.get("reason", "export_failed"), 200,
                              hint=r.get("hint"))
        body = r["data"]
        try:
            self.send_response(200)
            self.send_header("Content-Type", r["content_type"])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{r["filename"]}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        return 200, len(body)

    def _post_import(self):
        """
        导入数据。body:
          {"watch":[...]} / {"trades":[...]} / {"account":{...}}
          或整体 state 包（export(what=all) 的产物），加 "_replace":true 才覆盖。
        默认**合并**，不覆盖历史（INV-7）。
        """
        body, err = self._read_body()
        if err:
            return self._fail(err, 400)
        try:
            from qcore import transfer
        except Exception as e:  # noqa: BLE001
            return self._fail("import_error", 500, detail=str(e)[:160])
        r = transfer.import_payload(body if isinstance(body, dict) else {})
        if not r.get("ok"):
            return self._fail(r.get("reason", "import_failed"), 400,
                              hint=r.get("hint"))
        return self._ok(r)

    # ---------------- 路由：POST ----------------

    def do_OPTIONS(self):
        """CORS 预检。只回白名单里的 Origin。"""
        try:
            from qcore import net_guard as NG
        except Exception:
            NG = None
        self.send_response(204)
        if NG is not None:
            for k, v in (NG.cors_headers(self.headers.get("Origin") or "") or {}).items():
                self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        # 必须在任何分支返回前把请求体读干净。
        # 否则在 keep-alive 连接上，残留的 body 会被当成下一个请求的起始行，
        # 表现为 501 "Unsupported method ('{...}POST')"。
        # Cloudflare / 反向代理会复用连接，所以这个错位是必现的（实测 50%）。
        # 每次请求都要重新 drain。handler 实例在 keep-alive 连接上是复用的，
        # 上一个请求缓存的 _raw_body 必须清掉，否则会跳过读取、残留 body。
        self._raw_body = None
        self._body_error = None
        if not self._drain_body():
            return
        _a = self._auth_check()
        if _a is not None:
            return
        _g = self._guard_request()
        if _g is not None:
            return
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
        if path == "/api/fees/save":
            return self._post_fees_save()
        if path == "/api/paper/rebalance":
            return self._post_paper_rebalance()
        if path == "/api/login":
            return self._api_login()
        if path == "/api/logout":
            return self._api_logout()
        if path == "/api/sync/task":
            return self._post_sync_task()
        if path == "/api/sync/result":
            return self._post_sync_result()
        if path == "/api/sync/push":
            return self._post_sync_push()
        if path == "/api/model/import":
            return self._post_model_import()
        if path == "/api/model/export":
            return self._post_model_export()
        if path == "/api/model/rollback":
            return self._post_model_rollback()
        if path == "/api/import":
            return self._post_import()
        if path == "/api/watch":
            return self._post_watch()
        if path == "/api/account/config":
            return self._post_account_config()
        if path == "/api/update/apply":
            return self._post_update_apply()
        if path.startswith("/api/"):
            return self._fail("unknown_endpoint", 404, path=path)
        return self._fail("not_found", 404, path=path)

    def _drain_body(self) -> bool:
        """
        把请求体一次性读进内存并缓存。返回 False 表示已经发过错误响应。
        超长时继续读空（不中断连接），但标记为 too_large。
        """
        if getattr(self, "_raw_body", None) is not None:
            return True      # 本次请求内重复调用（如 _read_body 兜底）
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            self._raw_body = b""
            self._body_error = "bad_length"
            return True
        if length <= 0:
            self._raw_body = b""
            self._body_error = None
            return True
        if length > MAX_BODY_BYTES:
            # 读完丢弃，保持连接可用；再由 _read_body 返回错误
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self._raw_body = b""
            self._body_error = "body_too_large"
            return True
        try:
            self._raw_body = self.rfile.read(length)
        except Exception:
            self._raw_body = b""
            self._body_error = "read_error"
        self._body_error = None
        return True

    def _read_body(self):
        """返回 (obj, error)。body 已在 do_POST 入口读完并缓存。"""
        if getattr(self, "_raw_body", None) is None:
            self._drain_body()
        if getattr(self, "_body_error", None):
            return None, self._body_error
        raw = getattr(self, "_raw_body", b"") or b""
        if not raw:
            return None, "empty_body"
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
            if not self._is_internal():
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
            if not self._is_internal():
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
        try:
            from qcore import net_guard as _NG
            client = _NG.real_ip(self.client_address[0] if self.client_address else "",
                                 {k: v for k, v in (self.headers or {}).items()}) or "-"
        except Exception:
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
