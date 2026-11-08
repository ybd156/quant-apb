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
}
LIVE_TRADES_FILE = "live_trades.jsonl"

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
        for key in ("rows", "items", "data", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


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
        if path == "/api/health":
            return self._ok({"state_dir": str(STATE_DIR), "static_dir": str(STATIC_DIR),
                             "run_enabled": ALLOW_RUN})
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

        expected = os.environ.get("APB_TOKEN", "")
        if not expected:
            return self._fail("token_not_configured", 403,
                              hint="服务端未设置环境变量 APB_TOKEN，/api/run 一律拒绝")
        given = self.headers.get("X-Token") or ""
        if not hmac.compare_digest(given, expected):
            return self._fail("unauthorized", 403, hint="X-Token 与 APB_TOKEN 不匹配")

        with _RUN_LOCK:
            if _RUN["state"] == "running":
                return self._fail("busy", 409, hint="已有一次运行在进行中")
            _RUN.update({"state": "running", "started_at": _now_iso(),
                         "finished_at": None, "returncode": None,
                         "duration_s": None, "tail": "", "error": None})
        threading.Thread(target=_run_engine, name="apb-run", daemon=True).start()
        return self._ok({"started": True, "cmd": _run_cmd()})

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
