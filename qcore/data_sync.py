"""
行情数据同步 —— 让 Qlib 社区数据保持最新，解决「陈旧 N 天」。

## 问题

`data/qlib/` 是我一次性下载的静态包。下载那天之后就不会再更新，
所以 `status.json` 的 `as_of` 一直停在下载日，UI 上显示「陈旧 3 天」。

## 解法

`chenditc/investment_data` 每天在 GitHub Releases 发新包（约 545MB）。
本模块：
  1. 查远端最新 tag（YYYY-MM-DD 形式）
  2. 与本地记录比对，不新就跳过
  3. 需要时下载 + 解压到 `data/qlib/x/qlib_bin`
  4. 写 `data/state/data_sync.json` 记录同步状态

## 网络

GitHub 在国内经常不可达。本模块用 `qcore.net` 的**自动选路**：
先直连，失败走代理（实测 NAS 上必须走代理才能下载 545MB 的包）。

## 磁盘

解压后约 3.7GB。解压前检查剩余空间，不够就中止并报错，不要写坏磁盘。

## 原子性

先解压到 `qlib_bin.new`，成功后再原子替换 `qlib_bin`。
中途失败不会破坏现有数据。
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
import urllib.error
from datetime import datetime
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
QLIB_DIR = os.path.join(DATA, "qlib")


def _target() -> str:
    """
    数据落地目录。**必须尊重 APB_QLIB_DIR** —— 否则用户指了别的数据目录，
    这里还傻傻去下载 545MB（实测踩过：--skip-data 失效，明明有数据还下载）。
    """
    env = os.environ.get("APB_QLIB_DIR", "").strip()
    if env:
        return env
    return os.path.join(QLIB_DIR, "x", "qlib_bin")


TARGET = _target()
STAMP = os.path.join(DATA, "state", "data_sync.json")
TARBALL = os.path.join(QLIB_DIR, "qlib_bin.tar.gz")

API = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
DL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
MIN_FREE_GB = 6.0          # 解压 3.7GB + 压缩包 0.55GB + 余量


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_stamp() -> Dict:
    try:
        with open(STAMP, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def local_date() -> Optional[str]:
    """本地数据最后一天（从日历读）。"""
    p = os.path.join(_target(), "calendars", "day.txt")
    if not os.path.isfile(p):
        return None
    try:
        last = None
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    last = s
        return last
    except Exception:
        return None


def remote_tag(timeout: int = 20) -> Optional[str]:
    """远端最新数据版本（YYYY-MM-DD）。

    先查 GitHub release tag；拿不到就**问 NAS**（`/api/data/version`）。
    为什么必须回退：训练机在国内常常连不上 GitHub，`tag_name` 拿不到 →
    上层判定「远端版本未知」→ 跳过下载 → 数据永远停在旧日期。
    实测：训练机停在 2026-11-06，而 NAS 已有 2026-11-09，中间差两个交易日。
    """
    from . import net
    body, _mode = net.get(API, timeout_direct=10, timeout_proxy=20)
    if body:
        try:
            d = json.loads(body)
            tag = d.get("tag_name")
            if tag:
                return tag
        except Exception:  # noqa: BLE001
            pass
    # 回退：NAS 自己的数据版本
    for base in _mirror_urls():
        try:
            b = base.split("/api/")[0] if "/api/" in base else base.rstrip("/")
            body2, _m2 = net.get(f"{b}/api/data/version", timeout_direct=8, timeout_proxy=8)
            if not body2:
                continue
            d2 = json.loads(body2)
            v = ((d2.get("data") or {}) if isinstance(d2, dict) else {}).get("date")
            if v:
                return str(v)
        except Exception:  # noqa: BLE001
            continue
    return None


def _free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        try:
            import shutil as _sh
            return _sh.disk_usage(path).free / (1024 ** 3)
        except Exception:
            return 999.0


def _token() -> str:
    """
    取访问 NAS 用的 token。优先级：
      1. 环境变量 APB_TOKEN
      2. apb_defaults.json 里的 token 字段
    """
    t = os.environ.get("APB_TOKEN", "").strip()
    if t:
        return t
    for root in (ROOT, os.path.join(ROOT, "tools")):
        pj = os.path.join(root, "apb_defaults.json")
        if os.path.isfile(pj):
            try:
                with open(pj, encoding="utf-8") as f:
                    d = json.load(f) or {}
                t = str(d.get("token") or d.get("nas_token") or "").strip()
                if t:
                    return t
            except Exception:
                pass
    return ""


def _mirror_urls() -> list:
    """
    下载源，按优先级：
      1. NAS 自己（训练机只连 NAS 就行，不需要能访问 GitHub）
      2. GitHub 直连
      3. GitHub 镜像

    实测价值：国内训练机访问 GitHub 常报 remote_unreachable，
    而 NAS 上本来就有这份数据，局域网几秒就传完。
    """
    urls = []
    ns = ""
    try:
        ns = os.environ.get("APB_NAS", "").strip()
        # 1) apb_defaults.json（训练机套件里才有）
        if not ns:
            for root in (ROOT, os.path.join(ROOT, "tools")):
                pj = os.path.join(root, "apb_defaults.json")
                if os.path.isfile(pj):
                    with open(pj, encoding="utf-8") as f:
                        ns = str((json.load(f) or {}).get("nas_url") or "").strip()
                    if ns:
                        break
        # 2) tools/defaults.py 里的兜底常量（源码仓库也有）
        if not ns:
            try:
                import sys as _sys
                _tp = os.path.join(ROOT, "tools")
                if _tp not in _sys.path:
                    _sys.path.insert(0, _tp)
                import defaults as _DEF  # type: ignore
                ns = _DEF.nas()
            except Exception:
                ns = ""
    except Exception:
        pass
    # APB_DATA_NO_SELF=1：跳过 NAS 自己这个源。
    # NAS 上跑定时同步时必须这样 —— 否则它从自己的 /api/data/qlib 下载，
    # 那个端点打包的就是本地这份旧数据，同步完还是旧的（自己喂自己）。
    _no_self = os.environ.get("APB_DATA_NO_SELF", "").strip() in ("1", "true", "yes")
    if ns and not _no_self:
        # 请求精简包：全量 546MB 走公网要 19 分钟，实测 0.47MB/s。
        # csi300 的 300 只票压缩后 44MB，1.5 分钟能下完（实测）。
        # 股票池可通过 APB_UNIVERSE / APB_MAX_STOCKS 覆盖。
        uv = os.environ.get("APB_UNIVERSE", "csi300").strip() or "csi300"
        n = os.environ.get("APB_MAX_STOCKS", "300").strip() or "300"
        urls.append(f"{ns.rstrip('/')}/api/data/qlib?universe={uv}&max_stocks={n}")
    urls.append(DL)
    urls.append("https://ghproxy.net/" + DL)
    return [u for u in urls if u]


def download(verbose: bool = True) -> Dict:
    """
    下载 tarball。支持断点续传 + 多源回退（NAS -> GitHub -> 镜像）。
    返回 {"ok", "bytes", "mode", "reason", "url"}。
    """
    from . import net
    os.makedirs(QLIB_DIR, exist_ok=True)
    have = os.path.getsize(TARBALL) if os.path.isfile(TARBALL) else 0
    t0 = time.time()
    mode_used = "none"

    candidates = _mirror_urls()
    if verbose:
        print(f"    {len(candidates)} 个下载源，依次尝试", flush=True)

    for src_url in candidates:
      # 每个源开始前重新读本地文件大小：416 之后 have 可能是旧的，
      # 而且不同源（全量 vs 精简包）大小差 10 倍，不能共用 have
      try:
          have = os.path.getsize(TARBALL) if os.path.isfile(TARBALL) else 0
      except Exception:
          have = 0
      if verbose:
          print(f"    -> {src_url[:70]}", flush=True)
      for attempt in range(3):
        # 先试直连再试代理（net.get 内部已有选路，但大文件要自己控制流式写盘）
        for mode, proxy in (("direct", None), ("proxy", net.discover_proxy())):
            if mode == "proxy" and not proxy:
                continue
            try:
                import urllib.request
                headers = {"User-Agent": "apb", "Accept": "application/octet-stream"}
                # NAS 上的 /api/data/qlib 需要鉴权（服务设了 APB_PASSWORD 后
                # 所有非公开端点都要求登录或 token）。训练机用 token 访问。
                # 实测：不带 token 直接 401，请求立刻失败，**没有任何下载流量**，
                # 用户以为是"没连上"，其实是鉴权挡住了。
                _tk = _token()
                if _tk:
                    headers["X-APB-Token"] = _tk
                    headers["X-Token"] = _tk
                if have > 0:
                    headers["Range"] = f"bytes={have}-"
                req = urllib.request.Request(src_url, headers=headers)
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy
                                                else {}))
                with opener.open(req, timeout=60) as r, open(TARBALL, "ab" if have else "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0) + have
                    done = have
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if verbose and total and done % (100 << 20) < (8 << 20):
                            print(f"    {done/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"{done/max(time.time()-t0,1)/1e6:.1f} MB/s", flush=True)
                size = os.path.getsize(TARBALL)
                # 完整性阈值：全量包 546MB，精简包 44MB。
                # 低于 20MB 视为不完整（可能只传了错误页）。
                if size > 20 * 1024 * 1024:
                    return {"ok": True, "bytes": size, "mode": mode,
                            "url": src_url,
                            "seconds": round(time.time() - t0)}
                have = size
                mode_used = mode
            except urllib.error.HTTPError as e:  # noqa: F821
                # 416 = 请求的 Range 起点 >= 远端文件大小。
                # 这说明**本地文件已经下载完整了**，不是错误。
                #
                # 实测踩过：TARBALL 已是最新的 571MB，但代码每次都拿它的大小
                # 去发 Range 续传 -> 每次 416 -> 重试 3 次 -> 判定 download_failed。
                # 明明下好了却报失败，然后回退到 GitHub（也不通），最后整个同步失败。
                if getattr(e, "code", 0) == 416:
                    try:
                        size = os.path.getsize(TARBALL)
                    except Exception:
                        size = 0
                    if size > 20 * 1024 * 1024:
                        if verbose:
                            print(f"    416 -> 本地已有完整文件 {size/1e6:.0f} MB，"
                                  f"跳过下载", flush=True)
                        return {"ok": True, "bytes": size, "mode": mode,
                                "url": src_url, "reused": True,
                                "seconds": round(time.time() - t0)}
                if verbose:
                    print(f"    {mode} HTTP {getattr(e, 'code', '?')}: "
                          f"{str(e)[:60]}", flush=True)
                time.sleep(1 + attempt)
                continue
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    {mode} 下载失败(第{attempt+1}次): "
                          f"{type(e).__name__}: {str(e)[:60]}", flush=True)
                # RemoteDisconnected / IncompleteRead 是常见的瞬时断连，
                # 等一下再试；有断点续传（Range）所以不会从头开始
                time.sleep(2 + attempt * 2)
                continue
        time.sleep(2)
    return {"ok": False, "reason": "download_failed", "bytes": have, "mode": mode_used}


def extract(verbose: bool = True) -> Dict:
    """解压到 qlib_bin.new，成功后原子替换。"""
    if not os.path.isfile(TARBALL):
        return {"ok": False, "reason": "no_tarball"}
    free = _free_gb(QLIB_DIR)
    if free < MIN_FREE_GB:
        return {"ok": False, "reason": "insufficient_disk",
                "free_gb": round(free, 1), "need_gb": MIN_FREE_GB}
    tgt = _target()
    new_dir = tgt + ".new"
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir, ignore_errors=True)
    os.makedirs(new_dir, exist_ok=True)
    t0 = time.time()
    try:
        with tarfile.open(TARBALL, "r:gz") as t:
            t.extractall(new_dir)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"extract_failed: {type(e).__name__}: {str(e)[:60]}"}
    # tarball 顶层是 qlib_bin/
    inner = os.path.join(new_dir, "qlib_bin")
    src = inner if os.path.isdir(inner) else new_dir
    if os.path.isdir(tgt):
        old = tgt + ".old"
        shutil.rmtree(old, ignore_errors=True)
        try:
            os.rename(tgt, old)
        except Exception:
            pass
    os.makedirs(os.path.dirname(tgt), exist_ok=True)
    try:
        os.rename(src, tgt)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"swap_failed: {type(e).__name__}"}
    shutil.rmtree(new_dir, ignore_errors=True)
    shutil.rmtree(tgt + ".old", ignore_errors=True)
    if verbose:
        print(f"    解压完成 {time.time()-t0:.0f}s -> {tgt}", flush=True)
    return {"ok": True, "seconds": round(time.time() - t0)}


def sync(force: bool = False, verbose: bool = True) -> Dict:
    """
    完整同步：查远端 → 需要时下载 + 解压。
    远端不新且本地有数据时直接跳过（省 545MB 流量）。
    """
    local = local_date()
    remote = remote_tag()
    stamp = _load_stamp()
    out = {"checked_at": _now(), "local_date": local, "remote_tag": remote,
           "synced_at": stamp.get("synced_at")}

    # 关键：remote_tag() 只查 GitHub。国内拿不到时它是 None，
    # 但**不能因此就放弃下载** —— 本地没数据时那是唯一机会。
    #
    # 实测踩过：训练机报 `remote_unreachable`，而 NAS 下载源明明是通的，
    # 因为这一行在 download() 之前就 return 了，多源回退根本没执行。
    if remote is None:
        if local:
            # 本地有数据：保守跳过，不白下 44MB
            out.update({"ok": True, "action": "skip",
                        "reason": "remote_version_unknown_but_local_exists",
                        "hint": "查不到远端版本（GitHub 不可达），本地已有数据 "
                                f"{local}，跳过。要强制更新用 --force"})
            _atomic(STAMP, {**stamp, **out})
            if verbose:
                print(f"[data_sync] 远端版本未知，本地已有 {local}，跳过下载", flush=True)
            return out
        # 本地没数据：必须继续尝试多源下载，不能因为查不到版本就卡死
        if verbose:
            print("[data_sync] 远端版本未知（GitHub 不可达），"
                  "本地无数据 —— 直接尝试多源下载", flush=True)

    # 远端 tag 形如 2026-11-08，与本地日历最后一天比
    if remote and not force and local and remote <= local:
        out.update({"ok": True, "action": "skip", "reason": "already_latest"})
        _atomic(STAMP, {**stamp, **out})
        if verbose:
            print(f"[data_sync] 已是最新：本地 {local} / 远端 {remote}", flush=True)
        return out

    if verbose:
        print(f"[data_sync] 需要更新：本地 {local or '无'} / 远端 {remote or '未知'}", flush=True)
    d = download(verbose=verbose)
    if not d.get("ok"):
        out.update({"ok": False, "reason": d.get("reason"), "download": d})
        _atomic(STAMP, {**stamp, **out})
        return out
    e = extract(verbose=verbose)
    if not e.get("ok"):
        out.update({"ok": False, "reason": e.get("reason"), "extract": e})
        _atomic(STAMP, {**stamp, **out})
        return out

    new_local = local_date()
    out.update({"ok": True, "action": "updated", "new_local_date": new_local,
                "download": d, "extract": e, "synced_at": _now()})
    _atomic(STAMP, out)
    if verbose:
        print(f"[data_sync] 完成：本地数据更新到 {new_local}", flush=True)
    return out


def summary() -> Dict:
    s = _load_stamp()
    return {"ok": True, "local_date": local_date(),
            "tarball_mb": round(os.path.getsize(TARBALL) / 1e6, 1)
            if os.path.isfile(TARBALL) else 0,
            "data_dir": _target(),
            "exists": os.path.isdir(_target()),
            "free_gb": round(_free_gb(QLIB_DIR), 1),
            "last_sync": s}


if __name__ == "__main__":
    import sys
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
    if "--sync" in sys.argv:
        print()
        print(json.dumps(sync(force="--force" in sys.argv), ensure_ascii=False, indent=1))
