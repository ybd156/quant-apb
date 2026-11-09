#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练机自检：对比 NAS 上的模型，把缺失的周期补上。

## 为什么需要

`daemon.bat` 之前只是「等 NAS 派任务」，队列空了就一直空转 ——
但真正缺的是「某个周期根本没模型」，这不是任务，是常态缺口。

结果：用户开着 daemon 一小时，h=15 还是没模型。

## 这个脚本做什么

1. 问 NAS：哪些周期有模型、哪些没有
2. 本地判断：缺的、或者 NAS 上超龄的
3. 调用 train_one_click.py 只训这些周期，训完自动上传
4. 输出「还缺什么」，供 daemon 决定下一轮

## 用法

    python tools/train_missing.py                    # 补所有缺失周期
    python tools/train_missing.py --dry-run          # 只看缺什么
    python tools/train_missing.py --max-age 7        # 超龄 7 天也重训
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import defaults as DEF
except Exception:
    DEF = None


def _nas() -> str:
    ns = os.environ.get("APB_NAS", "").strip()
    if ns:
        return ns.rstrip("/")
    pj = os.path.join(ROOT, "apb_defaults.json")
    if os.path.isfile(pj):
        try:
            with open(pj, encoding="utf-8") as f:
                v = str((json.load(f) or {}).get("nas_url") or "").strip()
            if v:
                return v.rstrip("/")
        except Exception:
            pass
    if DEF:
        try:
            return DEF.nas().rstrip("/")
        except Exception:
            pass
    return ""


def _token() -> str:
    t = os.environ.get("APB_TOKEN", "").strip()
    if t:
        return t
    pj = os.path.join(ROOT, "apb_defaults.json")
    if os.path.isfile(pj):
        try:
            with open(pj, encoding="utf-8") as f:
                d = json.load(f) or {}
            return str(d.get("token") or d.get("nas_token") or "").strip()
        except Exception:
            pass
    return ""


def nas_models(nas: str, token: str, timeout: int = 25):
    """问 NAS 要模型清单。失败返回 None（网络不通就跳过，不阻塞）。"""
    url = nas + "/api/model/local"
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}
    if token:
        hdrs["X-APB-Token"] = token
        hdrs["X-Token"] = token
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        return (d.get("data") or {}).get("models") or []
    except Exception as e:  # noqa: BLE001
        print(f"  [!] 问 NAS 失败: {type(e).__name__}: {str(e)[:60]}")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="补训缺失周期")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不训练")
    ap.add_argument("--max-age", type=float, default=None,
                    help="NAS 上模型超过这么多天也重训")
    ap.add_argument("--nas", default="")
    ap.add_argument("--token", default="")
    a = ap.parse_args(argv)

    nas = (a.nas or _nas()).rstrip("/")
    token = a.token or _token()

    # 目标周期（默认全部）
    targets = [1, 3, 5, 10, 15]
    if DEF:
        try:
            targets = [int(x) for x in DEF.horizons()]
        except Exception:
            pass

    print("=" * 62)
    print("  补训缺失周期")
    print("=" * 62)
    print(f"  NAS   : {nas or '(未配置)'}")
    print(f"  目标  : {targets}")

    if not nas:
        print("  [!] 没有 NAS 地址，无法判断缺什么。")
        print("      设置环境变量 APB_NAS 或在 apb_defaults.json 里写 nas_url")
        return 2

    models = nas_models(nas, token)
    if models is None:
        print("  [!] 拿不到 NAS 的模型清单（网络/鉴权问题）")
        print("      如果你确定要训练，直接用：")
        print(f"      python tools/train_one_click.py --horizons {','.join(map(str, targets))}")
        return 2

    have = {}
    for m in models:
        try:
            h = int(m.get("horizon"))
        except (TypeError, ValueError):
            continue
        have[h] = m

    missing, stale = [], []
    print()
    print(f"  {'周期':<8}{'NAS 状态':<14}{'训练时间':<22}{'动作'}")
    print("  " + "-" * 58)
    for h in targets:
        m = have.get(h) or {}
        exists = bool(m.get("exists"))
        ts = str(m.get("trained_at") or "")
        age = m.get("age_days")
        act = ""
        if not exists:
            missing.append(h)
            act = "缺模型 -> 训练"
        elif a.max_age is not None and age is not None and float(age) > a.max_age:
            stale.append(h)
            act = f"超龄 {age:.1f} 天 -> 重训"
        else:
            act = "已有，跳过"
        print(f"  h{h:<7}{'有' if exists else '缺':<14}{ts[:19]:<22}{act}")

    todo = missing + stale
    print()
    if not todo:
        print("  所有周期都有模型，无需训练。")
        return 0

    print(f"  需要训练: {todo}")
    if a.dry_run:
        print("  --dry-run：未训练。")
        return 0

    # 调用 train_one_click
    cmd = [sys.executable, os.path.join(ROOT, "tools", "train_one_click.py"),
           "--horizons", ",".join(map(str, todo)),
           "--skip-data",
           "--nas", nas]
    if token:
        cmd += ["--token", token]
    print(f"  执行: {' '.join(cmd[-6:])}")
    print()
    try:
        rc = subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n  已中断")
        return 130
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
