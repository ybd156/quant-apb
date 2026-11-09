#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动时从快照恢复缺失的环境变量。

## 为什么需要

fnOS 的应用管理器会按它自己的配置重建容器 —— 手工 `docker run -e APB_PASSWORD=...`
设的变量会被**覆盖掉**。实测：容器重建后 `APB_PASSWORD` 消失，
服务变成完全公开（任何人都能看到持仓和策略）。

`env_guard` 会把关键变量快照到 `data/state/env_snapshot.json`，
而 `data/` 是挂载卷，重建不丢。所以启动时从快照读回来即可自愈。

## 用法（在 start.sh 里）

    eval "$(python /app/scripts/restore_env.py)"

只输出 `export K='V'` 行，不改任何文件。
已存在的环境变量优先（不覆盖），避免快照里的旧值盖掉新配置。
"""
from __future__ import annotations

import json
import os

SNAPSHOT = "/app/data/state/env_snapshot.json"
PREFIX = "APB_"
# 这些即使快照里有也不要恢复（会干扰容器自身标识）
SKIP = {"APB_IMAGE", "APB_CONTAINER_NAME"}


def _sh_quote(v: str) -> str:
    return "'" + str(v).replace("'", "'\\''") + "'"


def main() -> int:
    try:
        with open(SNAPSHOT, encoding="utf-8") as f:
            snap = json.load(f) or {}
    except Exception:
        return 0

    env = snap.get("env") or {}
    restored = []
    for k, v in env.items():
        if not str(k).startswith(PREFIX):
            continue
        if k in SKIP:
            continue
        if os.environ.get(k):
            continue                      # 已有值优先
        v = str(v)
        if not v:
            continue
        print(f"export {k}={_sh_quote(v)}")
        restored.append(k)

    # 恢复结果写到 stderr（不影响 eval），方便排查
    if restored:
        import sys
        print(f"[restore_env] 从快照恢复 {len(restored)} 个变量: "
              f"{', '.join(restored)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
