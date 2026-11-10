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
    # 换行/回车必须清掉：`export K='a
    # b'` 会让 b 被当成命令执行。
    v = str(v).replace("\r", " ").replace("\n", " ")
    return "'" + v.replace("'", "'\\''") + "'"


def main() -> int:
    try:
        with open(SNAPSHOT, encoding="utf-8") as f:
            snap = json.load(f) or {}
    except Exception:
        return 0

    env = snap.get("env") or {}
    restored = []
    empty_in_snapshot = []
    for k, v in env.items():
        if not str(k).startswith(PREFIX):
            continue
        if k in SKIP:
            continue
        if os.environ.get(k):
            continue                      # 已有值优先
        v = str(v)
        if not v:
            empty_in_snapshot.append(k)
            continue
        print(f"export {k}={_sh_quote(v)}")
        restored.append(k)

    import sys
    if restored:
        print("", file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        print(f"[restore_env] 当前环境缺少 {len(restored)} 个关键变量，"
              f"已从快照恢复：", file=sys.stderr)
        print(f"  {', '.join(restored)}", file=sys.stderr)
        print("  来源：data/state/env_snapshot.json", file=sys.stderr)
        print("  若不是你预期的行为，请检查容器的 -e 参数。", file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        print("", file=sys.stderr)
    if empty_in_snapshot:
        print(f"[restore_env] 快照里这些变量是空值，不恢复："
              f"{', '.join(empty_in_snapshot)}", file=sys.stderr)
    return 0


def refresh() -> int:
    """
    把当前生效的环境变量回写快照。

    为什么需要：快照原来只由更新流程写入。用户改了密码但容器没重建时，
    快照还是旧值 —— 下次重建丢变量就会恢复出**已退役的旧密码**。
    启动成功后调用一次，让快照跟随现状。
    """
    import json as _j
    env = {k: v for k, v in os.environ.items()
           if k.startswith(PREFIX) and k not in SKIP and v}
    if not env:
        return 0
    try:
        with open(SNAPSHOT, encoding="utf-8") as f:
            snap = _j.load(f) or {}
    except Exception:
        snap = {}
    old = snap.get("env") or {}
    if old == env:
        return 0
    snap["env"] = env
    from datetime import datetime
    snap["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        tmp = SNAPSHOT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _j.dump(snap, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SNAPSHOT)
        try:
            os.chmod(SNAPSHOT, 0o600)
        except OSError:
            pass
        import sys
        print(f"[restore_env] 快照已回写 {len(env)} 个变量", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"[restore_env] 回写失败：{type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import sys as _sys
    if "--refresh" in _sys.argv:
        raise SystemExit(refresh())
    raise SystemExit(main())
