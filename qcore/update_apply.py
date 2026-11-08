"""
应用内自动更新 —— 在**独立的辅助容器**里完成自我替换。

## 为什么必须是独立容器

主容器不能自己停自己：一旦 `docker stop apb`，正在执行更新的进程也被杀掉，
更新就半途而废（旧的删了、新的没起）。

所以流程是：
    主容器（Web）收到 /api/update/apply
        → 通过 docker.sock 启动一个**辅助容器**（同镜像，跑本模块）
        → 立即返回 202
    辅助容器（本模块）：
        1. 拉新镜像（失败则中止，主容器不受影响）
        2. 读取旧容器配置（端口/挂载/环境变量）
        3. 停旧、删旧
        4. 用新镜像按原配置重建并启动
        5. 失败则用旧镜像回滚
        6. 删除自己

## 关键：配置从旧容器读，不写死

端口、挂载路径、环境变量都从旧容器 inspect 出来，这样用户改过 compose
或 fnOS 改过路径，更新后仍然保持原样。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from . import docker_api as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "update.log")


def _log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _split_image(ref: str):
    """ghcr.io/owner/repo:tag -> (ghcr.io/owner/repo, tag)"""
    if ":" in ref.rsplit("/", 1)[-1]:
        repo, tag = ref.rsplit(":", 1)
        return repo, tag
    return ref, "latest"


def apply_update(target: str, image_ref: str, tag: Optional[str] = None,
                 sock: str = D.SOCK, dry_run: bool = False) -> Dict:
    """执行更新。target=容器名，image_ref=新镜像（可含 tag）。"""
    repo, tag0 = _split_image(image_ref)
    tag = tag or tag0
    _log(f"=== 开始更新 {target} -> {repo}:{tag} ===")

    if not D.available(sock):
        return {"ok": False, "reason": "docker_socket_unavailable",
                "hint": "容器未挂载 /var/run/docker.sock，无法应用内更新"}

    old = D.find_container(target, sock)
    if old is None:
        return {"ok": False, "reason": "target_not_found", "target": target}
    old_id = old["Id"]
    old_image = old.get("Image", "")
    _log(f"旧容器 {old_id[:12]} image={old_image}")

    # 保存旧配置（端口/挂载/环境变量/命令）
    info = D.inspect(old_id, sock)
    cfg = info.get("Config") or {}
    hc = info.get("HostConfig") or {}
    env = cfg.get("Env") or []
    cmd = cfg.get("Cmd") or []
    binds = hc.get("Binds") or []
    port_bindings = hc.get("PortBindings") or {}
    restart = ((hc.get("RestartPolicy") or {}).get("Name")) or "unless-stopped"
    _log(f"配置: binds={len(binds)} env={len(env)} ports={list(port_bindings)}")

    # 1) 拉新镜像
    pr = D.pull(repo, tag, sock=sock)
    if not pr.get("ok"):
        _log(f"拉取失败: {pr.get('error')}")
        return {"ok": False, "reason": "pull_failed", "error": pr.get("error")}
    _log("镜像拉取完成")

    new_ref = f"{repo}:{tag}"
    if dry_run:
        return {"ok": True, "dry_run": True, "would_use": new_ref}

    # 2) 停 + 删旧
    _log("停止旧容器")
    D.stop(old_id, sock)
    time.sleep(2)
    _log("删除旧容器")
    D.remove(old_id, sock)
    time.sleep(1)

    # 3) 用新镜像重建
    created = D.create(target, new_ref, ports=port_bindings, env=env,
                       binds=binds, restart=restart, sock=sock)
    if not created.get("ok"):
        _log(f"重建失败: {created.get('error')}，尝试回滚")
        rb = D.create(target, old_image, ports=port_bindings, env=env,
                      binds=binds, restart=restart, sock=sock)
        if rb.get("ok"):
            D.start(rb["id"], sock)
            _log("回滚成功，已用旧镜像恢复")
        else:
            _log(f"回滚也失败: {rb.get('error')} —— 需要手工介入")
        return {"ok": False, "reason": "recreate_failed",
                "error": created.get("error"), "rollback_ok": bool(rb.get("ok"))}

    # 4) 启动
    ok = D.start(created["id"], sock)
    _log(f"启动{'成功' if ok else '失败'}")
    return {"ok": bool(ok), "container": target, "image": new_ref,
            "id": (created.get("id") or "")[:12]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="APB 应用内更新")
    ap.add_argument("--target", default=os.environ.get("APB_TARGET_NAME", "apb"))
    ap.add_argument("--image", default=os.environ.get("APB_IMAGE",
                                                      "ghcr.io/ybd156/quant-apb:latest"))
    ap.add_argument("--tag", default=os.environ.get("APB_TAG"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    r = apply_update(a.target, a.image, a.tag, dry_run=a.dry_run)
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
