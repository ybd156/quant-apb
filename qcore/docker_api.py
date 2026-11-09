"""
极简 Docker Engine API 客户端（走 unix socket，不依赖 docker CLI）。

## 为什么需要它

容器镜像里只有 python + numpy + pandas，**没有 docker 命令**。
但"应用内一键更新"必须能操作宿主机 Docker。

Docker Engine 的 API 就是一个跑在 `/var/run/docker.sock` 上的 HTTP 服务，
用 `http.client` 自定义 `connect()` 即可直连，不需要任何第三方库。

## 用到的最小接口

    POST /images/create?fromImage=<img>&tag=<tag>   拉镜像（流式）
    POST /containers/{id}/stop                      停容器
    DELETE /containers/{id}?force=1                 删容器
    POST /containers/create?name=<name>             创建
    POST /containers/{id}/start                     启动
    GET  /containers/json?all=1                     列表

## 安全说明

挂载 docker.sock 等于给容器 root 级权限。这是"应用内自更新"的必要代价，
文档里必须写清楚。不想承担这个风险的用户可以不挂 socket，
此时更新只能由宿主机 `docker compose pull && up -d` 完成。
"""
from __future__ import annotations

import http.client
import json
import socket
from typing import Dict, List, Optional, Tuple

SOCK = "/var/run/docker.sock"


class _UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path: str = SOCK, timeout: int = 600):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def available(path: str = SOCK) -> bool:
    import os
    return os.path.exists(path) and os.access(path, os.W_OK | os.R_OK)


def request(method: str, path: str, body: Optional[dict] = None,
            timeout: int = 600, sock: str = SOCK) -> Tuple[int, str]:
    conn = _UnixHTTP(sock, timeout=timeout)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, path, body=payload, headers=headers)
        r = conn.getresponse()
        data = r.read().decode("utf-8", errors="replace")
        return r.status, data
    finally:
        conn.close()


def list_containers(all: bool = True, sock: str = SOCK) -> List[Dict]:
    st, data = request("GET", f"/containers/json?all={1 if all else 0}", sock=sock)
    if st != 200:
        return []
    try:
        return json.loads(data)
    except Exception:
        return []


def find_container(name: str, sock: str = SOCK) -> Optional[Dict]:
    for c in list_containers(True, sock):
        names = [n.lstrip("/") for n in (c.get("Names") or [])]
        if name in names:
            return c
    return None


def pull(image: str, tag: str = "latest", sock: str = SOCK,
         timeout: int = 900) -> Dict:
    """拉镜像。Docker 返回流式 JSON，逐行看是否有 error。"""
    st, data = request("POST", f"/images/create?fromImage={image}&tag={tag}",
                       timeout=timeout, sock=sock)
    err = None
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if j.get("error"):
            err = j["error"]
            break
    return {"ok": st == 200 and not err, "status": st, "error": err}


def stop(container_id: str, sock: str = SOCK, timeout: int = 60) -> bool:
    st, _ = request("POST", f"/containers/{container_id}/stop?t=20",
                    timeout=timeout, sock=sock)
    return st in (204, 304)


def remove(container_id: str, sock: str = SOCK) -> bool:
    st, _ = request("DELETE", f"/containers/{container_id}?force=1", sock=sock)
    return st in (204, 200)


def create(name: str, image: str, ports: Optional[dict] = None,
           env: Optional[List[str]] = None, binds: Optional[List[str]] = None,
           restart: str = "unless-stopped", entrypoint: Optional[List[str]] = None,
           cmd: Optional[List[str]] = None, sock: str = SOCK,
           group_add: Optional[List[str]] = None,
           network_mode: Optional[str] = None,
           extra_hosts: Optional[List[str]] = None,
           cap_add: Optional[List[str]] = None,
           devices: Optional[List[dict]] = None,
           user: Optional[str] = None) -> Dict:
    """
    创建容器。

    ## 为什么参数这么多

    只写死 RestartPolicy/Binds/PortBindings 是危险的：重建一次就丢一批配置。
    实测踩过：漏了 GroupAdd -> 容器不再属于 docker 组 ->
    访问不了 /var/run/docker.sock -> **下次更新再也做不了**（自我毁灭）。

    凡是旧容器 HostConfig 里有的、会影响运行的字段，都必须能传进来。
    """
    hc: Dict = {
        "RestartPolicy": {"Name": restart},
        "Binds": binds or [],
        "PortBindings": ports or {},
    }
    if group_add:
        hc["GroupAdd"] = [str(g) for g in group_add]
    if network_mode:
        hc["NetworkMode"] = network_mode
    if extra_hosts:
        hc["ExtraHosts"] = extra_hosts
    if cap_add:
        hc["CapAdd"] = cap_add
    if devices:
        hc["Devices"] = devices

    body = {
        "Image": image,
        "Env": env or [],
        "HostConfig": hc,
    }
    # 以指定用户运行。辅助更新容器必须用 root —— 它要访问 docker.sock，
    # 而 socket 是 root:994 且 660，非 root 用户即使有 GroupAdd 也可能因
    # fnOS 重建容器丢失组权限而失败（实测踩过）。
    if user:
        body["User"] = str(user)
    # 覆盖入口：辅助容器需要跑 update_apply，而不是镜像默认的 start.sh
    if entrypoint is not None:
        body["Entrypoint"] = entrypoint
    if cmd is not None:
        body["Cmd"] = cmd
    st, data = request("POST", f"/containers/create?name={name}", body=body, sock=sock)
    try:
        j = json.loads(data)
    except Exception:
        j = {}
    return {"ok": st in (201, 200), "status": st, "id": j.get("Id"), "error": j.get("message")}


def start(container_id: str, sock: str = SOCK) -> bool:
    st, _ = request("POST", f"/containers/{container_id}/start", sock=sock)
    return st in (204, 304)


def inspect(container_id: str, sock: str = SOCK) -> Dict:
    st, data = request("GET", f"/containers/{container_id}/json", sock=sock)
    try:
        return json.loads(data) if st == 200 else {}
    except Exception:
        return {}
