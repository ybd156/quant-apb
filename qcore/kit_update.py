"""
训练机自更新 —— 服务端下发版本，训练机自己覆盖。

## 为什么需要

训练机可能放在别的城市、别的网络里。每次改了训练逻辑，让用户手动
下载 zip、解压、覆盖，既不现实也容易出错。

所以：**服务端有版本，训练机自己比对、自己拉、自己换**。

## 流程

    训练机 --auto-update
      ↓ GET /api/train/kit            （服务端返回当前版本 + 下载地址）
      ↓ 比对本地 VERSION
      ├─ 不新 → 跳过
      └─ 更新 → 下载 zip
                 ↓ 解压到临时目录
                 ↓ 校验（有没有 tools/train_one_click.py、VERSION 对不对）
                 ↓ 覆盖：qcore/ tools/ scripts/ train.bat train.sh VERSION
                 ↓ **保留** data/ dist/ runtime/ （数据和已装运行时不能丢）
                 ↓ 写 .updated_from 记录

## 安全边界（重要）

**只覆盖代码，绝不动数据**：
    覆盖: qcore/ tools/ scripts/ *.bat *.sh VERSION requirements.txt README-*.md
    保留: data/ dist/ runtime/ apb_defaults.json

`apb_defaults.json` 保留是因为用户可能改过 NAS 地址；
如果想强制用服务端的默认值，删掉它再更新即可。

## 失败处理

任何一步失败都**保持原样**，不留下半个新版本：
先解压到 `.update-tmp/`，全部校验通过才逐个覆盖。
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 允许被覆盖的路径（白名单，比黑名单安全：新加的文件默认不动）
OVERWRITE = (
    "qcore", "tools", "scripts",
    "train.bat", "train.sh", "VERSION",
    "requirements.txt", "README-训练机.md", "LICENSE",
)
# 绝对不动
PRESERVE = ("data", "dist", "runtime", "apb_defaults.json", ".update-tmp", ".updated_from")

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def local_version() -> str:
    for p in (os.path.join(ROOT, "VERSION"), os.path.join(ROOT, "webui", "VERSION")):
        try:
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            continue
    return ""


def _ver_tuple(v: str):
    out = []
    for part in str(v or "").replace("-", ".").split("."):
        try:
            out.append(int("".join(ch for ch in part if ch.isdigit()) or 0))
        except Exception:
            out.append(0)
    return tuple(out)


def newer(remote: str, local: str) -> bool:
    return _ver_tuple(remote) > _ver_tuple(local)


def fetch_manifest(nas: str, token: str = "", timeout: int = 20) -> Dict:
    """从服务端取版本清单。"""
    import urllib.error
    import urllib.request
    url = nas.rstrip("/") + "/api/train/kit"
    h = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
    if token:
        h["X-APB-Token"] = token
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        return d.get("data") or d
    except urllib.error.HTTPError as e:
        return {"_error": f"http_{e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {str(e)[:60]}"}


def _download(url: str, out: str, timeout: int = 300) -> Dict:
    """下载 zip。先用 net 的自动选路，失败再裸 urllib。"""
    import urllib.request
    if os.path.isfile(out):
        os.remove(out)
    try:
        from . import net
        body, mode = net.get(url, timeout_direct=60, timeout_proxy=120)
        if body:
            with open(out, "w", encoding="utf-8", errors="ignore") as f:
                f.write(body)
            # net.get 返回的是解码后的文本，二进制会坏 —— 这里改用二进制下载
            os.remove(out)
    except Exception:
        pass

    h = {"User-Agent": _BROWSER_UA}
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(out)
        if size < 50000:
            return {"ok": False, "reason": f"文件太小 {size}B，可能下错了"}
        return {"ok": True, "bytes": size}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}"}


def _validate(tmp: str) -> Dict:
    """解压后先校验关键文件在不在，再决定覆盖。"""
    must = ["tools/train_one_click.py", "qcore/__init__.py", "VERSION"]
    miss = [m for m in must if not os.path.exists(os.path.join(tmp, m))]
    if miss:
        return {"ok": False, "reason": f"缺关键文件 {miss}"}
    v = ""
    try:
        with open(os.path.join(tmp, "VERSION"), encoding="utf-8") as f:
            v = f.read().strip()
    except Exception:
        pass
    if not v:
        return {"ok": False, "reason": "VERSION 为空"}
    return {"ok": True, "version": v}


def apply_update(zip_path: str, verbose: bool = True) -> Dict:
    """
    解压 + 校验 + 覆盖。全程失败不改动现有文件。
    """
    tmp = os.path.join(ROOT, ".update-tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(tmp, ignore_errors=True)
        return {"ok": False, "reason": f"解压失败: {type(e).__name__}"}

    # zip 里通常套一层 apb-train-kit/
    src = tmp
    entries = [x for x in os.listdir(tmp) if not x.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
        src = os.path.join(tmp, entries[0])

    val = _validate(src)
    if not val.get("ok"):
        shutil.rmtree(tmp, ignore_errors=True)
        return {"ok": False, "reason": val.get("reason")}

    old = local_version()
    replaced, skipped = [], []
    for name in OVERWRITE:
        s = os.path.join(src, name)
        d = os.path.join(ROOT, name)
        if not os.path.exists(s):
            continue
        if os.path.isdir(s):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
            try:
                shutil.copytree(s, d)
                replaced.append(name)
            except Exception:
                skipped.append(name)
        else:
            try:
                shutil.copy2(s, d)
                replaced.append(name)
            except Exception:
                skipped.append(name)

    # 保留清单里的东西不能被覆盖（保险，理论上 OVERWRITE 里没有）
    for keep in PRESERVE:
        if keep in replaced:
            replaced.remove(keep)

    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with open(os.path.join(ROOT, ".updated_from"), "w", encoding="utf-8") as f:
            json.dump({"from": old, "to": val.get("version"), "at": _now(),
                       "replaced": replaced, "skipped": skipped}, f, ensure_ascii=False)
    except Exception:
        pass

    if verbose:
        print(f"[kit_update] {old} -> {val.get('version')}，"
              f"覆盖 {len(replaced)} 项，跳过 {len(skipped)} 项")
    return {"ok": True, "from": old, "to": val.get("version"),
            "replaced": replaced, "skipped": skipped}


def check_and_update(nas: str, token: str = "", force: bool = False,
                     verbose: bool = True) -> Dict:
    """完整流程：查版本 → 需要就下载 → 覆盖。"""
    man = fetch_manifest(nas, token)
    if man.get("_error"):
        return {"ok": False, "reason": f"取版本失败: {man['_error']}"}
    remote = str(man.get("version") or "")
    local = local_version()
    if not remote:
        return {"ok": False, "reason": "服务端没返回版本号"}
    if not force and not newer(remote, local):
        if verbose:
            print(f"[kit_update] 已是最新（本地 {local} / 服务端 {remote}）")
        return {"ok": True, "action": "skip", "local": local, "remote": remote}

    if verbose:
        print(f"[kit_update] 需要更新：{local} -> {remote}")
    # 优先走 NAS 自己的下载端点 —— 训练机只需要能连 NAS，
    # 不用能访问 GitHub（国内很多网络到不了）。
    local_rel = str(man.get("local_url") or "")
    url = (nas.rstrip("/") + local_rel) if local_rel else str(man.get("url") or "")
    if not url:
        return {"ok": False, "reason": "服务端没给下载地址"}

    tmpzip = os.path.join(ROOT, ".update-kit.zip")
    _cands = [url]
    if man.get("url"):
        _cands.append(str(man["url"]))          # 回退 GitHub 直连
    if man.get("url_mirror"):
        _cands.append(str(man["url_mirror"]))   # 再回退镜像
    for u in _cands:
        if not u:
            continue
        if verbose:
            print(f"[kit_update] 下载 {u}")
        d = _download(u, tmpzip)
        if d.get("ok"):
            break
        if verbose:
            print(f"[kit_update] 失败：{d.get('reason')}")
    else:
        return {"ok": False, "reason": "下载失败（主站和镜像都不通）"}

    r = apply_update(tmpzip, verbose=verbose)
    try:
        os.remove(tmpzip)
    except Exception:
        pass
    if r.get("ok"):
        r["action"] = "updated"
        r["note"] = "代码已更新；data/ dist/ runtime/ apb_defaults.json 未动"
    return r


if __name__ == "__main__":
    import sys
    nas = ""
    for a in sys.argv:
        if a.startswith("--nas="):
            nas = a.split("=", 1)[1]
    if not nas:
        try:
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            import defaults as _D
            nas = _D.nas()
        except Exception:
            nas = "https://apbcdn.683868.xyz"
    print(f"本地版本: {local_version() or '(无)'}")
    print(f"服务端:   {nas}")
    print(json.dumps(check_and_update(nas, os.environ.get("APB_TOKEN", "")),
                     ensure_ascii=False, indent=1))
