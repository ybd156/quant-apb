"""
模型版本管理 + 数据备份 —— 出问题能回滚，丢盘能恢复。

## 为什么必须有

现在 `gbdt.pkl` 是**覆盖式保存**：新模型一旦比旧模型差，你回不去。
而 `data/` 里是模型、台账、自选、成交记录——硬盘挂了全没。

这两件事平时看不出价值，出事那天价值无限。

## 版本策略

- 每次训练自动存一份带时间戳的快照到 `data/model/versions/`
- 同时记录当时的 IC / 回测指标 / 训练样本数，便于对比
- 只保留最近 `keep` 份（默认 12），自动清理最旧的
- **当前生效的版本**用 `data/model/current.json` 指向，不靠文件名猜

## 回滚

```python
from qcore.versioning import rollback
rollback("20261109-081500")   # 回到那个版本
```

回滚前会自动把当前模型存成一份新快照，所以回滚本身也可回滚。

## 备份

打包 `data/state` + `data/model`（不含 3.7GB 的 Qlib 原始数据，
那部分可以从 GitHub 重新下载）。默认存到 `data/backup/`。
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "data", "model")
VER_DIR = os.path.join(MODEL_DIR, "versions")
STATE = os.path.join(ROOT, "data", "state")
BACKUP_DIR = os.path.join(ROOT, "data", "backup")
CURRENT = os.path.join(MODEL_DIR, "current.json")
MODEL_PATH = os.path.join(MODEL_DIR, "gbdt.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


def paths(horizon: Optional[int] = None) -> tuple:
    """按周期实例化的模型路径。

    h=1 沿用 `gbdt.pkl` / `meta.json` / `current.json`（向后兼容）；
    h≠1 用 `gbdt_h{h}.pkl` / `meta_h{h}.json` / `current_h{h}.json`
    —— 与 `live_model._paths(h)`、`data/model/meta_h5.json` 的既有约定对齐。
    旧实现 save() 没有 horizon 参数，无论训哪个周期都覆盖 h=1 槽位（陷阱 13）。
    """
    h = int(horizon or 1)
    if h == 1:
        return MODEL_PATH, META_PATH, CURRENT
    return (os.path.join(MODEL_DIR, f"gbdt_h{h}.pkl"),
            os.path.join(MODEL_DIR, f"meta_h{h}.json"),
            os.path.join(MODEL_DIR, f"current_h{h}.json"))

KEEP = 12


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _atomic_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(model, meta: Dict, tag: str = "", horizon: Optional[int] = None) -> Dict:
    """
    保存模型：写当前生效文件 + 存一份带时间戳快照 + 更新 current.json。
    返回 {"version","path","meta"}。

    horizon 缺省时从 meta["horizon"] 取，再缺省为 1。**不再静默覆盖 h=1 槽位。**
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VER_DIR, exist_ok=True)
    h = int(horizon or (meta or {}).get("horizon") or 1)
    model_path, meta_path, cur_path = paths(h)
    ver = _stamp() + (f"-{tag}" if tag else "")

    # 1) 当前生效
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=4)

    m = dict(meta or {})
    m["version"] = ver
    m["horizon"] = h
    m["saved_at"] = _now()
    _atomic_json(meta_path, m)

    # 2) 快照（文件名带周期，避免 h=1 与 h=5 的快照互相覆盖）
    snap = os.path.join(VER_DIR, f"gbdt_h{h}_{ver}.pkl")
    shutil.copy2(model_path, snap)
    _atomic_json(os.path.join(VER_DIR, f"gbdt_h{h}_{ver}.meta.json"), m)

    # 3) 指针
    _atomic_json(cur_path, {"version": ver, "horizon": h, "saved_at": m["saved_at"],
                            "file": os.path.basename(snap)})
    prune(KEEP)
    return {"version": ver, "path": snap, "meta": m}


def list_versions() -> List[Dict]:
    out = []
    if not os.path.isdir(VER_DIR):
        return out
    for fn in sorted(os.listdir(VER_DIR)):
        if not fn.endswith(".pkl"):
            continue
        # 兼容两种快照名：gbdt_{ver}.pkl（旧，h=1）与 gbdt_h{h}_{ver}.pkl（新）
        hz = 1
        body = fn[len("gbdt_"):-len(".pkl")]
        if body.startswith("h"):
            head, _, rest = body.partition("_")
            if head[1:].isdigit() and rest:
                hz, body = int(head[1:]), rest
        ver = body
        mp = os.path.join(VER_DIR, f"gbdt_{ver}.meta.json")
        if hz != 1:
            mp = os.path.join(VER_DIR, f"gbdt_h{hz}_{ver}.meta.json")
        meta = {}
        try:
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
        out.append({"version": ver, "horizon": hz, "file": fn,
                    "size_mb": round(os.path.getsize(os.path.join(VER_DIR, fn)) / 1e6, 2),
                    "saved_at": meta.get("saved_at"),
                    "ic": meta.get("ic"), "sharpe": meta.get("sharpe"),
                    "train_end": meta.get("train_end"),
                    "n_trees": meta.get("n_trees")})
    return sorted(out, key=lambda x: x["version"], reverse=True)


def current_version(horizon: Optional[int] = None) -> Optional[str]:
    try:
        with open(paths(horizon)[2], encoding="utf-8") as f:
            return json.load(f).get("version")
    except Exception:
        return None


def rollback(version: str, horizon: Optional[int] = None) -> Dict:
    """
    回滚到指定版本。回滚前先把当前模型另存一份，所以回滚本身也可回滚。
    """
    model_path, meta_path, _cur = paths(horizon)
    src = os.path.join(VER_DIR, f"gbdt_{version}.pkl")
    if not os.path.isfile(src):
        # 新版快照文件名带周期：gbdt_h{h}_{ver}.pkl
        for h_try in (horizon, 1, 5, 10, 20, 3, 15):
            if h_try is None:
                continue
            cand = os.path.join(VER_DIR, f"gbdt_h{int(h_try)}_{version}.pkl")
            if os.path.isfile(cand):
                src = cand
                break
    if not os.path.isfile(src):
        avail = [v["version"] for v in list_versions()[:5]]
        return {"ok": False, "reason": "version_not_found", "version": version,
                "available": avail}
    cur = current_version(horizon)
    if cur and os.path.isfile(model_path):
        # 备份当前
        try:
            with open(model_path, "rb") as f:
                m = pickle.load(f)
            meta = {}
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
            save(m, meta, tag="before-rollback", horizon=horizon)
        except Exception:
            pass
    shutil.copy2(src, model_path)
    mp = os.path.join(VER_DIR, f"gbdt_{version}.meta.json")
    meta = {}
    try:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass
    _atomic_json(META_PATH, meta)
    _atomic_json(CURRENT, {"version": version, "saved_at": _now(),
                           "file": os.path.basename(src), "rolled_back_from": cur})
    return {"ok": True, "version": version, "rolled_back_from": cur}


def prune(keep: int = KEEP) -> int:
    """只保留最近 keep 份快照，返回删除数量。"""
    vs = list_versions()
    if len(vs) <= keep:
        return 0
    cur = current_version()
    removed = 0
    for v in vs[keep:]:
        if v["version"] == cur:
            continue
        for fn in (f"gbdt_{v['version']}.pkl", f"gbdt_{v['version']}.meta.json"):
            p = os.path.join(VER_DIR, fn)
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    removed += 1
            except Exception:
                pass
    return removed


# ------------------------------------------------------------------ 备份
def backup(include_qlib: bool = False, out_dir: Optional[str] = None) -> Dict:
    """
    打包 state + model（可选含 Qlib 原始数据）。
    默认不含 Qlib —— 那 3.7GB 可以从 GitHub 重新下载。
    """
    od = out_dir or BACKUP_DIR
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, f"apb-backup-{_stamp()}.zip")
    n_files, n_bytes = 0, 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base in (STATE, MODEL_DIR):
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        rel = os.path.relpath(fp, ROOT)
                        z.write(fp, rel)
                        n_files += 1
                        n_bytes += os.path.getsize(fp)
                    except Exception:
                        continue
        if include_qlib:
            qd = os.path.join(ROOT, "data", "qlib", "x", "qlib_bin", "features")
            if os.path.isdir(qd):
                for root, _dirs, files in os.walk(qd):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            z.write(fp, os.path.relpath(fp, ROOT))
                            n_files += 1
                            n_bytes += os.path.getsize(fp)
                        except Exception:
                            continue
    size = os.path.getsize(path)
    return {"ok": True, "path": path, "files": n_files,
            "raw_mb": round(n_bytes / 1e6, 1), "zip_mb": round(size / 1e6, 1),
            "ratio": round(size / max(n_bytes, 1), 3),
            "include_qlib": include_qlib}


def list_backups() -> List[Dict]:
    out = []
    if not os.path.isdir(BACKUP_DIR):
        return out
    for fn in sorted(os.listdir(BACKUP_DIR), reverse=True):
        if not fn.endswith(".zip"):
            continue
        p = os.path.join(BACKUP_DIR, fn)
        out.append({"file": fn, "size_mb": round(os.path.getsize(p) / 1e6, 2),
                    "mtime": datetime.fromtimestamp(os.path.getmtime(p))
                    .astimezone().isoformat(timespec="seconds")})
    return out


def restore(zip_path: str, dry_run: bool = True) -> Dict:
    """
    从备份恢复。dry_run=True 时只列出会被覆盖的文件，不动磁盘。
    """
    if not os.path.isfile(zip_path):
        return {"ok": False, "reason": "file_not_found"}
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            will = [n for n in names if n.startswith(("data/state/", "data/model/"))]
            if dry_run:
                return {"ok": True, "dry_run": True, "n": len(will),
                        "sample": will[:10],
                        "note": "dry_run=True 未改动任何文件；传 False 才会真正覆盖"}
            for n in will:
                z.extract(n, ROOT)
            return {"ok": True, "restored": len(will)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"[:120]}


def summary() -> Dict:
    vs = list_versions()
    return {"ok": True, "current": current_version(), "n_versions": len(vs),
            "versions": vs[:5], "backups": list_backups()[:5],
            "keep": KEEP}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
    if os.path.isfile(paths()[0]):
        r = backup()
        print("\n备份:", json.dumps(r, ensure_ascii=False))
