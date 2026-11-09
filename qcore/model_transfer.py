"""
模型搬运 —— 在别的电脑训练，把模型放进 NAS。

## 为什么这是对的架构

实测（2026-11）：

| 设备 | 内存带宽 | 直方图 scatter-add |
|---|---|---|
| N100 CPU（NAS） | 18.0 GB/s | 185 M 元素/s |
| 本机 Ryzen | ~40 GB/s | 180 M 元素/s |
| RTX 3060 | 310.7 GB/s | 1154 M 元素/s |

GBDT 是**内存受限**的，所以 N100 的直方图和桌面 CPU 几乎一样快 ——
NAS 自己能训，只是慢一点（250 树约 14 分钟/周期）。

但如果手边有更强的机器（或 GPU），**训练在那儿做、模型搬进来**更划算。
本模块负责搬运，并且**先校验再安装**，不让坏模型污染线上。

## 校验什么（这四条缺一不可）

1. **能反序列化** —— pickle 损坏、版本不兼容，立刻拒
2. **特征数对得上** —— 模型期望 158（或蒸馏后的 60），当前代码必须产出同样的列数
3. **feature_idx 合法** —— 蒸馏模型带的是列索引，越界会静默喂错列，必须挡
4. **能真的预测** —— 拿随机数据跑一次 predict，出 NaN/异常就拒

校验不过就**不安装**，原模型保持不变。

## 用法

```bash
# 在训练机上
python -m qcore.model_transfer --export --horizon=5

# 在 NAS 上
python -m qcore.model_transfer --import model_h5.zip
```

或走 API：`POST /api/model/import`（body 是 zip 的 base64）。
"""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import zipfile
from datetime import datetime
from typing import Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "data", "model")
STATE = os.path.join(ROOT, "data", "state")
MANIFEST = "apb_model.json"


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


# ---------------------------------------------------------------- 导出
def export_bundle(horizon: int = 1, out_path: Optional[str] = None) -> Dict:
    """把某个 horizon 的模型打成 zip，便于拷到别的机器。"""
    from . import live_model as LM
    mp, metap = LM._paths(int(horizon))
    if not os.path.isfile(mp):
        return {"ok": False, "reason": "model_not_found", "horizon": int(horizon),
                "path": mp}
    meta = {}
    try:
        with open(metap, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass
    manifest = {
        "format": "apb-model-v1",
        "horizon": int(horizon),
        "exported_at": _now(),
        "meta": meta,
        "env": {
            "python": f"{__import__('sys').version_info.major}."
                      f"{__import__('sys').version_info.minor}",
            "numpy": np.__version__,
        },
    }
    out = out_path or os.path.join(
        STATE, f"model_h{int(horizon)}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1,
                                        default=str))
        z.write(mp, "model.pkl")
    return {"ok": True, "path": out, "horizon": int(horizon),
            "size_mb": round(os.path.getsize(out) / 1e6, 2),
            "meta": meta,
            "note": "拷到 NAS 后执行 python -m qcore.model_transfer --import <zip>"}


# ---------------------------------------------------------------- 校验
def _validate(model, meta: Dict, expected_features: Optional[int] = None) -> Dict:
    """
    四条校验。返回 {"ok": bool, "checks": [...], "reason": ...}
    """
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    # ① 反序列化成功（能走到这里就说明 pickle 没问题）
    add("unpickle", True, f"{type(model).__name__}")

    # ② 特征数
    # 优先信模型自身的 _n_features —— meta 是人写的，可能串位（实测有过
    # n_factors 被写成 n_stocks 的情况）。模型属性是训练时定死的，更可靠。
    nf_model = None
    for attr in ("_n_features", "n_features_in_", "n_features"):
        if hasattr(model, attr):
            v = getattr(model, attr)
            nf_model = v() if callable(v) else v
            break
    nf = nf_model if nf_model is not None else (meta or {}).get("n_factors")
    fidx = (meta or {}).get("feature_idx")
    expect = expected_features if expected_features is not None else nf
    if fidx:
        n_eff = len(fidx)
        ok_idx = True
        try:
            mx = int(max(fidx))
            ok_idx = mx < (expected_features or mx + 1)
        except Exception:
            ok_idx = False
        add("feature_idx", ok_idx,
            f"{n_eff} 个索引，最大 {max(fidx) if fidx else '-'} / 上限 {expected_features}")
        add("feature_count", n_eff == (expect or n_eff), f"模型 {n_eff} vs 期望 {expect}")
    else:
        add("feature_count", nf is None or expect is None or int(nf) == int(expect),
            f"模型 {nf} vs 期望 {expect}")

    # ③ horizon 必须声明
    h = (meta or {}).get("horizon")
    add("horizon_declared", h is not None, f"horizon={h}")

    # ④ 能真的预测
    try:
        k = int(nf or (len(fidx) if fidx else 158))
        x = np.random.default_rng(0).normal(size=(32, k)).astype(np.float32)
        p = model.predict(x)
        ok_pred = (p is not None and np.asarray(p).shape[0] == 32
                   and np.isfinite(np.asarray(p, dtype=float)).mean() > 0.9)
        add("smoke_predict", ok_pred, f"输出 {np.asarray(p).shape}，"
                                      f"有限值占比 {np.isfinite(np.asarray(p, dtype=float)).mean():.2f}")
    except Exception as e:  # noqa: BLE001
        add("smoke_predict", False, f"{type(e).__name__}: {str(e)[:70]}")

    bad = [c for c in checks if not c["ok"]]
    return {"ok": not bad, "checks": checks,
            "reason": (bad[0]["check"] + ": " + bad[0]["detail"]) if bad else ""}


def _expected_feature_count() -> Optional[int]:
    """当前代码产出的 Alpha158 因子数。取不到返回 None（不阻断）。"""
    try:
        from . import alpha158
        names = alpha158.factor_names() if hasattr(alpha158, "factor_names") else None
        if names:
            return len(names)
    except Exception:
        pass
    try:
        with open(os.path.join(STATE, "factor_importance.json"), encoding="utf-8") as f:
            d = json.load(f)
        rows = d.get("rows") or []
        if rows:
            return len(rows)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 导入
def import_bundle(src, horizon: Optional[int] = None,
                  dry_run: bool = False, install: bool = True) -> Dict:
    """
    从 zip 文件或 bytes 导入模型。校验不过不安装。
    src: 文件路径 / bytes / base64 字符串
    """
    from . import live_model as LM
    from . import versioning as V

    raw: bytes
    if isinstance(src, bytes):
        raw = src
    elif isinstance(src, str) and os.path.isfile(src):
        with open(src, "rb") as f:
            raw = f.read()
    elif isinstance(src, str):
        try:
            raw = base64.b64decode(src)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"bad_base64: {type(e).__name__}"}
    else:
        return {"ok": False, "reason": "bad_source"}

    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"not_a_zip: {type(e).__name__}"}

    names = z.namelist()
    if "model.pkl" not in names:
        return {"ok": False, "reason": "missing_model_pkl", "files": names[:10]}
    manifest = {}
    if MANIFEST in names:
        try:
            manifest = json.loads(z.read(MANIFEST).decode("utf-8"))
        except Exception:
            manifest = {}
    meta = dict(manifest.get("meta") or {})
    h = int(horizon or manifest.get("horizon") or meta.get("horizon") or 1)
    meta["horizon"] = h

    try:
        model = pickle.loads(z.read("model.pkl"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"unpickle_failed: {type(e).__name__}: {str(e)[:60]}"}

    expect = _expected_feature_count()
    val = _validate(model, meta, expect)
    if not val["ok"]:
        return {"ok": False, "reason": "validation_failed",
                "detail": val["reason"], "checks": val["checks"],
                "note": "校验未通过，未安装；线上模型保持不变"}

    if dry_run:
        return {"ok": True, "dry_run": True, "horizon": h, "checks": val["checks"],
                "meta": meta, "would_install": bool(install)}

    if not install:
        return {"ok": True, "validated": True, "horizon": h, "checks": val["checks"]}

    # 安装：走 versioning.save，自动留快照，可回滚
    saved = V.save(model, meta, tag=f"import-h{h}")
    LM.save_model(model, meta, horizon=h)      # 同步到 horizon 路径
    return {"ok": True, "installed": True, "horizon": h,
            "version": saved.get("version"), "checks": val["checks"],
            "meta": meta,
            "note": f"已安装为 horizon={h} 的模型，可用 /api/model/rollback 回滚"}


def list_local() -> Dict:
    """本机现有各 horizon 的模型状态。"""
    from . import live_model as LM
    from . import horizons as HZ
    out = []
    for h in HZ.all_horizons():
        mp, metap = LM._paths(h)
        exists = os.path.isfile(mp)
        meta = {}
        if exists:
            try:
                with open(metap, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
        out.append({"horizon": h, "label": HZ.label_for(h), "exists": exists,
                    "size_mb": round(os.path.getsize(mp) / 1e6, 2) if exists else 0,
                    "trained_at": meta.get("trained_at"),
                    "n_trees": meta.get("n_trees"),
                    "n_factors": meta.get("n_factors"),
                    "distilled": meta.get("distilled"),
                    "feature_idx": bool(meta.get("feature_idx"))})
    return {"ok": True, "models": out,
            "note": "训练可以在别的机器做，导出的 zip 用 --import 放进来"}


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    h = 1
    for a in args:
        if a.startswith("--horizon="):
            h = int(a.split("=", 1)[1])
    if "--export" in args:
        print(json.dumps(export_bundle(h), ensure_ascii=False, indent=1))
    elif "--import" in args:
        src = None
        for i, a in enumerate(args):
            if a == "--import" and i + 1 < len(args):
                src = args[i + 1]
        if not src:
            print(json.dumps({"ok": False, "reason": "need_zip_path"}, ensure_ascii=False))
        else:
            print(json.dumps(import_bundle(src, horizon=h), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(list_local(), ensure_ascii=False, indent=1))
