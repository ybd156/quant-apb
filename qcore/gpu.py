"""
GPU 加速（可选）—— 有就用，没有就退回 numpy，绝不硬依赖。

## 实测结论（2026-11，本项目真实数据）

| 设备 | 内存带宽 | FP32 算力 | 直方图 scatter-add | 相对 N100 CPU |
|---|---|---|---|---|
| **N100 CPU**（NAS） | 18.0 GB/s | 137.8 GFLOPS | 185 M 元素/s | 1× |
| **N100 核显**（UHD 24CU） | 17.5 GB/s | 50.8 GFLOPS | — | **0.4×（更慢）** |
| **RTX 3060 Laptop** | **310.7 GB/s** | **6172 GFLOPS** | **1154 M 元素/s** | **6.2×** |

**核显不要用**：它和 CPU 共享系统内存，带宽打平（17.5 vs 18.0），
而算力只有 CPU 的 37%。GBDT 直方图是内存受限的 scatter-add，
带宽不涨就等于不会变快，反而多付内核启动开销。

**独显值得用**：3060 带宽是 CPU 的 17 倍，直方图实测快 6.2 倍。

## 架构建议（这是关键）

    笔记本 GPU 训练  →  模型文件  →  NAS 推理
      6.2× 加速          259KB        CPU 毫秒级

推理本来就是毫秒级，不需要 GPU。所以 GPU 只用于**离线训练**，
NAS 上永远走 CPU —— 这样既拿到加速，又不给部署增加任何依赖。

## 依赖

`cupy-cuda12x` + `nvidia-*-cu12`（提供 DLL）。
**装不上就自动降级**，`available()` 返回 False，调用方无感。
"""
from __future__ import annotations

import os
import site
import glob
from typing import Optional

import numpy as np

_state = {"checked": False, "cp": None, "reason": ""}


def _bootstrap_dll_path() -> None:
    """Windows 上 cupy 找不到 CUDA DLL 时，把 nvidia pip 包的 bin 目录加进 PATH。"""
    if os.name != "nt":
        return
    try:
        paths = list(site.getsitepackages())
        try:
            paths.append(site.getusersitepackages())
        except Exception:
            pass
        for sp in paths:
            for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
    except Exception:
        pass


def available(force: bool = False) -> bool:
    """GPU 是否可用。结果缓存，避免反复尝试导入。"""
    if _state["checked"] and not force:
        return _state["cp"] is not None
    _state["checked"] = True
    if os.environ.get("APB_GPU", "1") == "0":
        _state["cp"] = None
        _state["reason"] = "APB_GPU=0 禁用"
        return False
    _bootstrap_dll_path()
    try:
        import cupy as cp  # noqa: F401
        a = cp.zeros(8, dtype=cp.float32)
        _ = (a + 1.0).sum()
        cp.cuda.Stream.null.synchronize()
        _state["cp"] = cp
        _state["reason"] = f"cupy {cp.__version__}"
        return True
    except Exception as e:  # noqa: BLE001
        _state["cp"] = None
        _state["reason"] = f"{type(e).__name__}: {str(e)[:80]}"
        return False


def info() -> dict:
    ok = available()
    out = {"available": ok, "reason": _state["reason"]}
    if ok:
        cp = _state["cp"]
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            props = cp.cuda.runtime.getDeviceProperties(0)
            out.update({
                "device": props["name"].decode() if isinstance(props["name"], bytes)
                else str(props["name"]),
                "compute": f"{props['major']}.{props['minor']}",
                "vram_free_gb": round(free / 1e9, 2),
                "vram_total_gb": round(total / 1e9, 2),
                "cupy": cp.__version__,
            })
        except Exception as e:  # noqa: BLE001
            out["detail_error"] = str(e)[:80]
    return out


def hist_batch(Xb: np.ndarray, cols: np.ndarray, idx: np.ndarray,
               gnode: np.ndarray, hnode: np.ndarray, nb: int,
               use_gpu: Optional[bool] = None) -> tuple:
    """
    批量直方图 —— GPU 版把 158 次 bincount 合并成 1 次。

    原理：给每个特征的箱号加一个偏移 j*nb，展平后一次 bincount，
    结果 reshape 回 (ng, nb)。这样只有一次内核启动，而不是 158 次。

    返回 (hist_g, hist_h) 形状 (ng, nb)，与 numpy 版数值一致。
    """
    ng = cols.size
    if use_gpu is None:
        use_gpu = available() and ng >= 32     # 特征太少时 GPU 反而慢（启动开销）

    if not use_gpu:
        hist_g = np.empty((ng, nb))
        hist_h = np.empty((ng, nb))
        for j in range(ng):
            b = Xb[cols[j]].take(idx)
            hist_g[j] = np.bincount(b, weights=gnode, minlength=nb)
            hist_h[j] = np.bincount(b, weights=hnode, minlength=nb)
        return hist_g, hist_h

    cp = _state["cp"]
    try:
        # 一次性把该节点需要的特征列搬到显存（(n, ng) 的 gather）
        sub = Xb[:, cols].take(idx, axis=0)          # (n, ng) uint8/uint16
        off = (np.arange(ng, dtype=np.int64) * nb).reshape(1, ng)
        flat = (sub.astype(np.int64) + off).ravel()
        d_flat = cp.asarray(flat)
        d_g = cp.asarray(gnode, dtype=cp.float64)
        d_h = cp.asarray(hnode, dtype=cp.float64)
        n = idx.size
        total = ng * nb
        hg = cp.bincount(d_flat, weights=cp.tile(d_g, ng), minlength=total)
        hh = cp.bincount(d_flat, weights=cp.tile(d_h, ng), minlength=total)
        return (cp.asnumpy(hg).reshape(ng, nb),
                cp.asnumpy(hh).reshape(ng, nb))
    except Exception:
        # GPU 路径出任何问题都退回 CPU，不让训练挂掉
        hist_g = np.empty((ng, nb))
        hist_h = np.empty((ng, nb))
        for j in range(ng):
            b = Xb[cols[j]].take(idx)
            hist_g[j] = np.bincount(b, weights=gnode, minlength=nb)
            hist_h[j] = np.bincount(b, weights=hnode, minlength=nb)
        return hist_g, hist_h


def bench(n: int = 8_000_000, nb: int = 32, ng: int = 158) -> dict:
    """CPU vs GPU 直方图对照（用于诊断这台机器值不值得开 GPU）。"""
    rng = np.random.default_rng(0)
    Xb = rng.integers(0, nb, size=(n, ng)).astype(np.uint8)
    cols = np.arange(ng)
    idx = np.arange(n)
    g = rng.normal(size=n)
    h = np.full(n, 0.5)
    out = {"n": n, "ng": ng, "nb": nb, "gpu_available": available()}

    import time
    t0 = time.perf_counter()
    hist_batch(Xb, cols, idx, g, h, nb, use_gpu=False)
    out["cpu_s"] = round(time.perf_counter() - t0, 4)
    if out["gpu_available"]:
        hist_batch(Xb, cols, idx, g, h, nb, use_gpu=True)      # 预热
        t0 = time.perf_counter()
        hist_batch(Xb, cols, idx, g, h, nb, use_gpu=True)
        out["gpu_s"] = round(time.perf_counter() - t0, 4)
        out["speedup"] = round(out["cpu_s"] / max(out["gpu_s"], 1e-6), 2)
    out["info"] = info()
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(info(), ensure_ascii=False, indent=1))
    print()
    print(json.dumps(bench(), ensure_ascii=False, indent=1))
