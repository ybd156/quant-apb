"""
模型自举 —— 缺什么自动补什么，不需要人盯着。

## 为什么需要

用户问得很直接：「没有模型预测就一定要停止更新然后才能训练？」

不是。**系统应该自愈。**

之前的逻辑：`LM.train()` 默认只训 horizon=1。
结果 h=3 / h=10 / h=15 永远没有模型，UI 上一片「未训」，
必须有人记得去训练机跑一遍。

这不是"设计选择"，是**漏了**。

## 这个模块做什么

    bootstrap(only_missing=True)   检查所有周期，缺的训，超龄的重训
    ensure(horizon)                单个周期：没模型就训，有模型就跳过

每一步都：
  1. 低优先级（调用方用 nice -19）
  2. 训完立刻 predict，让 UI 马上能用
  3. 失败不影响其他周期（一个崩了继续下一个）

## 训练时长（NAS N100，300 只 / 250 树）

    h=1  ~5 分钟
    单周期 5-8 分钟，5 个周期全补约 40 分钟

所以只在夜间窗口跑，不跟盘中抢 CPU。
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from . import live_model as LM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")


def _log(msg: str) -> None:
    print(msg, flush=True)


def horizons() -> List[int]:
    try:
        from . import horizons as HZ
        return [h for h in HZ.all_horizons() if h > 0]
    except Exception:
        return [1, 3, 5, 10, 15]


def model_exists(horizon: int) -> bool:
    try:
        m, meta = LM.load_model(horizon)
        return m is not None
    except Exception:
        return False


def picks_exists(horizon: int) -> bool:
    fn = "picks.json" if int(horizon) == 1 else f"picks_h{int(horizon)}.json"
    return os.path.isfile(os.path.join(STATE, fn))


def age_days(horizon: int) -> Optional[float]:
    try:
        return LM.model_age_days(horizon)
    except Exception:
        return None


def status() -> List[Dict]:
    """每个周期的模型/预测状态。"""
    max_age = float(os.environ.get("APB_LIVE_MAX_AGE", "7"))
    out = []
    for h in horizons():
        a = age_days(h)
        out.append({
            "horizon": h,
            "has_model": model_exists(h),
            "has_picks": picks_exists(h),
            "age_days": None if a is None else round(a, 1),
            "stale": bool(a is not None and a > max_age),
        })
    return out


def ensure(horizon: int, force: bool = False, verbose: bool = True) -> Dict:
    """
    确保某个周期有模型且新鲜。缺了/超龄就训练，训完立刻 predict。
    """
    h = int(horizon)
    has = model_exists(h)
    a = age_days(h) if has else None
    max_age = float(os.environ.get("APB_LIVE_MAX_AGE", "7"))

    need_train = force or (not has) or (a is not None and a > max_age)
    if not need_train:
        # 模型在但没预测 -> 只补预测，很快
        if not picks_exists(h):
            if verbose:
                _log(f"[bootstrap] h{h} 有模型但没预测，补一次 predict")
            try:
                LM.predict(horizon=h, verbose=verbose)
                return {"ok": True, "horizon": h, "action": "predicted_only"}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "horizon": h, "action": "predict",
                        "reason": f"{type(e).__name__}: {str(e)[:80]}"}
        if verbose:
            _log(f"[bootstrap] h{h} 模型新鲜（{a:.1f} 天），跳过")
        return {"ok": True, "horizon": h, "action": "skip"}

    if verbose:
        why = ("没有模型" if not has else f"模型已 {a:.1f} 天")
        _log(f"[bootstrap] h{h} {why} -> 开始训练")

    # 内存约束：NAS 上 300 只 × 250 树实测要 4GB+，会被 OOM 杀掉
    # （日志停在"标准化"之后，没有任何 traceback —— 典型的 OOM）
    # 用更保守的规模，模型效果差别很小（Alpha158 的 alpha 主要来自因子，不是树的数量）
    stocks = int(os.environ.get("APB_BOOT_STOCKS", "200"))
    trees = int(os.environ.get("APB_BOOT_TREES", "120"))
    try:
        import resource as _rs
        _avail_mb = None
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("MemAvailable:"):
                        _avail_mb = int(ln.split()[1]) // 1024
                        break
        except Exception:
            pass
        if verbose and _avail_mb:
            _log(f"[bootstrap] h{h} 可用内存 {_avail_mb} MB，"
                 f"用 {stocks} 只 / {trees} 树（可用内存 < 3000MB 时自动再降）")
        if _avail_mb and _avail_mb < 3000:
            stocks = min(stocks, 120)
            trees = min(trees, 80)
            if verbose:
                _log(f"[bootstrap] h{h} 内存偏紧，降到 {stocks} 只 / {trees} 树")
    except Exception:
        pass

    t0 = time.time()
    try:
        r = LM.train(horizon=h, stocks=stocks, trees=trees, verbose=verbose)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "horizon": h, "action": "train",
                "reason": f"{type(e).__name__}: {str(e)[:80]}"}
    if not r.get("ok"):
        return {"ok": False, "horizon": h, "action": "train",
                "reason": r.get("reason") or r.get("hint")}

    dt = time.time() - t0
    if verbose:
        _log(f"[bootstrap] h{h} 训练完成 {dt/60:.1f} 分钟，生成预测 ...")
    try:
        LM.predict(horizon=h, verbose=verbose)
        pred_ok = True
    except Exception as e:  # noqa: BLE001
        pred_ok = False
        if verbose:
            _log(f"[bootstrap] h{h} 预测失败: {type(e).__name__}: {str(e)[:80]}")
    return {"ok": True, "horizon": h, "action": "trained",
            "seconds": round(dt, 1), "predict_ok": pred_ok,
            "train_n": r.get("train_n"), "n_trees": r.get("n_trees")}


def bootstrap(only_missing: bool = True, force: bool = False,
              verbose: bool = True) -> Dict:
    """
    补齐所有周期的模型。

    only_missing=True  只处理「没模型」的（首次自举，最快上线）
    only_missing=False 额外的还处理「超龄」的（定期维护）
    """
    hs = horizons()
    todo = []
    for h in hs:
        has = model_exists(h)
        a = age_days(h) if has else None
        max_age = float(os.environ.get("APB_LIVE_MAX_AGE", "7"))
        if force:
            todo.append(h)
        elif not has:
            todo.append(h)
        elif not only_missing and a is not None and a > max_age:
            todo.append(h)
        elif not picks_exists(h):
            todo.append(h)          # 有模型没预测，也算待处理

    if verbose:
        _log(f"[bootstrap] 检查 {len(hs)} 个周期，待处理 {todo or '无'}")
    if not todo:
        return {"ok": True, "action": "nothing_to_do", "checked": hs}

    results = []
    for h in todo:
        r = ensure(h, force=force, verbose=verbose)
        results.append(r)
        # 一个失败不影响后面
        time.sleep(2)

    ok_n = sum(1 for r in results if r.get("ok"))
    if verbose:
        _log(f"[bootstrap] 完成 {ok_n}/{len(results)}")
    return {"ok": ok_n > 0, "checked": hs, "processed": todo,
            "ok_count": ok_n, "results": results}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if "--status" in a:
        import json
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif "--horizon" in a:
        h = 1
        for x in a:
            if x.startswith("--horizon="):
                h = int(x.split("=", 1)[1])
        print(ensure(h, force=("--force" in a)))
    else:
        bootstrap(only_missing=("--all" not in a), force=("--force" in a))
