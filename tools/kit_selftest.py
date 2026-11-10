"""
训练机套件自检：确认这台机器能承接**所有**任务类型。

## 为什么需要它

套件是精简打包的（只含训练/外包任务真正需要的模块）。精简的代价是：
一旦漏了某个动态依赖，训练机到用户手上才发现跑不起来 —— 那时候
用户只看到一堆看不懂的 ImportError。

所以在**构建时**和**首次运行时**都跑一遍这个自检：
  · 每个任务入口能否 import
  · 每个任务能否真的执行到「需要数据之前」那一步
  · 依赖（numpy / pandas）版本对不对

## 三类任务

    train     训练模型并上传           qcore.live_model.train
    pipeline  全流程回测（NAS 外包）    qcore.pipeline.run
    prewarm   预热外部数据缓存          tools.prewarm_external.run

## 用法

    python tools/kit_selftest.py            # 全检
    python tools/kit_selftest.py --json     # 机器可读
    python tools/kit_selftest.py --job train  # 只查某一类

退出码 0 = 全部可承接；非 0 = 有任务接不了（并打印具体原因）。
"""
from __future__ import annotations

import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _imp(mod: str, attr: str = "") -> tuple:
    """import 一个模块（可指定属性），返回 (ok, 详情)。"""
    try:
        m = importlib.import_module(mod)
        if attr:
            if not hasattr(m, attr):
                return False, f"{mod} 缺少 {attr}"
            return True, f"{mod}.{attr}"
        return True, mod
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:100]}"


#: 任务类型 -> 需要 import 的 (模块, 属性)
JOBS = {
    "train": [
        ("qcore.live_model", "train"),
        ("qcore.gbdt", ""),
        ("qcore.qlib_data", ""),
        ("qcore.data_sync", ""),
        ("qcore.model_transfer", "export_bundle"),
        ("qcore.versioning", ""),
        ("tools.train_one_click", "train_horizons"),
        ("tools.defaults", ""),
    ],
    "pipeline": [
        ("qcore.pipeline", ""),
        ("qcore.dataset", "build_dataset"),
        ("qcore.engine", "run_walkforward"),
        ("qcore.signals", "signal_zoo"),
        ("qcore.analysis", ""),
        ("qcore.stats", ""),
    ],
    "prewarm": [
        ("qcore.datasource", ""),
        ("qcore.event_data", "load_events"),
        ("qcore.flow_factor", "fetch_flow"),
        ("tools.prewarm_external", "run"),
    ],
    "sync": [
        # sync_agent 通过 HTTP 跟 NAS 说话，**不导入 qcore.sync**
        # （qcore.sync 是 NAS 侧的队列实现）。实测确认过，别凭印象加依赖。
        ("tools.sync_agent", "run_task"),
        ("qcore.kit_update", "check_and_update"),
        ("qcore.model_transfer", "import_bundle"),
    ],
}


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    only = ""
    if "--job" in argv:
        i = argv.index("--job")
        only = argv[i + 1] if len(argv) > i + 1 else ""

    print("[1] 运行时依赖")
    try:
        import numpy as _np
        import pandas as _pd
        check("numpy", True, _np.__version__)
        check("pandas", True, _pd.__version__)
    except Exception as e:  # noqa: BLE001
        check("numpy / pandas 可导入", False, f"{type(e).__name__}: {e}")

    print("\n[2] 版本与包结构")
    try:
        import qcore
        check("qcore 可导入", True, getattr(qcore, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        check("qcore 可导入", False, f"{type(e).__name__}: {e}")
    check("VERSION 文件存在", os.path.isfile(os.path.join(ROOT, "VERSION")))

    print("\n[3] 各任务类型的入口")
    results = {}
    for job, mods in JOBS.items():
        if only and job != only:
            continue
        print(f"  -- {job}")
        bad = []
        for mod, attr in mods:
            ok, detail = _imp(mod, attr)
            if not ok:
                bad.append(f"{mod}{('.' + attr) if attr else ''}: {detail}")
        results[job] = bad
        check(f"{job} 可承接", not bad,
              "全部入口就绪" if not bad else "; ".join(bad)[:220])

    print("\n[4] 数据与配置")
    try:
        from qcore import data_sync as DS
        check("能读到本地数据日期", True, str(DS.local_date()))
    except Exception as e:  # noqa: BLE001
        check("能读到本地数据日期", False, f"{type(e).__name__}: {str(e)[:80]}")
    try:
        import defaults as DEF
        check("apb_defaults 可读", True, f"nas={DEF.nas()[:40]}")
        check("已配置鉴权（token 或 password）",
              bool(DEF.token() or DEF.password()),
              "有 token" if DEF.token() else ("有 password" if DEF.password() else
                                              "都没有 —— 上传会 login_required"))
    except Exception as e:  # noqa: BLE001
        check("apb_defaults 可读", False, f"{type(e).__name__}: {str(e)[:80]}")

    out = {"ok": FAIL == 0, "passed": PASS, "failed": FAIL,
           "jobs": {k: (not v) for k, v in results.items()},
           "unsupported": {k: v for k, v in results.items() if v}}
    if "--json" in argv:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print(f"\nRESULT: {PASS} passed, {FAIL} failed")
        if out["unsupported"]:
            print("接不了的任务：")
            for k, v in out["unsupported"].items():
                print(f"  {k}: {v[:3]}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
