"""
训练机套件的模块清单：**只打包训练/外包任务真正需要的模块**。

## 为什么要做这件事

套件原来把整个 `qcore/` 拷进去 —— 126 个模块 / 42,222 行，
但训练机实际只需要 61 个 / 22,784 行（46% 是 Web UI、盘中、告警、
报告这类只在 NAS 上跑的东西）。

副作用不只是包大：
  · 训练机 `import` 时的 .pyc 编译、杀毒扫描、首次启动都变慢
  · 用户看到一堆用不上的文件，报错时也分不清哪些是相关代码
  · 训练机是「给别人拷过去就能跑」的东西，越干净越不容易出错

## 怎么算的（机械、可复现）

从**任务入口**出发做传递闭包（AST 解析 `from . import x` / `import qcore.x`）：

    train    qcore/live_model.py, tools/train_one_click.py, ...
    pipeline qcore/pipeline.py        ← job=pipeline 外包全流程回测
    prewarm  qcore/datasource.py, qcore/event_data.py, tools/prewarm_external.py

闭包外的模块一律不打包。清单写进 `kit_manifest.json`，构建时可复核。

## 用法

    python -m scripts.kit_manifest                 # 打印清单与统计
    python -m scripts.kit_manifest --json          # 机器可读
    python -m scripts.kit_manifest --verify DIR    # 校验某个套件目录是否缺模块
"""
from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC = os.path.join(ROOT, "qcore")

#: 训练机要执行的三类任务的入口
ENTRIES: List[str] = [
    # --- train：训练模型并上传
    "qcore/live_model.py",
    "qcore/gbdt.py",
    "qcore/qlib_data.py",
    "qcore/data_sync.py",
    "qcore/kit_update.py",
    "qcore/kit_doctor.py",
    "qcore/versioning.py",
    "qcore/model_transfer.py",
    "qcore/horizons.py",
    "qcore/env_guard.py",
    "qcore/daily_append.py",
    "qcore/net.py",
    "tools/train_one_click.py",
    "tools/sync_agent.py",
    "tools/train_missing.py",
    "tools/defaults.py",
    "scripts/benchmark_alpha158.py",
    # --- pipeline：外包全流程回测（NAS 不做重活）
    "qcore/pipeline.py",
    # --- prewarm：预热外部数据缓存
    "qcore/datasource.py",
    "qcore/event_data.py",
    "tools/prewarm_external.py",
    # --- 静默后台守护：持续跟随版本更新 + 崩溃自动拉起
    "tools/agent_daemon.py",
    # --- 配置向导：窗口填令牌 / 测连接 / 保存
    "tools/kit_setup.py",
    # --- 唯一入口：一屏状态 + 菜单
    "tools/kit_status.py",
    "tools/kit_menu.py",
]

#: 无论闭包是否覆盖，都必须打包的（`__init__` 是包本身）
ALWAYS = ["qcore/__init__.py"]

#: 明确不打包的（只在 NAS 侧用，或属于开发工具）
NEVER = [
    "qcore/docker_api.py",      # 容器自更新，训练机不更新容器
    "qcore/update_apply.py",    # 同上
    "qcore/update_progress.py",
    "qcore/auth.py",            # 登录态是 NAS 的事
    "qcore/tokens.py",
    "qcore/net_guard.py",
]


def _mods_of(path: str) -> Set[str]:
    """一个文件里 `from . import x` / `import qcore.x` 引用的本包模块。"""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:  # noqa: BLE001
        return set()
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
            out.add(n.module.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.level == 1:
            for a in n.names:
                out.add(a.name.split(".")[0])
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("qcore."):
                    out.add(a.name.split(".")[1])
    return out


def closure() -> Dict:
    """计算训练机需要的文件闭包。"""
    seen: Set[str] = set()
    todo = list(ENTRIES) + list(ALWAYS)
    while todo:
        rel = os.path.normpath(todo.pop())
        if rel in seen:
            continue
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            continue
        seen.add(rel)
        for m in _mods_of(full):
            p = os.path.normpath(os.path.join("qcore", m + ".py"))
            if os.path.isfile(os.path.join(ROOT, p)) and p not in seen:
                todo.append(p)

    all_qcore = sorted(f for f in os.listdir(QC) if f.endswith(".py"))
    needed_qcore = sorted(os.path.basename(p) for p in seen if p.startswith("qcore" + os.sep))
    excluded = [f for f in all_qcore if f not in needed_qcore]

    def lines(rels: List[str]) -> int:
        n = 0
        for r in rels:
            try:
                n += len(open(os.path.join(ROOT, r), encoding="utf-8").read().splitlines())
            except Exception:  # noqa: BLE001
                pass
        return n

    needed_rel = sorted(seen)
    excluded_rel = [os.path.join("qcore", f) for f in excluded]
    return {
        "ok": True,
        "entries": ENTRIES,
        "always": ALWAYS,
        "never": NEVER,
        "modules": needed_qcore,
        "files": needed_rel,
        "excluded": excluded,
        "stats": {
            "qcore_total": len(all_qcore),
            "qcore_needed": len(needed_qcore),
            "qcore_excluded": len(excluded),
            "lines_needed": lines(needed_rel),
            "lines_excluded": lines(excluded_rel),
            "cut_ratio": round(lines(excluded_rel) / max(1, lines(needed_rel) + lines(excluded_rel)), 3),
        },
    }


def verify(kit_dir: str, do_import: bool = True) -> Dict:
    """校验已构建的套件目录。

    两层：
      1. **文件层**：闭包里的模块是否都在（快，抓漏拷）
      2. **导入层**：把每个模块真 import 一遍（慢，但抓得到动态依赖）
         —— 静态 AST 分析看不见 `importlib.util.spec_from_file_location`
            这类动态加载，也看不见字符串拼出来的 import。
            训练机要能承接**所有**任务，光文件齐不够，必须能跑起来。
    """
    man = closure()
    missing = [f for f in man["files"] if not os.path.isfile(os.path.join(kit_dir, f))]
    extra = []
    qc_dir = os.path.join(kit_dir, "qcore")
    if os.path.isdir(qc_dir):
        for f in sorted(os.listdir(qc_dir)):
            if f.endswith(".py") and f not in man["modules"] and f not in NEVER_BASENAMES:
                extra.append(f)
    out = {"ok": not missing, "missing": missing, "extra": extra,
           "n_expected": len(man["files"])}
    if do_import:
        import subprocess
        names = [x[:-3] for x in man["modules"]]
        code = "\n".join([
            "import importlib, json",
            "bad = []",
            "for n in " + repr(names) + ":",
            "    try:",
            "        importlib.import_module('qcore.' + n)",
            "    except Exception as e:",
            "        bad.append(n + ': ' + type(e).__name__ + ': ' + str(e)[:80])",
            "print(json.dumps({'failed': bad}))",
        ])
        try:
            r = subprocess.run([sys.executable, "-c", code], cwd=kit_dir,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=180)
            data = json.loads((r.stdout or "{}").strip().splitlines()[-1] or "{}")
            out["import_failed"] = data.get("failed") or []
            out["ok"] = bool(out["ok"]) and not out["import_failed"]
        except Exception as e:  # noqa: BLE001
            out["import_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            out["ok"] = False
    return out


NEVER_BASENAMES = {os.path.basename(p) for p in NEVER}


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--verify" in a:
        d = a[a.index("--verify") + 1]
        r = verify(d, do_import=("--no-import" not in a))
        print(json.dumps(r, ensure_ascii=False, indent=1))
        raise SystemExit(0 if r["ok"] else 1)
    m = closure()
    if "--json" in a:
        print(json.dumps(m, ensure_ascii=False, indent=1))
    else:
        s = m["stats"]
        print(f"qcore 总数   {s['qcore_total']} 个")
        print(f"需要打包     {s['qcore_needed']} 个 / {s['lines_needed']} 行")
        print(f"裁掉         {s['qcore_excluded']} 个 / {s['lines_excluded']} 行 "
              f"（{s['cut_ratio']*100:.0f}%）")
        print("\n需要的模块：")
        print("  " + " ".join(x[:-3] for x in m["modules"]))
        print("\n裁掉的模块：")
        print("  " + " ".join(x[:-3] for x in m["excluded"]))
