#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_model_admission.py —— 模型准入（R36/R40）与点对点成分（R4）

守住五件事：
  1. 训练机 kit 版本**低于**本机 → 拒绝（旧因子集不许覆盖）
  2. 新模型 `train_end` **早于**线上 → 拒绝（旧数据不许盖新模型）
  3. 缺 `qa.holdout_ic` → 标 `quality_unverified`，但**不阻断**（先有证据再加码）
  4. `APB_MODEL_ALLOW_OLD=1` 可显式放行，且 detail 留痕
  5. 点对点成分比"测试起点快照"多出**显著数量**的历史成员（未来条件选择的量级）

用法：python -m tests.test_model_admission
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import model_transfer as MT  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("model admission（R36/R40）+ pit universe（R4）")
    print("=" * 70)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp", f"admit_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    # 造一个"线上已部署"的 meta，让 train_end 比较有基准
    from qcore import live_model as LM
    mp, metap = LM._paths(1)
    backup = None
    if os.path.isfile(metap):
        backup = open(metap, "rb").read()
    try:
        with open(metap, "w", encoding="utf-8") as f:
            json.dump({"train_end": "2026-10-10", "n_trees": 20, "n_stocks": 60}, f)

        # 1) 版本更低 → 硬拒
        r = MT._admission({"train_end": "2026-11-01"},
                          {"env": {"kit_version": "0.1.0"}, "qa": {"holdout_ic": 0.01}}, 1)
        ver = next(c for c in r["checks"] if c["check"] == "kit_version")
        check("kit 版本更低 → 拒绝", ver["ok"] is False, ver["detail"])
        check("硬门未通过", r["hard_ok"] is False)

        # 2) train_end 更早 → 硬拒
        r = MT._admission({"train_end": "2026-09-01"},
                          {"env": {"kit_version": "99.0.0"}, "qa": {"holdout_ic": 0.01}}, 1)
        te = next(c for c in r["checks"] if c["check"] == "train_end")
        check("train_end 更早 → 拒绝", te["ok"] is False, te["detail"])

        # 3) 缺质量证据 → 只标注，不阻断
        r = MT._admission({"train_end": "2026-11-01"},
                          {"env": {"kit_version": "99.0.0"}, "qa": {}}, 1)
        check("缺 holdout_ic → quality_unverified", r["quality_unverified"] is True)
        check("缺质量证据**不**阻断安装", r["hard_ok"] is True)

        # 4) 显式放行
        os.environ["APB_MODEL_ALLOW_OLD"] = "1"
        r = MT._admission({"train_end": "2026-09-01"},
                          {"env": {"kit_version": "0.1.0"}, "qa": {"holdout_ic": 0.01}}, 1)
        check("APB_MODEL_ALLOW_OLD=1 → 放行", r["hard_ok"] is True)
        check("放行留痕", any("APB_MODEL_ALLOW_OLD" in c["detail"] for c in r["checks"]),
              str([c["detail"] for c in r["checks"] if "ALLOW" in c["detail"]])[:80])
        os.environ.pop("APB_MODEL_ALLOW_OLD", None)

    finally:
        if backup is not None:
            open(metap, "wb").write(backup)
        shutil.rmtree(tmp, ignore_errors=True)

    # 5) 点对点成分 vs 快照
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ltb", os.path.join(root, "scripts", "lowturnover_bt.py"))
        ltb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ltb)
        from qcore import qlib_data as QD
        pit = ltb.pit_codes("csi300", "2008-01-01", "2021-01-01")
        snap = QD.universe_codes("csi300", start="2021-01-01")
        extra = set(pit) - set(snap)
        check("点对点成分 ≥ 快照成分", len(pit) >= len(snap), f"{len(pit)} vs {len(snap)}")
        check("快照漏掉了大量历史成员（≥50 只）", len(extra) >= 50,
              f"仅快照覆盖 {len(snap)}，点对点 {len(pit)}，多出 {len(extra)}")
    except Exception as e:  # noqa: BLE001
        check("点对点成分可计算", False, f"{type(e).__name__}: {str(e)[:80]}")

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
