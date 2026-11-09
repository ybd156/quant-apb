"""
训练机套件自检 —— 开跑前确认「这套代码够不够新」。

## 为什么需要

实测踩过：用户拿着几小时前下载的 kit 跑，症状是
    「数据准备失败: remote_unreachable」
    numpy 2.5.4（网络装的，说明没有 wheels 目录）

但 Release 上早就有新版了。旧版不会自己报告"我太旧了"，
用户只能看到一堆看不懂的错。

这个模块做两件事：
  1. 检查几个关键能力是否具备（wheels 目录、NAS 下载源、导出的函数）
  2. 缺了就打印**明确的一句话**：去下哪个文件
"""
from __future__ import annotations

import os
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check() -> Dict:
    """
    返回 {"ok": bool, "version": str, "problems": [...], "advice": str}

    检查项都是「本版才有的能力」，缺了说明套件太旧：
      · wheels/ 目录存在且含 numpy wheel  -> 否则装依赖要联网，很慢
      · data_sync 有 _mirror_urls          -> 否则不会从 NAS 下数据
      · paper_auto 有 reset                -> 否则模拟盘不能清空重设
      · webui 里有 update_progress         -> 否则更新没进度
    """
    problems: List[str] = []
    version = ""
    try:
        vp = os.path.join(ROOT, "VERSION")
        if os.path.isfile(vp):
            version = open(vp, encoding="utf-8").read().strip()
    except Exception:
        pass

    # 只有「训练套件」才要求 wheels（源码仓库本来就没有，不该误报）
    is_kit = os.path.isfile(os.path.join(ROOT, "apb_defaults.json"))

    # 1) 预打包 wheel
    wd = os.path.join(ROOT, "wheels")
    has_np = False
    if os.path.isdir(wd):
        try:
            has_np = any(f.startswith("numpy-") and f.endswith(".whl")
                         for f in os.listdir(wd))
        except Exception:
            has_np = False
    if is_kit and not has_np:
        problems.append("缺 wheels/ 里的 numpy 包 —— 装依赖会走网络，可能十几分钟")

    # 2) NAS 下载源
    try:
        from qcore import data_sync as DS
        if not hasattr(DS, "_mirror_urls"):
            problems.append("data_sync 不支持多源回退 —— 只能从 GitHub 下，国内会失败")
        else:
            try:
                urls = DS._mirror_urls()
                if not any("/api/data/qlib" in str(u) for u in urls):
                    problems.append("数据下载源里没有 NAS —— 仍依赖 GitHub")
            except Exception:
                problems.append("data_sync._mirror_urls 调用失败")
    except Exception as e:  # noqa: BLE001
        problems.append(f"导入 data_sync 失败: {type(e).__name__}")

    # 3) 模拟盘重置
    try:
        from qcore import paper_auto as PA
        if not hasattr(PA, "reset"):
            problems.append("模拟盘不支持清空重设（旧版）")
    except Exception:
        pass

    # 4) 更新进度
    if not os.path.isfile(os.path.join(ROOT, "qcore", "update_progress.py")):
        problems.append("没有更新进度模块（旧版）")

    ok = not problems
    advice = ""
    if not ok:
        _v = version or "最新"
        advice = (
            "这套代码太旧了，请重新下载最新训练套件：\n"
            f"    https://github.com/ybd156/quant-apb/releases/latest\n"
            "    文件名：apb-train-kit_<版本>.zip\n"
            "下载后解压到一个**新的空目录**，不要覆盖旧目录（残留文件会互相干扰）。\n"
            f"当前套件版本：{version or '未知'}"
        )
    return {"ok": ok, "version": version, "problems": problems, "advice": advice}


def print_report() -> bool:
    r = check()
    if r["ok"]:
        print(f"  [OK] 套件版本 {r['version'] or '(未标注)'} —— 具备 NAS 下载、"
              f"预打包依赖、模拟盘重设等能力")
        return True
    print()
    print("  " + "=" * 58)
    print("  [!] 这套训练套件太旧，缺以下能力：")
    for p in r["problems"]:
        print("      - " + p)
    print()
    for line in r["advice"].splitlines():
        print("  " + line)
    print("  " + "=" * 58)
    return False


if __name__ == "__main__":
    raise SystemExit(0 if print_report() else 1)
