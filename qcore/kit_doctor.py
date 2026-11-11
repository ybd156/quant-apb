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

    只检查「训练/同步真的会用到、且缺失会直接导致跑不动」的能力：
      · wheels/ 里有 numpy wheel   -> 否则装依赖要走网络，可能十几分钟
      · data_sync 有 _mirror_urls  -> 否则只会从 GitHub 下数据，国内失败
      · paper_auto 有 reset        -> 否则模拟盘不能清空重设

    ## 不该出现在这里的检查

    任何「只有 NAS / 容器侧才用到」的模块都**不能**作为套件新旧判据 ——
    精简打包（`scripts/kit_manifest.py`）本来就不会带它们，
    会稳定误报「太旧」。版本新旧交给 `qcore/kit_update.py` 跟服务端比对。
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

    # 4) 更新进度 —— **故意不检查**。
    #
    # `update_progress.py` 是 NAS 容器自更新的进度模块，训练机不跑容器、
    # 永远用不到它，所以 `scripts/kit_manifest.py` 把它列进了 NEVER（不打包）。
    #
    # 原来这里用「文件在不在」判断套件新旧 —— 于是精简打包之后必然误报：
    # 用户下载的明明是最新的 0.32.6，却被告知「这套训练套件太旧」，
    # 而且同一个页面同时写着「当前套件版本：0.32.6」，自相矛盾（实测截图）。
    #
    # 版本新旧应该由 `qcore/kit_update.py` 跟服务端比对，不是靠猜文件。
    # 这里只检查「训练真的需要的能力」。

    ok = not problems
    advice = ""
    if not ok:
        # 不再说「太旧」—— 缺失可能是打包不全、目录残留、或真旧，
        # 断言成「太旧」会误导用户反复重下同一个版本（实测踩过）。
        advice = (
            "这套套件缺少运行所需的能力（上面列出的项）。\n"
            "  1) 确认解压到**新的空目录**，不要覆盖旧目录（残留文件会互相干扰）\n"
            "  2) 仍不行就重新下载：https://github.com/ybd156/quant-apb/releases/latest\n"
            f"     文件名：apb-train-kit_<版本>.zip\n"
            f"  当前套件版本：{version or '未知'}"
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
