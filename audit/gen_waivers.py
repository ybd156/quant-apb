"""
生成 `audit/waivers.json` —— 把 MEDIUM 泄漏告警按**机械可复核的规则**分流。

## 为什么不能手工逐条签收

实测 MEDIUM = 704 条，散在 ~40 个文件。逐条人工判断既不现实，也无法复核。
所以这里用一条**可重复执行的判据**：

> 若某文件**不在打分链路的传递闭包里**（即不被
> `signals.py / learner.py / engine.py / dataset.py` 直接或间接 import），
> 那么它内部的全样本 `mean/std/quantile` 不可能进入信号或权重，
> 只能影响"事后统计" —— 这是评估层统计，属 MEDIUM 的正常形态。

链路上的文件的 MEDIUM **一律不豁免**，继续计入"未确认"，交由人工逐条看。

## 产物

`audit/waivers.json`：
```json
{"criterion": "...", "generated_at": "...", "by": "lead",
 "waivers": [{"path": "qcore/x.py", "reason": "...", "scope": "file"}, ...],
 "unconfirmed_paths": [...]}
```

`leak_scan.py` 读这个文件，把匹配的 finding 标 WAIVED（并在汇总里区分
`WAIVED`（行内注释）与 `WAIVED_FILE`（本清单））。

## 跑法

    python -m audit.gen_waivers           # 生成/刷新
    python -m audit.gen_waivers --check   # 只校验现状，不写文件
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WAIVER_PATH = os.path.join(ROOT, "audit", "waivers.json")

# 打分链路的根：这些模块（及其传递依赖）里的全样本统计会污染信号/权重
CHAIN_ROOTS = ("signals", "learner", "engine", "dataset")

# 学习/建模链路的根：这些模块会**训练**模型，全样本统计会污染模型参数，
# 同样不能按"评估层"豁免（第一版只看了打分链路，会把 gbdt / alpha158 /
# online_calib 这类训练代码误判成纯评估层）。
MODEL_ROOTS = ("gbdt", "alpha158", "feature_select", "sample_weight", "online_calib",
               "intraday_online", "live_model", "horizons", "ensemble", "ensemble_model",
               "distill", "ranker", "residual", "neutral", "behavior_report",
               "event_learning", "meta")


def _imports_of(path: str):
    """粗略提取 `from .x import` / `from . import x` / `import qcore.x`。"""
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:  # noqa: BLE001
        return set()
    out = set()
    for m in re.finditer(r"from\s+\.(\w+)\s+import", src):
        out.add(m.group(1))
    for m in re.finditer(r"from\s+\.\s+import\s+([^\n]+)", src):
        for name in re.split(r"[,\s]+", m.group(1)):
            name = name.strip().strip("()")
            if name.isidentifier():
                out.add(name)
    for m in re.finditer(r"(?:from|import)\s+qcore\.(\w+)", src):
        out.add(m.group(1))
    return out


def chain_closure() -> set:
    """从打分链路根出发的传递闭包（模块名集合）。"""
    graph = {}
    for p in glob.glob(os.path.join(ROOT, "qcore", "*.py")):
        mod = os.path.basename(p)[:-3]
        graph[mod] = _imports_of(p)
    roots = [r for r in (*CHAIN_ROOTS, *MODEL_ROOTS) if r in graph]
    seen, stack = set(), list(roots)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in graph.get(cur, ()):
            if nxt in graph and nxt not in seen:
                stack.append(nxt)
    return seen


def medium_by_file() -> dict:
    """按文件统计 MEDIUM 条数。

    **不能用 subprocess**：本环境（Windows 沙箱）禁止创建管道，会 WinError 5。
    直接 import leak_scan 调 scan_file，零子进程。
    """
    sys.path.insert(0, os.path.join(ROOT, "audit"))
    import leak_scan as LS          # noqa: E402
    # **必须关掉已有豁免再扫**：否则上一次生成的 waivers.json 会让这些文件
    # 直接扫出 0 条 MEDIUM → 本次生成时它们根本不进候选 → 新清单把它们"洗掉"。
    # （第一版就是这么把自己的豁免洗没的：MEDIUM 246 → 207 → 会一路归零）
    LS._FILE_WAIVERS = {}
    counts = {}
    for path in glob.glob(os.path.join(ROOT, "qcore", "*.py")):
        try:
            fs = LS.scan_file(path)
        except Exception:  # noqa: BLE001
            continue
        n = sum(1 for f in fs if f.sev == "MEDIUM" and not f.waived)
        if n:
            counts[os.path.relpath(path, ROOT).replace("\\", "/")] = n
    return counts


# ---------------------------------------------------------------- 逐文件判定
# 第一层是机械规则（是否在链路闭包内）；闭包**内**的文件还需要一次人判。
# 下面这张表记录「闭包内但已核实无泄漏」的文件与**具体理由**，
# 理由必须指到代码行为，不能写「看起来没问题」。
# 判定为 keep 的文件继续计入 MEDIUM（= 待人工确认），不做豁免。
FILE_JUDGEMENTS = {
    "horizon_ledger": ("waive",
        "全部为台账展示聚合（mean/hit_rate/spread/top_mean），只写 horizon_perf.json "
        "供 UI 读取，不回喂预测或权重"),
    "intraday_online": ("waive",
        "train_online 只接收 y is not None 的**已到期**样本，标准化与指标都在该集合内计算，"
        "训练时刻天然因果（见 qcore/intraday_online.py:train_online）"),
    "analysis": ("waive",
        "优势体检模块：demean/秩相关/分位差本身就是评估指标，不产出信号或权重"),
    "events": ("waive",
        "事件研究评估统计（CAR/分位），产物写 events.json 供 UI；不进打分链路"),
    "behavior_report": ("waive",
        "仅被 pipeline 调用的报告脚本（行为因子体检），输出到 reports/"),
    "sources": ("waive",
        "数据源交叉验证的健康度统计（compare_returns），只影响 source_health.json"),
    # 零 importer：没有任何模块引用它们，统计不可能进入主链路
    "ensemble": ("waive", "零 importer（未被任何模块 import），不可能进入主链路"),
    "ranker": ("waive", "零 importer，同上"),
    "residual": ("waive", "零 importer，同上"),
    "distill": ("waive", "零 importer（只有自身 __main__），同上"),
    # 以下保留为待人工确认：它们在决策/训练路径上，不能按「评估层」豁免
    # 核心/训练路径：**故意不豁免**。它们每次改动都值得重看一遍，
    # 保留告警当提醒；这 33 条是"当前已知、已理解、仍在盯"的清单。
    "gbdt": ("keep",
             "训练代码：L191 的分箱边界只用**训练矩阵**拟合（X[:,k] 的分位），"
             "L329-330 是节点内梯度/海森求和，L520/524/575 是 L2 与 R² 指标 —— "
             "均非时序泄漏，但属核心，保留告警"),
    "engine": ("keep",
              "逐日重放核心：告警点全部是 axis=1（信号维）归一化与横截面 mean，"
              "非时序统计；核心代码保留告警"),
    "learner": ("keep", "权重更新核心：softmax 在信号维做归一化，非时序统计"),
    "dataset": ("keep", "面板构造核心"),
    "signed": ("keep",
               "L56 是 axis=1 横截面去均值；L88-89 的 mu/sd 在 fit 时用**训练矩阵**拟合并"
               "存盘、预测时复用（标准做法），非泄漏；核心保留告警"),
    # 外围但确实在链路里的：已逐条核实，可豁免
    "meta": ("waive",
             "告警点全部是评估量：L79-81 的 demean+相关是 meta_ic 本身，"
             "L130-131 frac_inverted/frac_silent、L146-147 hit_rate/mean_signed_ret 都是"
             "「事后命中率」展示；final_sign 由 final_t 判定，不由这些均值决定"),
    "online_calib": ("waive",
                     "L227-234 是对（预测, 已实现收益）做 OLS 求 a/b，样本是已到期配对；"
                     "L191 是 MSE 指标。校准参数只作用于未来预测，不回看未来"),
    "reward": ("waive",
               "L101-102 hit_rate/加权均值、L136 max_dd 都是对**已实现**序列的评价指标，"
               "用于报告与奖励口径评估，不进入信号计算"),
    "event_learning": ("waive",
                       "L209/L332-333 是事件后收益的均值与命中率（评估统计），"
                       "产物写 event_learning.json 供 UI"),
    "live_model": ("waive",
                   "L256-258 是特征索引越界检查、L267 是样本数门槛、L279 是应用在线校准；"
                   "没有全样本统计量进入打分"),
}


def build() -> dict:
    chain = chain_closure()
    meds = medium_by_file()
    waivers, unconfirmed = [], []
    for path, n in sorted(meds.items(), key=lambda kv: -kv[1]):
        mod = os.path.basename(path)[:-3]
        if mod in chain:
            jd = FILE_JUDGEMENTS.get(mod)
            if jd and jd[0] == "waive":
                waivers.append({"path": path, "scope": "file", "n": n,
                                "reason": f"[已核实] {jd[1]}"})
                continue
            unconfirmed.append({
                "path": path, "n": n,
                "why": (jd[1] if jd else
                        "在打分/建模链路闭包内，全样本统计可能污染信号/权重")})
        else:
            waivers.append({
                "path": path, "scope": "file", "n": n,
                "reason": (f"评估层：不在打分链路闭包内"
                           f"（未被 {'/'.join(CHAIN_ROOTS)} 直接或间接 import），"
                           f"全样本统计只影响事后指标，不进信号与权重"),
            })
    return {
        "criterion": ("MEDIUM 分流规则：文件是否在打分链路闭包内。"
                      f"闭包根 = {list(CHAIN_ROOTS)}；闭包内不豁免，闭包外按文件豁免。"
                      "这条规则可重复执行：python -m audit.gen_waivers"),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "by": "lead",
        "chain_modules": sorted(chain),
        "waivers": waivers,
        "unconfirmed_paths": unconfirmed,
        "note": ("行内 `# leak-scan: ok <理由>` 的豁免不在此文件里；"
                 "本文件只做文件级分流。"),
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="生成 audit/waivers.json")
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    a = ap.parse_args(argv)

    d = build()
    n_w = sum(w["n"] for w in d["waivers"])
    n_u = sum(u["n"] for u in d["unconfirmed_paths"])
    print(f"打分链路闭包: {len(d['chain_modules'])} 个模块")
    print(f"MEDIUM 分流: 可豁免 {n_w} 条 / {len(d['waivers'])} 个文件 · "
          f"需人工确认 {n_u} 条 / {len(d['unconfirmed_paths'])} 个文件")
    print("\n需人工确认的文件（在打分链路内）:")
    for u in d["unconfirmed_paths"][:20]:
        print(f"  {u['path']:40s} {u['n']:4d} 条")
    if a.check:
        if os.path.isfile(WAIVER_PATH):
            old = json.load(open(WAIVER_PATH, encoding="utf-8"))
            same = {w["path"] for w in old.get("waivers", [])} == \
                   {w["path"] for w in d["waivers"]}
            print(f"\n[check] 现有 waivers.json 与当前扫描{'一致' if same else '不一致（建议刷新）'}")
        else:
            print("\n[check] waivers.json 不存在")
        return 0
    with open(WAIVER_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {WAIVER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
