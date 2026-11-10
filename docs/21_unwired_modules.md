# 21 · 未接线模块清单（2026-11-10 实测）

> 生成方式：对 `qcore/` 全部 120 个模块做引用扫描 —— 分类判据是
> **① 被其它 .py import** ／ **② 有 `__main__` 或被 bat/sh/yml/md 引用（CLI 入口）** ／
> **③ 两者皆无**。第三类才是「死代码候选」，前两类只是「不在主链路」。

## 汇总

| 类别 | 模块数 | 行数 | 含义 |
|---|---|---|---|
| 被 import | 91 | 32,643 | 主链路，改它们要跑全套回归 |
| CLI 入口 | 10 | 3,016 | 有 `__main__` 或被脚本调用，可手动跑，不进自动流程 |
| **零引用且无入口** | **19** | **4,033** | **死代码候选，需要「接/删/归档」决策** |

## 一、CLI 入口（不算死代码，但也不在自动流程里）

`alert(641)` `pipeline(414)` `claims(358)` `attribution(332)` `brief(277)`
`update_apply(241)` `model_bootstrap(230)` `crossmarket(195)` `live(193)` `probe_sources(135)`

## 二、零引用且无入口 —— 需要决策

| 模块 | 行数 | 与 AGENTS 的关系 | 建议 |
|---|---|---|---|
| `account.py` | 369 | 未登记 | 与 `account_config.py` 职责重叠，二选一 |
| `advanced.py` | 312 | 未登记 | 归档 |
| `bt_rules.py` | 305 | 未登记 | 归档（回测规则已被 `backtest.py` 覆盖） |
| `sentiment.py` | 247 | 未登记 | 归档（情绪数据源不可回测） |
| `position.py` | 246 | 未登记 | 与 `risk.py` 重叠，判重后删 |
| `stress.py` | 235 | 未登记 | 归档（压力测试未接入报告） |
| `altdata.py` | 216 | 未登记 | 归档（另类数据无长历史） |
| `retail_index.py` | 203 | 未登记 | 归档 |
| `optimizer.py` | 200 | 未登记 | 归档（`risk.py` 已有配权） |
| `ensemble.py` | 199 | 未登记 | **删**（AI Engineer 实测「分 regime 更差」） |
| `tactics.py` | 198 | 未登记 | 归档 |
| `model.py` | 190 | 未登记 | 与 `live_model.py` / `versioning.py` 重叠，判重后删 |
| **`forecast.py`** | **188** | **§2 表里写成「前景与空间测算（Grinold）」核心模块** | **文档在撒谎**：零引用。要么接线，要么从 §2 删除 |
| `ranker.py` | 186 | 未登记 | 归档 |
| `multihorizon.py` | 164 | 未登记 | 与 `horizons.py` 重叠，判重后删 |
| `survival.py` | 160 | 未登记 | 归档 |
| `residual.py` | 152 | 未登记 | 归档（中性化在 `behavior_report` 里另有一份） |
| `drift.py` | 140 | 未登记 | 与 `learner` 的 PH 漂移检测重叠 |
| `combo.py` | 123 | 未登记 | **含全库最高 IC 声称 +0.0352(t=+5.14)，无调用方、未登记 claims** |

## 三、这份清单不能当「删掉就行」

三点必须先说清：

1. **零引用 ≠ 没价值**。`combo.py` 的 +0.0352 若真成立，它就是全项目最强的线索；
   问题不是它没被调用，而是**它的结论从未过四件套门禁**。处置顺序是
   「先复算，再决定接/删」，不是「没人用就删」。
2. **`forecast.py` 是文档缺陷，不是代码缺陷**。§2 把它写成核心模块，实际零引用。
   这类漂移比死代码更危险 —— 读者会以为 Grinold 基本法则在跑。
3. **删之前先确认没有外部调用**：`deploy/`、`docker/`、用户自己的 bat/ps1
   都可能 `python -m qcore.xxx`。本清单只扫了仓库内文件。

## 四、复现命令

```bash
# 逐模块引用扫描（无依赖，纯文本）
python - <<'EOF'
import glob, os, re
mods = sorted(os.path.basename(p)[:-3] for p in glob.glob("qcore/*.py") if not p.endswith("__init__.py"))
py = [p for pat in ("qcore/*.py","webui/*.py","scripts/*.py","tools/*.py","tests/*.py","audit/*.py") for p in glob.glob(pat)]
txt = {p: open(p, encoding="utf-8", errors="ignore").read() for p in py}
for m in mods:
    own = f"qcore/{m}.py"
    pat = re.compile(rf"(from\s+\.{m}\s+import|from\s+qcore\.{m}\s+import|import\s+qcore\.{m}\b)")
    if not any(p != own and pat.search(t) for p, t in txt.items()):
        print(m, len(open(own, encoding="utf-8", errors="ignore").read().splitlines()))
EOF
```
