"""
数据清洗 —— 把脏数据挡在模型外面，并且**说清楚挡掉了什么**。

## 为什么必须有

本项目已经踩过的数据坑（AGENTS.md §7 里记了 10 条，其中 4 条是数据问题）：

| 坑 | 后果 |
|---|---|
| 前复权减法产生负价 | 收益率 ±10000% |
| 停牌复牌跳变 | 单日 +394%/+622% 被 clip 成满格喂进权重 |
| 东财 clist 单页上限 100 | 股票池退化成超级大盘股 |
| NaN 写进 JSON | 浏览器 `JSON.parse` 整页崩 |

而 Qlib 社区数据是**多源合并**（Wind/Caihui/Tushare/AKShare），
复权口径混用的风险更高——`AGENTS.md` INV-5 要求每次扩池子都跑数据守卫，
之前一直没做。

## 九项检查

1. **价格非正**：`close <= 0` —— 复权错误的直接证据
2. **极端跳变**：`|日收益| > 50%` —— 除权未复权、拆股、数据错位
3. **停牌零量**：`volume == 0` 或连续多日价格完全不变
4. **重复日期**：同一天多条记录
5. **时间穿越**：日期晚于今天
6. **复权因子异常**：`factor` 非正、或单日跳变 > 20%
7. **缺失过长**：连续缺失 > 20 个交易日
8. **异常量能**：`volume > 20 × 60日中位数`
9. **高低价矛盾**：`high < low`、`close` 不在 `[low, high]` 内

## 处理原则

- **可修复的修**（去重、排序、剔除非法 bar）
- **不可修复的标记**（整只票剔除，并说明原因）
- **永远不猜**（不用插值填补价格，宁可不交易）
- 每只票出一份体检报告，汇总成 `data/state/data_quality.json`

## 用法

```python
from qcore.cleaning import audit, clean_panel, summary
audit("sh600519")          # 单只体检
clean_panel(codes)         # 批量清洗，返回 {code: df} + 报告
summary()                  # 最近一次清洗的汇总
```
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
REPORT = os.path.join(STATE, "data_quality.json")

# 阈值（集中在这里，改阈值不改逻辑）
MAX_ABS_RET = 0.50        # 单日 |收益| 上限
MAX_FACTOR_JUMP = 0.20    # 复权因子单日跳变上限
MAX_GAP_DAYS = 20         # 连续缺失上限（交易日）
VOL_OUTLIER = 20.0        # 量能异常倍数
STALE_PRICE_RUN = 10      # 连续价格不变天数（疑似停牌）


def _max_run(mask: np.ndarray) -> int:
    """最长连续 True 段长度。"""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(best)


def _q(code: str) -> str:
    c = str(code).lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c
    c6 = c.zfill(6)
    if c6[0] in ("6", "9"):
        return "sh" + c6
    if c6[:2] in ("43", "83", "87", "92"):
        return "bj" + c6
    return "sz" + c6


def audit_one(df: pd.DataFrame, code: str = "") -> Dict:
    """
    单只股票体检。返回 {code, ok, issues:[{kind,severity,n,detail}], clean_rows}
    不改数据，只诊断。
    """
    issues: List[Dict] = []
    if df is None or len(df) == 0:
        return {"code": code, "ok": False,
                "issues": [{"kind": "empty", "severity": "fatal", "n": 0,
                            "detail": "无数据"}], "rows": 0}

    d = df.copy()
    n0 = len(d)
    need = {"date", "open", "high", "low", "close"}
    miss = need - set(d.columns)
    if miss:
        return {"code": code, "ok": False,
                "issues": [{"kind": "missing_columns", "severity": "fatal",
                            "n": len(miss), "detail": f"缺列 {sorted(miss)}"}],
                "rows": n0}

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    bad_date = int(d["date"].isna().sum())
    if bad_date:
        issues.append({"kind": "bad_date", "severity": "high", "n": bad_date,
                       "detail": "日期无法解析"})

    # 1) 价格非正
    for col in ("open", "high", "low", "close"):
        v = pd.to_numeric(d[col], errors="coerce")
        bad = int((v <= 0).sum())
        if bad:
            issues.append({"kind": f"nonpositive_{col}", "severity": "high", "n": bad,
                           "detail": f"{col} <= 0 的行（复权错误的典型症状）"})

    # 2) 高低价矛盾
    h = pd.to_numeric(d["high"], errors="coerce")
    lo = pd.to_numeric(d["low"], errors="coerce")
    c = pd.to_numeric(d["close"], errors="coerce")
    hl = int((h < lo).sum())
    if hl:
        issues.append({"kind": "high_lt_low", "severity": "high", "n": hl,
                       "detail": "high < low"})
    out_of_range = int(((c > h + 1e-9) | (c < lo - 1e-9)).sum())
    if out_of_range:
        issues.append({"kind": "close_out_of_range", "severity": "mid",
                       "n": out_of_range, "detail": "close 不在 [low, high] 内"})

    # 3) 重复日期
    dup = int(d["date"].duplicated().sum())
    if dup:
        issues.append({"kind": "duplicate_date", "severity": "mid", "n": dup,
                       "detail": "同一天多条记录"})

    # 4) 时间穿越
    future = int((d["date"] > pd.Timestamp(datetime.now().date())).sum())
    if future:
        issues.append({"kind": "future_date", "severity": "fatal", "n": future,
                       "detail": "日期晚于今天"})

    # 5) 极端跳变
    cs = c.where(c > 0)
    ret = cs.pct_change()
    jump = int((ret.abs() > MAX_ABS_RET).sum())
    if jump:
        worst = float(ret.abs().max())
        issues.append({"kind": "extreme_return", "severity": "high", "n": jump,
                       "detail": f"|日收益| > {MAX_ABS_RET:.0%} 共 {jump} 次，最大 {worst:.1%}"})

    # 6) 复权因子
    if "factor" in d.columns:
        f = pd.to_numeric(d["factor"], errors="coerce")
        badf = int((f <= 0).sum())
        if badf:
            issues.append({"kind": "nonpositive_factor", "severity": "high", "n": badf,
                           "detail": "factor <= 0"})
        fj = int((f.pct_change().abs() > MAX_FACTOR_JUMP).sum())
        if fj:
            issues.append({"kind": "factor_jump", "severity": "mid", "n": fj,
                           "detail": f"factor 单日跳变 > {MAX_FACTOR_JUMP:.0%}"})

    # 7) 停牌零量 / 价格不动
    if "volume" in d.columns:
        v = pd.to_numeric(d["volume"], errors="coerce")
        zero = int((v <= 0).sum())
        if zero:
            issues.append({"kind": "zero_volume", "severity": "mid", "n": zero,
                           "detail": "成交量为 0（停牌）"})
        med = float(v[v > 0].median()) if (v > 0).any() else 0.0
        if med > 0:
            out = int((v > med * VOL_OUTLIER).sum())
            if out:
                issues.append({"kind": "volume_outlier", "severity": "low", "n": out,
                               "detail": f"成交量 > {VOL_OUTLIER:.0f}×中位数"})

    # 最长连续「价格不变」段（不是总次数——总次数会把正常的小幅重复也算进去）
    _unchanged = (cs.diff() == 0).to_numpy()
    _run = _max_run(_unchanged)
    if _run > STALE_PRICE_RUN:
        issues.append({"kind": "stale_price", "severity": "mid", "n": _run,
                       "detail": f"最长连续价格不变 {_run} 天 > {STALE_PRICE_RUN}（疑似停牌）"})

    # 8) 缺失过长
    if "date" in d.columns and d["date"].notna().any():
        d2 = d.dropna(subset=["date"]).sort_values("date")
        if len(d2) > 1:
            gaps = d2["date"].diff().dt.days.fillna(0)
            big = int((gaps > MAX_GAP_DAYS * 1.5).sum())
            if big:
                issues.append({"kind": "long_gap", "severity": "low", "n": big,
                               "detail": f"相邻记录间隔 > {int(MAX_GAP_DAYS*1.5)} 自然日"})

    fatal = [i for i in issues if i["severity"] == "fatal"]
    high = [i for i in issues if i["severity"] == "high"]
    return {"code": code, "ok": not fatal, "rows": n0,
            "n_issues": len(issues), "n_fatal": len(fatal), "n_high": len(high),
            "issues": issues,
            "verdict": ("不可用" if fatal else
                        "需清洗" if high else "有瑕疵但可用" if issues else "干净")}


def clean_one(df: pd.DataFrame, code: str = "") -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    清洗单只。返回 (清洗后 df 或 None, 报告)。
    原则：能修的修（去重/排序/剔除非法 bar），不能修的整只丢弃，绝不插值。
    """
    rep = audit_one(df, code)
    if not rep["ok"]:
        return None, {**rep, "action": "dropped"}
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    n0 = len(d)
    d = d.dropna(subset=["date"])
    d = d.drop_duplicates("date").sort_values("date")
    for col in ("open", "high", "low", "close"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d["close"] > 0) & (d["open"] > 0) & (d["high"] > 0) & (d["low"] > 0)]
    d = d[(d["high"] >= d["low"])]
    d = d[(d["close"] <= d["high"] * 1.001) & (d["close"] >= d["low"] * 0.999)]
    d = d.reset_index(drop=True)
    removed = n0 - len(d)
    return (d if len(d) else None), {**rep, "action": "cleaned",
                                     "rows_in": n0, "rows_out": len(d),
                                     "removed": removed}


def clean_panel(codes: List[str], loader=None, write_report: bool = True,
                verbose: bool = True) -> Dict:
    """批量清洗。返回 {"panel": {code: df}, "report": {...}}。"""
    if loader is None:
        from . import qlib_data as QD
        loader = QD.load_stock

    panel: Dict[str, pd.DataFrame] = {}
    reports: List[Dict] = []
    t0 = datetime.now()
    for i, c in enumerate(codes, 1):
        try:
            df = loader(c)
        except Exception as e:  # noqa: BLE001
            reports.append({"code": c, "ok": False, "verdict": "加载失败",
                            "issues": [{"kind": "load_error", "severity": "fatal",
                                        "n": 0, "detail": str(e)[:60]}]})
            continue
        out, rep = clean_one(df, c)
        reports.append(rep)
        if out is not None:
            panel[c] = out
        if verbose and i % 100 == 0:
            print(f"  cleaned {i}/{len(codes)}", flush=True)

    ok = [r for r in reports if r.get("ok")]
    dropped = [r for r in reports if not r.get("ok")]
    kinds: Dict[str, int] = {}
    for r in reports:
        for it in r.get("issues", []):
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1

    summary_doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_s": round((datetime.now() - t0).total_seconds(), 1),
        "n_input": len(codes), "n_ok": len(ok), "n_dropped": len(dropped),
        "n_clean": sum(1 for r in reports if r.get("verdict") == "干净"),
        "rows_in": sum(r.get("rows_in", r.get("rows", 0)) for r in reports),
        "rows_out": sum(r.get("rows_out", r.get("rows", 0)) for r in reports),
        "issue_kinds": dict(sorted(kinds.items(), key=lambda x: -x[1])),
        "dropped": [{"code": r["code"], "why": r.get("issues", [{}])[0].get("detail", "")}
                    for r in dropped[:20]],
        "worst": sorted([r for r in reports if r.get("n_issues")],
                        key=lambda r: -r.get("n_issues", 0))[:10],
    }
    if write_report:
        os.makedirs(STATE, exist_ok=True)
        tmp = REPORT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary_doc, f, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, REPORT)
    return {"panel": panel, "report": summary_doc}


def summary() -> Dict:
    try:
        with open(REPORT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"ok": False, "reason": "no_report",
                "hint": "跑一次 clean_panel 生成 data_quality.json"}


if __name__ == "__main__":
    import json
    from qcore import qlib_data as QD

    print("=" * 70)
    print("数据清洗体检 —— Qlib 社区数据（多源合并，复权口径风险高）")
    print("=" * 70)
    codes = QD.universe_codes("csi300", start="2021-01-01")[:120]
    r = clean_panel(codes, verbose=False)
    rep = r["report"]
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("dropped", "worst")}, ensure_ascii=False, indent=1))
    print("\n问题类型分布:", rep["issue_kinds"])
    if rep["dropped"]:
        print("\n被整只剔除（前 5）:")
        for d in rep["dropped"][:5]:
            print(f"  {d['code']}: {d['why']}")
    print("\n问题最多的 5 只:")
    for w in rep["worst"][:5]:
        print(f"  {w['code']}: {w['verdict']} ({w['n_issues']} 项) "
              f"{[i['kind'] for i in w['issues'][:3]]}")
