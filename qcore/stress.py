"""
压力测试 · 收益归因 · 拥挤度 —— 三个"没人问但会咬人"的东西。

## 一、压力测试：分段看，而不是看总收益

总收益 -28% 是个没信息量的数字。真正该问的是：
**它在 2015 股灾、2018 熊市、2020 疫情、2024 微盘踩踏里各亏了多少？**
如果某个策略在 4 次危机里都亏，那它不是"暂时回撤"，是"结构性脆弱"。

## 二、Brinson 归因：钱是行业给的，还是选股给的

    R_p - R_b = Σ (w_p,i - w_b,i)·R_b,i       ← 配置效应（行业超配/低配）
              + Σ w_b,i·(R_p,i - R_b,i)       ← 选择效应（同行业内选得准不准）
              + Σ (w_p,i - w_b,i)(R_p,i - R_b,i)  ← 交互

**大多数"看起来很厉害"的策略，收益全在配置效应里** —— 也就是押对了行业，
而不是选对了股。押行业靠 beta，选股才是 alpha。分不清这两者，就会把
"我运气好赶上了新能源"误当成"我的模型很准"。

## 三、拥挤度：所有人都在用同一个信号时，它会失效

信号拥挤的表现：**大量信号同时指向同一方向**。
这时候不是"共识很强所以更可信"，而是"边际买盘已经用完"。
实测（本项目）：条件相关 p90 = 0.95，有效独立信号仅 12/73 —— 已经是拥挤状态。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ 压力测试
# 已知的 A 股极端区间（人工标注，来源为公开市场事实）
SCENARIOS: List[Tuple[str, str, str, str]] = [
    ("2015 股灾", "2015-06-12", "2015-09-30", "去杠杆 + 千股跌停，两融强平"),
    ("2016 熔断", "2016-01-04", "2016-02-29", "熔断机制四天两次触发后暂停"),
    ("2018 熊市", "2018-01-24", "2018-12-28", "贸易摩擦 + 去杠杆，全年单边下跌"),
    ("2020 疫情", "2020-01-23", "2020-03-23", "新冠爆发，节后首日 -8%"),
    ("2021 抱团瓦解", "2021-02-18", "2021-03-31", "核心资产杀估值"),
    ("2022 俄乌+封控", "2022-02-24", "2022-04-26", "地缘 + 疫情，沪指破 3000"),
    ("2024 微盘踩踏", "2024-01-02", "2024-02-07", "小市值流动性危机，量化集体回撤"),
    ("2025 关税冲击", "2025-04-01", "2025-04-30", "外部关税预期扰动"),
]


def scenario_table(scores: np.ndarray, RET: np.ndarray, dates: np.ndarray,
                   h: int = 5, n_groups: int = 5,
                   scenarios: Optional[Sequence[Tuple]] = None) -> pd.DataFrame:
    """
    在每个历史极端区间内，算多空组合（非重叠 h 日）与全市场等权的表现。
    scores (N,T)；RET (T,N)。
    """
    sc = np.asarray(scores, dtype=float)
    T, N = RET.shape
    d = pd.DatetimeIndex(dates)
    rows = []
    for name, t0, t1, note in (scenarios or SCENARIOS):
        m = (d >= pd.Timestamp(t0)) & (d <= pd.Timestamp(t1))
        idx = np.where(m)[0]
        if idx.size < h + 2:
            rows.append({"scenario": name, "period": f"{t0}~{t1}", "ok": False,
                         "reason": f"区间内仅 {idx.size} 个交易日"})
            continue
        ls, ew = [], []
        for t in idx[:-1 - h:h]:
            s = sc[:, t]
            y = RET[t]
            v = np.isfinite(s) & np.isfinite(y)
            if v.sum() < n_groups * 2:
                continue
            ii = np.where(v)[0]
            order = ii[np.argsort(s[ii])]
            g = len(order) // n_groups
            ls.append(float(y[order[-g:]].mean() - y[order[:g]].mean()))
            ew.append(float(y[v].mean()))
        if not ls:
            rows.append({"scenario": name, "period": f"{t0}~{t1}", "ok": False,
                         "reason": "无有效样本"})
            continue
        rows.append({
            "scenario": name, "period": f"{t0}~{t1}", "ok": True,
            "n_periods": len(ls),
            "long_short": float(np.mean(ls)), "ls_cum": float(np.prod(1 + np.array(ls)) - 1),
            "market_ew": float(np.mean(ew)), "ew_cum": float(np.prod(1 + np.array(ew)) - 1),
            "note": note,
        })
    return pd.DataFrame(rows)


def stress_verdict(tbl: pd.DataFrame) -> Dict:
    """给压力测试一个结论：是"暂时回撤"还是"结构性脆弱"。"""
    ok = tbl[tbl["ok"] == True] if "ok" in tbl else pd.DataFrame()
    if len(ok) == 0:
        return {"ok": False, "reason": "无有效区间"}
    neg = int((ok["long_short"] < 0).sum())
    worst = ok.loc[ok["long_short"].idxmin()]
    return {
        "ok": True, "n_scenarios": int(len(ok)),
        "n_negative": neg,
        "worst_scenario": worst["scenario"], "worst_ls": float(worst["long_short"]),
        "verdict": (f"{neg}/{len(ok)} 个危机区间亏损 → 结构性脆弱"
                    if neg >= len(ok) * 0.6 else
                    f"{neg}/{len(ok)} 个危机区间亏损 → 具备一定抗压性"),
    }


# ------------------------------------------------------------------ Brinson 归因
def brinson(scores: np.ndarray, RET: np.ndarray, sectors: Sequence[str],
            h: int = 5, top_k: int = 10) -> Dict:
    """
    简化 Brinson：把组合超额拆成"行业配置"与"行业内选股"。
    scores (N,T)；RET (T,N)；sectors (N,)。
    """
    sc = np.asarray(scores, dtype=float)
    T, N = RET.shape
    sec = np.array(sectors, dtype=object)
    uniq = sorted(set(sec.tolist()))

    alloc, select, inter, active = [], [], [], []
    for t in range(T - 1 - h):
        s = sc[:, t]
        y = RET[t]
        v = np.isfinite(s) & np.isfinite(y)
        if v.sum() < top_k * 2:
            continue
        ii = np.where(v)[0]
        pick = ii[np.argsort(-s[ii])][:top_k]
        w_p = np.zeros(N); w_p[pick] = 1.0 / len(pick)
        w_b = v.astype(float) / v.sum()          # 基准 = 全池等权

        a = se = it = 0.0
        for k in uniq:
            m = (sec == k) & v
            if m.sum() == 0:
                continue
            wp, wb = w_p[m].sum(), w_b[m].sum()
            rp = float((w_p[m] * y[m]).sum() / wp) if wp > 0 else 0.0
            rb = float(y[m].mean())
            a += (wp - wb) * rb
            se += wb * (rp - rb)
            it += (wp - wb) * (rp - rb)
        alloc.append(a); select.append(se); inter.append(it)
        yy = np.nan_to_num(y, nan=0.0)
        active.append(float((w_p * yy).sum() - (w_b * yy).sum()))

    if not active:
        return {"ok": False, "reason": "样本不足"}
    a, s_, i_ = np.mean(alloc), np.mean(select), np.mean(inter)
    tot = a + s_ + i_
    return {
        "ok": True, "n_periods": len(active),
        "total_excess": float(np.mean(active)),
        "allocation": float(a), "selection": float(s_), "interaction": float(i_),
        "alloc_share": float(a / tot) if abs(tot) > 1e-12 else float("nan"),
        "select_share": float(s_ / tot) if abs(tot) > 1e-12 else float("nan"),
        "verdict": ("**选股效应为负** → 模型在同行业内选错股，收益全靠行业配置/交互（beta）"
                    if s_ < 0 else
                    ("收益主要来自**行业配置**（beta 押注），不是选股能力"
                     if abs(a) > abs(s_) else
                     "收益主要来自**行业内选股**（更接近真 alpha）")),
    }


# ------------------------------------------------------------------ 拥挤度
def crowding(scores: np.ndarray, DIR: np.ndarray, warmup: int = 120) -> Dict:
    """
    信号拥挤度：当天有多少信号指向同一方向。
    DIR (T,N,S) int8。返回每日拥挤度序列与当前值。
    """
    T, N, S = DIR.shape
    conc = np.full(T, np.nan)
    for t in range(T):
        d = DIR[t].astype(np.float32)
        act = d != 0
        n_act = act.sum(axis=1)                     # (N,) 每股发声数
        if not (n_act > 0).any():
            continue
        # 每股"净方向一致性" = |sum(dir)| / 发声数
        net = np.abs(d.sum(axis=1))
        cons = np.where(n_act > 0, net / np.maximum(n_act, 1), np.nan)
        conc[t] = float(np.nanmean(cons))
    cur = float(conc[-1]) if np.isfinite(conc[-1]) else float("nan")
    hist = conc[warmup:]
    pct = float(pd.Series(hist).rank(pct=True).iloc[-1]) if np.isfinite(cur) and len(hist) > 20 else float("nan")
    return {
        "ok": bool(np.isfinite(cur)),
        "current": cur, "percentile": pct,
        "mean": float(np.nanmean(hist)) if len(hist) else float("nan"),
        "level": ("极高（拥挤）" if pct > 0.8 else "偏高" if pct > 0.6 else
                  "中性" if pct > 0.4 else "偏低" if pct > 0.2 else "极低（分歧大）"),
        "verdict": ("信号高度同向 → 边际买盘可能耗尽，降低仓位"
                    if pct > 0.8 else
                    "信号分歧大 → 方向不明，减少下注"
                    if pct < 0.2 else "拥挤度正常"),
    }


def format_report(stress_tbl: pd.DataFrame, stress_v: Dict,
                  attr: Dict, crowd: Dict) -> str:
    L = ["## 压力测试 · 归因 · 拥挤度", ""]
    L.append("### 一、极端情景表现")
    L.append("| 情景 | 区间 | 多空累计 | 市场等权累计 | 期数 |")
    L.append("|---|---|---|---|---|")
    for r in stress_tbl.itertuples():
        if not getattr(r, "ok", False):
            L.append(f"| {r.scenario} | {r.period} | — | — | {getattr(r,'reason','')} |")
            continue
        L.append(f"| {r.scenario} | {r.period} | {r.ls_cum:+.1%} | {r.ew_cum:+.1%} | {r.n_periods} |")
    L.append("")
    if stress_v.get("ok"):
        L.append(f"**{stress_v['verdict']}**；最差：{stress_v['worst_scenario']} "
                 f"({stress_v['worst_ls']:+.2%})")
    L.append("")
    L.append("### 二、收益归因（Brinson）")
    if attr.get("ok"):
        L.append(f"- 总超额（每期）：{attr['total_excess']:+.4%}")
        L.append(f"- 行业配置：{attr['allocation']:+.4%}（占 {attr['alloc_share']:.0%}）")
        L.append(f"- 行业内选股：{attr['selection']:+.4%}（占 {attr['select_share']:.0%}）")
        L.append(f"- 交互：{attr['interaction']:+.4%}")
        L.append("")
        L.append(f"**{attr['verdict']}**")
    else:
        L.append("样本不足。")
    L.append("")
    L.append("### 三、信号拥挤度")
    if crowd.get("ok"):
        L.append(f"- 当前 {crowd['current']:.2%}（历史分位 {crowd['percentile']:.0%}）→ "
                 f"**{crowd['level']}**")
        L.append(f"- {crowd['verdict']}")
    else:
        L.append("无数据。")
    return "\n".join(L)
