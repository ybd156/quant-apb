"""
报告渲染层（report）—— 纯 Markdown，无 matplotlib / 无外部模板依赖。

用法
----
    from qcore.metrics import evaluate_all
    from qcore.backtest import run_long_short, run_baselines
    from qcore.report import render

    metrics = evaluate_all(res, horizons=(1, 5, 20), n_groups=5)
    bt = {'long_short': run_long_short(res, 5, 5, 10),
          'baselines': run_baselines(res, res_baseline, 5, 10)}
    render(res, metrics, bt, 'reports/report.md')

产出为 UTF-8 中文 Markdown，表格全部用管道表格，便于直接贴进 README 或文档。
「结论 / 局限」两节留占位段落，由 Lead 按实际结果填写。
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

__all__ = ["render", "render_string", "md_table"]

DASH = "—"

# 基线表列名 -> 带单位的展示列名（内部列名保持简洁，便于程序消费）
_BASE_UNITS = {
    "年化收益": "年化收益%",
    "年化波动": "年化波动%",
    "最大回撤": "最大回撤%",
    "胜率": "胜率%",
    "平均换手": "平均换手(倍/期)",
    "累计成本bps": "累计成本(bps)",
}


# --------------------------------------------------------------------------- #
# 格式化工具
# --------------------------------------------------------------------------- #
def _f(x, nd: int = 3, pct: bool = False, signed: bool = False) -> str:
    """安全格式化：NaN/None/非数 -> 破折号。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return DASH
    if not np.isfinite(v):
        return DASH
    if pct:
        v = v * 100.0
        return f"{v:+.{nd}f}%" if signed else f"{v:.{nd}f}%"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _i(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return DASH


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """把行列表渲染成 Markdown 管道表格。"""
    hs = [str(h) for h in headers]
    lines = ["| " + " | ".join(hs) + " |",
             "|" + "|".join(["---"] * len(hs)) + "|"]
    n = 0
    for r in rows:
        cells = [str(c) for c in r]
        if len(cells) < len(hs):
            cells += [""] * (len(hs) - len(cells))
        lines.append("| " + " | ".join(cells[:len(hs)]) + " |")
        n += 1
    if n == 0:
        lines.append("| " + " | ".join([DASH] * len(hs)) + " |")
    return "\n".join(lines)


def _regime_name(r: int) -> str:
    return {0: "下跌低波", 1: "下跌高波", 2: "上涨低波", 3: "上涨高波"}.get(int(r), f"R{int(r)}")


# --------------------------------------------------------------------------- #
# 各节
# --------------------------------------------------------------------------- #
def _sec_header(res: dict, title: str) -> List[str]:
    meta = res.get("meta") or {}
    cfg = res.get("config") or meta.get("config") or {}
    out = [f"# {title}", ""]
    out.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
               f"·　评估口径：全样本走样外（point-in-time），T+1 开盘成交")
    out.append("")
    return out


def _sec_config(res: dict) -> List[str]:
    meta = res.get("meta") or {}
    cfg = (res.get("config") or meta.get("config")
           or meta.get("weight_cfg") or {})
    out = ["## 一、配置摘要", ""]
    rows = []
    if isinstance(cfg, dict) and cfg:
        for k in sorted(cfg.keys()):
            rows.append([f"`{k}`", str(cfg[k])])
    else:
        rows.append(["config", "未提供（res['config'] / res['meta']['config'] 为空）"])
    out.append(md_table(["参数", "取值"], rows))
    out.append("")
    if meta:
        keep = [(k, v) for k, v in meta.items() if k != "config" and not isinstance(v, (dict, list))]
        if keep:
            out.append("**引擎元信息**")
            out.append("")
            out.append(md_table(["字段", "值"], [[f"`{k}`", str(v)] for k, v in keep]))
            out.append("")
    return out


def _sec_sample(res: dict) -> List[str]:
    dates = res.get("dates")
    scores = np.asarray(res.get("scores"))
    regime = res.get("REGIME")
    codes = res.get("codes") or []
    sids = res.get("sids") or []
    T = scores.shape[1] if scores.ndim == 2 else 0
    N = scores.shape[0] if scores.ndim == 2 else 0

    d0 = d1 = "?"
    if dates is not None and len(dates):
        try:
            d0 = str(pd.Timestamp(np.asarray(dates)[0]).date())
            d1 = str(pd.Timestamp(np.asarray(dates)[-1]).date())
        except Exception:
            pass

    tradable = int(np.isfinite(scores).sum())
    rows = [
        ["样本区间", f"{d0} ~ {d1}"],
        ["交易日数 T", _i(T)],
        ["股票数 N", _i(N)],
        ["信号数 S", _i(len(sids))],
        ["有效(可交易)样本点", _i(tradable)],
        ["平均每日可交易股票", _f(tradable / T, 1) if T else DASH],
        ["股票池", ", ".join(map(str, codes[:12])) + (" …" if len(codes) > 12 else "")],
    ]
    out = ["## 二、样本区间", "", md_table(["项目", "取值"], rows), ""]

    if regime is not None and len(regime):
        rg = np.asarray(regime).ravel()
        vals, cnts = np.unique(rg, return_counts=True)
        out.append("**市场状态（regime）分布**")
        out.append("")
        out.append(md_table(
            ["regime", "含义", "交易日数", "占比"],
            [[int(v), _regime_name(v), _i(c), _f(c / max(rg.size, 1), 1, pct=True)]
             for v, c in zip(vals, cnts)]))
        out.append("")
    return out


def _sec_hit(metrics: dict) -> List[str]:
    hr = metrics.get("hit_rate") or {}
    if not hr:
        return ["## 三、方向命中率", "", "_（无数据）_", ""]
    rows = []
    for h in sorted(hr.keys()):
        m = hr[h]
        rows.append([f"h={h}", _f(m.get("hit_rate"), 4), _i(m.get("n_samples")),
                     _i(m.get("n_days")), _f(m.get("mean_signed_ret"), 4, signed=True),
                     _f(m.get("coverage"), 1, pct=True)])
    out = ["## 三、方向命中率", "",
           "口径：`sign(score) == sign(r_h)`，仅统计 |score| > 阈值的样本；"
           "零收益计为未命中。", "",
           md_table(["周期", "命中率", "样本数", "交易日数", "带符号平均收益", "覆盖率"], rows), ""]

    # 分 regime
    rrows = []
    regs = set()
    for h in hr:
        regs |= set((hr[h].get("by_regime") or {}).keys())
    for r in sorted(regs):
        row = [_regime_name(r)]
        for h in sorted(hr.keys()):
            v = (hr[h].get("by_regime") or {}).get(r)
            row.append(_f(v, 4))
        rrows.append(row)
    if rrows:
        out += ["**按市场状态**", "",
                md_table(["regime"] + [f"h={h}" for h in sorted(hr.keys())], rrows), ""]
    return out


def _sec_ic(metrics: dict) -> List[str]:
    ic = metrics.get("rank_ic") or {}
    if not ic:
        return ["## 四、Rank IC", "", "_（无数据）_", ""]
    rows = []
    for h in sorted(ic.keys()):
        m = ic[h]
        rows.append([f"h={h}", _f(m.get("mean"), 4), _f(m.get("std"), 4),
                     _f(m.get("icir"), 3), _f(m.get("icir_ann"), 2),
                     _f(m.get("t_stat"), 2),
                     _f(m.get("positive_ratio"), 1, pct=True), _i(m.get("n_days"))])
    out = ["## 四、Rank IC（每日横截面 Spearman）", "",
           "ICIR = mean(IC)/std(IC)；`ICIR年化` = ICIR × √252。"
           "IC>0 占比即「IC 胜率」。", "",
           md_table(["周期", "均值", "标准差", "ICIR", "ICIR年化", "t值", "IC>0占比", "有效天数"], rows), ""]
    return out


def _sec_decile(metrics: dict) -> List[str]:
    dec = metrics.get("decile") or {}
    if not dec:
        return ["## 五、分组收益", "", "_（无数据）_", ""]
    out = ["## 五、分组收益（组 1 = 分数最低）", ""]
    for h in sorted(dec.keys()):
        df = dec[h]
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        attrs = getattr(df, "attrs", {}) or {}
        rows = [[r["label"], _f(r["mean_ret_pct"], 3, signed=True), _f(r["hit_rate"], 4),
                 _i(r["n_obs"]), _i(r["n_days"])] for _, r in df.iterrows()]
        out.append(f"**h={h}**　组间单调性 Spearman = {_f(attrs.get('monotonicity'), 3)}"
                   f"　·　多空价差(高-低) = {_f(attrs.get('top_minus_bottom'), 4, pct=True, signed=True)}")
        out.append("")
        out.append(md_table(["组", "平均收益", "命中率", "样本数", "交易日数"], rows))
        out.append("")
    return out


def _sec_bt(bt: dict) -> List[str]:
    main = bt.get("long_short") if isinstance(bt, dict) and "long_short" in bt else bt
    base = bt.get("baselines") if isinstance(bt, dict) else None
    out = ["## 六、多空组合绩效", ""]
    if not isinstance(main, dict) or not main.get("ok"):
        reason = main.get("reason") if isinstance(main, dict) else "no_backtest"
        return out + [f"_回测未完成（{reason}）_", ""]

    rows = [
        ["持有周期 h", f"{main['h']} 日"],
        ["分组数", _i(main["n_groups"])],
        ["双边成本", f"{main['cost_bps']:.1f} bps"],
        ["有效交易日", _i(main["n_days"])],
        ["调仓次数", _i(main["n_periods"])],
        ["累计收益", _f(main["total_return"], 2, pct=True, signed=True)],
        ["年化收益（多空）", _f(main["ann_return"], 2, pct=True, signed=True)],
        ["年化收益（多头腿）", _f(main["ann_long"], 2, pct=True, signed=True)],
        ["年化收益（空头腿）", _f(main["ann_short"], 2, pct=True, signed=True)],
        ["年化波动", _f(main["ann_vol"], 2, pct=True)],
        ["Sharpe", _f(main["sharpe"], 2)],
        ["最大回撤", _f(main["max_drawdown"], 2, pct=True)],
        ["Calmar", _f(main["calmar"], 2)],
        ["日胜率", _f(main["win_rate"], 1, pct=True)],
        ["平均换手（两腿合计/期）", _f(main["turnover_avg"], 3)],
        ["年化换手", _f(main["turnover_ann"], 2)],
        ["累计成本", f"{_f(main['cost_total_bps'], 1)} bps"],
        ["年化成本拖累", f"{_f(main['cost_ann_bps'], 1)} bps"],
    ]
    gs = main.get("group_sizes") or {}
    if "long" in gs:
        rows.append(["多头组股票数(均/最小/最大)",
                     f"{gs['long']['mean']:.1f} / {gs['long']['min']} / {gs['long']['max']}"])
    if "short" in gs:
        rows.append(["空头组股票数(均/最小/最大)",
                     f"{gs['short']['mean']:.1f} / {gs['short']['min']} / {gs['short']['max']}"])
    out.append(md_table(["项目", "取值"], rows))
    out.append("")

    if isinstance(base, pd.DataFrame) and len(base):
        out.append("**基线对比**")
        out.append("")
        disp = base.rename(columns=_BASE_UNITS)
        cols = list(disp.columns)
        rows2 = []
        for _, r in disp.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if c.startswith(("年化收益", "年化波动", "最大回撤", "胜率")):
                    cells.append(_f(v, 2, signed=c.startswith("年化收益")))
                elif c in ("Sharpe", "平均换手(倍/期)"):
                    cells.append(_f(v, 3))
                elif c == "累计成本(bps)":
                    cells.append(_f(v, 1))
                elif c == "期数":
                    cells.append(_i(v))
                else:
                    cells.append(str(v))
            rows2.append(cells)
        out.append(md_table(cols, rows2))
        out.append("")
        out.append("> 注：多空腿收益为「多头腿独立复利 − 空头腿独立复利」口径，"
                   "与多空净值的日频复合略有差异，以「年化收益（多空）」为准。")
        out.append("")
        try:
            lr = base.loc[base["策略"] == "学习器(多空)", "年化收益"]
            ew = base.loc[base["策略"] == "等权集成(多空)", "年化收益"]
            if len(lr) and len(ew) and np.isfinite(lr.iloc[0]) and np.isfinite(ew.iloc[0]):
                verdict = "跑赢" if lr.iloc[0] > ew.iloc[0] else "**跑输**"
                out.append(f"> 学习器 vs 等权集成：{verdict} "
                           f"（{lr.iloc[0]:+.2f}% vs {ew.iloc[0]:+.2f}%）。"
                           "按 SPEC §6，跑不赢等权集成即说明学习器无效，须如实报告。")
                out.append("")
        except Exception:
            pass
    return out


def _sec_weights(res: dict, metrics: dict) -> List[str]:
    conc = metrics.get("concentration") or {}
    wh = res.get("weights_hist")
    sids = list(res.get("sids") or [])
    out = ["## 七、权重集中度与信号权重排名", ""]

    if isinstance(conc, dict) and conc and not conc.get("insufficient"):
        rows = [
            ["Herfindahl（均值）", _f(conc.get("hhi_mean"), 5)],
            ["Herfindahl（中位）", _f(conc.get("hhi_median"), 5)],
            ["Herfindahl（末值）", _f(conc.get("hhi_last"), 5)],
            ["有效信号数 1/HHI（均值）", _f(conc.get("effective_n_mean"), 2)],
            [f"Top{conc.get('top_k', 5)} 权重占比（均值）",
             _f(conc.get("topk_mean"), 2, pct=True)],
            ["HHI 趋势（后半段 − 前半段）", _f(conc.get("hhi_trend"), 5, signed=True)],
            ["信号总数 S", _i(conc.get("n_signals"))],
        ]
        out.append(md_table(["指标", "取值"], rows))
        out.append("")
        if conc.get("hhi_trend") is not None and np.isfinite(conc.get("hhi_trend", np.nan)):
            t = conc["hhi_trend"]
            out.append(f"> 权重集中度{'上升（向少数信号集中，需警惕过拟合）' if t > 0 else '下降（权重分散，接近等权）'}。")
            out.append("")

    if wh is not None:
        w = np.asarray(wh, dtype=np.float64)
        if w.ndim == 2 and w.shape[0] > 0:
            last = w[-1]
            if sids and len(sids) != w.shape[1]:
                sids = []
            names = sids or [f"s{i}" for i in range(w.shape[1])]
            order = np.argsort(-np.nan_to_num(last, nan=-1.0))
            top = order[:10]
            bot = order[::-1][:10]
            out.append("**最新一日权重 Top10**")
            out.append("")
            out.append(md_table(["#", "信号", "权重"],
                                [[j + 1, names[i], _f(last[i], 4)] for j, i in enumerate(top)]))
            out.append("")
            out.append("**最新一日权重 末位10**")
            out.append("")
            out.append(md_table(["#", "信号", "权重"],
                                [[j + 1, names[i], _f(last[i], 4)] for j, i in enumerate(bot)]))
            out.append("")
    return out


def _sec_stability(metrics: dict) -> List[str]:
    hr = metrics.get("hit_rate") or {}
    ic = metrics.get("rank_ic") or {}
    years = set()
    for h in hr:
        years |= set((hr[h].get("by_year") or {}).keys())
    for h in ic:
        years |= set((ic[h].get("by_year") or {}).keys())
    out = ["## 八、分年度稳定性", ""]
    if not years:
        return out + ["_（dates 缺失或样本不足一年，无法分年）_", ""]
    hs = sorted(set(hr.keys()) | set(ic.keys()))
    rows = []
    for y in sorted(years):
        row = [str(y)]
        for h in hs:
            row.append(_f((hr.get(h, {}).get("by_year") or {}).get(y), 4))
        for h in hs:
            row.append(_f((ic.get(h, {}).get("by_year") or {}).get(y), 4))
        rows.append(row)
    headers = ["年份"] + [f"命中率 h={h}" for h in hs] + [f"IC h={h}" for h in hs]
    out.append(md_table(headers, rows))
    out.append("")
    out.append("> 稳定性判据：任一年份 IC 符号翻转或命中率跌破 50%，即视为该年度失效，"
               "不得用全样本均值掩盖。")
    out.append("")
    return out


def _sec_placeholder() -> List[str]:
    return [
        "## 九、结论",
        "",
        "<!-- LEAD_FILL:结论 -->",
        "",
        "_（待填写：学习器是否跑赢等权集成；IC 与命中率是否具备经济显著性；"
        "分年度是否稳定；成本后是否仍有正 alpha。）_",
        "",
        "## 十、局限与缺陷",
        "",
        "<!-- LEAD_FILL:局限 -->",
        "",
        "_（待填写：生存者偏差程度、交易成本与冲击成本假设、信号同质性、"
        "样本期覆盖的牛熊结构、多空可行性（A股融券约束）、停牌与涨跌停不可成交等。）_",
        "",
        "---",
        "",
        f"_本报告由 `qcore/report.py` 自动生成，指标口径见 `docs/02_SPEC.md` §6。_",
        "",
    ]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def render_string(res: dict, metrics: dict, bt: dict,
                  title: str = "APB 走样外评估报告") -> str:
    """生成完整 Markdown 文本（不落盘）。"""
    lines: List[str] = []
    lines += _sec_header(res, title)
    lines += _sec_config(res)
    lines += _sec_sample(res)
    lines += _sec_hit(metrics)
    lines += _sec_ic(metrics)
    lines += _sec_decile(metrics)
    lines += _sec_bt(bt)
    lines += _sec_weights(res, metrics)
    lines += _sec_stability(metrics)
    lines += _sec_placeholder()
    return "\n".join(lines)


def render(res: dict, metrics: dict, bt: dict, out_path: str,
           title: str = "APB 走样外评估报告") -> str:
    """
    渲染中文 Markdown 报告到 out_path（UTF-8）。

    metrics 期望 evaluate_all() 的返回值；bt 期望
    {'long_short': run_long_short(...), 'baselines': run_baselines(...)}。
    缺少的部分会降级为占位行而不是抛异常。
    返回写入的绝对路径。
    """
    txt = render_string(res, metrics, bt, title=title)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    return os.path.abspath(out_path)
