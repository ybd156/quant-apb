"""报告附录：统计严谨性章节（决定"这套系统到底能不能用"）。

单独成模块的原因：这一节的所有数字都必须来自 stats.py / analysis.py 的
HAC 校正结果，不允许在报告层重新算一遍——避免"报告好看"污染"事实"。
"""
from __future__ import annotations

from typing import Dict


def _f(x, nd: int = 4, sign: bool = True) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _pct(x, nd: int = 2, sign: bool = True) -> str:
    try:
        v = float(x) * 100
    except Exception:
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def append_stats(path: str, edge: Dict, eq_cmp: Dict, meta: Dict, meta_cmp: Dict,
                 meta_ic: Dict, mon: Dict) -> None:
    L = []
    A = L.append

    A("")
    A("---")
    A("")
    A("## 统计严谨性检验（决定这套系统到底能不能用）")
    A("")
    A("> 重叠 h 日收益会把有效样本数高估约 h 倍，t 值随之虚高。")
    A("> 本节所有 t 值均使用 Newey-West (HAC) 标准误，滞后阶取 h−1。")
    A("> 这是本节与普通回测报告最本质的区别——普通回测的 t 值在这里基本都会缩水一半以上。")
    A("")

    A("### 1. 横截面预测力（Rank IC）显著性")
    A("")
    A("| 周期 | IC 均值 | HAC 标准误 | t 值 | p 值 | 95% 置信区间 | 有效样本 | 判定 |")
    A("|---|---|---|---|---|---|---|---|")
    for h in ("1", "5", "20"):
        e = edge[h]["ic"]
        ci = e["ci95"]
        A(f"| h={h} | {_f(e['mean'])} | {_f(e['hac_se'], 4, False)} | {_f(e['t'], 2)} | "
          f"{_f(e['p'], 3, False)} | [{_f(ci[0])}, {_f(ci[1])}] | {e['effective_n']} | "
          f"{edge[h]['ic_verdict']} |")
    A("")
    A("**多空组合（非重叠 h 日调仓，扣双边成本）**")
    A("")
    A("| 周期 | 每期均值 | t 值 | 年化 | Sharpe | 期数 | Deflated Sharpe | 通过多重检验 |")
    A("|---|---|---|---|---|---|---|---|")
    for h in ("1", "5", "20"):
        ls = edge[h]["long_short"]
        A(f"| h={h} | {_pct(ls['mean_per_period'])} | {_f(ls['t'], 2)} | "
          f"{_pct(ls['annualized'] or 0)} | {_f(ls['sharpe'], 2)} | {ls['n_periods']} | "
          f"{_f(ls['dsr'] or 0, 3, False)} | {'是' if ls['dsr_significant'] else '否'} |")
    A("")
    A("> Deflated Sharpe 把「我试了 N 个信号」的运气扣掉（N = 信号总数）。")
    A("> 未通过即表示：看到的 Sharpe 无法与「多次试验必然产生的最大值」区分。")
    A("")

    A("### 2. 学习器 vs 等权集成（硬门槛）")
    A("")
    A("| 口径 | IC 均值 | t 值 |")
    A("|---|---|---|")
    A(f"| 自学习权重 | {_f(eq_cmp['learned_ic'])} | {_f(eq_cmp['learned_t'], 2)} |")
    A(f"| 等权集成 | {_f(eq_cmp['equal_ic'])} | {_f(eq_cmp['equal_t'], 2)} |")
    A(f"| 差值（学习器 − 等权） | {_f(eq_cmp['diff_ic'])} | {_f(eq_cmp['diff_t'], 2)} |")
    A("")
    A(f"**判定：{eq_cmp['verdict']}**")
    A("")
    A("> 口径：Sharpe = 每期算术均值/标准差 × √252；年化收益为**几何**复合，两者不可相除验算。")
    A("")
    A("> 学习器若跑不赢等权集成，说明权重调整没有带来信息，只是把噪声重新分配了一遍。")
    A("> 这条是「自学习」这个卖点的生死线，必须如实报告。")
    A("")

    A("### 3. Meta 层自我纠错")
    A("")
    A("| 项目 | 结果 |")
    A("|---|---|")
    A(f"| 当前判定 | **{meta['verdict']}**（sign = {meta['final_sign']:+d}） |")
    A(f"| 到期 IC 均值 / HAC t | {_f(meta['final_mu'])} / {_f(meta['final_t'], 2)} |")
    A(f"| 反向运行时间占比 | {_pct(meta['frac_inverted'], 1, False)} |")
    A(f"| 不表态（空仓）时间占比 | {_pct(meta['frac_silent'], 1, False)} |")
    A(f"| 命中率：原始 → 纠错后 | {_pct((meta_cmp.get('raw') or {}).get('hit_rate'), 2, False)} → "
      f"{_pct((meta_cmp.get('meta') or {}).get('hit_rate'), 2, False)} |")
    A(f"| 纠错后 IC / t | {_f(meta_ic['mean'])} / {_f(meta_ic['t'], 2)} |")
    A("")
    A("> Meta 层只在「横截面 IC 显著异于 0」（|t| ≥ 2）时才改变方向，否则**不表态**。")
    A("> 这是「对一次两次不算什么」的工程解法：没有统计证据就不下注，")
    A("> 而不是把噪声当信号硬做。")
    A("")

    A("### 4. 长期稳定性（分年度 IC）")
    A("")
    sub = edge["5"].get("subperiods") or []
    if sub:
        A("| 年份 | IC 均值 | t 值 | 样本 |")
        A("|---|---|---|---|")
        for r in sub:
            flag = " ⚠️" if r["mean"] < 0 else ""
            A(f"| {r['year']} | {_f(r['mean'])}{flag} | {_f(r['t'], 2)} | {r['n']} |")
        neg = sum(1 for r in sub if r["mean"] < 0)
        A("")
        A(f"共 {len(sub)} 个年度，其中 **{neg} 年 IC 为负**。")
        A("任一年份符号翻转，都说明这个优势不是「长期稳定」的。")
    else:
        A("样本不足以分年统计。")
    A("")

    A("### 5. 滚动优势监控（自我纠错开关）")
    A("")
    A(f"- 优势「在线」时间占比：{_pct(mon.get('frac_active', 0), 1, False)}")
    A(f"- 当前状态：{'有优势' if mon.get('final_state') == 1 else ('已休眠' if mon.get('final_state') == -1 else '未判定')}")
    A(f"- 当前滚动 HAC t：{_f(mon.get('final_t'), 2)}")
    A("")
    A("> 状态机：`unknown → active（t>2）→ dormant（t<0.5）`。")
    A("> dormant 时下游仓位应归零——这就是「优势消失后自动停手」。")
    A("")

    A("### 6. 结论")
    A("")
    e5 = edge["5"]
    A(f"- **h=5 横截面 IC = {_f(e5['ic']['mean'])}，HAC t = {_f(e5['ic']['t'], 2)}** → {e5['ic_verdict']}")
    A(f"- {e5['conclusion']}")
    A(f"- 学习器 vs 等权：{eq_cmp['verdict']}")
    A(f"- Meta 自纠错：{meta['verdict']}，命中率 "
      f"{_pct(meta_cmp['raw']['hit_rate'], 2, False)} → {_pct(meta_cmp['meta']['hit_rate'], 2, False)}，"
      f"纠错后 IC t = {_f(meta_ic['t'], 2)}")
    A("")
    A("**这套系统当前的正确定位：它是一台「测量仪」，不是「印钞机」。**")
    A("它最大的价值是诚实地告诉你——现在有没有优势、优势有多大、还能不能信、")
    A("什么时候该停手。而不是保证赚钱。当 IC 不显著时，它应该输出「不表态」，")
    A("而不是硬给一个信号。")
    A("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))


def append_costs(path: str, market: str, cost_bps: float, rt: Dict, CO) -> None:
    """追加"交易成本"章节 —— 很多策略死在成本上，但报告里常常只有一个 10bps。"""
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A("## 交易成本模型")
    A("")
    A("> 三个市场的成本结构差一个数量级。同一个策略在 A 股能活，搬到港股可能直接亏死。")
    A("")

    A("### 1. 分市场费率（单边，不含滑点）")
    A("")
    A("| 市场 | 佣金 | 印花税 | 印花税方向 | 其他规费 | 半价差 | **往返合计** | 核对日期 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in CO.summary_table():
        A(f"| {r['name']} | {r['commission_bps']:.4f}% | {r['stamp_duty_bps']:.2f}% | "
          f"{r['stamp_side']} | {r['other_bps']:.4f}% | {r['slippage_bps']:.1f}bps | "
          f"**{r['round_trip_bps']:.1f} bps** | {r['verified_on']} |")
    A("")
    A("> A 股印花税**仅卖出单边**收 0.05%；港股**双边**各 0.1%，这是港股往返成本高的主因。")
    A("> 美股零佣金，真实成本主要是价差与冲击成本。")
    A("")

    A(f"### 2. 本报告使用的成本口径（{market}，按 100 万本币名义额）")
    A("")
    A("| 项目 | 买入 | 卖出 |")
    A("|---|---|---|")
    b = rt["breakdown"]["buy"]
    s_ = rt["breakdown"]["sell"]
    for k in b:
        A(f"| {k} | {b[k]:.2f} | {s_[k]:.2f} |")
    A(f"| **合计（元）** | {sum(b.values()):.2f} | {sum(s_.values()):.2f} |")
    A("")
    A(f"**往返成本 = {cost_bps:.2f} bps**（买入 {rt['buy_bps']:.2f} + 卖出 {rt['sell_bps']:.2f}）")
    A("")

    A("### 3. 规模敏感性（最低佣金让小资金成本率更高）")
    A("")
    A("| 名义额 | 市场 | 往返 bps |")
    A("|---|---|---|")
    for r in CO.cost_scenarios():
        n = r["notional"]
        unit = f"{n/1e4:.0f} 万" if n < 1e8 else f"{n/1e8:.0f} 亿"
        A(f"| {unit} | {r['market']} | {r['round_trip_bps']:.2f} |")
    A("")
    A("> 小资金被「最低佣金」拖累，大资金被「冲击成本」拖累，中间有一段最优区间。")
    A("> 只看一个固定 bps 的回测，等于假装这两个效应都不存在。")
    A("")

    drag = CO.annual_cost_drag(market, 0.7, 5)
    A("### 4. 年化拖累")
    A("")
    A(f"按每 {5} 日调仓、每次换手 70% 估算：**每年成本拖累约 {drag['annual_cost_pct']:.2f}%**")
    A(f"（往返 {drag['round_trip_bps']:.1f} bps × 每年 {drag['periods_per_year']:.0f} 次调仓）")
    A("")
    A("> 对照：技术信号的横截面 IC 通常在 0.02~0.03，按基本法则算出的毛 IR 也就 0.3~0.5。")
    A("> 每年 7% 左右的成本，足以把这点优势吃干净。**这就是为什么很多回测好看、实盘不赚钱。**")
    A("")
    A("**费率会变**（A 股印花税 2023-08-28 刚从 0.1% 减半到 0.05%），"
      "实盘前必须用 `qcore/costs.py` 里的 `verified_on` 重新核对。")
    A("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))
