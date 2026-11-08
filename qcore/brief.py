"""
每日研判简报 —— 把"当前形态 + 预测结论 + 证据链"压成一页。

输出：
  reports/daily_brief.html   （含 5-6 张 PNG，可直接浏览器打开/手机看）
  reports/img/*.png

设计原则：
  1. 结论必须带**可信度**与**证伪条件**，不给裸预测
  2. 每个数字都能追到出处（哪张图、哪段检验）
  3. 没有优势时明说"不表态"，不硬给方向
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import chart as CH
from . import stats as ST
from . import analysis as AN
from . import confidence as CF
from . import meta as MT
from .data import get_pool, load_kline, load_index
from .dataset import build_dataset
from .engine import RunConfig, run_walkforward
from .context import now_context, analog_forecast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
IMG_DIR = os.path.join(REPORT_DIR, "img")
os.makedirs(IMG_DIR, exist_ok=True)

REGIME_NAME = {0: "下跌·低波", 1: "下跌·高波", 2: "上涨·低波", 3: "上涨·高波"}


def _with_family(rows):
    """给权重行补上信号族标签（weighter 本身不存族信息）。"""
    try:
        from .signals import signal_zoo
        fam = {s.sid: s.family for s in signal_zoo()}
    except Exception:
        fam = {}
    out = []
    for r in rows:
        d = dict(r)
        d["family"] = fam.get(d.get("sid", ""), "-")
        out.append(d)
    return out


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=max(5, w // 4)).mean().to_numpy()


def build(pool_n: int = 300, start: str = "2013-01-01", horizon: int = 5,
          focus_code: Optional[str] = None, out_name: str = "daily_brief.html") -> Dict:
    pool = get_pool(top_n=pool_n)
    codes = pool["code"].tolist()
    p3d = build_dataset(codes, start=start, horizon=horizon, extra_horizons=[1, 20],
                        verbose=False)
    cfg = RunConfig(horizon=horizon, warmup=120, record_ledger=False)
    res = run_walkforward(p3d, cfg)
    res_eq = run_walkforward(p3d, RunConfig(horizon=horizon, warmup=120, mode="equal",
                                            record_ledger=False))

    dates = p3d.dates
    dstr = [str(pd.Timestamp(d).date()) for d in dates]
    T, N = p3d.RET[horizon].shape

    # ---------------- 1. 净值曲线
    ls = AN.long_short_series(res["scores"], p3d.RET[horizon].T, horizon)
    cost = 0.7 * 19.28 / 1e4
    nav_ls = np.cumprod(1 + np.nan_to_num(ls - cost)) if ls.size else np.array([1.0])
    # 基准必须用**日频**收益连乘；用 h 日重叠收益连乘会把同一段行情算 h 次（实测会飙到 56 倍）
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = p3d.CLOSE[1:] / p3d.CLOSE[:-1] - 1.0
    ew = np.nanmean(np.where(np.isfinite(daily), daily, np.nan), axis=1)
    nav_pool = np.cumprod(1.0 + np.nan_to_num(ew))
    idx = load_index("000300").set_index("date")["close"].reindex(pd.DatetimeIndex(dates))
    nav_idx = (idx / idx.iloc[0]).to_numpy()
    CH.line_png(os.path.join(IMG_DIR, "nav.png"),
                {"多空组合(扣成本)": (None, nav_ls),
                 "全池等权(基准)": (None, nav_pool[:len(nav_ls)] if len(nav_pool) >= len(nav_ls) else nav_pool),
                 "沪深300": (None, nav_idx[:len(nav_ls)])},
                title="累计净值", subtitle=f"非重叠 {horizon} 日调仓 · 双边成本 19.28bps · 起点=1",
                fmt="{:.2f}", hline=1.0)

    # ---------------- 2. 滚动 IC
    ic = AN.daily_ic_series(res["scores"], p3d.RET[horizon].T)
    ic_r = _rolling_mean(ic, 250)
    CH.line_png(os.path.join(IMG_DIR, "ic.png"),
                {"逐日IC(淡)": (None, np.clip(ic, -0.3, 0.3)),
                 "滚动250日IC": (None, ic_r)},
                title="横截面 Rank IC",
                subtitle="白线=滚动250日均值（长期在0以下即信号方向系统性做反）；暗线=逐日IC",
                fmt="{:.3f}", hline=0.0,
                colors={"逐日IC(淡)": (70, 34, 38), "滚动250日IC": (226, 232, 240)})

    # ---------------- 3. 权重演化（Top 8）
    wh = res["weights_hist"]
    top = np.argsort(-wh[-1])[:8]
    series = {}
    for i in top:
        series[res["sids"][i]] = (None, wh[:, i])
    CH.line_png(os.path.join(IMG_DIR, "weights.png"), series,
                title="信号权重演化（当前 Top8）",
                subtitle="Hedge 乘性更新 + 贝叶斯收缩；权重变化=学习器在重新分配信任",
                fmt="{:.4f}")

    # ---------------- 4. 市场状态色带
    CH.regime_band_png(os.path.join(IMG_DIR, "regime.png"), dstr, p3d.REGIME[-500:])

    # ---------------- 5. 当前形态（个股 K 线）
    last_scores = res["scores"][:, -1]
    if focus_code is None:
        order = np.argsort(-np.nan_to_num(last_scores, nan=-9))
        focus_code = codes[order[0]]
    k = load_kline(focus_code)
    k = k.tail(160).reset_index(drop=True)
    ma = {"MA20": k["close"].rolling(20).mean().to_numpy(),
          "MA60": k["close"].rolling(60).mean().to_numpy()}
    ctx = now_context(k, name=focus_code)
    ana = analog_forecast(k, h=horizon, k=30)
    CH.candle_png(os.path.join(IMG_DIR, f"candle_{focus_code}.png"),
                  [str(pd.Timestamp(d).date()) for d in k["date"]],
                  k["open"].to_numpy(), k["high"].to_numpy(), k["low"].to_numpy(),
                  k["close"].to_numpy(), k["volume"].to_numpy(), ma,
                  title=f"{focus_code} 当前形态", subtitle="近 160 日 · MA20/MA60 · 下方为成交量")

    # ---------------- 6. 预测结论
    last = res["scores"][:, -1]
    last_idx = int(np.nanargmax(np.where(np.isfinite(last), last, -np.inf)))
    sc = float(last[last_idx])
    reg = int(p3d.REGIME[-1])

    cal = CF.calibrate(res["scores_TN"], p3d.RET[horizon], horizon, warmup=cfg.warmup)
    p_conf = CF.combine(cal["confidence"], res["concord"].T, res["quality"].T)
    conf_last = float(p_conf[-1, last_idx]) if np.isfinite(p_conf[-1, last_idx]) else float("nan")
    n_sup = int(cal["n_support"][-1, last_idx])
    lvl = CF.level_of(conf_last, n_sup)
    meta = MT.learn_meta(res["scores_TN"], p3d.RET[horizon], horizon, warmup=cfg.warmup)
    sign = int(meta["final_sign"])

    ic_ok = ic[np.isfinite(ic)]
    st_ic = ST.mean_tstat(ic_ok, lags=horizon - 1)

    verdict = _verdict(sign, st_ic, lvl)
    html = _render_html(
        codes=codes, focus=focus_code, score=sc, conf=conf_last, level=lvl, n_sup=n_sup,
        sign=sign, meta_t=meta["final_t"], regime=reg, ctx=ctx, ana=ana,
        st_ic=st_ic, n_days=T, n_stocks=N, horizon=horizon,
        top_sig=_with_family(res["weighter"].top(10)), verdict=verdict,
        as_of=dstr[-1])
    path = os.path.join(REPORT_DIR, out_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return {"ok": True, "report": path, "images": os.listdir(IMG_DIR),
            "focus": focus_code, "score": sc, "confidence": conf_last,
            "level": lvl, "meta_sign": sign, "ic_t": float(st_ic["t"]),
            "verdict": verdict}


def _verdict(sign: int, st_ic: Dict, lvl: str) -> str:
    t = st_ic.get("t", float("nan"))
    if not np.isfinite(t):
        return "样本不足，不表态"
    if lvl in ("insufficient", "no_edge"):
        return "统计上无优势 → 建议不表态（空仓或持有基准）"
    if sign < 0:
        return (f"信号方向系统性做反（IC t={t:.2f}）→ 系统已自动反向使用；"
                "但反向后的优势尚未达到显著，只可小仓位验证")
    if sign > 0:
        return f"信号方向有效（IC t={t:.2f}）→ 可按可信度分级使用"
    return "Meta 层判定不显著 → 不表态"


def _render_html(**kw) -> str:
    ctx = kw["ctx"]
    ana = kw["ana"]
    st = kw["st_ic"]
    rows = "".join(
        f"<tr><td>{d['sid']}</td><td>{d['family']}</td>"
        f"<td class='num'>{d['weight']:.4f}</td>"
        f"<td class='num'>{d['hit_rate']:.3f}</td>"
        f"<td class='num'>{d['n_updates']}</td></tr>"
        for d in kw["top_sig"])
    analog_html = ""
    if ana.get("ok"):
        analog_html = (f"<p>历史类比：找到 <b>{ana['n_analogs']}</b> 个相似时点，"
                       f"其后 {ana['h']} 日中位数 <b>{ana['median']:+.2%}</b>，"
                       f"上涨概率 <b>{ana['hit_rate']:.0%}</b>，"
                       f"区间 [{ana['q10']:+.1%}, {ana['q90']:+.1%}]，"
                       f"类比可信度 <b>{ana['credibility']:.0%}</b>。</p>")
    now = ctx.get("now", {}) if ctx else {}
    now_html = ""
    if now.get("ok"):
        now_html = (f"<p>趋势 <b>{ {'up':'向上','down':'向下','sideways':'震荡'}.get(now['trend'],'?') }</b>、"
                    f"波动 <b>{ {'high':'偏高','low':'偏低','mid':'中性'}.get(now['vol_state'],'?') }</b>；"
                    f"距 250 日高点 <b>{now['drawdown_from_250d_high']:.1%}</b>，"
                    f"20 日 <b>{now['ret20']:+.1%}</b>，60 日 <b>{now['ret60']:+.1%}</b>。</p>")

    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>APB 每日研判 · {kw['as_of']}</title>
<style>
body{{background:#11141a;color:#e2e8f0;font:14px/1.7 -apple-system,"Microsoft YaHei",sans-serif;margin:0;padding:24px}}
.wrap{{max-width:1040px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 10px;border-left:3px solid #60a5fa;padding-left:10px}}
.muted{{color:#8c98aa;font-size:12px}}
.card{{background:#181c24;border:1px solid #2d3440;border-radius:10px;padding:16px;margin:12px 0}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap}}
.kpi div{{flex:1;min-width:150px;background:#181c24;border:1px solid #2d3440;border-radius:10px;padding:14px}}
.kpi b{{display:block;font-size:22px;margin-top:4px}}
img{{width:100%;border-radius:8px;margin:8px 0;border:1px solid #2d3440}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid #2d3440;text-align:left}}
th{{color:#8c98aa;font-weight:500}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.warn{{background:#3a2d12;border-color:#7a5a12}} .ok{{background:#12301f;border-color:#1f5a34}}
</style></head><body><div class="wrap">
<h1>APB 每日研判</h1>
<div class="muted">决策日 {kw['as_of']} · 只用当日及以前数据 · 样本 {kw['n_stocks']} 只 / {kw['n_days']} 个交易日 · 预测周期 {kw['horizon']} 日</div>

<h2>一、结论</h2>
<div class="card {'warn' if kw['level'] in ('insufficient','no_edge') else ''}">
<p><b>{kw['verdict']}</b></p>
<p class="muted">Meta 层方向：{ {1:'顺势用',-1:'反着用',0:'不表态'}[kw['sign']] }（IC t={kw['meta_t']:+.2f}）</p>
</div>

<h2>二、关键数字</h2>
<div class="kpi">
<div>当前市场状态<b>{REGIME_NAME.get(kw['regime'],'?')}</b></div>
<div>关注标的<b>{kw['focus']}</b><span class="muted">综合分 {kw['score']:+.3f}</span></div>
<div>可信度<b>{kw['conf']:.1%}</b><span class="muted">{kw['level']} · n={kw['n_sup']}</span></div>
<div>横截面 IC<b>{st['mean']:+.4f}</b><span class="muted">HAC t={st['t']:+.2f}</span></div>
</div>

<h2>三、当前形态</h2>
<div class="card">{now_html}{analog_html}</div>
<img src="img/candle_{kw['focus']}.png" alt="当前形态">

<h2>四、市场状态与净值</h2>
<img src="img/regime.png" alt="市场状态">
<img src="img/nav.png" alt="净值">

<h2>五、预测力与权重演化</h2>
<img src="img/ic.png" alt="IC">
<img src="img/weights.png" alt="权重">

<h2>六、当前最受信任的信号</h2>
<table><thead><tr><th>信号</th><th>族</th><th>权重</th><th>命中率</th><th>更新次数</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>七、这张图怎么看 / 怎么用</h2>
<div class="card">
<p>1. <b>先看结论</b>：如果写着"统计上无优势"，就<b>不要下单</b>——这不是保守，这是纪律。</p>
<p>2. <b>再看 IC 图</b>：滚动 250 日 IC 长期在 0 以下，说明技术信号方向是反的，系统会自动反向使用（Meta 层）。</p>
<p>3. <b>然后看形态</b>：趋势/波动/距高点 + 历史类比分布，比"我觉得会涨"更有信息量。</p>
<p>4. <b>最后看权重</b>：权重变化说明学习器在重新分配信任，突然放大的信号值得单独复核。</p>
<p class="muted">本页为研究工具输出，不构成投资建议。所有结论均附统计检验，样本外表现未知。</p>
</div>
</div></body></html>"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--focus", default=None)
    a = ap.parse_args()
    out = build(a.pool, horizon=a.horizon, focus_code=a.focus)
    print(json.dumps({k: v for k, v in out.items() if k != "images"}, ensure_ascii=False))
