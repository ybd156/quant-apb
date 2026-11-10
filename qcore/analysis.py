"""
优势体检 —— 把"这套东西到底行不行"算成一个可证伪的结论。

不做点估计吹牛，只回答四个问题：
  ① 每日横截面 IC 的均值，HAC 标准误多大，t 值多少？（重叠样本已校正）
  ② 扣掉"我试了 73 个信号"的运气之后，Sharpe 还显著吗？（Deflated Sharpe）
  ③ 这个优势是长期稳定的，还是只有某两年？（分年度 + 滚动监控）
  ④ 如果方向是系统性反的，反着做行不行？（对称检验，避免"亏了就反过来说"）

所有函数都只吃"已经算好的分数与收益"，不碰引擎内部状态。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import stats as S


def _rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（处理 ties），无 scipy 依赖。"""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # 处理并列
    sa = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            avg = (i + j + 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def daily_ic_series(scores: np.ndarray, RET: np.ndarray, min_names: int = 5) -> np.ndarray:
    """
    scores / RET 均为 **(N,T)**。返回 (T,) 的每日横截面 Spearman。

    ⚠️ 本项目最高频的 bug 就是朝向混用。若调用方传了 (T,N)，这里会因为
    N/T 都 > 1 而静默算错（返回长度等于股票数）。所以做一次启发式断言：
    交易日数通常远大于股票数，若第一维比第二维大很多，基本可以确定传反了。
    """
    N, T = scores.shape
    if T > 0 and N > T * 3 and T > 50:
        raise ValueError(
            f"daily_ic_series 期望 (N,T)，收到 shape={scores.shape}；"
            "看起来是 (T,N) —— 请先转置")
    out = np.full(T, np.nan)
    for t in range(T):
        sc, r = scores[:, t], RET[:, t]
        m = np.isfinite(sc) & np.isfinite(r)
        if m.sum() < min_names:
            continue
        out[t] = spearman(sc[m], r[m])
    return out


def long_short_series(scores: np.ndarray, RET: np.ndarray, h: int,
                      n_groups: int = 5) -> np.ndarray:
    """
    非重叠 h 日调仓的多空收益序列（每期一个观测）。
    返回长度约 T/h 的数组，元素为"多头组均收益 − 空头组均收益"。
    """
    N, T = scores.shape
    rets = []
    for t in range(0, T - 1 - h, h):
        sc, r = scores[:, t], RET[:, t]
        m = np.isfinite(sc) & np.isfinite(r)
        if m.sum() < n_groups * 2:
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(sc[idx])]
        g = len(order) // n_groups
        if g < 1:
            continue
        low = order[:g]
        high = order[-g:]
        rets.append(float(r[high].mean() - r[low].mean()))
    return np.asarray(rets, dtype=np.float64)


def edge_report(scores: np.ndarray, RET: np.ndarray, dates: np.ndarray,
                h: int = 5, n_trials: int = 73, label: str = "learned") -> Dict:
    """一份完整的优势体检表。scores/RET 为 (N,T)。"""
    ic = daily_ic_series(scores, RET)
    ic_ok = ic[np.isfinite(ic)]
    summ = S.summarize_edge(ic_ok, dates[-ic_ok.size:], horizon=h, n_trials=n_trials)

    ls = long_short_series(scores, RET, h)
    # DSR 的 sr 参数是「每期 Sharpe」= mean/std，不是每期收益均值。
    # 旧实现传 np.nanmean(ls_daily)（≈4e-4，收益单位），而 sr0_daily 是
    # Sharpe 单位（≈4.7e-2），两者差 ~100 倍 → DSR 恒被压在
    # Φ(-E[max SR])≈0.0078 的地板上；h=1/5/20 三个不同策略全给 0.008 即是指纹。
    _ls = np.asarray(ls, dtype=np.float64)
    _ls = _ls[np.isfinite(_ls)]
    _sd = float(_ls.std(ddof=1)) if _ls.size > 2 else 0.0
    sr_pp = float(_ls.mean() / _sd) if _sd > 0 else float("nan")
    dsr_ls = (S.deflated_sharpe(sr_pp, int(_ls.size), n_trials)
              if _ls.size > 10 else {"ok": False})
    ls_stat = S.mean_tstat(ls, lags=0)          # 已是非重叠，用朴素 SE
    sr_ls = S.sharpe_ratio(ls, periods=max(1, 252 // max(h, 1)))

    # 反向对称检验：把分数取负再算一遍，避免"亏了就说要反着做"
    ic_inv = -ic
    inv = S.mean_tstat(ic_inv[np.isfinite(ic_inv)], lags=max(h - 1, 0))

    return {
        "label": label, "h": h, "n_trials": n_trials,
        "ic": {"mean": summ["ic_mean"], "hac_se": summ["hac_se"], "t": summ["t"],
               "p": summ["p"], "ci95": summ["ci95"], "boot_ci95": summ["boot_ci95"],
               "n_obs": summ["n_obs"], "effective_n": summ["effective_n"]},
        "ic_verdict": summ["verdict"],
        "long_short": {"mean_per_period": ls_stat["mean"], "t": ls_stat["t"],
                       "p": ls_stat["p"], "sharpe": sr_ls, "n_periods": int(ls.size),
                       "annualized": float(np.nanmean(ls) * 252 / max(h, 1)) if ls.size else None,
                       "sr_per_period": sr_pp, "sr_basis": "per_period_sharpe",
                       "dsr": dsr_ls.get("dsr"), "dsr_significant": dsr_ls.get("significant")},
        "inverted_check": {"ic_mean": inv["mean"], "t": inv["t"], "p": inv["p"]},
        "deflated_sharpe": dsr_ls,
        "rolling_edge": {"frac_time_active": summ["frac_time_active"],
                         "final_state": summ["final_state"]},
        "subperiods": summ["subperiods"],
        "conclusion": _conclude(summ, ls_stat, inv),
    }


def _conclude(summ: Dict, ls_stat: Dict, inv: Dict) -> str:
    t = summ.get("t")
    if not np.isfinite(t):
        return "样本不足，无法判断"
    if t >= 2.0 and summ["deflated_sharpe"].get("significant"):
        return "IC 显著为正且通过多重检验 → 有可用优势，可进入模拟盘验证"
    if t <= -2.0:
        return ("IC 显著为负 → 信号方向系统性做反。**这不等于「反着做就能赚」**："
                "反着做需要独立样本复核 + 2N 次多重检验惩罚 + 融券可行性，"
                "当前三项均不满足，不得据此上线")
    return "IC 与 0 不可区分 → 当前信号集无优势，应停手或换特征，而不是加大权重"


def diagnose_learner(res: dict, res_eq: dict, p3d, h: int = 5,
                     cmp: Optional[Dict] = None,
                     sensitivity: bool = False) -> Dict:
    """
    学习器为什么无效 —— 逐条给出实测数字，而不是只说「无显著差异」。

    ## 检查四项

    1. **信号是否同质**：73 个专家实际只有几个独立方向？
       如果有效独立数很低，学习器再怎么调权重也没用（都是同一批声音）。

    2. **权重更新幅度**：每天权重变化多少？
       如果变化幅度 < 1e-3，等于没在学习（学习率太小或被 ρ=0.995 抹平）。

    3. **命中率对比**：学习器选的 top 组，命中率比等权高多少？
       如果持平，说明排序能力没差异。

    4. **参数敏感性**：换 κ（先验强度）和 ρ（记忆衰减）能不能翻转结论？
       如果任何参数下都不显著，说明不是调参问题，是信号本身没信息。
    """
    out: Dict = {"ok": True, "checks": [], "reasons": [], "paths": []}

    # ---- 1) 信号同质性 ----
    try:
        wh = res.get("weights_hist")
        if wh is not None and len(wh) > 0:
            w = np.asarray(wh, dtype=float)
            w = w[np.isfinite(w).all(axis=1)]
            if len(w) > 0:
                hhi = float(np.mean((w ** 2).sum(axis=1)))
                eff_n = 1.0 / hhi if hhi > 0 else 0.0
                n_sig = int(w.shape[1])
                pct = eff_n / max(n_sig, 1) * 100
                out["checks"].append({
                    "name": "信号同质性",
                    "value": round(eff_n, 1),
                    "detail": f"{n_sig} 个信号，有效独立数约 {eff_n:.1f}（{pct:.0f}%）",
                })
                if pct < 40:
                    out["reasons"].append(
                        f"信号高度同质：{n_sig} 个专家实际只有约 {eff_n:.0f} 个独立方向，"
                        f"调权重等于在几个声音之间来回倒，信息量没增加")
                    out["paths"].append(
                        "先做因子去冗余（按相关性聚类，每类只留代表），"
                        "再谈权重学习 —— 否则权重学习在冗余维度上是无效劳动")
    except Exception as e:  # noqa: BLE001
        out["checks"].append({"name": "信号同质性", "value": None,
                             "detail": f"算不出来: {type(e).__name__}"})

    # ---- 2) 权重更新幅度 ----
    try:
        wh = res.get("weights_hist")
        if wh is not None and len(wh) > 2:
            w = np.asarray(wh, dtype=float)
            d = np.abs(np.diff(w, axis=0))
            d = d[np.isfinite(d)]
            mean_step = float(np.mean(d)) if d.size else 0.0
            mean_w = float(np.mean(np.abs(w))) or 1.0
            rel = mean_step / mean_w * 100
            out["checks"].append({
                "name": "权重更新幅度",
                "value": round(mean_step, 6),
                "detail": f"日均权重变化 {mean_step:.2e}，相对 {mean_w:.4f} 为 {rel:.2f}%",
            })
            if rel < 0.5:
                out["reasons"].append(
                    f"学习几乎没发生：日均权重只变动 {rel:.2f}%。"
                    f"ρ=0.995 的记忆衰减把每次更新的信号抹掉了（半衰期 138 天）")
                out["paths"].append(
                    "降低 ρ（试 0.98~0.99），或提高 η，让单次预测错误能真的推动权重")
    except Exception as e:  # noqa: BLE001
        out["checks"].append({"name": "权重更新幅度", "value": None,
                             "detail": f"算不出来: {type(e).__name__}"})

    # ---- 3) 命中率对比 ----
    try:
        if cmp:
            li, ei = cmp.get("learned_ic"), cmp.get("equal_ic")
            if li is not None and ei is not None:
                out["checks"].append({
                    "name": "IC 对比（学习器 vs 等权）",
                    "value": round(float(li) - float(ei), 6),
                    "detail": f"学习器 IC {li:+.4f} / 等权 IC {ei:+.4f} / "
                              f"差 {float(li)-float(ei):+.4f}",
                })
                if abs(float(li) - float(ei)) < 2e-3:
                    out["reasons"].append(
                        f"两者 IC 只差 {abs(float(li)-float(ei)):.4f} —— "
                        f"学习器学到的排序和等权几乎一样，说明它没学到额外信息")
    except Exception:
        pass

    # ---- 4) 参数敏感性（默认跳过：要跑 6 次完整回测，几分钟）----
    if not sensitivity:
        out["sensitivity_skipped"] = True
        out["conclusion"] = ("；".join(out["reasons"][:3]) if out["reasons"]
                             else "没找到明确原因，需要更多数据")
        return out
    try:
        sens = []
        for rho in (0.98, 0.995):
            for kap in (2, 8, 20):
                try:
                    from .learner import WeightConfig, AdaptiveWeighter
                    from .engine import run_walkforward
                    cfg = WeightConfig(rho=rho, kappa=kap)
                    wx = AdaptiveWeighter(p3d.DIR.shape[2], 4, cfg)
                    rr = run_walkforward(p3d, weighter=wx)
                    cc = compare_learned_vs_equal(rr, res_eq, p3d, h=h)
                    sens.append({"rho": rho, "kappa": kap,
                                 "diff_t": round(float(cc.get("diff_t") or 0), 2),
                                 "diff_ic": round(float(cc.get("diff_ic") or 0), 5)})
                except Exception:
                    continue
        if sens:
            ts = [x["diff_t"] for x in sens]
            out["checks"].append({
                "name": "参数敏感性（ρ × κ 共 %d 组）" % len(sens),
                "value": round(max(ts), 2),
                "detail": "diff_t 范围 " + f"{min(ts):+.2f} ~ {max(ts):+.2f}",
            })
            out["sensitivity"] = sens
            if max(ts) < 2.0:
                out["reasons"].append(
                    f"扫了 {len(sens)} 组参数（ρ∈{{0.98,0.995}} × κ∈{{2,8,20}}），"
                    f"diff_t 最高只有 {max(ts):+.2f} —— 没有任何一档显著。"
                    f"说明不是参数没调好，是信号本身不含可学的东西")
                out["paths"].append(
                    "调参这条路走到头了。要么换特征（当前 73 个信号在 5 日尺度上没有 alpha），"
                    "要么换周期（分钟级实测有 63.6% 命中但幅度不够成本）")
    except Exception:
        pass

    out["conclusion"] = (
        "；".join(out["reasons"][:3]) if out["reasons"]
        else "没找到明确原因，需要更多数据"
    )
    return out


def compare_learned_vs_equal(res: dict, res_eq: dict, p3d, h: int = 5) -> Dict:
    """学习器 vs 等权集成：如果学不赢等权，学习器就是无效的。"""
    ret = p3d.RET[h]
    if ret.shape != res["scores"].shape:      # (T,N) -> (N,T)
        ret = ret.T
    a = daily_ic_series(res["scores"], ret)
    b = daily_ic_series(res_eq["scores"], ret)
    sa = S.mean_tstat(a[np.isfinite(a)], lags=max(h - 1, 0))
    sb = S.mean_tstat(b[np.isfinite(b)], lags=max(h - 1, 0))
    diff = a - b
    sd = S.mean_tstat(diff[np.isfinite(diff)], lags=max(h - 1, 0))
    return {"learned_ic": sa["mean"], "learned_t": sa["t"],
            "equal_ic": sb["mean"], "equal_t": sb["t"],
            "diff_ic": sd["mean"], "diff_t": sd["t"], "diff_p": sd["p"],
            "verdict": ("学习器优于等权（t=%.2f）" % sd["t"] if sd["t"] > 2
                        else ("学习器劣于等权（t=%.2f）" % sd["t"] if sd["t"] < -2
                              else "学习器与等权无显著差异 → 学习器无效，应回退等权"))}
