"""
归因 —— 预测对了为什么对，错了为什么错。

## 为什么不能只记"命中/未中"

`horizon_ledger` 现在只记 `top_ret` 和 `hit`。但这不够：

    命中，是因为选对了行业（板块 beta），还是选对了行业里的个股（alpha）？
    未中，是行业逆风拖累，还是行业内选错了？

这两种"未中"的处理方式完全相反：
- **板块拖累** → 模型没错，是逆风，不该惩罚信号
- **选股失误** → 模型确实看错了，该降权

不区分就等于把噪声和真错混在一起学。

## 分解方式（Brinson 风格）

对每只票，把持有期收益拆成三段：

    个股收益 = 市场 + 行业超额 + 个股超额

    市场     = 全市场等权收益（同期）
    行业超额 = 该股所属行业等权收益 − 市场
    个股超额 = 个股收益 − 行业等权收益

然后对 top 组（模型最看好的那批）求和：
- 如果 `行业超额` 贡献 > 60% → **板块驱动**
- 如果 `个股超额` 贡献 > 60% → **选股驱动**
- 否则 → **混合**

## 输出

每条对账记录多一个 `attribution` 字段：
    {"market": x, "industry": y, "stock": z,
     "driver": "板块驱动|选股驱动|混合", "reason": "一句话说明"}

汇总给出：命中里多少来自板块、多少来自选股；未中里多少是逆风、多少是选错。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
OUT = os.path.join(STATE, "attribution.json")

_cache: Dict[str, object] = {}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- 行业映射
def industry_of(code: str) -> Optional[str]:
    """个股 -> 行业。取不到返回 None（不猜）。"""
    key = "ind"
    if key not in _cache:
        m: Dict[str, str] = {}
        try:
            from .data import get_pool
            pool = get_pool(top_n=1500)
            for r in pool.itertuples(index=False):
                c = str(getattr(r, "code", "")).zfill(6)
                ind = getattr(r, "industry", None)
                if c and isinstance(ind, str) and ind:
                    m[c] = ind
        except Exception:
            pass
        _cache[key] = m
    m = _cache[key]  # type: ignore[assignment]
    return m.get(str(code).zfill(6))


def _industry_members() -> Dict[str, List[str]]:
    key = "members"
    if key not in _cache:
        out: Dict[str, List[str]] = {}
        try:
            from .data import get_pool
            pool = get_pool(top_n=1500)
            for r in pool.itertuples(index=False):
                c = str(getattr(r, "code", "")).zfill(6)
                ind = getattr(r, "industry", None)
                if c and isinstance(ind, str) and ind:
                    out.setdefault(ind, []).append(c)
        except Exception:
            pass
        _cache[key] = out
    return _cache[key]  # type: ignore[return-value]


# ---------------------------------------------------------------- 收益
def _ret(code: str, d0: str, d1: str) -> Optional[float]:
    """个股区间收益（不复权真实价）。"""
    try:
        from .data import fetch_kline
        df = fetch_kline(code, adjust=0, klt=101, beg="20240101")
        if df is None or not len(df):
            return None
        d = df.copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        a = d[d["date"] == str(d0)[:10]]
        b = d[d["date"] == str(d1)[:10]]
        if a.empty or b.empty:
            return None
        p0 = float(a["close"].iloc[-1])
        p1 = float(b["close"].iloc[-1])
        return (p1 / p0 - 1.0) if p0 > 0 else None
    except Exception:
        return None


def _market_ret(d0: str, d1: str) -> Optional[float]:
    """全市场等权收益的近似：用沪深300 指数。"""
    try:
        from . import data as D
        df = D.fetch_index_kline("sh000300", beg="20240101") if hasattr(D, "fetch_index_kline") else None
        if df is None:
            return None
        d = df.copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        a = d[d["date"] == str(d0)[:10]]
        b = d[d["date"] == str(d1)[:10]]
        if a.empty or b.empty:
            return None
        p0, p1 = float(a["close"].iloc[-1]), float(b["close"].iloc[-1])
        return (p1 / p0 - 1.0) if p0 > 0 else None
    except Exception:
        return None


def _industry_ret(ind: str, d0: str, d1: str, max_n: int = 25) -> Optional[float]:
    """行业等权收益（最多取 25 只代表，够用且快）。"""
    members = _industry_members().get(ind) or []
    if not members:
        return None
    rs = []
    for c in members[:max_n]:
        r = _ret(c, d0, d1)
        if r is not None:
            rs.append(r)
    return float(np.mean(rs)) if len(rs) >= 3 else None


# ---------------------------------------------------------------- 单票归因
def attribute(code: str, d0: str, d1: str,
              mkt: Optional[float] = None) -> Dict:
    """
    把一只票的区间收益拆成 市场 / 行业超额 / 个股超额。
    数据不全就返回 available=False，不编造。
    """
    r = _ret(code, d0, d1)
    if r is None:
        return {"available": False, "reason": "个股行情取不到"}
    if mkt is None:
        mkt = _market_ret(d0, d1)
    if mkt is None:
        return {"available": False, "reason": "基准取不到", "ret": round(r, 6)}

    ind = industry_of(code)
    if not ind:
        return {"available": False, "reason": "行业未知", "ret": round(r, 6)}
    ir = _industry_ret(ind, d0, d1)
    if ir is None:
        return {"available": False, "reason": f"行业 {ind} 成分不足", "ret": round(r, 6)}

    ind_ex = ir - mkt
    stk_ex = r - ir
    return {"available": True, "code": str(code).zfill(6), "industry": ind,
            "ret": round(r, 6), "market": round(mkt, 6),
            "industry_excess": round(ind_ex, 6), "stock_excess": round(stk_ex, 6),
            "check": round(mkt + ind_ex + stk_ex, 9)}


def explain(record: Dict, max_n: int = 20) -> Dict:
    """
    给一条已对账的台账记录加归因，并说明"为什么命中/为什么没中"。
    record 是 horizon_ledger 里 status=done 的那条。
    """
    res = record.get("result") or {}
    items = res.get("items") or []
    d0 = record.get("entry_date") or record.get("as_of")
    d1 = record.get("exit_date")
    if not items or not d0 or not d1:
        return {"ok": False, "reason": "记录不完整"}

    mkt = _market_ret(d0, d1)
    rows = []
    for it in items[:max_n]:
        a = attribute(it["code"], d0, d1, mkt=mkt)
        if a.get("available"):
            rows.append(a)
    if len(rows) < 5:
        return {"ok": False, "reason": f"可归因样本不足 {len(rows)}"}

    # 按分数排序，取 top 三分之一
    by_score = sorted(rows, key=lambda x: -float(
        next((i.get("score") or 0) for i in items if str(i.get("code")).zfill(6) == x["code"]), 0))
    k = max(1, len(by_score) // 3)
    top = by_score[:k]

    def _avg(key: str) -> float:
        return float(np.mean([x[key] for x in top]))

    m_avg, i_avg, s_avg = _avg("market"), _avg("industry_excess"), _avg("stock_excess")
    total = abs(i_avg) + abs(s_avg)
    if total < 1e-9:
        driver, reason = "无解释力", "top 组与市场几乎同步，看不出超额来源"
    elif abs(i_avg) / total > 0.6:
        driver = "板块驱动"
        reason = (f"top 组超额主要来自行业（行业超额 {i_avg*100:+.2f}%），"
                  f"选股贡献仅 {s_avg*100:+.2f}% —— 模型押对了板块")
    elif abs(s_avg) / total > 0.6:
        driver = "选股驱动"
        reason = (f"top 组超额主要来自行业内选股（个股超额 {s_avg*100:+.2f}%），"
                  f"行业贡献 {i_avg*100:+.2f}% —— 模型挑对了票")
    else:
        driver = "混合"
        reason = (f"行业 {i_avg*100:+.2f}% 与选股 {s_avg*100:+.2f}% 各占一半，"
                  f"没有单一主导因素")

    hit = bool(res.get("hit"))
    # 未中时区分"逆风"和"选错"
    if not hit:
        if i_avg < 0 and abs(i_avg) > abs(s_avg):
            verdict = "行业逆风"
            advice = "模型没错，是板块拖累。这类未中不该重罚信号"
        elif s_avg < 0:
            verdict = "选股失误"
            advice = "行业没拖后腿，是个股选错了。这类未中该降权"
        else:
            verdict = "综合未中"
            advice = "多因素叠加，需要更多样本判断"
    else:
        verdict = driver
        advice = ("靠板块，注意风格切换时会不会反过来" if driver == "板块驱动"
                  else "靠选股，这类 alpha 更稳" if driver == "选股驱动"
                  else "混合来源")

    return {"ok": True, "as_of": record.get("as_of"), "exit": d1,
            "horizon": record.get("horizon"), "hit": hit,
            "n_attr": len(rows), "top_k": k,
            "market": round(m_avg, 6), "industry_excess": round(i_avg, 6),
            "stock_excess": round(s_avg, 6),
            "driver": driver, "verdict": verdict,
            "reason": reason, "advice": advice}


# ---------------------------------------------------------------- 汇总
def run(horizons: Optional[List[int]] = None, verbose: bool = True) -> Dict:
    """对每个周期已对账的记录跑归因，汇总落盘。"""
    try:
        from . import horizon_ledger as HL
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}"}

    hs = horizons or HL._all_horizons()
    per_h = []
    for h in hs:
        rows = [r for r in HL._read(h)
                if r.get("status") == "done" and r.get("result")]
        expl = []
        for r in rows[-20:]:                    # 最近 20 期够看趋势
            e = explain(r)
            if e.get("ok"):
                expl.append(e)
        if not expl:
            per_h.append({"horizon": h, "n": 0, "note": "没有可归因的记录"})
            continue
        hits = [e for e in expl if e["hit"]]
        miss = [e for e in expl if not e["hit"]]
        def _cnt(rows_, key, val):
            return sum(1 for x in rows_ if x.get(key) == val)
        per_h.append({
            "horizon": h, "n": len(expl),
            "hit_n": len(hits), "miss_n": len(miss),
            "hit_board": _cnt(hits, "driver", "板块驱动"),
            "hit_stock": _cnt(hits, "driver", "选股驱动"),
            "hit_mixed": _cnt(hits, "driver", "混合"),
            "miss_headwind": _cnt(miss, "verdict", "行业逆风"),
            "miss_pickerr": _cnt(miss, "verdict", "选股失误"),
            "avg_industry_excess": round(float(np.mean([e["industry_excess"] for e in expl])), 6),
            "avg_stock_excess": round(float(np.mean([e["stock_excess"] for e in expl])), 6),
            "recent": expl[-5:],
        })

    payload = {"generated_at": _now(), "horizons": per_h,
               "note": ("归因用 Brinson 分解：个股收益 = 市场 + 行业超额 + 个股超额。"
                        "命中里区分『押对板块』和『挑对个股』；未中里区分『行业逆风』和『选股失误』。"
                        "这两类的处理方式相反 —— 逆风不该重罚信号，选错才该降权。")}
    _atomic(OUT, payload)
    if verbose:
        print(f"{'周期':<14}{'样本':>5}{'命中':>5}{'板块':>5}{'选股':>5}{'逆风':>5}{'选错':>5}")
        for p in per_h:
            if not p.get("n"):
                print(f"  h{p['horizon']:<12}{0:>5}")
                continue
            print(f"  h{p['horizon']:<12}{p['n']:>5}{p['hit_n']:>5}"
                  f"{p['hit_board']:>5}{p['hit_stock']:>5}"
                  f"{p['miss_headwind']:>5}{p['miss_pickerr']:>5}")
    return {"ok": True, **payload}


def summary() -> Dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return run(verbose=False)


if __name__ == "__main__":
    run(verbose=True)
    print()
    print("说明：样本 < 10 时归因只能当观察，不能当结论。")
