"""
导出模型产物给 Web UI：model.json / picks.json / factor_importance.json。

## 产物契约

- `model.json`              模型元信息 + 回测摘要
- `picks.json`              当前推荐持仓（含名称、分数、权重、是否持有中）
- `factor_importance.json`  GBDT 特征重要性（158 个因子）

## 名称从哪来

Qlib 数据的代码是 `sh600519` / `sz000001` 这种带交易所前缀的，
不带中文名。所以剥前缀后去 `data/state/names.json` 查（由主流水线导出）。
查不到就只显示代码，不编造。

## 为什么要单独一个脚本

训练一次要几分钟，不该塞进 Web 服务的请求路径里。
由流水线或定时任务调用，UI 只读 JSON（INV-8）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import qlib_data as QD
from qcore import names as NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")


def _bare(code: str) -> str:
    """sh600519 -> 600519；sz000001 -> 000001；bj430017 -> 430017"""
    c = str(code).lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c[2:]
    return c


def _load_name_map() -> dict:
    m = NAMES.load()
    out = {}
    for k, v in (m or {}).items():
        out[_bare(k)] = v.get("name") if isinstance(v, dict) else str(v)
    return out


def _write(name: str, payload: dict) -> str:
    os.makedirs(STATE, exist_ok=True)
    p = os.path.join(STATE, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)
    return p


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="csi300")
    ap.add_argument("--stocks", type=int, default=300)
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--hold-days", type=int, default=20)
    ap.add_argument("--band", type=float, default=1.0)
    ap.add_argument("--train-end", default="2019-12-31")
    a = ap.parse_args(argv)

    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "bench", os.path.join(here, "benchmark_alpha158.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    spec2 = importlib.util.spec_from_file_location(
        "ltb", os.path.join(here, "lowturnover_bt.py"))
    ltb = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ltb)
    from qcore import bt as BT

    print(f"训练 {a.universe} / 截至 {a.train_end} / {a.trees} 棵 ...", flush=True)
    d = ltb.prepare(a.universe, a.stocks, train_end=a.train_end, trees=a.trees)
    if d is None:
        print("准备失败")
        return 1

    model = d["model"]
    scores, close, dates, codes = d["scores"], d["close"], d["dates"], d["codes"]

    # ---- 回测摘要（用最佳参数）----
    r = BT.backtest(scores, close, dates, None,
                    BT.BTConfig(top_k=a.top_k, hold_days=a.hold_days, band=a.band))
    m = r["metrics"]

    # ---- 信号质量 ----
    ic = bench._ic(scores, d["y"])

    # 名称映射要用足够大的池子，否则 CSI300 里大半查不到名字
    nm = _load_name_map()
    try:
        from qcore.data import get_pool as _gp
        _big = _gp(top_n=1200)
        for r in _big.itertuples(index=False):
            c = str(getattr(r, "code", "")).zfill(6)
            n_ = getattr(r, "name", None)
            if c and isinstance(n_, str) and n_ and c not in nm:
                nm[c] = n_
        print(f"  名称池扩充到 {len(nm)} 条", flush=True)
    except Exception as e:
        print(f"  名称池扩充失败（用原映射）: {e}", flush=True)

    # ---- 最新一期推荐 ----
    last = len(dates) - 1
    s = scores[last]
    ok = np.isfinite(s)
    order = np.argsort(-np.where(ok, s, -np.inf))[:a.top_k]
    prev = scores[last - 1] if last > 0 else s
    prev_top = set(np.argsort(-np.where(np.isfinite(prev), prev, -np.inf))[:a.top_k].tolist())

    rows = []
    for rank, j in enumerate(order, 1):
        code = codes[j]
        bare = _bare(code)
        px = close[last, j]
        prev_px = close[last - 1, j] if last > 0 else px
        chg = (px / prev_px - 1.0) if (np.isfinite(px) and np.isfinite(prev_px) and prev_px) else None
        # 单日 |涨跌| > 21% 必是复权/除权伪影（A股涨跌停 10%/20%），置 None 不展示假数
        if chg is not None and abs(chg) > 0.21:
            chg = None
        rows.append({
            "rank": rank, "code": bare, "qlib_code": code,
            "name": nm.get(bare) or nm.get(bare.zfill(6)) or "",
            "score": round(float(s[j]), 6),
            "close": round(float(px), 4) if np.isfinite(px) else None,
            "chg_pct": round(float(chg) * 100, 2) if chg is not None else None,
            "weight": round(1.0 / len(order), 4),
            "held": j in prev_top,
        })

    _write("model.json", {
        "model": "Alpha158+GBDT", "version": "v" + open(os.path.join(ROOT, "VERSION")).read().strip() if os.path.isfile(os.path.join(ROOT, "VERSION")) else "v0.6.0",
        "n_factors": 158, "n_trees": int(a.trees),
        "train_period": f"2008-01-01..{a.train_end}",
        "test_period": f"{str(dates[0].date())}..{str(dates[-1].date())}",
        "ic": round(float(ic["ic"]), 6), "ic_t": round(float(ic["t"]), 3),
        "rank_ic": round(float(ic["rank_ic"]), 6),
        "universe": a.universe, "n_stocks": len(codes),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backtest": {
            "annual": round(float(m["annual"]), 6),
            # 夏普显示「超额夏普」——绝对夏普含 beta，对选股策略没有意义
            "sharpe": round(float(m["excess_sharpe"]), 4),
            "abs_sharpe": round(float(m["sharpe"]), 4),
            "mdd": round(float(m["mdd"]), 6),
            "turnover": round(float(m["avg_turnover"]), 4),
            "hold_days": a.hold_days, "top_k": a.top_k, "band": a.band,
            "bench_annual": round(float(m["bench_annual"]), 6),
            "excess_annual": round(float(m["excess_annual"]), 6),
            "excess_sharpe": round(float(m["excess_sharpe"]), 4),
            "excess_mdd": round(float(m["excess_mdd"]), 6),
            "excess": round(float(m["excess_annual"]), 6),
            "cost_bps": 19.28,
        },
    })

    _write("picks.json", {
        "as_of": str(dates[-1].date()), "rows": rows, "n": len(rows),
        "top_k": a.top_k, "hold_days": a.hold_days, "band": a.band,
        "note": "按 GBDT 分数排序；held=true 表示上一期已在推荐中（缓冲带内不动）",
    })

    imp = getattr(model, "feature_importance", None)
    fn = bench.factor_names()
    if imp is not None and len(imp) == len(fn):
        order_i = np.argsort(-np.asarray(imp))
        _write("factor_importance.json", {
            "rows": [{"factor": fn[i], "importance": round(float(imp[i]), 6)}
                     for i in order_i],
            "n": len(fn),
        })
        print(f"因子重要性已导出（{len(fn)} 个）")
    else:
        print("警告：feature_importance 不可用，跳过")

    print(f"model.json / picks.json 已写入 {STATE}")
    print(f"  IC={ic['ic']:+.4f} (t={ic['t']:+.2f})  超额={m['excess_annual']*100:+.2f}% "
          f"超额夏普={m['excess_sharpe']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
