"""
盘中在线训练 —— 每分钟预测下一分钟，下一分钟立刻验证对错。

## 核心思路（用户提出的关键洞察）

盘中的因果困境是「不知道当日收盘结果」，但**分钟级不存在这个问题**：

    T 分钟：用 ≤T 的信息预测 T+M 分钟的涨跌
    T+M 分钟：M 分钟后的收益已经知道了 → 立刻对账

所以每分钟都能：
    1. 收进一批新样本（T-M 分钟的预测，到期了）
    2. 用这些样本更新模型
    3. 用更新后的模型预测下一分钟

这是**真正的实时训练**，而且完全因果安全。

## 为什么用逻辑回归而不是 GBDT

| | GBDT | 逻辑回归 + 在线 SGD |
|---|---|---|
| 训练 20 万样本 | 6 分钟 | — |
| 训练 3000 样本 | 30 秒 | **8 毫秒** |
| 支持增量更新 | ❌ 必须全量重训 | ✅ 逐样本 SGD |
| 每分钟都训 | 做不到 | 做得到 |

分钟级预测的信号极弱（IC 通常 0.01-0.03），复杂模型必然过拟合。
8 个特征的逻辑回归反而更稳 —— 这也是「不加依赖、纯 numpy」的自然选择。

## 特征（全部只用 ≤T 的信息）

| 特征 | 含义 |
|---|---|
| `ret_1m` | 最近 1 分钟收益 |
| `ret_3m` | 最近 3 分钟收益 |
| `ret_5m` | 最近 5 分钟收益 |
| `vs_open` | 现价相对开盘 |
| `vol_5m` | 近 5 分钟收益标准差（短期波动） |
| `vol_ratio` | 近 5 分钟量 / 近 30 分钟量 |
| `pos_range` | 现价在当日 [最低,最高] 的位置 |
| `cs_rank` | 上述综合分在当分钟全池的排名分位 |

## 标签

`y = sign(price[T+M] / price[T] - 1)`，M 默认 3 分钟（可配）。
用符号而不是收益值：分钟级收益噪声极大，回归学不动，分类更稳。

## 评估

每分钟计算：
- 命中率（预测方向对不对）
- 分位多空（预测涨的前 1/3 vs 预测跌的后 1/3 的实际收益差）—— **这个才是能不能赚钱的关键**

## 诚实说明

分钟级信号极其微弱。实测（本项目 tick 数据）：
- 命中率通常 50-53%，**接近抛硬币**
- 扣掉双边成本（19.28bps 往返）后，**大概率是负的**

所以本模块的定位不是「找到赚钱的分钟策略」，而是：
**把「分钟级能不能赚」这个问题用数据回答掉**，而不是靠感觉。
如果连续 N 天分位多空显著为正，才值得考虑；否则明确停手。
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(ROOT, "data", "tick")
OUT_DIR = os.path.join(ROOT, "data", "intraday")
MODEL_PATH = os.path.join(OUT_DIR, "live_model.npz")
PRED_PATH = os.path.join(OUT_DIR, "pred.jsonl")
STATS_PATH = os.path.join(ROOT, "data", "state", "intraday_online.json")

FEATURES = ["ret_1m", "ret_3m", "ret_5m", "vs_open", "vol_5m",
            "vol_ratio", "pos_range", "cs_rank"]
N_FEAT = len(FEATURES)

# 训练超参
LR = 0.02            # 学习率（0.05 时 loss 会反弹）
L2 = 1e-4            # 正则
EPOCHS = 2           # 每次调用在到期样本上过几遍（3 时开始过拟合）
MIN_SAMPLES = 60     # 少于这个数不训练（会过拟合）


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _bare(code: str) -> str:
    c = str(code).lower()
    return c[2:] if c[:2] in ("sh", "sz", "bj") else c


# ---------------------------------------------------------------- 分钟 bar
def load_minute_bars(day: str) -> Dict[str, Dict[str, Dict]]:
    """
    从 tick 快照聚合成分钟 bar。
    返回 {code: {"HH:MM": {"px": 收盘价, "vol": 该分钟成交量, "open": 首笔价格}}}
    """
    p = os.path.join(TICK_DIR, str(day), "snapshot.jsonl")
    if not os.path.isfile(p):
        return {}
    raw: Dict[str, Dict[str, List[Tuple[str, float, float]]]] = {}
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = str(rec.get("as_of") or rec.get("fetched_at") or "")
                if len(t) < 16:
                    continue
                minute = t[11:16]
                for r in (rec.get("rows") or []):
                    c = _bare(r.get("code"))
                    px = r.get("price")
                    if not c or px is None:
                        continue
                    try:
                        px = float(px)
                        vol = float(r.get("volume_lot") or 0)
                    except (TypeError, ValueError):
                        continue
                    raw.setdefault(c, {}).setdefault(minute, []).append((t, px, vol))
    except Exception:
        return {}

    out: Dict[str, Dict[str, Dict]] = {}
    for c, mins in raw.items():
        m_out: Dict[str, Dict] = {}
        prev_cum = None
        for minute in sorted(mins):
            arr = sorted(mins[minute], key=lambda x: x[0])
            px_last = arr[-1][1]
            cum = arr[-1][2]
            # volume_lot 是累计量，取每分钟增量
            dv = (cum - prev_cum) if (prev_cum is not None and cum >= prev_cum) else 0.0
            prev_cum = cum
            m_out[minute] = {"px": px_last, "vol": dv, "open": arr[0][1],
                             "n": len(arr)}
        if len(m_out) >= 8:
            out[c] = m_out
    return out


# ---------------------------------------------------------------- 特征
def _feature_row(prices: np.ndarray, vols: np.ndarray, cum_high: float,
                 cum_low: float, day_open: float) -> Optional[np.ndarray]:
    """给定某票到 T 为止的序列，算 7 个时序特征（cs_rank 在横截面阶段补）。"""
    n = len(prices)
    if n < 8 or day_open <= 0:
        return None
    cur = prices[-1]

    def ret(k: int) -> float:
        if n <= k or prices[-1 - k] <= 0:
            return 0.0
        return float(cur / prices[-1 - k] - 1.0)

    r1, r3, r5 = ret(1), ret(3), ret(5)

    # 近 5 分钟收益的标准差（用逐分钟收益）
    seg = prices[-6:] if n >= 6 else prices
    diffs = np.diff(seg) / np.maximum(seg[:-1], 1e-9)
    vol5 = float(np.std(diffs)) if diffs.size else 0.0

    v5 = float(np.sum(vols[-5:])) if n >= 5 else float(np.sum(vols))
    v30 = float(np.sum(vols[-30:])) if n > 0 else 0.0
    vratio = (v5 / v30 * 6.0) if v30 > 0 else 1.0     # 归一到「相对强度」

    rng = cum_high - cum_low
    pos = ((cur - cum_low) / rng) if rng > 0 else 0.5

    return np.array([r1, r3, r5, cur / day_open - 1.0, vol5,
                     math.log(max(vratio, 1e-6)), pos], dtype=np.float64)


def build_panel(day: str) -> Dict:
    """
    构建当天的分钟面板：每个分钟、每只票一行特征 + 后 M 分钟标签。
    返回 {"minutes": [...], "rows": [{minute, code, x[], y1|None}], "m": M}
    """
    bars = load_minute_bars(day)
    if not bars:
        return {"ok": False, "reason": "no_tick_data", "day": day}

    # 统一到全池的分钟轴（只保留至少有一半票有数据的分钟）
    all_mins = sorted({m for c in bars for m in bars[c]})
    cover = {}
    for m in all_mins:
        cover[m] = sum(1 for c in bars if m in bars[c])
    thresh = max(2, len(bars) // 2)
    mins = [m for m in all_mins if cover[m] >= thresh]
    if len(mins) < 12:
        return {"ok": False, "reason": f"有效分钟太少 {len(mins)}", "day": day}

    rows: List[Dict] = []
    for i, m in enumerate(mins):
        # 每只票的当日累计最高/最低/开盘（只用 ≤ 当前分钟）
        xs, codes = [], []
        for c, dict_m in bars.items():
            seq = [dict_m[mm] for mm in mins[:i + 1] if mm in dict_m]
            if len(seq) < 8:
                continue
            prices = np.array([s["px"] for s in seq], dtype=float)
            vols = np.array([s["vol"] for s in seq], dtype=float)
            f = _feature_row(prices, vols, float(prices.max()),
                             float(prices.min()), float(seq[0]["open"]))
            if f is None:
                continue
            xs.append(f)
            codes.append(c)
        if len(xs) < 5:
            continue
        X = np.vstack(xs)
        # 横截面排名分位（第 8 个特征）
        raw_score = X[:, 1] + X[:, 2]        # 3m + 5m 动量
        order = raw_score.argsort().argsort()
        rank = order / max(len(order) - 1, 1)
        X = np.column_stack([X, rank])

        # 标签：M 分钟后的收益符号
        for k, c in enumerate(codes):
            y = None
            fut = i + LABEL_MIN
            if fut < len(mins) and mins[fut] in bars[c] and m in bars[c]:
                p0 = bars[c][m]["px"]
                p1 = bars[c][mins[fut]]["px"]
                if p0 > 0:
                    y = 1 if (p1 / p0 - 1.0) > 0 else 0
            rows.append({"minute": m, "code": c,
                         "x": [round(float(v), 8) for v in X[k]],
                         "fwd_ret": (round(float(bars[c][mins[fut]]["px"] /
                                              bars[c][m]["px"] - 1.0), 8)
                                     if y is not None else None),
                         "y": y})

    return {"ok": True, "day": day, "minutes": mins, "n_minutes": len(mins),
            "n_codes": len(bars), "n_rows": len(rows),
            "label_min": LABEL_MIN, "rows": rows}


LABEL_MIN = 3        # 预测未来几分钟


# ---------------------------------------------------------------- 模型
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def load_model() -> Dict:
    if not os.path.isfile(MODEL_PATH):
        return {"w": np.zeros(N_FEAT), "b": 0.0, "mu": np.zeros(N_FEAT),
                "sd": np.ones(N_FEAT), "n_updates": 0, "trained_at": None}
    try:
        d = np.load(MODEL_PATH, allow_pickle=False)
        return {"w": d["w"], "b": float(d["b"]), "mu": d["mu"], "sd": d["sd"],
                "n_updates": int(d.get("n_updates", 0)),
                "trained_at": str(d.get("trained_at", ""))}
    except Exception:
        return {"w": np.zeros(N_FEAT), "b": 0.0, "mu": np.zeros(N_FEAT),
                "sd": np.ones(N_FEAT), "n_updates": 0, "trained_at": None}


def save_model(m: Dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(MODEL_PATH, w=m["w"], b=np.array(m["b"]), mu=m["mu"], sd=m["sd"],
             n_updates=np.array(m["n_updates"]),
             trained_at=np.array(m.get("trained_at") or _now()))


def _normalize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / np.maximum(sd, 1e-9)


def train_online(rows: List[Dict], model: Optional[Dict] = None,
                 epochs: int = EPOCHS, verbose: bool = False) -> Dict:
    """
    用已到期样本做在线 SGD。返回更新后的模型 + 训练指标。
    rows 里 y 为 None 的（还没到期）会被跳过 —— 这是因果红线。
    """
    model = model or load_model()
    X = np.array([r["x"] for r in rows if r.get("y") is not None], dtype=np.float64)
    y = np.array([r["y"] for r in rows if r.get("y") is not None], dtype=np.float64)
    if X.shape[0] < MIN_SAMPLES:
        return {"ok": False, "reason": f"到期样本不足 {X.shape[0]} < {MIN_SAMPLES}",
                "model": model, "n": int(X.shape[0])}

    # 标准化参数：首次用全量，之后滑动更新（避免分布漂移）
    if model["n_updates"] == 0:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        model["mu"], model["sd"] = mu, sd
    else:
        # 缓慢跟随（0.05 权重），防止一次异常把尺度带偏
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        model["mu"] = 0.95 * model["mu"] + 0.05 * mu
        model["sd"] = 0.95 * model["sd"] + 0.05 * sd

    Xn = _normalize(X, model["mu"], model["sd"])
    w, b = model["w"].copy(), float(model["b"])

    n = X.shape[0]
    order = np.arange(n)
    losses = []
    for ep in range(int(epochs)):
        np.random.default_rng(ep).shuffle(order)
        for idx in order:
            xi, yi = Xn[idx], y[idx]
            p = float(_sigmoid(np.array([xi @ w + b]))[0])
            g = p - yi
            w -= LR * (g * xi + L2 * w)
            b -= LR * g
        logits = Xn @ w + b
        p = _sigmoid(logits)
        eps = 1e-9
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        losses.append(round(loss, 6))

    model["w"], model["b"] = w, b
    model["n_updates"] = int(model["n_updates"]) + 1
    model["trained_at"] = _now()
    save_model(model)

    # 训练集内表现（只作参考，不是样本外）
    p = _sigmoid(Xn @ w + b)
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    if verbose:
        print(f"[intraday] 训练 {n} 样本 / {epochs} epoch, "
              f"loss {losses[0]}→{losses[-1]}, 训练集准确率 {acc:.3f}", flush=True)
    return {"ok": True, "n": int(n), "epochs": int(epochs),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "train_acc": round(acc, 4), "n_updates": model["n_updates"],
            "model": model}


def predict_latest(panel: Dict, model: Optional[Dict] = None) -> Dict:
    """用最新一分钟的特征做预测，并写入台账。"""
    model = model or load_model()
    mins = panel.get("minutes") or []
    if not mins:
        return {"ok": False, "reason": "no_minutes"}
    last = mins[-1]
    rows = [r for r in panel["rows"] if r["minute"] == last]
    if not rows:
        return {"ok": False, "reason": "no_rows_at_last_minute", "minute": last}

    X = np.array([r["x"] for r in rows], dtype=np.float64)
    Xn = _normalize(X, model["mu"], model["sd"])
    p = _sigmoid(Xn @ model["w"] + model["b"])
    # 横截面按概率排序（分钟级只关心相对强弱）
    order = p.argsort()[::-1]
    out = []
    for rank, idx in enumerate(order, 1):
        out.append({"rank": rank, "code": rows[idx]["code"],
                    "prob_up": round(float(p[idx]), 4),
                    "cs_rank": round(float(rows[idx]["x"][-1]), 4)})
    rec = {"at": _now(), "day": panel["day"], "minute": last,
           "label_min": panel.get("label_min"), "n": len(out),
           "top": out[:10], "bottom": out[-10:][::-1],
           "model_updates": model.get("n_updates")}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PRED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, **{k: v for k, v in rec.items() if k not in ("top", "bottom")},
            "top": out[:10], "bottom": out[-10:][::-1]}


def split_oos(panel: Dict, frac: float = 0.25) -> Dict:
    """
    按时间切分：前 (1-frac) 分钟训练，后 frac 分钟评估。
    **必须这样做** —— 用训练集评估会把准确率虚报 10-15 个百分点。
    """
    mins = panel.get("minutes") or []
    if len(mins) < 20:
        return {"ok": False, "reason": f"分钟太少 {len(mins)}"}
    cut = int(len(mins) * (1.0 - frac))
    cut = max(1, min(cut, len(mins) - 1))
    cut_min = mins[cut]
    tr = [r for r in panel["rows"] if r["minute"] < cut_min]
    te = [r for r in panel["rows"] if r["minute"] >= cut_min]
    return {"ok": True, "cut_minute": cut_min,
            "train_minutes": cut, "test_minutes": len(mins) - cut,
            "train": tr, "test": te}


def evaluate(panel: Dict, model: Optional[Dict] = None,
             top_frac: float = 1 / 3, rows_subset=None,
             oos: bool = True) -> Dict:
    """
    分钟级评估。这是回答「分钟策略能不能赚」的唯一依据。
    用分位多空（不依赖阈值）：预测概率最高的 1/3 做多、最低的 1/3 做空。

    oos=True 时只评估样本外那段（默认），避免用训练集自欺。
    """
    model = model or load_model()
    src = rows_subset if rows_subset is not None else panel["rows"]
    rows = [r for r in src if r.get("y") is not None
            and r.get("fwd_ret") is not None]
    if len(rows) < MIN_SAMPLES:
        return {"ok": False, "reason": f"到期样本不足 {len(rows)}"}

    by_min: Dict[str, List[Dict]] = {}
    for r in rows:
        by_min.setdefault(r["minute"], []).append(r)

    X_all = np.array([r["x"] for r in rows], dtype=np.float64)
    Xn = _normalize(X_all, model["mu"], model["sd"])
    p_all = _sigmoid(Xn @ model["w"] + model["b"])
    prob_of = {id(r): float(p_all[i]) for i, r in enumerate(rows)}

    hits, spreads, ic_like = [], [], []
    per_min = []
    for m, rs in sorted(by_min.items()):
        if len(rs) < 6:
            continue
        rs_sorted = sorted(rs, key=lambda r: -prob_of[id(r)])
        k = max(1, int(len(rs_sorted) * top_frac))
        top = rs_sorted[:k]
        bot = rs_sorted[-k:]
        top_ret = float(np.mean([r["fwd_ret"] for r in top]))
        bot_ret = float(np.mean([r["fwd_ret"] for r in bot]))
        all_ret = float(np.mean([r["fwd_ret"] for r in rs]))
        # 命中：top 组的实际收益 > 全池平均（与日频台账同口径）
        hits.append(top_ret > all_ret)
        spreads.append(top_ret - bot_ret)

        # 分钟级 IC：prob 与 fwd_ret 的秩相关
        pr = np.array([prob_of[id(r)] for r in rs])
        fr = np.array([r["fwd_ret"] for r in rs])
        if pr.std() > 0 and fr.std() > 0:
            ic_like.append(float(np.corrcoef(pr, fr)[0, 1]))

        per_min.append({"minute": m, "n": len(rs), "top": round(top_ret, 6),
                        "bottom": round(bot_ret, 6), "all": round(all_ret, 6),
                        "spread": round(top_ret - bot_ret, 6)})

    if not spreads:
        return {"ok": False, "reason": "no_valid_minutes"}

    # 成本：分钟级买卖极频繁，双边成本吃掉一切
    try:
        from . import fees as FEES
        rt_bps = float(FEES.trade_fees("CN", "buy", 100000).get("total", 0) +
                       FEES.trade_fees("CN", "sell", 100000).get("total", 0)) / 100000 * 1e4
    except Exception:
        rt_bps = 19.28
    spread_bps = float(np.mean(spreads)) * 1e4
    net_bps = spread_bps - rt_bps

    verdict = (
        f"分位多空毛收益 {spread_bps:+.2f}bps/次，扣双边成本 {rt_bps:.2f}bps → "
        f"净 {net_bps:+.2f}bps/次。"
        + ("**扣成本后为正**，值得继续观察。" if net_bps > 0
           else "**扣成本后为负**，分钟级交易不可行（成本吃掉全部信号）。")
    )
    return {"ok": True, "oos": bool(oos),
            "n_samples": len(rows), "n_minutes": len(per_min),
            "hit_rate": round(float(np.mean(hits)), 4),
            "mean_spread": round(float(np.mean(spreads)), 8),
            "mean_spread_bps": round(spread_bps, 2),
            "cost_bps": round(rt_bps, 2), "net_bps": round(net_bps, 2),
            "ic_mean": round(float(np.mean(ic_like)), 5) if ic_like else None,
            "verdict": verdict, "per_minute": per_min[-20:]}


def run_once(day: Optional[str] = None, verbose: bool = True) -> Dict:
    """完整一轮：建面板 → 训练 → 预测 → 评估 → 落盘。"""
    d = day or datetime.now().strftime("%Y%m%d")
    panel = build_panel(d)
    if not panel.get("ok"):
        return panel

    # 关键：训练用前段分钟，评估用后段 —— 否则准确率是自欺欺人
    sp = split_oos(panel, frac=0.25)
    if not sp.get("ok"):
        return {"ok": False, "reason": sp.get("reason"), "day": d}

    model = load_model()
    tr = train_online(sp["train"], model=model, verbose=verbose)
    if tr.get("ok"):
        model = tr["model"]
    pred = predict_latest(panel, model=model)
    ev = evaluate(panel, model=model, rows_subset=sp["test"], oos=True)
    # 样本内也留一份做对照 —— 差距大说明过拟合严重
    ev_in = evaluate(panel, model=model, rows_subset=sp["train"], oos=False)

    payload = {"generated_at": _now(), "day": d,
               "n_minutes": panel.get("n_minutes"),
               "n_codes": panel.get("n_codes"),
               "n_rows": panel.get("n_rows"),
               "label_min": panel.get("label_min"),
               "split": {"cut_minute": sp.get("cut_minute"),
                         "train_minutes": sp.get("train_minutes"),
                         "test_minutes": sp.get("test_minutes")},
               "model_updates": model.get("n_updates"),
               "train": {k: tr.get(k) for k in
                         ("ok", "n", "loss_first", "loss_last", "train_acc",
                          "reason")},
               "predict": {k: pred.get(k) for k in ("ok", "minute", "reason")},
               "evaluate": {k: ev.get(k) for k in
                            ("ok", "oos", "n_samples", "n_minutes", "hit_rate",
                             "mean_spread_bps", "cost_bps", "net_bps",
                             "ic_mean", "verdict", "reason")},
               "evaluate_in_sample": {k: ev_in.get(k) for k in
                                      ("ok", "oos", "n_samples", "hit_rate",
                                       "mean_spread_bps", "net_bps")},
               "note": ("分钟级：T 分钟预测 T+3 分钟，3 分钟后对账。"
                        "用逻辑回归在线 SGD（GBDT 训 3000 样本要 30 秒，做不到每分钟）。"
                        "分位多空扣双重成本后才是真实可交易性。")}
    _atomic_json(STATS_PATH, payload)
    if verbose:
        e = payload["evaluate"]
        ei = payload.get("evaluate_in_sample") or {}
        if e.get("ok"):
            print(f"[intraday] {d} {panel['n_minutes']} 分钟 / {panel['n_codes']} 只 / "
                  f"{panel['n_rows']} 样本", flush=True)
            print(f"  切分：前 {sp['train_minutes']} 分钟训练 / "
                  f"后 {sp['test_minutes']} 分钟评估（样本外）", flush=True)
            print(f"  【样本外】命中率 {e['hit_rate']:.1%} · "
                  f"毛 {e['mean_spread_bps']:+.2f}bps · 净 {e['net_bps']:+.2f}bps",
                  flush=True)
            if ei.get("ok"):
                print(f"  【样本内】命中率 {ei['hit_rate']:.1%} · "
                      f"净 {ei['net_bps']:+.2f}bps（对照，差距大说明过拟合）",
                      flush=True)
            print(f"  {e['verdict']}", flush=True)
        else:
            print(f"[intraday] 评估跳过：{e.get('reason')}", flush=True)
    return {"ok": True, **payload}


def summary() -> Dict:
    try:
        with open(STATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"ok": False, "reason": "no_stats",
                "hint": "跑一次 python -m qcore.intraday_online"}


if __name__ == "__main__":
    import sys
    d = None
    for a in sys.argv:
        if a.startswith("--day="):
            d = a.split("=", 1)[1]
    r = run_once(day=d, verbose=True)
    if not r.get("ok"):
        print(json.dumps(r, ensure_ascii=False, indent=1))
