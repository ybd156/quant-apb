"""
滑点追踪 —— 回答唯一能验证回测可信度的问题：**回测假设用收盘价成交，实盘实际差多少？**

## 为什么这是"唯一诚实的验证"

回测里的成交价是假设出来的（本项目默认「信号日收盘价成交」）。
真实世界里，信号在收盘后才知道，成交最早发生在**次日开盘或盘中**，
中间隔着：隔夜跳空、开盘集合竞价冲击、盘中买卖价差、自己的成交量冲击。

这些项**没有历史数据可以回测**（我们没有留存的历史盘口），
所以只能靠**实盘对账**：

    信号价 ref_price（T 日收盘）  ──记录──▶  fills（真实成交价 price）

    滑点(bps) = (成交价 / 信号价 - 1) × 10000     买入
    滑点(bps) = (信号价 / 成交价 - 1) × 10000     卖出（反向，正数同样表示"成本变高"）

符号约定：**正 = 买入成本变高 / 卖出收入变少 = 对我们不利**。
这样 `mean_slippage_bps` 可以直接和 `costs.round_trip_bps()` 的模型假设对比：
实盘均值 > 模型单边假设 ⇒ 回测偏乐观（真实成本更高），反之偏保守。

## 数据落盘（只追加，不改历史）

    data/slippage/signals.jsonl   {"recorded_at","as_of","rows":[{code,ref_price,score}]}
    data/slippage/fills.jsonl     {"recorded_at","code","price","ts","source","side"}

用 jsonl 而不是一个 json：信号每天一条、成交每笔一条，天然是**事件流**，
追加写不会互相覆盖，也不会因为一条坏记录毁掉整个文件。

## 严格因果

配对只允许「信号时间 ≤ 成交时间」。用 `recorded_at` 做因果判据（墙钟时间），
`as_of` 只作为业务日期展示。绝不允许拿成交之后的信号去解释成交。

## 隔离测试

目录可用环境变量 `APB_SLIPPAGE_DIR` 覆盖（`__main__` 自检就是这么做的），
避免自检把假数据写进真实账本。
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_DIR = os.path.join(ROOT, "data", "slippage")
TICK_DIR = os.path.join(ROOT, "data", "tick")

# 回测里隐含的往返成本（bps）。优先从 costs 模块现算，取不到才用这个兜底值。
# 19.282 = costs.round_trip_bps("CN", 1e5) 的实测值（含佣金/印花税/过户/规费/冲击）。
ROUND_TRIP_FALLBACK_BPS = 19.282

SIDES = ("buy", "sell")


# ------------------------------------------------------------------ 基础设施
def _now() -> str:
    """全项目统一的时间口径：本地时区、秒级精度。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _dir() -> str:
    return os.environ.get("APB_SLIPPAGE_DIR") or DEF_DIR


def _p(name: str) -> str:
    return os.path.join(_dir(), name)


def _bare(code: str) -> str:
    c = str(code).strip().lower()
    return c[2:] if c[:2] in ("sh", "sz", "bj") else c.zfill(6)


def _qlib(code: str) -> str:
    c = str(code).strip().lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c
    c6 = c.zfill(6)
    if c6[0] in ("6", "9"):
        return "sh" + c6
    if c6[:2] in ("43", "83", "87", "92"):
        return "bj" + c6
    return "sz" + c6


def _clean(o):
    """递归把 NaN/Inf 转 None —— 写进 JSON 的裸 NaN 会让浏览器整页崩。"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, np.floating):
        v = float(o)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _atomic_write(path: str, text: str) -> None:
    """先写同目录 .tmp，fsync 后 os.replace —— 断电只会留下旧文件，不会留下半截。"""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        raise


def _append_jsonl(path: str, rec: Dict) -> None:
    """
    追加一行到 jsonl —— 但仍然是**原子**的。

    为什么不是 open(path,"a") 直接写：追加写不是原子的（进程被杀会留半行）。
    这里把「旧内容 + 新行」整体写 .tmp 再 replace。
    账本文件很小（每天几条到几十条），整体重写的代价可以忽略，
    换来的是「任何时刻读到的都是完整 JSONL」。
    """
    old = ""
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = f.read()
        except Exception:
            old = ""
    if old and not old.endswith("\n"):
        old += "\n"
    line = json.dumps(_clean(rec), ensure_ascii=False, allow_nan=False)
    _atomic_write(path, old + line + "\n")


def _read_jsonl(path: str) -> List[Dict]:
    """容错读：坏行跳过，不让一条烂记录毁掉整个账本。"""
    if not os.path.isfile(path):
        return []
    out: List[Dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    except Exception:
        return out
    return out


def _parse_ts(s) -> Optional[datetime]:
    """
    宽松解析时间。支持：
      "2026-11-06"                       → 当天 00:00（本地）
      "2026-11-06T14:30:00+08:00"        → 带时区
      "2026-11-06 14:30:00"              → 无时区，按本地时区补
      "2026-11-06T14:30"                 → 分钟精度
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(t)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                    "%Y%m%d", "%Y/%m/%d"):
            try:
                d = datetime.strptime(t, fmt)
                break
            except Exception:
                continue
        else:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return d


def _rid(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


# ------------------------------------------------------------------ 基准价
# 价格口径。这是本模块最容易出错的地方，所以显式记录、显式对拍：
#   "hfq" 后复权（qlib / 东财日 K）—— 和真实成交价差 10 倍以上（茅台 353 vs 1453）
#   "raw" 真实成交价（手工填 / 券商导出 / 秒级快照）
# 两种口径**不能相减**，否则算出来的"滑点"是 30000bps 的假数（实测踩过）。
BASIS_HFQ = "hfq"
BASIS_RAW = "raw"
SUSPECT_BPS = 3000.0     # 单笔滑点超过 30% 基本只能是口径错了（创业板 20% 涨跌停也只到 2000）


def _ref_price(code: str) -> Tuple[Optional[float], str]:
    """
    取「信号日收盘价」作为滑点基准，并**返回它的口径**。

    三级降级：
      1. qlib 本地社区数据（离线、最快）→ 后复权 hfq
      2. 东财日 K（后复权）           → 后复权 hfq
      3. 秒级快照的最新价             → 真实价 raw

    因为 1/2 是后复权，和真实成交价不在同一尺度，analyze() 会把
    「hfq 基准 + raw 成交」的配对判为口径不一致并剔除。
    要真正对账，请用 record_signal(..., prices={code: 真实收盘价}) 直接给原始价。
    """
    c = _bare(code)
    try:
        from qcore import qlib_data as QD
        if QD.available():
            df = QD.load_stock(_qlib(c))
            if df is not None and len(df) and "close" in df:
                v = float(df["close"].iloc[-1])
                if np.isfinite(v) and v > 0:
                    return v, BASIS_HFQ
    except Exception:
        pass
    try:
        from qcore import data as D
        df = D.load_kline(c)
        if df is not None and len(df) and "close" in df:
            v = float(df["close"].iloc[-1])
            if np.isfinite(v) and v > 0:
                return v, BASIS_HFQ
    except Exception:
        pass
    try:
        from qcore import tick as T
        p = T.snapshot([c])
        if p.get("ok"):
            for r in p.get("rows") or []:
                if _bare(r.get("code", "")) == c and r.get("price"):
                    return float(r["price"]), BASIS_RAW
    except Exception:
        pass
    return None, "none"


# ------------------------------------------------------------------ 记录
def record_signal(codes: Iterable[str], scores: Optional[Dict] = None,
                  as_of: Optional[str] = None,
                  prices: Optional[Dict[str, float]] = None) -> str:
    """
    记录一次「信号快照」：T 日收盘时的价格，作为后续计算滑点的基准。

    经济含义：这是回测里的那个"假设成交价"。把它钉在账本上，
    后面每一笔真实成交都能算出"假设 vs 现实"的差额。

    prices: 可选，{code: 真实收盘价}。**强烈建议传**——
            不传就自动取，而自动取到的是后复权价，无法和真实成交价对账。
            自己填真实收盘价（口径 raw），配对才是有效的。

    返回记录 id（形如 sig-20261108T214700-a1b2c3），便于日志追踪。
    """
    codes = [_bare(c) for c in (codes or [])]
    scores = scores or {}
    prices = {_bare(k): v for k, v in (prices or {}).items()}
    as_of = as_of or datetime.now().strftime("%Y-%m-%d")
    rows = []
    for c in codes:
        if c in prices and prices[c]:
            px, basis = float(prices[c]), BASIS_RAW
        else:
            px, basis = _ref_price(c)
        sc = scores.get(c)
        if sc is None:
            sc = scores.get(_qlib(c))
        rows.append({"code": c,
                     "ref_price": (round(float(px), 6) if px else None),
                     "basis": basis,
                     "score": (round(float(sc), 6) if isinstance(sc, (int, float, np.floating)) else None)})
    rec = {"recorded_at": _now(), "as_of": str(as_of), "rows": rows}
    _append_jsonl(_p("signals.jsonl"), rec)
    return _rid("sig")


def record_fill(code: str, price: float, ts: Optional[str] = None,
                source: str = "manual", side: str = "buy",
                basis: str = BASIS_RAW) -> str:
    """
    记录一次「实际成交」。

    side="buy"  买入：滑点 = (成交价/信号价 - 1) × 1e4
    side="sell" 卖出：滑点 = (信号价/成交价 - 1) × 1e4（反向，正数仍=不利）

    source 用来区分数据可信度：manual（手工填真实成交，最可信）
    / paper（模拟盘）/ broker（券商导出）。**只有 manual 和 broker 能用来评判回测。**

    basis 默认 raw（真实价）。若你填的是后复权价，请显式传 basis="hfq"，
    否则 analyze() 会因为口径不一致把它判成可疑配对。
    """
    side = str(side or "buy").strip().lower()
    if side not in SIDES:
        side = "buy"
    b = str(basis or BASIS_RAW).strip().lower()
    if b not in (BASIS_RAW, BASIS_HFQ):
        b = BASIS_RAW
    rec = {"recorded_at": _now(), "code": _bare(code),
           "price": (round(float(price), 6) if price else None),
           "ts": str(ts) if ts else _now(),
           "source": str(source or "manual"), "side": side, "basis": b}
    _append_jsonl(_p("fills.jsonl"), rec)
    return _rid("fill")


# ------------------------------------------------------------------ 滑点计算
def slip_bps(ref_price: float, price: float, side: str = "buy") -> float:
    """
    单笔滑点（bps）。**正 = 对我们不利**，买卖同号，方便直接和成本模型比。

    买入：(成交/信号 - 1)·1e4   —— 买贵了 10bp 就是 +10
    卖出：(信号/成交 - 1)·1e4   —— 卖便宜了 10bp 也是 +10
    """
    if not ref_price or not price or ref_price <= 0 or price <= 0:
        return float("nan")
    if str(side).lower() == "sell":
        return (ref_price / price - 1.0) * 1e4
    return (price / ref_price - 1.0) * 1e4


def _event_time(rec: Dict) -> Optional[datetime]:
    """
    信号的「信息可得时间」—— 因果判据用它，不用 recorded_at。

    为什么不用 recorded_at：批量补录信号时（例如今天一次性补录过去 30 天的信号），
    recorded_at 全是今天，但每条的 as_of 是各自的历史日期。
    那是**记账顺序**，不是**信息顺序**。真正决定"这笔成交能不能用这条信号解释"的，
    是参考价属于哪一天 —— 你只能在那个收盘之后成交。

    所以：as_of 是纯日期 → 取当天 23:59:59（收盘后）；带时刻 → 原样；
    都没有才退回 recorded_at。
    """
    raw = rec.get("as_of")
    t = _parse_ts(raw)
    if t is not None:
        s = str(raw or "").strip()
        if len(s) <= 10 and ":" not in s:      # 纯日期 → 视为该日收盘后
            t = t.replace(hour=23, minute=59, second=59)
        return t
    return _parse_ts(rec.get("recorded_at"))


def _pair_fills(days: int) -> Tuple[List[Dict], int, int, List[Dict]]:
    """
    把每笔成交配到「它之前最近的一次信号」。

    严格因果：只允许「信号的信息时间 ≤ 成交时间」，见 _event_time 的说明。
    口径守卫：后复权基准价 vs 真实成交价的配对会被剔除（见 SUSPECT_BPS 上方注释）。

    返回 (配对列表, 未配对成交数, 信号总数, 可疑配对列表)
    """
    sigs = _read_jsonl(_p("signals.jsonl"))
    fills = _read_jsonl(_p("fills.jsonl"))
    # 展开成 (时间, code, ref_price, score) 的事件表
    events: List[Dict] = []
    for s in sigs:
        t = _event_time(s)
        if t is None:
            continue
        for r in (s.get("rows") or []):
            px = r.get("ref_price")
            if not px:
                continue
            events.append({"t": t, "code": _bare(r.get("code", "")),
                           "ref_price": float(px), "score": r.get("score"),
                           "basis": str(r.get("basis") or "unknown"),
                           "as_of": s.get("as_of")})
    events.sort(key=lambda e: e["t"])

    cutoff = datetime.now().astimezone() - timedelta(days=int(max(days, 1)))
    pairs: List[Dict] = []
    suspect: List[Dict] = []
    unmatched = 0
    for f in fills:
        ft = _parse_ts(f.get("ts")) or _parse_ts(f.get("recorded_at"))
        if ft is None or ft < cutoff:
            continue
        code = _bare(f.get("code", ""))
        price = f.get("price")
        if not price:
            continue
        # 该股在该成交时刻之前、最近的一条信号
        cand = [e for e in events if e["code"] == code and e["t"] <= ft]
        if not cand:
            unmatched += 1
            continue
        e = cand[-1]
        bps = slip_bps(e["ref_price"], float(price), f.get("side", "buy"))
        if not np.isfinite(bps):
            unmatched += 1
            continue
        # ---- 口径守卫 ----
        # 1) 显式口径冲突：后复权基准 vs 真实成交（茅台 353 vs 1453 → 假 32000bps）
        # 2) 量级兜底：单笔滑点 >30%（创业板 20% 涨跌停也只到 2000bps）
        fbasis = str(f.get("basis") or BASIS_RAW)
        bad = ""
        if {e["basis"], fbasis} == {BASIS_HFQ, BASIS_RAW}:
            bad = f"口径不一致：信号 {e['basis']} / 成交 {fbasis}"
        elif abs(bps) > SUSPECT_BPS:
            bad = f"滑点 {bps:.0f}bps 超量级上限 {SUSPECT_BPS:.0f}bps，疑似口径错误"
        if bad:
            suspect.append({"code": code, "slippage_bps": round(float(bps), 2),
                            "ts": str(f.get("ts")), "reason": bad})
            continue
        pairs.append({"code": code, "ref_price": e["ref_price"],
                      "price": float(price), "slippage_bps": round(float(bps), 2),
                      "ts": str(f.get("ts")), "side": str(f.get("side", "buy")),
                      "source": f.get("source"), "signal_as_of": e.get("as_of"),
                      "hour": ft.astimezone().hour})
    return pairs, unmatched, len(sigs), suspect


def _round_trip_bps() -> float:
    """回测隐含的往返成本。现算优先，失败用兜底常量。"""
    try:
        from qcore import costs
        v = float(costs.round_trip_bps("CN", 1e5)["round_trip_bps"])
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return ROUND_TRIP_FALLBACK_BPS


def analyze(days: int = 30) -> Dict:
    """
    对比 signal 的 ref_price 与 fill 的 price，算滑点（bps）。

    经济含义：这是"回测可信度"的唯一硬证据。
      mean  ≈ 平均每次交易被市场多收了多少
      p90   ≈ 倒霉时候的成本（风控要按这个留余量，不是按 mean）
      worst ≈ 极端个案（通常是开盘一字板 / 流动性枯竭 / 大单冲击）

    返回的 verdict 直接和 costs 模型的往返成本对拍：
    实测单边均值 > 模型单边假设 ⇒ 回测偏乐观，实盘收益要往下修。
    """
    pairs, unmatched, n_sigs, suspect = _pair_fills(days)
    if not pairs:
        reason = "no_data" if (n_sigs == 0) else "no_pairs"
        out = {"ok": False, "reason": reason, "days": int(days),
               "n_signals": n_sigs, "n_unmatched_fills": unmatched,
               "n_suspect": len(suspect),
               "hint": "先 record_signal 记录信号价，再 record_fill 记录真实成交"}
        if suspect:
            # 有配对但全被口径守卫剔除 —— 这不是"没数据"，是"数据用错了"
            out["reason"] = "basis_mismatch"
            out["suspect"] = suspect[:5]
            out["hint"] = ("信号基准价是后复权（自动取得），成交价是真实价，两者不可相减。"
                           "请用 record_signal(..., prices={code: 真实收盘价}) 重新记录")
        return out

    bps = np.array([p["slippage_bps"] for p in pairs], dtype=float)
    bps = bps[np.isfinite(bps)]
    if bps.size == 0:
        return {"ok": False, "reason": "no_pairs", "days": int(days)}

    mean = float(np.mean(bps))
    median = float(np.median(bps))
    # p90 用线性插值分位数：小样本下比"取第 90 个百分位点"更稳
    p90 = float(np.percentile(bps, 90))

    worst = sorted(pairs, key=lambda p: -abs(p["slippage_bps"]))[:5]
    worst = [{"code": p["code"], "slippage_bps": p["slippage_bps"],
              "ts": p["ts"], "side": p["side"], "price": p["price"],
              "ref_price": p["ref_price"]} for p in worst]

    # 按成交小时分组 —— 开盘 30 分钟和尾盘的滑点结构完全不同（流动性 U 型）
    by_hour: List[Dict] = []
    for h in sorted({p["hour"] for p in pairs}):
        g = [p["slippage_bps"] for p in pairs if p["hour"] == h]
        by_hour.append({"hour": int(h), "mean_bps": round(float(np.mean(g)), 2),
                        "n": len(g)})

    rt = _round_trip_bps()
    one_way = rt / 2.0
    optimistic = mean > one_way
    verdict = (f"平均滑点 {mean:.1f}bps，回测假设的 {rt:.2f}bps 往返成本"
               f"（单边 {one_way:.2f}bps）{'偏乐观' if optimistic else '偏保守'}："
               f"实测单边 {mean:.1f}bps "
               f"{'高于' if optimistic else '低于'}模型假设 "
               f"{abs(mean - one_way):.1f}bps，"
               f"按 5 日调仓年化约 {abs(mean - one_way) * 2 * 50.4 / 100:.1f}%"
               f"{'额外拖累' if optimistic else '额外收益（勿当真，样本可能不足）'}")

    return {
        "ok": True,
        "days": int(days),
        "n_pairs": int(bps.size),
        "n_signals": n_sigs,
        "n_unmatched_fills": unmatched,
        "n_suspect": len(suspect),
        "mean_slippage_bps": round(mean, 2),
        "median_slippage_bps": round(median, 2),
        "p90_slippage_bps": round(p90, 2),
        "mean_abs_slippage_bps": round(float(np.mean(np.abs(bps))), 2),
        "pct_negative": round(float(np.mean(bps < 0)), 4),   # 负 = 成交反而更便宜
        "worst": worst,
        "by_hour": by_hour,
        "assumed_round_trip_bps": round(rt, 3),
        "verdict": verdict,
    }


# ------------------------------------------------------------------ 开盘冲击
def _open_30m_avg(codes: List[str], day: Optional[str] = None) -> Dict[str, float]:
    """
    从当天已落盘的 tick 快照里，算「开盘后 30 分钟成交均价」的近似。

    tick.save_tick 每 3 秒追加一行，所以 09:30~10:00 大约有 600 条快照。
    这里对每只股票取该区间的**价格均值**（等权，不是成交量加权 ——
    快照里有 volume_lot 但累计量的差分在跨源混写时不可靠，宁可用等权也不造假）。

    没有当天 tick 文件（未开盘 / 未轮询）→ 返回 {}，调用方退回"用现价代替"。
    """
    day = day or datetime.now().strftime("%Y%m%d")
    p = os.path.join(TICK_DIR, day, "snapshot.jsonl")
    if not os.path.isfile(p):
        return {}
    want = {_bare(c) for c in codes}
    acc: Dict[str, List[float]] = {}
    try:
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                for r in rec.get("rows") or []:
                    c = _bare(r.get("code", ""))
                    if c not in want:
                        continue
                    ts = r.get("ts")
                    if not ts:
                        continue
                    t = datetime.fromtimestamp(float(ts))
                    hm = (t.hour, t.minute)
                    if not ((9, 30) <= hm <= (10, 0)):
                        continue
                    px = r.get("price")
                    if px and float(px) > 0:
                        acc.setdefault(c, []).append(float(px))
    except Exception:
        return {}
    return {c: float(np.mean(v)) for c, v in acc.items() if v}


def estimate_open_impact(codes: Iterable[str], top_n: int = 20) -> Dict:
    """
    用当前 tick 快照对比昨收，估算如果现在下单相对昨收的偏离（bps）。

    把偏离拆成两段，因为它们的经济含义完全不同：
      隔夜跳空 gap_bps  = 开盘价 / 昨收 - 1
          → 你**拿不到**的那部分：信号在收盘后发出，跳空在你下单前已经发生。
            这是"收盘价成交"这个回测假设最大的谎言来源。
      开盘冲击 drift_bps = 现价 / 开盘价 - 1
          → 开盘后市场继续走的这段，你**部分能拿到**，但也可能被反噬。

    如果当天有 tick 落盘，再给出 avg30（开盘 30 分钟均价）与相对它的偏离，
    这是更接近"开盘后 30 分钟分批下单"的真实基准。

    返回按 |现价偏离| 降序的前 top_n 只。
    """
    codes = [_bare(c) for c in (codes or [])]
    if not codes:
        return {"ok": False, "reason": "empty"}
    try:
        from qcore import tick as T
        snap = T.snapshot(codes)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "tick_failed", "detail": str(e)[:80]}
    if not snap.get("ok"):
        return {"ok": False, "reason": snap.get("reason", "tick_failed")}

    avg30 = _open_30m_avg(codes)
    rows: List[Dict] = []
    for r in snap.get("rows") or []:
        c = _bare(r.get("code", ""))
        px = r.get("price")
        prev = r.get("prev_close")
        op = r.get("open")
        if not px or not prev or float(prev) <= 0:
            continue
        px = float(px)
        prev = float(prev)
        gap = (float(op) / prev - 1.0) * 1e4 if op else None
        drift = (px / float(op) - 1.0) * 1e4 if op else None
        now_bps = (px / prev - 1.0) * 1e4
        a30 = avg30.get(c)
        rows.append({
            "code": c, "name": r.get("name"),
            "prev_close": round(prev, 4), "open": (round(float(op), 4) if op else None),
            "price": round(px, 4),
            "gap_bps": (round(gap, 2) if gap is not None else None),
            "drift_bps": (round(drift, 2) if drift is not None else None),
            "now_bps": round(now_bps, 2),
            "avg30": (round(a30, 4) if a30 else None),
            # 相对"开盘 30 分钟均价"的偏离 = 现在下单比那半小时的均价贵/便宜多少
            "vs_avg30_bps": (round((px / a30 - 1.0) * 1e4, 2) if a30 else None),
        })
    if not rows:
        return {"ok": False, "reason": "no_data",
                "detail": "快照缺 prev_close/open，无法拆分跳空与冲击"}

    rows.sort(key=lambda x: -abs(x["now_bps"]))
    rows = rows[:max(int(top_n), 1)]
    nows = np.array([x["now_bps"] for x in rows], dtype=float)
    gaps = np.array([x["gap_bps"] for x in rows if x["gap_bps"] is not None], dtype=float)
    drifts = np.array([x["drift_bps"] for x in rows if x["drift_bps"] is not None], dtype=float)
    return {
        "ok": True,
        "as_of": snap.get("as_of"),
        "source": snap.get("source"),
        "stale": snap.get("stale"),
        "latency_s": snap.get("latency_s"),
        "n": len(rows),
        "has_30m_avg": bool(avg30),
        "rows": rows,
        "summary": {
            "mean_now_bps": round(float(np.mean(nows)), 2),
            "median_now_bps": round(float(np.median(nows)), 2),
            "mean_gap_bps": (round(float(np.mean(gaps)), 2) if gaps.size else None),
            "mean_drift_bps": (round(float(np.mean(drifts)), 2) if drifts.size else None),
            "n_up": int((nows > 0).sum()), "n_down": int((nows < 0).sum()),
        },
        "basis": "现在下单相对昨收的偏离；gap=隔夜跳空(拿不到)，"
                 "drift=开盘后冲击(部分能拿到)；has_30m_avg=True 时 vs_avg30_bps 更接近真实可执行价",
    }


# ------------------------------------------------------------------ 自检
if __name__ == "__main__":
    import shutil

    print("=" * 74)
    print("滑点模块自检 —— 3 条假信号 + 3 条假成交，与手算对拍")
    print("=" * 74)

    tmp = tempfile.mkdtemp(prefix="apb_slip_")
    old_env = os.environ.get("APB_SLIPPAGE_DIR")
    os.environ["APB_SLIPPAGE_DIR"] = tmp
    try:
        # ---- 造假信号：绕过网络，直接写 ref_price
        # basis="raw" 表示这是真实收盘价（和真实成交同口径，可比）
        def fake_signal(as_of: str, rows: List[Tuple[str, float, float]],
                        basis: str = BASIS_RAW) -> str:
            rec = {"recorded_at": _now(), "as_of": as_of,
                   "rows": [{"code": c, "ref_price": px, "score": sc, "basis": basis}
                            for c, px, sc in rows]}
            _append_jsonl(_p("signals.jsonl"), rec)
            return "sig-fake"

        fake_signal("2026-11-05", [("600519", 100.0, 0.9),
                                   ("000001", 10.0, 0.5),
                                   ("300750", 200.0, 0.7)])

        # ---- 造假成交（时间都在信号之后 → 因果成立）
        # 600519: 100.00 -> 100.05  买入  = (100.05/100.00-1)*1e4 = +5.00 bps
        # 000001:  10.00 ->  10.02  买入  = (10.02/10.00-1)*1e4 = +20.00 bps
        # 300750: 200.00 -> 199.00  买入  = (199.00/200.00-1)*1e4 = -50.00 bps
        record_fill("600519", 100.05, "2026-11-06T09:35:00+08:00", "manual", "buy")
        record_fill("000001", 10.02, "2026-11-06T10:05:00+08:00", "manual", "buy")
        record_fill("300750", 199.00, "2026-11-06T14:30:00+08:00", "manual", "buy")

        r = analyze(days=30)
        print("\n--- analyze(days=30) ---")
        print(json.dumps(r, ensure_ascii=False, indent=1))

        # ---- 手算对拍
        exp = {"600519": 5.00, "000001": 20.00, "300750": -50.00}
        got = {}
        # 重新读账本，逐笔核对（不信任 analyze 的聚合，直接看原始配对）
        pairs, unmatched, nsig, suspect = _pair_fills(30)
        for p in pairs:
            got[p["code"]] = p["slippage_bps"]
        exp_mean = round((5.0 + 20.0 - 50.0) / 3, 2)          # -8.33
        exp_median = 5.0                                        # 排序 [-50,5,20] 中位 = 5
        exp_p90 = 5.0 + 0.8 * (20.0 - 5.0)                       # 线性插值 = 17.0
        exp_worst = "300750"                                     # |−50| 最大

        checks = [
            ("每笔滑点", got, exp),
            ("n_pairs", r["n_pairs"], 3),
            ("mean", r["mean_slippage_bps"], exp_mean),
            ("median", r["median_slippage_bps"], exp_median),
            ("p90", r["p90_slippage_bps"], exp_p90),
            ("worst[0].code", r["worst"][0]["code"], exp_worst),
            ("by_hour 小时集合", sorted(x["hour"] for x in r["by_hour"]), [9, 10, 14]),
            ("by_hour[0]", (r["by_hour"][0]["hour"], r["by_hour"][0]["n"]), (9, 1)),
        ]
        print("\n--- 手算对拍 ---")
        ok_all = True
        for name, gotv, expv in checks:
            ok = (gotv == expv) if not isinstance(expv, float) else (abs(float(gotv) - expv) < 1e-6)
            ok_all &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<16} got={gotv}  expected={expv}")
        print(f"  unmatched_fills={unmatched}  n_signals={nsig}")

        # ---- 卖出方向符号检查（不写入账本，直接验函数）
        # 卖出：信号价 100，实际只卖到 99.5 → 少收 0.5% → 应为正（不利）
        # 注意买/卖不是严格对称：卖出口径 (信号/成交-1) 是倒数关系，
        # 同样的 0.5% 不利变动得 +50.00 vs +50.25bps，差 O(bps²/1e4)≈0.25bp，可忽略。
        s_buy = slip_bps(100.0, 100.5, "buy")
        s_sell = slip_bps(100.0, 99.5, "sell")
        exp_sell = (100.0 / 99.5 - 1.0) * 1e4
        ok_sign = abs(s_buy - 50.0) < 1e-9 and abs(s_sell - exp_sell) < 1e-9 and s_sell > 0
        ok_all &= ok_sign
        print(f"  [{'PASS' if ok_sign else 'FAIL'}] 买卖同号       "
              f"buy={s_buy:.2f} sell={s_sell:.2f} (都应为正 = 不利，"
              f"sell 期望 {exp_sell:.2f})")

        # ---- 缺数据降级
        os.environ["APB_SLIPPAGE_DIR"] = os.path.join(tmp, "empty")
        r2 = analyze(days=30)
        ok_empty = (r2.get("ok") is False and r2.get("reason") == "no_data")
        ok_all &= ok_empty
        print(f"  [{'PASS' if ok_empty else 'FAIL'}] 空目录降级     {r2}")

        # ---- 口径守卫：后复权基准(茅台 353) + 真实成交(1453) 必须被剔除
        os.environ["APB_SLIPPAGE_DIR"] = os.path.join(tmp, "basis")
        os.makedirs(os.environ["APB_SLIPPAGE_DIR"], exist_ok=True)
        fake_signal("2026-11-05", [("600519", 353.38, 0.9)], basis=BASIS_HFQ)
        record_fill("600519", 1453.0, "2026-11-06T10:00:00+08:00", "manual", "buy")
        r3 = analyze(days=30)
        ok_basis = (r3.get("ok") is False and r3.get("reason") == "basis_mismatch")
        ok_all &= ok_basis
        print(f"  [{'PASS' if ok_basis else 'FAIL'}] 口径守卫       "
              f"reason={r3.get('reason')} n_suspect={r3.get('n_suspect')} "
              f"（后复权 353 vs 真实 1453，若强行相减会得到 +32447bps 假滑点）")
        os.environ["APB_SLIPPAGE_DIR"] = tmp

        print("\n" + "=" * 74)
        print(f"自检结果：{'全部 PASS' if ok_all else '存在 FAIL'}")
        print("=" * 74)
        print("\n说明：estimate_open_impact() 需要实时快照（联网），自检不调用；")
        print("      用法：python -c \"from qcore import slippage as S; "
              "import json; print(json.dumps(S.estimate_open_impact(['600519','000001']), "
              "ensure_ascii=False, indent=1))\"")
    finally:
        if old_env is None:
            os.environ.pop("APB_SLIPPAGE_DIR", None)
        else:
            os.environ["APB_SLIPPAGE_DIR"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)
