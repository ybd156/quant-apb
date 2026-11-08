"""
多数据源接入与交叉验证 —— 不把鸡蛋放在一个篮子里。

## 为什么必须多源

单一数据源有三个致命风险，本项目已经全部踩过：
1. **口径错误**：东财 fqt=1 是减法复权，老股价格被压成负数（601088 最低 −9.55）。
2. **接口限流**：连续请求会被直接断连（实测 3 次后 RemoteDisconnected）。
3. **数据缺失**：美股 amount 在 2017-07 前恒为 0，看着有值其实是空。

所以设计成：**多源并取 → 交叉验证 → 健康度加权 → 自动切换**。

## 关键洞察：比价格，不如比收益率

不同源的复权基准完全不同（同一只 600519 在 2026-06-11：
东财后复权、新浪、腾讯 qfq 的绝对值各不相同，差 2% 以上）。
直接比价格会误判为"数据冲突"。

正确做法：**比日收益率**。只要收益率序列一致，复权基准的常数差异不影响任何下游计算
（我们的信号全部是相对量）。

## 健康度

每个源记录：成功次数、失败次数、平均延迟、最近错误。
健康度 = 成功率 × 延迟惩罚，作为选源优先级。
状态落盘 `data/state/source_health.json`，重启后不丢。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
os.makedirs(STATE_DIR, exist_ok=True)
HEALTH_PATH = os.path.join(STATE_DIR, "source_health.json")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://finance.sina.com.cn"}
_MIN_INTERVAL = 0.22
_last = [0.0]


def _throttle() -> None:
    wait = _MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def _get(url: str, headers: Optional[dict] = None, retries: int = 3,
         timeout: int = 25, encoding: str = "utf-8") -> str:
    last = None
    for i in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers=headers or UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode(encoding, errors="replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} :: {last}")


# ------------------------------------------------------------------ 适配器
def _em_secid(code: str) -> str:
    c = str(code).zfill(6)
    return f"1.{c}" if c[0] in "569" else f"0.{c}"


def fetch_eastmoney(code: str, adjust: int = 2) -> pd.DataFrame:
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={_em_secid(code)}&fields1=f1,f2,f3,f4,f5,f6"
           "&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt="
           f"{adjust}&beg=19900101&end=20500101&lmt=100000")
    d = json.loads(_get(url))
    rows = ((d.get("data") or {}).get("klines") or [])
    if not rows:
        raise RuntimeError("eastmoney empty")
    df = pd.DataFrame([r.split(",") for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume", "amount"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_sina(code: str, days: int = 1023) -> pd.DataFrame:
    c = str(code).zfill(6)
    sym = ("sh" if c[0] in "569" else "sz") + c
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
    txt = _get(url)
    data = json.loads(txt) if txt.strip().startswith("[") else []
    if not data:
        raise RuntimeError("sina empty")
    df = pd.DataFrame(data).rename(columns={"day": "date"})
    for c2 in ("open", "high", "low", "close", "volume"):
        df[c2] = pd.to_numeric(df[c2], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = np.nan
    return df[["date", "open", "close", "high", "low", "volume", "amount"]] \
        .sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_tencent(code: str, days: int = 1000) -> pd.DataFrame:
    c = str(code).zfill(6)
    sym = ("sh" if c[0] in "569" else "sz") + c
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={sym},day,,,{days},qfq")
    d = json.loads(_get(url))
    node = (d.get("data") or {}).get(sym) or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise RuntimeError("tencent empty")
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    for c2 in ("open", "close", "high", "low", "volume"):
        df[c2] = pd.to_numeric(df[c2], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = df["volume"] * 100.0        # 腾讯是手，统一乘 100 成股
    df["amount"] = np.nan
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


ADAPTERS: Dict[str, Callable[..., pd.DataFrame]] = {
    "eastmoney": fetch_eastmoney,
    "sina": fetch_sina,
    "tencent": fetch_tencent,
}


# ------------------------------------------------------------------ 健康度
def _load_health() -> Dict:
    if os.path.exists(HEALTH_PATH):
        try:
            with open(HEALTH_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_health(h: Dict) -> None:
    tmp = HEALTH_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=1)
    os.replace(tmp, HEALTH_PATH)


def _record(source: str, ok: bool, latency_ms: float, error: str = "") -> None:
    h = _load_health()
    s = h.setdefault(source, {"ok": 0, "fail": 0, "latency_ms": 0.0, "last_error": ""})
    if ok:
        s["ok"] += 1
        s["latency_ms"] = 0.8 * s["latency_ms"] + 0.2 * latency_ms if s["latency_ms"] else latency_ms
    else:
        s["fail"] += 1
        s["last_error"] = error[:200]
    _save_health(h)


def health() -> Dict[str, Dict]:
    h = _load_health()
    out = {}
    for k, v in h.items():
        n = v["ok"] + v["fail"]
        rate = v["ok"] / n if n else 0.5
        lat_pen = 1.0 / (1.0 + v["latency_ms"] / 1000.0)
        out[k] = {**v, "success_rate": round(rate, 4),
                  "score": round(rate * lat_pen, 4)}
    return out


# 优先级：历史最全的源在前。健康度只用于"把明显坏掉的源降级"，
# 不用于"把只给 1023 根的源提上来"——实测新浪只回 1023 根，
# 若纯按延迟排序会把 600519 的 6037 根历史砍掉 83%。
PRIORITY = ["eastmoney", "tencent", "sina"]


def ranked_sources(min_success: float = 0.5) -> List[str]:
    """固定优先级 + 健康度降级（成功率低于阈值的源排到最后）。"""
    h = health()
    good, bad = [], []
    for name in PRIORITY:
        score = h.get(name, {}).get("success_rate", 1.0)
        (bad if score < min_success else good).append(name)
    return good + bad


# ------------------------------------------------------------------ 交叉验证
def compare_returns(df_a: pd.DataFrame, df_b: pd.DataFrame, tol: float = 0.005,
                    exdiv_like: float = 0.02) -> Dict:
    """
    比**收益率**而不是价格（各源复权基准不同，价格差常数倍是正常的）。

    鲁棒判定：用中位数与 p95 而不是均值 —— 因为**除权日**的收益率天然不同
    （一个源做了复权调整、另一个没做），这类日子的差异不是数据错误。
    实测 600519 的"不一致日"全部落在每年 6 月下旬的分红除权日。
    """
    a = df_a.set_index("date")["close"].pct_change()
    b = df_b.set_index("date")["close"].pct_change()
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 30:
        return {"ok": False, "reason": "insufficient_overlap", "n": len(j)}
    diff = (j["a"] - j["b"]).abs()
    med = float(diff.median())
    p95 = float(diff.quantile(0.95))
    suspected = diff[diff > exdiv_like]
    clean = diff[diff <= exdiv_like]
    # 三档判定：一致 / 口径差异（可用但要标注）/ 不一致（不可用）
    if med <= 5e-4 and p95 <= tol:
        verdict = "consistent"
    elif med <= 2e-3 and p95 <= 1e-2:
        verdict = "basis_mismatch"      # 复权口径不同，但序列本身没错
    else:
        verdict = "inconsistent"

    return {
        "ok": verdict != "inconsistent",
        "verdict": verdict,
        "n": int(len(j)),
        "median_abs_diff": med,
        "p95_abs_diff": p95,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "clean_frac": float(len(clean) / len(diff)),
        "suspected_adjustment_days": [str(pd.Timestamp(d).date()) for d in suspected.index[:5]],
        "n_suspected": int(len(suspected)),
        "note": ("疑似除权/复权口径差异日（非数据错误）" if len(suspected) else "无明显口径差异"),
    }


@dataclass
class MultiSourceResult:
    code: str
    df: Optional[pd.DataFrame] = None
    source: str = ""
    ok: bool = False
    attempts: List[Dict] = field(default_factory=list)
    consensus: Optional[Dict] = None
    health_snapshot: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"code": self.code, "source": self.source, "ok": self.ok,
                "rows": 0 if self.df is None else int(len(self.df)),
                "attempts": self.attempts, "consensus": self.consensus,
                "health": self.health_snapshot}


def load_multi(code: str, market: str = "CN", min_rows: int = 200,
               validate: bool = True, prefer: Optional[List[str]] = None) -> MultiSourceResult:
    """
    多源取数：
      validate=True  → 取**两个**源，选历史更长的当主源，另一个做交叉验证
      validate=False → 按健康度顺序取第一个成功的（批量场景，省请求）

    选主源的原则是"历史最长"而不是"最快"——实测新浪只给 1023 根，
    若按健康度选源会把 600519 的 6037 根历史砍掉 83%，等于自废武功。
    """
    res = MultiSourceResult(code=str(code).zfill(6), health_snapshot=health())
    order = prefer or ranked_sources()
    got: List[Tuple[str, pd.DataFrame]] = []
    need = 2 if validate else 1

    for name in order:
        if len(got) >= need:
            break
        fn = ADAPTERS.get(name)
        if fn is None:
            continue
        t0 = time.perf_counter()
        try:
            df = fn(res.code)
            lat = (time.perf_counter() - t0) * 1000
            if df is None or len(df) < min_rows:
                _record(name, False, lat, f"rows={0 if df is None else len(df)}<{min_rows}")
                res.attempts.append({"source": name, "ok": False,
                                     "error": f"rows={0 if df is None else len(df)} < {min_rows}"})
                continue
            _record(name, True, lat)
            res.attempts.append({"source": name, "ok": True, "rows": int(len(df)),
                                 "latency_ms": round(lat, 1)})
            got.append((name, df))
        except Exception as e:  # noqa: BLE001
            _record(name, False, 0.0, f"{type(e).__name__}: {e}")
            res.attempts.append({"source": name, "ok": False,
                                 "error": f"{type(e).__name__}: {e}"[:160]})

    if not got:
        res.health_snapshot = health()
        return res

    got.sort(key=lambda x: -len(x[1]))          # 历史最长者为主源
    res.source, res.df = got[0][0], got[0][1]
    res.ok = True
    if len(got) > 1:
        res.consensus = compare_returns(res.df, got[1][1])
        res.consensus["primary"] = res.source
        res.consensus["secondary"] = got[1][0]
    res.health_snapshot = health()
    return res


def source_risk_report(codes: List[str], sample: int = 5) -> Dict:
    """抽样做多源一致性体检，给出"数据源风险"结论。"""
    rows = []
    for c in codes[:sample]:
        r = load_multi(c, validate=True)
        rows.append({"code": c, "source": r.source, "ok": r.ok,
                     "consensus": r.consensus,
                     "n_sources_tried": len(r.attempts)})
    ok_cons = [r for r in rows if r["consensus"] and r["consensus"].get("ok")]
    return {
        "sampled": len(rows),
        "consistent": len(ok_cons),
        "inconsistent": len([r for r in rows if r["consensus"]
                             and not r["consensus"].get("ok")]),
        "rows": rows,
        "health": health(),
        "verdict": ("多源一致，数据可信" if len(ok_cons) >= max(1, len(rows) - 1)
                    else "存在源间不一致，下游结论需标注不确定性"),
    }
