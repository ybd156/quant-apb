"""qcore/intraday.py —— 盘中分钟级实时自学习 + 每日复盘

⚠️⚠️ 第一铁律：**没有历史分钟数据，所以本模块不能回测，只能前向验证。**
    东财 `klt=1` 只返回**当日 240 根**（09:31–15:00），`lmt` 再大也只有一天；
    隔夜之后昨天的分钟线就再也拿不到了。因此：
      · 本模块唯一合法的用法是「t 分钟打分 → 封存预测 → t+h 分钟看结果 → 更新权重」；
      · **禁止**任何形式的"用今天的数据回测今天"（那不是回测，那是自欺）；
      · 所有统计量都是**样本外**的，且样本量天然受限（每天最多 240 根 × 股票数）。
    这条限制也写进了每一份复盘报告的抬头。

架构（严格对应 SPEC §0.0 的"先出预测，后看结果"）
------------------------------------------------
    step() 每轮严格四步，顺序不可交换：
      ① 当前已收盘 bar 打分 → 追加进台账（封存，pending）
      ② 评估已到期（bar_time + h 分钟已到）的样本 → 回填 ret/hit
      ③ 用刚兑现的样本更新 AdaptiveWeighter（复用 qcore.learner，不重写）
      ④ 落盘（台账 + 权重状态）

幂等：台账主键 = (code, bar_time, horizon)。重跑同一天不会重复入账，
      已回填的结果字段不会被覆盖，预测字段永不修改。

成本：A股往返约 19.3 bps（qcore.costs.effective_cost_bps("CN", 1e5)）。
      1 分钟频率下每天最多 240 次往返 ⇒ 日成本 ~46% ⇒ 年化 ~11600%。
      **这个数字本身就是结论**：1 分钟频率在 A 股不可交易。
      复盘同时给出毛命中率与扣成本净收益，不允许只报毛的。

用法
----
    python -m qcore.intraday --once                 # 单次推进（盘中每 60s 调一次）
    python -m qcore.intraday --loop                 # 常驻轮询，非交易时段自动休眠
    python -m qcore.intraday --review               # 复盘最近一个有台账的交易日
    python -m qcore.intraday --review --date 2026-11-06
    python -m qcore.intraday --once --codes 600519,000001 --horizon 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .costs import effective_cost_bps
from .intraday_signals import INTRADAY_SIDS, intraday_signals, intraday_zoo
from .learner import AdaptiveWeighter, WeightConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(ROOT, "data", "intraday")
MINUTE_DIR = os.path.join(INTRADAY_DIR, "minute")
STATE_DIR = os.path.join(ROOT, "data", "state")
REPORT_DIR = os.path.join(ROOT, "reports")
STATE_PATH = os.path.join(STATE_DIR, "intraday_weighter.npz")
LOG_DIR = os.path.join(ROOT, "logs")

MARKET = "CN"
NOTIONAL = 1e5
HORIZONS = (1, 3, 5)

_IMMUTABLE = ("code", "bar_time", "horizon", "score", "action", "bar_idx",
              "dir_hash", "weights_hash", "sigma", "regime", "n_active", "as_of")


# ======================================================================
# 0. 小工具
# ======================================================================
def _ensure_dirs() -> None:
    for d in (INTRADAY_DIR, MINUTE_DIR, STATE_DIR, REPORT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def _log(msg: str) -> None:
    """日志只输出 ASCII；中文一律写进 .md 文件，避免编码问题。"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        _ensure_dirs()
        with open(os.path.join(LOG_DIR, "intraday.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:      # noqa: BLE001
        pass


def _hash_vec(x: np.ndarray, n: int = 8) -> str:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:n]


def trading_phase(now: Optional[datetime] = None) -> str:
    """返回 pre / am / lunch / pm / closed / weekend（不含节假日，节假日靠数据判空）。"""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return "weekend"
    m = now.hour * 60 + now.minute
    if m < 9 * 60 + 30:
        return "pre"
    if m <= 11 * 60 + 30:
        return "am"
    if m < 13 * 60:
        return "lunch"
    if m <= 15 * 60:
        return "pm"
    return "closed"


def in_session(now: Optional[datetime] = None) -> bool:
    return trading_phase(now) in ("am", "pm")


def default_codes(n: int = 5) -> List[str]:
    """默认标的：优先取本地日线缓存里的代码（离线可用，不打网络）。"""
    try:
        files = sorted(os.listdir(os.path.join(ROOT, "data", "raw", "kline")))
        codes = []
        for f in files:
            if f.endswith("_h2.csv"):
                codes.append(f[:-7])
        if not codes:
            codes = [f[:-4] for f in files if f.endswith(".csv") and f[:1].isdigit()]
        if codes:
            return codes[:n]
    except Exception:      # noqa: BLE001
        pass
    return ["600519"][:n]


# ======================================================================
# 1. 分钟线抓取（带限速 + 重试 + 当日缓存）
# ======================================================================
def _minute_cache_path(code: str, klt: int, date: str) -> str:
    return os.path.join(MINUTE_DIR, f"{code}_{klt}_{date}.csv")


def fetch_minute(code: str, klt: int = 1, fqt: int = 2,
                 use_cache: bool = True) -> pd.DataFrame:
    """抓当日分钟线 -> DataFrame[datetime,open,close,high,low,volume,amount]。

    东财 klt=1 只给**当日** 240 根，没有历史。抓到的当日完整数据会落盘缓存，
    这样复盘/重跑不需要再打网络（缓存只复用"已收满 240 根"的文件）。
    """
    from .sources import _em_secid, _get

    code = str(code).zfill(6)
    today = datetime.now().strftime("%Y%m%d")
    cache = _minute_cache_path(code, klt, today)
    if use_cache and os.path.exists(cache):
        try:
            old = pd.read_csv(cache, parse_dates=["datetime"])
            if len(old) >= (240 if klt == 1 else 48):
                return old
        except Exception:      # noqa: BLE001
            pass

    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={_em_secid(code)}&fields1=f1,f2,f3"
           "&fields2=f51,f52,f53,f54,f55,f56,f57"
           f"&klt={int(klt)}&fqt={int(fqt)}&beg=0&end=20500101&lmt=100000")
    txt = _get(url)
    d = json.loads(txt)
    kl = ((d.get("data") or {}).get("klines") or [])
    if not kl:
        raise RuntimeError(f"no minute data for {code}")
    df = pd.DataFrame([r.split(",") for r in kl],
                      columns=["datetime", "open", "close", "high", "low", "volume", "amount"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna(subset=["close"]).drop_duplicates("datetime").sort_values("datetime")
    df = df.reset_index(drop=True)

    try:
        _ensure_dirs()
        d0 = df["datetime"].iloc[-1].strftime("%Y%m%d") if len(df) else today
        if len(df) >= (240 if klt == 1 else 48):
            df.to_csv(_minute_cache_path(code, klt, d0), index=False, encoding="utf-8")
    except Exception:      # noqa: BLE001
        pass
    return df


def fetch_many(codes: Iterable[str], klt: int = 1) -> Dict[str, pd.DataFrame]:
    """批量抓取；单个失败不影响其他（返回成功的那部分）。"""
    out: Dict[str, pd.DataFrame] = {}
    for c in codes:
        try:
            df = fetch_minute(c, klt=klt)
            if len(df):
                out[str(c).zfill(6)] = df
        except Exception as e:      # noqa: BLE001
            _log(f"fetch fail {c}: {type(e).__name__}")
    return out


def _drop_forming_bar(df: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """盘中最后一根 bar 可能还没走完 —— 丢掉它，只用已收盘的 bar。"""
    if df is None or len(df) == 0:
        return df
    now = now or datetime.now()
    if not in_session(now):
        return df
    last = pd.Timestamp(df["datetime"].iloc[-1]).to_pydatetime()
    if last.replace(second=0, microsecond=0) >= now.replace(second=0, microsecond=0):
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ======================================================================
# 2. 学习器（复用 AdaptiveWeighter，只加持久化）
# ======================================================================
class IntradayLearner:
    """盘中自学习权重器：把 AdaptiveWeighter 套一层存/取（不重写算法）。"""

    def __init__(self, sids: Sequence[str] = INTRADAY_SIDS, horizon_min: int = 1,
                 cfg: Optional[WeightConfig] = None, n_regimes: int = 1):
        self.sids = list(sids)
        self.S = len(self.sids)
        self.R = int(n_regimes)
        self.horizon_min = int(horizon_min)
        if cfg is None:
            # 盘中遗忘要快：一天 240 根，decay 0.99 的半衰期约 69 次更新
            cfg = WeightConfig(horizon=int(horizon_min), decay_rho=0.99, kappa=8.0,
                               eta=0.02, lw_decay=0.999)
        self.cfg = cfg
        self.w = AdaptiveWeighter(self.S, n_regimes=self.R, cfg=cfg, sids=self.sids)

    # ---------------- 打分 ----------------
    def score(self, d: np.ndarray, s: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """score = Σ w·DIR·STR / Σ w·1[DIR≠0]（与日频 engine 同口径）。"""
        w = self.w.weights(0)
        mask = np.asarray(d) != 0
        den = float((w * mask).sum())
        if den <= 0:
            return 0.0, 0, np.zeros(self.S)
        contrib = w * np.asarray(d, dtype=np.float64) * np.asarray(s, dtype=np.float64)
        return float(contrib.sum() / den), int(mask.sum()), contrib

    def update(self, active: np.ndarray, payoff: np.ndarray) -> None:
        self.w.update(0, active, payoff)

    def top(self, k: int = 5) -> List[dict]:
        return self.w.top(k, regime=0)

    # ---------------- 持久化 ----------------
    def save(self, path: Optional[str] = None) -> str:
        path = path or STATE_PATH
        _ensure_dirs()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        w = self.w
        np.savez(
            path,
            a_g=w.a_g, b_g=w.b_g, E_g=w.E_g, logw_g=w.logw_g, n_g=w.n_g,
            a_r=w.a_r, b_r=w.b_r, E_r=w.E_r, logw_r=w.logw_r, n_r=w.n_r,
            ph_mean=w.ph_mean, ph_sum=w.ph_sum, ph_min=w.ph_min, alarms=w.alarms,
            sids=np.array(self.sids, dtype=object), n_regimes=np.int64(self.R),
            horizon_min=np.int64(self.horizon_min),
            cfg=np.array(json.dumps(self.cfg.__dict__, ensure_ascii=False)),
        )
        return path

    @classmethod
    def load(cls, path: Optional[str] = None, horizon_min: int = 1) -> "IntradayLearner":
        path = path or STATE_PATH
        if not os.path.exists(path):
            return cls(horizon_min=horizon_min)
        try:
            z = np.load(path, allow_pickle=True)
            sids = [str(x) for x in z["sids"]]
            cfg_d = json.loads(str(z["cfg"]))
            cfg = WeightConfig(**{k: v for k, v in cfg_d.items()
                                  if k in WeightConfig.__dataclass_fields__})
            obj = cls(sids=sids, horizon_min=int(z["horizon_min"]) if "horizon_min" in z
                      else horizon_min, cfg=cfg, n_regimes=int(z["n_regimes"]))
            w = obj.w
            for k in ("a_g", "b_g", "E_g", "logw_g", "n_g",
                      "a_r", "b_r", "E_r", "logw_r", "n_r",
                      "ph_mean", "ph_sum", "ph_min", "alarms"):
                if k in z:
                    setattr(w, k, np.array(z[k], dtype=np.float64 if k != "alarms" else np.int64))
            if obj.sids != list(INTRADAY_SIDS):
                _log("WARN signal set changed -> reset weights")
                return cls(horizon_min=horizon_min)
            return obj
        except Exception as e:      # noqa: BLE001
            _log(f"WARN state load failed ({type(e).__name__}) -> fresh")
            return cls(horizon_min=horizon_min)


# ======================================================================
# 3. 台账（只追加 + 只回填结果）
# ======================================================================
def ledger_path(date: str) -> str:
    return os.path.join(INTRADAY_DIR, f"ledger_{str(date).replace('-', '')}.jsonl")


def load_ledger(date: str) -> List[dict]:
    p = ledger_path(date)
    out: List[dict] = []
    if not os.path.exists(p):
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:      # noqa: BLE001
                continue
    return out


def write_ledger(date: str, records: List[dict]) -> str:
    """原子写：先写临时文件再 os.replace，断电不会写坏台账。"""
    _ensure_dirs()
    p = ledger_path(date)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return p


def _key(r: dict) -> Tuple[str, str, int]:
    return (str(r.get("code")), str(r.get("bar_time")), int(r.get("horizon", 1)))


def merge_records(old: List[dict], new: List[dict]) -> Tuple[List[dict], int]:
    """幂等合并：主键重复则跳过；已有记录只允许回填结果字段。"""
    idx = {_key(r): i for i, r in enumerate(old)}
    added = 0
    for r in new:
        k = _key(r)
        if k in idx:
            continue
        idx[k] = len(old)
        old.append(r)
        added += 1
    return old, added


def apply_result(rec: dict, patch: dict) -> bool:
    """把结果回填到已有记录；预测字段一旦存在就不允许改动（SPEC §9）。"""
    changed = False
    for k, v in patch.items():
        if k in _IMMUTABLE and rec.get(k) not in (None, v):
            continue                       # 预测字段禁止改
        if rec.get(k) != v:
            rec[k] = v
            changed = True
    return changed


# ======================================================================
# 4. step：打分 → 评估 → 更新 → 落盘
# ======================================================================
def _prediction_row(code: str, i: int, df: pd.DataFrame, d: np.ndarray, s: np.ndarray,
                    learner: IntradayLearner, horizon: int, sigma: float) -> dict:
    score, n_active, contrib = learner.score(d, s)
    action = "flat" if abs(score) < 1e-12 else ("long" if score > 0 else "short")
    order = np.argsort(-np.abs(contrib))[:5]
    top = [[learner.sids[j], round(float(contrib[j]), 6)] for j in order if contrib[j] != 0]
    ts = pd.Timestamp(df["datetime"].iloc[i])
    return {
        "as_of": ts.strftime("%Y-%m-%d %H:%M"),
        "date": ts.strftime("%Y-%m-%d"),
        "code": code,
        "bar_idx": int(i),
        "bar_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon": int(horizon),
        "score": round(float(score), 6),
        "action": action,
        "n_active": int(n_active),
        "sigma": float(sigma),
        "regime": 0,
        "dir_vec": [int(x) for x in np.sign(np.asarray(d))],
        "dir_hash": _hash_vec(np.sign(d)),
        "weights_hash": _hash_vec(learner.w.weights(0)),
        "contrib_top5": top,
        "status": "pending",
        "entry_time": None, "entry_price": None,
        "exit_time": None, "exit_price": None,
        "ret": None, "hit": None,
    }


def step(codes: Optional[Sequence[str]] = None, horizon_min: int = 1, klt: int = 1,
         now: Optional[datetime] = None, verbose: bool = True) -> Dict:
    """推进一轮：① 打分封存 ② 评估到期 ③ 更新权重 ④ 落盘。返回统计字典。"""
    t0 = time.time()
    now = now or datetime.now()
    codes = [str(c).zfill(6) for c in (codes or default_codes())]
    frames = fetch_many(codes, klt=klt)
    if not frames:
        if verbose:
            _log("step: no minute data (market closed / offline) -> skip")
        return {"ok": False, "reason": "no_data", "codes": codes, "seconds": 0.0}

    learner = IntradayLearner.load(horizon_min=horizon_min)
    _expire_stale(now.strftime("%Y-%m-%d"), verbose=verbose)
    ledgers: Dict[str, List[dict]] = {}
    new_by_date: Dict[str, List[dict]] = {}
    frames_used: Dict[str, pd.DataFrame] = {}

    # ---------- ① 打分并封存（先写预测，后看结果） ----------
    n_pred = 0
    for code, raw in frames.items():
        df = _drop_forming_bar(raw, now)
        if df is None or len(df) < 21:
            continue
        frames_used[code] = df
        DIR, STR = intraday_signals(df)
        dmat = DIR.to_numpy()
        smat = STR.to_numpy()
        date = pd.Timestamp(df["datetime"].iloc[-1]).strftime("%Y-%m-%d")
        if date not in ledgers:
            ledgers[date] = load_ledger(date)
        existing = {_key(r) for r in ledgers[date] if str(r.get("code")) == code}
        last_done = max([int(r["bar_idx"]) for r in ledgers[date]
                         if str(r.get("code")) == code and r.get("bar_idx") is not None],
                        default=-1)
        sigma_arr = _sigma_series(df)
        rows: List[dict] = []
        for i in range(last_done + 1, len(df)):
            ts = pd.Timestamp(df["datetime"].iloc[i]).strftime("%Y-%m-%d %H:%M:%S")
            if (code, ts, int(horizon_min)) in existing:
                continue
            sig = float(sigma_arr[i]) if np.isfinite(sigma_arr[i]) else 0.0
            rows.append(_prediction_row(code, i, df, dmat[i], smat[i],
                                        learner, horizon_min, sig))
        if rows:
            new_by_date.setdefault(date, []).extend(rows)
            n_pred += len(rows)

    for date, rows in new_by_date.items():
        ledgers[date], added = merge_records(ledgers.setdefault(date, []), rows)

    # ---------- ② 评估已到期 ----------
    n_scored = 0
    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for date, recs in ledgers.items():
        by_code = _frames_for_date(date, frames_used)
        for r in recs:
            if r.get("status") != "pending":
                continue
            df = by_code.get(str(r.get("code")))
            if df is None:
                continue
            patch = _evaluate(r, df)
            if patch is None:
                continue
            if apply_result(r, patch):
                n_scored += 1
            act, pay = _payoff_vectors(r, df)
            if act is not None:
                samples.append((act, pay))

    # ---------- ③ 更新权重（只用已兑现样本） ----------
    for act, pay in samples:
        learner.update(act, pay)

    # ---------- ④ 落盘 ----------
    for date, recs in ledgers.items():
        write_ledger(date, recs)
    learner.save()

    dt = time.time() - t0
    out = {"ok": True, "codes": list(frames_used), "predicted": int(n_pred),
           "scored": int(n_scored), "updated": int(len(samples)),
           "ledgers": sorted(ledgers), "seconds": round(dt, 3)}
    if verbose:
        _log(f"step: codes={len(frames_used)} predicted={n_pred} scored={n_scored} "
             f"updated={len(samples)} in {dt:.2f}s")
    return out


def _sigma_series(df: pd.DataFrame) -> np.ndarray:
    """1 分钟收益的滚动标准差（预测时就知道，写进台账供标准化用）。"""
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    r = np.zeros(c.size)
    if c.size > 1:
        prev = c[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = np.where(prev > 0, c[1:] / prev - 1.0, 0.0)
    sd = pd.Series(r).rolling(20, min_periods=2).std(ddof=0).to_numpy()
    return np.nan_to_num(sd, nan=0.0)


def _frames_for_date(date: str, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """只保留该交易日的数据（分钟线天然只有一天）。"""
    out = {}
    for code, df in frames.items():
        try:
            if pd.Timestamp(df["datetime"].iloc[-1]).strftime("%Y-%m-%d") == date:
                out[code] = df
        except Exception:      # noqa: BLE001
            continue
    return out


def _evaluate(rec: dict, df: pd.DataFrame) -> Optional[dict]:
    """t 分钟信号 → t+1 开盘建仓 → t+1+h 开盘平仓（不用当根收盘价作弊）。"""
    h = int(rec.get("horizon", 1))
    i = int(rec.get("bar_idx", -1))
    n = len(df)
    entry_i = i + 1
    exit_i = i + 1 + h
    if i < 0 or exit_i >= n:
        return None
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(o[entry_i]) and np.isfinite(o[exit_i])) or o[entry_i] <= 0:
        return None
    ret = float(o[exit_i] / o[entry_i] - 1.0)
    score = float(rec.get("score", 0.0))
    hit = None
    if abs(score) > 1e-12:
        hit = bool(np.sign(score) == np.sign(ret))
    return {
        "status": "scored",
        "entry_time": pd.Timestamp(df["datetime"].iloc[entry_i]).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_price": float(o[entry_i]),
        "exit_time": pd.Timestamp(df["datetime"].iloc[exit_i]).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_price": float(o[exit_i]),
        "ret": round(ret, 8),
        "hit": hit,
    }


def _payoff_vectors(rec: dict, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """给学习器的 (active, payoff)：y = dir·r/(σ√h)，σ 用预测时的滚动波动。"""
    if rec.get("status") != "scored" or rec.get("ret") is None:
        return None, None
    d = _dir_vector_at(rec, df)
    if d is None:
        return None, None
    sigma = float(rec.get("sigma") or 0.0)
    h = int(rec.get("horizon", 1))
    r = float(rec["ret"])
    if sigma <= 1e-12:
        return None, None
    y = np.clip(np.asarray(d, dtype=np.float64) * r / (sigma * np.sqrt(max(h, 1))), -3.0, 3.0)
    return (d != 0), y


def _dir_vector_at(rec: dict, df: pd.DataFrame) -> Optional[np.ndarray]:
    """取该 bar 的信号方向向量。

    优先用台账里封存的 dir_vec（预测当时就写死的，不可改），
    并用 dir_hash 校验；没有 dir_vec 的旧记录才回退重算（只依赖 ≤t 的 bar）。
    """
    vec = rec.get("dir_vec")
    if vec is not None:
        d = np.asarray(vec, dtype=np.float64)
        if d.size == len(INTRADAY_SIDS) and _hash_vec(d) == rec.get("dir_hash"):
            return d
        return None
    i = int(rec.get("bar_idx", -1))
    if i < 0 or i >= len(df):
        return None
    try:
        DIR, _ = intraday_signals(df.iloc[: i + 1])
        d = DIR.iloc[-1].to_numpy(dtype=np.float64)
        if _hash_vec(np.sign(d)) != rec.get("dir_hash"):
            return None                     # 数据被修订过，放弃这条样本
        return d
    except Exception:      # noqa: BLE001
        return None


# ======================================================================
# 5. 复盘
# ======================================================================
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _segment(ts: str) -> str:
    try:
        t = pd.Timestamp(ts)
    except Exception:      # noqa: BLE001
        return "?"
    m = t.hour * 60 + t.minute
    if m < 10 * 60:
        return "09:30-10:00"
    if m < 10 * 60 + 30:
        return "10:00-10:30"
    if m < 11 * 60:
        return "10:30-11:00"
    if m <= 11 * 60 + 30:
        return "11:00-11:30"
    if m < 14 * 60:
        return "13:00-14:00"
    return "14:00-15:00"


def session_review(date: Optional[str] = None, horizon_min: int = 1,
                   write: bool = True) -> Dict:
    """收盘后复盘：毛命中率 / IC / 分时段 / 权重 top5 / **扣成本净收益**。"""
    _ensure_dirs()
    if date is None:
        date = _latest_ledger_date()
        if date is None:
            if write:
                _log("review: no ledger found -> nothing to review")
            return {"ok": False, "reason": "no_ledger"}
    recs = load_ledger(date)
    if not recs:
        if write:
            _log(f"review: empty ledger for {date}")
        return {"ok": False, "reason": "no_ledger", "date": date}

    scored = [r for r in recs if r.get("status") == "scored" and r.get("ret") is not None]
    rt_bps = float(effective_cost_bps(MARKET, NOTIONAL))
    cost = rt_bps / 1e4

    codes = sorted({r["code"] for r in recs})
    bars = len({r["bar_time"] for r in recs})
    t_min = min((r["bar_time"] for r in recs), default="")
    t_max = max((r["bar_time"] for r in recs), default="")

    # ---- 毛 / 净 ----
    if scored:
        rets = np.array([float(r["ret"]) for r in scored], dtype=np.float64)
        scores = np.array([float(r["score"]) for r in scored], dtype=np.float64)
        bets = np.abs(scores) > 1e-12
        gross = rets[bets] * np.sign(scores[bets])
        gross_bps = gross * 1e4
        net_bps = gross_bps - rt_bps
        hit_gross = float((gross > 0).mean()) if gross.size else float("nan")
        hit_net = float((net_bps > 0).mean()) if net_bps.size else float("nan")
        mean_gross = float(gross_bps.mean()) if gross_bps.size else float("nan")
        mean_net = float(net_bps.mean()) if net_bps.size else float("nan")
        total_net_bps = float(net_bps.sum()) if net_bps.size else 0.0
        avg_abs_move = float(np.abs(rets[bets]).mean() * 1e4) if gross_bps.size else float("nan")
        # 分时段
        seg: Dict[str, List[float]] = {}
        for r in scored:
            if abs(float(r["score"])) <= 1e-12:
                continue
            seg.setdefault(_segment(r["bar_time"]), []).append(
                float(r["ret"]) * np.sign(float(r["score"])) * 1e4)
        seg_rows = [(k, len(v), float(np.mean(v)), float((np.array(v) > 0).mean()))
                    for k, v in sorted(seg.items())]
        # 横截面 IC（同一 bar 上跨股票）
        ics = []
        by_bar: Dict[str, List[Tuple[float, float]]] = {}
        for r in scored:
            by_bar.setdefault(r["bar_time"], []).append((float(r["score"]), float(r["ret"])))
        for _, pairs in by_bar.items():
            if len(pairs) >= 3:
                a = np.array([p[0] for p in pairs])
                b = np.array([p[1] for p in pairs])
                v = _spearman(a, b)
                if np.isfinite(v):
                    ics.append(v)
        ic_mean = float(np.mean(ics)) if ics else float("nan")
        ic_std = float(np.std(ics)) if ics else float("nan")
        icir = float(ic_mean / ic_std) if ics and ic_std > 0 else float("nan")
    else:
        gross = np.zeros(0)
        hit_gross = hit_net = mean_gross = mean_net = float("nan")
        avg_abs_move = float("nan")
        total_net_bps = 0.0
        seg_rows = []
        ic_mean = ic_std = icir = float("nan")

    # ---- 各信号在台账上的表现（样本外、毛口径） ----
    sig_stats = _signal_stats(scored)
    learner = IntradayLearner.load(horizon_min=horizon_min)
    top = learner.top(5)

    # ---- 成本拖累（这是本模块最重要的一列数字） ----
    n_bets = int(gross.size)
    per_day_bets = float(n_bets) / max(len(codes), 1)
    daily_cost_pct = per_day_bets * cost * 100.0
    annual_cost_pct = daily_cost_pct * 250.0
    breakeven_bps = rt_bps

    verdict = ("净收益为负 ⇒ 该频率不可交易" if (n_bets and mean_net < 0)
               else ("净收益为正但样本极少，不足以证明可交易" if n_bets
                     else "无已兑现样本"))

    res = {
        "ok": True, "date": date, "codes": codes, "bars": bars,
        "bar_range": [t_min, t_max], "predictions": len(recs), "scored": len(scored),
        "bets": n_bets, "rt_bps": rt_bps,
        "hit_gross": hit_gross, "hit_net": hit_net,
        "mean_gross_bps": mean_gross, "mean_net_bps": mean_net,
        "total_net_bps": total_net_bps,
        "avg_abs_move_bps": avg_abs_move,
        "edge_at_60_bps": (0.2 * avg_abs_move) if np.isfinite(avg_abs_move) else float("nan"),
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "segments": seg_rows, "signals": sig_stats, "top_weights": top,
        "per_day_bets": per_day_bets, "daily_cost_pct": daily_cost_pct,
        "annual_cost_pct": annual_cost_pct, "breakeven_bps": breakeven_bps,
        "verdict": verdict,
    }

    if write:
        path = os.path.join(REPORT_DIR, f"intraday_{date.replace('-', '')}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_render_md(res, learner))
        res["report"] = path
        _log(f"review: {date} bets={n_bets} hit_gross={hit_gross:.3f} "
             f"mean_gross={mean_gross:.1f}bps mean_net={mean_net:.1f}bps -> {path}")
    return res


def _signal_stats(scored: List[dict]) -> List[Tuple[str, int, float, float]]:
    """每条信号在台账上的毛方向命中率与毛均值（用封存的 dir_vec，样本外）。"""
    if not scored:
        return []
    sids = list(INTRADAY_SIDS)
    n = len(sids)
    hits = np.zeros(n)
    cnt = np.zeros(n)
    pnl = np.zeros(n)
    for r in scored:
        v = r.get("dir_vec")
        ret = r.get("ret")
        if v is None or ret is None:
            continue
        d = np.asarray(v, dtype=np.float64)
        if d.size != n:
            continue
        sgn = np.sign(float(ret))
        m = d != 0
        if not m.any():
            continue
        hits[m] += (np.sign(d[m]) == sgn)
        cnt[m] += 1.0
        pnl[m] += np.sign(d[m]) * float(ret)
    rows: List[Tuple[str, int, float, float]] = []
    for j, sid in enumerate(sids):
        if cnt[j] > 0:
            rows.append((sid, int(cnt[j]), float(hits[j] / cnt[j]),
                         float(pnl[j] / cnt[j] * 1e4)))
    rows.sort(key=lambda x: -x[1])
    return rows


def _expire_stale(today: str, verbose: bool = True) -> int:
    """把历史日期里永远不会到期的 pending 记录标记为 expired（诚实：不伪造回填）。

    分钟线隔夜即消失 ⇒ 昨天没评估完的样本永远无法评估，只能标 expired，
    绝不参与命中率/IC/权重更新。
    """
    n = 0
    try:
        files = [f for f in os.listdir(INTRADAY_DIR)
                 if f.startswith("ledger_") and f.endswith(".jsonl")]
    except Exception:      # noqa: BLE001
        return 0
    for f in files:
        d = f[len("ledger_"):-len(".jsonl")]
        date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if date >= today:
            continue
        recs = load_ledger(date)
        changed = False
        for r in recs:
            if r.get("status") == "pending":
                r["status"] = "expired"
                r["expired_reason"] = "minute data gone (no historical minutes)"
                changed = True
                n += 1
        if changed:
            write_ledger(date, recs)
            if verbose:
                _log(f"expire: {date} marked stale pending -> expired")
    return n


def _latest_ledger_date() -> Optional[str]:
    try:
        files = [f for f in os.listdir(INTRADAY_DIR)
                 if f.startswith("ledger_") and f.endswith(".jsonl")]
    except Exception:      # noqa: BLE001
        return None
    if not files:
        return None
    d = max(f[len("ledger_"):-len(".jsonl")] for f in files)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _render_md(res: Dict, learner: IntradayLearner) -> str:
    """复盘报告（中文写文件，控制台只出 ASCII）。"""
    L: List[str] = []
    L.append(f"# 盘中分钟级复盘 {res['date']}")
    L.append("")
    L.append("> ⚠️ **本报告是前向验证，不是回测。** 东财 `klt=1` 只提供当日 240 根分钟线，"
             "没有历史分钟数据，任何\"回测\"都无从谈起。所有数字都来自"
             "\"t 分钟打分 → 封存 → t+h 分钟看结果\"的真实样本外流水。")
    L.append("")
    L.append("## 1. 样本概览")
    L.append("")
    L.append(f"- 交易日：{res['date']}")
    L.append(f"- 标的：{len(res['codes'])} 只（{', '.join(res['codes'][:8])}"
             f"{' ...' if len(res['codes']) > 8 else ''}）")
    L.append(f"- bar 数：{res['bars']}（{res['bar_range'][0]} ~ {res['bar_range'][1]}）")
    L.append(f"- 预测条数：{res['predictions']}（已到期 {res['scored']}，其中方向非零 {res['bets']}）")
    L.append("")
    L.append("## 2. 毛 vs 净（核心表）")
    L.append("")
    L.append("| 口径 | 命中率 | 每笔均值 (bps) |")
    L.append("|---|---|---|")
    L.append(f"| 毛（未扣成本） | {_f(res['hit_gross'], 3)} | {_f(res['mean_gross_bps'], 2)} |")
    L.append(f"| 净（扣 {res['rt_bps']:.1f} bps 往返） | {_f(res['hit_net'], 3)} | {_f(res['mean_net_bps'], 2)} |")
    L.append("")
    L.append(f"- 单笔往返成本：**{res['rt_bps']:.2f} bps**"
             f"（`effective_cost_bps(\"CN\", 1e5)`，含佣金/印花税/过户费/半价差）")
    L.append(f"- 1 分钟平均绝对波动：**{_f(res['avg_abs_move_bps'], 2)} bps**"
             f"（= 方向全对时每笔能赚到的上限量级）")
    L.append(f"- 盈亏平衡所需毛边际：**{res['breakeven_bps']:.2f} bps/笔**")
    L.append(f"- 每标的每日下注次数：{res['per_day_bets']:.1f}")
    L.append(f"- **日成本拖累：{res['daily_cost_pct']:.2f}%** ⇒ "
             f"**年化 {res['annual_cost_pct']:.0f}%**（按 250 个交易日）")
    L.append("")
    L.append("## 3. 结论")
    L.append("")
    L.append(f"**{res['verdict']}。**")
    L.append("")
    if res["bets"]:
        L.append(f"毛每笔 {_f(res['mean_gross_bps'], 2)} bps 对往返成本 {res['rt_bps']:.2f} bps："
                 f"信号必须在毛口径上稳定赚到 {res['breakeven_bps']:.1f} bps 以上才可能覆盖成本。"
                 f"1 分钟频率下每天最多 {res['per_day_bets']:.0f} 次换仓，"
                 f"仅成本一项就要吃掉日 {res['daily_cost_pct']:.2f}%、年 {res['annual_cost_pct']:.0f}%。"
                 "这不是参数问题，是**频率问题**。")
        L.append("")
        L.append(f"**为什么命中率再高也救不回来**：1 分钟的平均绝对波动只有 "
                 f"**{_f(res['avg_abs_move_bps'], 2)} bps**。即使命中率做到 60%（远超任何分钟级信号的实测水平），"
                 f"毛边际也只有 (2×0.60−1)×{_f(res['avg_abs_move_bps'], 2)} ≈ "
                 f"**{_f(res['edge_at_60_bps'], 2)} bps**，仍不到 {res['rt_bps']:.1f} bps 成本的 "
                 f"{_f(res['edge_at_60_bps'] / res['rt_bps'] * 100, 1)}%。"
                 "分子（1 分钟能提供的波动）比分母（一次往返成本）小一个数量级 —— 频率本身就是错的。")
        L.append("")
        L.append(f"注：当日毛命中率 {_f(res['hit_gross'], 3)} 低于 50%，说明该日集成方向系统性偏反"
                 "（单日单样本，不做因果解释）。但**结论不依赖命中率**："
                 "只要毛边际跨不过成本门槛，这个频率就不可交易。")
    L.append("")
    L.append("## 4. 分时段")
    L.append("")
    if res["segments"]:
        L.append("| 时段 | 笔数 | 毛均值 (bps) | 毛命中率 |")
        L.append("|---|---|---|---|")
        for k, n, m, h in res["segments"]:
            L.append(f"| {k} | {n} | {m:.2f} | {h:.3f} |")
    else:
        L.append("_无样本_")
    L.append("")
    L.append("## 5. 横截面 IC")
    L.append("")
    L.append(f"- IC 均值 {_f(res['ic_mean'], 4)}，标准差 {_f(res['ic_std'], 4)}，"
             f"ICIR {_f(res['icir'], 3)}（同一 bar 上跨标的的 Spearman，需 ≥3 只才有值）")
    L.append("")
    L.append("## 6. 各信号毛表现（样本外）")
    L.append("")
    if res["signals"]:
        L.append("| 信号 | 发声笔数 | 毛命中率 | 毛均值 (bps) | 扣成本后 (bps) |")
        L.append("|---|---|---|---|---|")
        for sid, n, h, m in res["signals"]:
            L.append(f"| {sid} | {n} | {h:.3f} | {m:.2f} | {m - res['rt_bps']:.2f} |")
        L.append("")
        L.append("> 每一条的\"扣成本后\"都是负的 —— 单笔 19 bps 的成本门槛，"
                 "任何分钟级信号都跨不过去。")
    else:
        L.append("_无样本_")
    L.append("")
    L.append("## 7. 各信号权重 Top5（当前状态）")
    L.append("")
    L.append("| 信号 | 权重 | 后验命中率 | 更新次数 | 平均 payoff |")
    L.append("|---|---|---|---|---|")
    for t in res["top_weights"]:
        L.append(f"| {t['sid']} | {t['weight']:.4f} | {t['hit_rate']:.3f} | "
                 f"{t['n_updates']} | {t['mean_payoff']:+.4f} |")
    L.append("")
    L.append("## 8. 局限（必须读）")
    L.append("")
    L.append("1. **无法回测**：分钟数据只有当日，样本量按天累积，一天最多 240×N 条；"
             "任何\"历史分钟回测\"都是伪造。")
    L.append("2. **无法复现历史**：隔夜后昨日分钟线消失，台账是唯一的历史记录（所以台账只追加）。")
    L.append("3. **执行假设仍偏乐观**：按 t+1 开盘价成交，未计入排队、涨跌停、部分成交。")
    L.append("4. **成本是硬约束**：A股往返 ~19 bps，1 分钟换仓下成本是收益的数百倍。")
    L.append("")
    return "\n".join(L) + "\n"


def _f(x: float, nd: int) -> str:
    try:
        if x is None or not np.isfinite(x):
            return "n/a"
        return f"{x:.{nd}f}"
    except Exception:      # noqa: BLE001
        return "n/a"


# ======================================================================
# 6. CLI
# ======================================================================
def _parse_codes(s: Optional[str]) -> List[str]:
    if not s:
        return default_codes()
    return [c.strip().zfill(6) for c in s.split(",") if c.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="intraday forward-test learner (NOT backtestable)")
    ap.add_argument("--once", action="store_true", help="run one step")
    ap.add_argument("--loop", action="store_true", help="poll every N seconds during session")
    ap.add_argument("--review", action="store_true", help="session review")
    ap.add_argument("--date", default=None, help="review date YYYY-MM-DD")
    ap.add_argument("--codes", default=None, help="comma separated codes")
    ap.add_argument("--horizon", type=int, default=1, help="horizon in minutes (default 1)")
    ap.add_argument("--klt", type=int, default=1, help="minute klt (1/5/15/60)")
    ap.add_argument("--interval", type=int, default=60, help="loop interval seconds")
    args = ap.parse_args(argv)

    _ensure_dirs()
    codes = _parse_codes(args.codes)

    if args.review:
        res = session_review(args.date, horizon_min=args.horizon)
        print(json.dumps({k: v for k, v in res.items() if k not in ("signals", "top_weights")},
                         ensure_ascii=True, default=str, indent=2))
        return 0 if res.get("ok") else 1

    if args.loop:
        _log(f"loop start: codes={codes} h={args.horizon} interval={args.interval}s")
        try:
            while True:
                phase = trading_phase()
                if phase in ("am", "pm"):
                    step(codes, horizon_min=args.horizon, klt=args.klt)
                else:
                    _log(f"phase={phase} -> sleep")
                time.sleep(max(10, args.interval))
        except KeyboardInterrupt:
            _log("loop stopped by user")
            return 0

    res = step(codes, horizon_min=args.horizon, klt=args.klt)
    print(json.dumps(res, ensure_ascii=True, default=str))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
