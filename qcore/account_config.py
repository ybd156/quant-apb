"""
模拟账户配置 —— 按市场分开管理。

## 为什么必须分开

A 股、美股、港股的资金、成本、最小交易单位、涨跌停规则完全不同：

| 市场 | 往返成本 | 最小单位 | 涨跌停 | 典型本金 |
|---|---|---|---|---|
| CN | 19.28 bps | 100 股 | ±10%/20% | 100 万 CNY |
| US | 3.28 bps | 1 股 | 无 | 10 万 USD |
| HK | 38.17 bps | **每手不同**（100/500/1000…） | 无 | 50 万 HKD |

用一套配置管三个市场，等于把港股 38bps 的成本套到美股上——必错。

## 结构

```json
{
  "default_market": "CN",
  "markets": {
    "CN": {"initial_capital": 5000.0, "top_k": 5, "max_weight": 0.35,
           "cost_bps": 19.28, "hold": 20, "currency": "CNY"},
    "US": {"initial_capital": 500.0,  "top_k": 5, "max_weight": 0.35,
           "cost_bps": 3.28,  "hold": 20, "currency": "USD"},
    "HK": {"initial_capital": 5000.0, "top_k": 5, "max_weight": 0.35,
           "cost_bps": 38.17, "hold": 20, "currency": "HKD"}
  },
  "updated_at": "..."
}
```

## 向后兼容

老文件是扁平结构（`initial_capital` 在顶层），读取时自动视为 `markets.CN`，
不会因为升级丢配置。

## 校验

每个市场独立校验，越界直接拒绝，不做静默纠正——悄悄改用户的钱数不可接受。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
PATH = os.path.join(STATE, "account.json")

MARKETS = ("CN", "US", "HK")

# 各市场的默认值（成本来自 qcore/costs.py 的实测值）
# 默认起始资金：小资金模式
#   A股 5000 元、美股 500 美元、港股 5000 港币
# 小资金下 top_k 不能太大 —— 5000 元分 20 只，每只 250 元，
# 很多股票一手（100 股）就超了。所以默认 top_k 调小到 5，
# 由 qcore/lot.py 的可行性检查兜底（买不起的会被跳过）。
# 每个周期可以单独设本金：不同周期的最优资金规模不一样。
#
# **不要**把 horizon_capital 放进 DEFAULTS —— 一旦预填，
# capital_for 就永远能取到值，用户显式设的 initial_capital 从此失效
# （实测：initial_capital=800000 + 预填全 5000 -> 实际用 5000）。
# 空 dict / 缺失 = 未设置，回落到 initial_capital。
#   T+1 换手极快，本金大了成本吃光收益；15 天可以放更多。
DEFAULT_HORIZON_CAPITAL = {1: 5_000.0, 3: 5_000.0, 5: 5_000.0,
                           10: 5_000.0, 15: 5_000.0}

DEFAULTS: Dict[str, Dict] = {
    "CN": {"initial_capital": 5_000.0, "top_k": 5, "max_weight": 0.35,
           "cost_bps": 19.28, "hold": 20, "currency": "CNY"},
    "US": {"initial_capital": 500.0, "top_k": 5, "max_weight": 0.35,
           "cost_bps": 3.28, "hold": 20, "currency": "USD"},
    "HK": {"initial_capital": 5_000.0, "top_k": 5, "max_weight": 0.35,
           "cost_bps": 38.17, "hold": 20, "currency": "HKD"},
}

# 字段 -> (下限, 上限)。initial_capital 按市场另设（见 CAPITAL_LIMITS）
LIMITS: Dict[str, Tuple[float, float]] = {
    "top_k": (1, 100),
    "max_weight": (0.0001, 1.0),
    "cost_bps": (0.0, 500.0),
    "hold": (1, 120),
}
# 下限放到 1000 元：小资金用户（2000/5000/8000）也要能设。
# 但要注意：本金低于约 34 万时，20 只推荐里有票买不起一手（见 qcore/lot.py），
# 系统会自动降级为「集中持有买得起的票」，不会假装分散。
CAPITAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "CN": (1_000.0, 1_000_000_000.0),
    "US": (100.0, 100_000_000.0),
    "HK": (1_000.0, 1_000_000_000.0),
}
INT_FIELDS = {"top_k", "hold"}

# 每市场的**成本物理下限**（bps，往返）。低于此值不是"激进假设"，是不可能。
#   CN：法定+佣金下限 = 3.141(买) + 8.141(卖) = 11.28 bps，还没算价差。
#   实测事故：account.json 曾出现 cost_bps=10.0，于是同一次 pipeline 里
#   回测用 19.28bps、模拟盘用 10.0bps —— 同一次输出两个成本基准。
#   港股双边 0.1% 印花税，下限更高；美股无印花税，给 0。
COST_FLOOR_BPS: Dict[str, float] = {"CN": 11.28, "US": 0.0, "HK": 11.0}


def _blank() -> Dict:
    return {"default_market": "CN",
            "markets": {m: dict(v) for m, v in DEFAULTS.items()},
            "updated_at": None}


def _migrate(raw: Dict) -> Dict:
    """把老扁平结构升级成按市场结构。"""
    if isinstance(raw.get("markets"), dict):
        out = _blank()
        for m in MARKETS:
            if isinstance(raw["markets"].get(m), dict):
                src = raw["markets"][m]
                out["markets"][m].update(
                    {k: v for k, v in src.items() if k in DEFAULTS[m]})
                # horizon_capital 不在 DEFAULTS 里（见上面的注释：故意不预填），
                # 所以旧代码这一行会把用户设的「各周期本金」**整段丢掉** ——
                # 实测：设了 h=5 20 万，读回来 `cfg["horizon_capital"]` 是 None，
                # `capital_for()` 永远回落到 initial_capital，设置形同虚设。
                hc = src.get("horizon_capital")
                if isinstance(hc, dict) and hc:
                    out["markets"][m]["horizon_capital"] = {
                        str(int(k)): v for k, v in hc.items()
                        if str(k).lstrip("-").isdigit()}
        dm = raw.get("default_market")
        if dm in MARKETS:
            out["default_market"] = dm
        out["updated_at"] = raw.get("updated_at")
        return out
    # 老格式：顶层就是 CN 的配置
    if any(k in raw for k in ("initial_capital", "top_k", "max_weight")):
        out = _blank()
        for k in ("initial_capital", "top_k", "max_weight", "cost_bps", "hold"):
            if k in raw:
                out["markets"]["CN"][k] = raw[k]
        out["updated_at"] = raw.get("updated_at")
        return out
    return _blank()


def load(market: Optional[str] = None) -> Dict:
    """
    读配置。market 给定时返回该市场的扁平配置（含 market/currency）；
    否则返回完整结构（default_market + markets）。
    文件缺失或损坏时回退默认值，不抛异常。
    """
    try:
        with open(PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}
    cfg = _migrate(raw if isinstance(raw, dict) else {})
    # 逐字段校验，越界的丢弃回默认
    for m in MARKETS:
        cur = cfg["markets"].setdefault(m, dict(DEFAULTS[m]))
        for k, (lo, hi) in LIMITS.items():
            v = cur.get(k)
            if not (isinstance(v, (int, float)) and lo <= float(v) <= hi):
                cur[k] = DEFAULTS[m][k]
            elif k in INT_FIELDS:
                cur[k] = int(v)
            else:
                cur[k] = float(v)
        clo, chi = CAPITAL_LIMITS[m]
        v = cur.get("initial_capital")
        if not (isinstance(v, (int, float)) and clo <= float(v) <= chi):
            cur["initial_capital"] = DEFAULTS[m]["initial_capital"]
        else:
            cur["initial_capital"] = float(v)
        cur["currency"] = DEFAULTS[m]["currency"]
        # 成本物理下限：低于法定下限直接回落到默认值，并留下审计痕迹。
        # 静默放行一个不可能的 10bps，会让回测与模拟盘站在两个基准上。
        floor = float(COST_FLOOR_BPS.get(m, 0.0))
        if float(cur.get("cost_bps") or 0.0) < floor:
            cur["cost_bps_clamped_from"] = float(cur.get("cost_bps") or 0.0)
            cur["cost_bps"] = float(DEFAULTS[m]["cost_bps"])
            cur["cost_bps_floor"] = floor

    if market is None:
        return cfg
    m = str(market).upper()
    if m not in MARKETS:
        m = cfg["default_market"]
    return {"market": m, **cfg["markets"][m], "updated_at": cfg.get("updated_at")}


def validate(payload: Dict, market: str) -> Tuple[Optional[Dict], Optional[str]]:
    """校验单个市场的提交。只接受白名单字段，越界返回明确错误。"""
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"
    m = str(market).upper()
    if m not in MARKETS:
        return None, f"market 必须是 {', '.join(MARKETS)} 之一，收到 {market}"
    clean: Dict = {}
    clo, chi = CAPITAL_LIMITS[m]
    if "initial_capital" in payload:
        v = payload["initial_capital"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, "initial_capital 必须是数字"
        v = float(v)
        if not (clo <= v <= chi):
            return None, f"initial_capital 必须在 [{clo}, {chi}] 之间，收到 {v}"
        clean["initial_capital"] = v
    for k, (lo, hi) in LIMITS.items():
        if k not in payload:
            continue
        v = payload[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, f"{k} 必须是数字"
        v = float(v)
        if not (lo <= v <= hi):
            return None, f"{k} 必须在 [{lo}, {hi}] 之间，收到 {v}"
        clean[k] = int(v) if k in INT_FIELDS else v
    # 每个周期单独的本金：{"horizon_capital": {"1": 5000, "5": 20000}}
    if "horizon_capital" in payload:
        hc = payload["horizon_capital"]
        if not isinstance(hc, dict):
            return None, "horizon_capital 必须是对象，如 {\"1\": 5000}"
        out = {}
        for k, v in hc.items():
            try:
                hh = int(k)
            except (TypeError, ValueError):
                return None, f"horizon_capital 的键必须是周期数字，收到 {k}"
            if hh not in (1, 3, 5, 10, 15):
                return None, f"周期只能是 1/3/5/10/15，收到 {hh}"
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, f"horizon_capital[{hh}] 必须是数字"
            v = float(v)
            if not (clo <= v <= chi):
                return None, f"horizon_capital[{hh}] 必须在 [{clo}, {chi}] 之间"
            out[str(hh)] = v
        if out:
            clean["horizon_capital"] = out

    if not clean:
        return None, "没有可更新的字段"
    return clean, None


def save(payload: Dict, market: str = "CN") -> Tuple[Optional[Dict], Optional[str]]:
    """校验后合并写入指定市场。返回 (该市场的新配置, error)。"""
    clean, err = validate(payload, market)
    if err:
        return None, err
    cfg = load()
    m = str(market).upper()
    # horizon_capital 要逐键合并：整对象替换会把用户留空的周期删掉，
    # 那个周期就静默回落到 initial_capital，量级可能差 100 倍。
    _hc = clean.pop("horizon_capital", None)
    cfg["markets"][m].update(clean)
    if _hc:
        cur = dict(cfg["markets"][m].get("horizon_capital") or {})
        cur.update(_hc)
        cfg["markets"][m]["horizon_capital"] = cur
    cfg["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(STATE, exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PATH)
    return load(m), None


def capital_explicit(market: str = "CN", horizon: int = 1) -> Optional[float]:
    """该 (市场, 周期) **用户显式设过**的本金；没设过返回 None。

    为什么需要它（实测事故）：用户在「各周期本金」里把 h=5 设成 20 万，
    然后点「清空重投」，结果本金变成旧账户的净值 —— 设置完全不生效。
    根因是 `capital_for()` **永远**有返回值（拿不到就用 initial_capital 兜底），
    于是调用方无法区分「用户设过」和「只是默认值」。

    这里只读**用户写进文件**的 horizon_capital，不合并 DEFAULTS。
    """
    try:
        with open(PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    # 支持两种结构：新（markets.CN.horizon_capital）与旧（顶层 horizon_capital）
    hc = None
    if isinstance(raw.get("markets"), dict) and isinstance(raw["markets"].get(market), dict):
        hc = raw["markets"][market].get("horizon_capital")
    if not isinstance(hc, dict):
        hc = raw.get("horizon_capital")
    if not isinstance(hc, dict):
        hc = {}
    v = None
    try:
        v = hc.get(str(int(horizon)), hc.get(int(horizon)))
    except (TypeError, ValueError):
        v = None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def capital_for(market: str = "CN", horizon: int = 1) -> float:
    """
    该 (市场, 周期) 的本金。优先用周期专属值，没有就退回市场默认。
    """
    try:
        cfg = load(market)
    except Exception:
        cfg = {}
    hc = cfg.get("horizon_capital") or {}
    try:
        v = hc.get(str(int(horizon)))
        if v is None:
            v = hc.get(int(horizon))
    except (TypeError, ValueError):
        v = None
    if v is not None:
        try:
            f = float(v)
            if f > 0:
                return f
        except (TypeError, ValueError):
            pass
    try:
        return float(cfg.get("initial_capital") or 5_000)
    except (TypeError, ValueError):
        return 5_000.0


def summary() -> Dict:
    """三个市场一览，给 UI 用。"""
    cfg = load()
    return {"ok": True, "default_market": cfg["default_market"],
            "markets": {m: {**cfg["markets"][m], "market": m} for m in MARKETS},
            "updated_at": cfg.get("updated_at")}


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
