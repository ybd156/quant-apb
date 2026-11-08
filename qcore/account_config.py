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
    "CN": {"initial_capital": 1000000.0, "top_k": 20, "max_weight": 0.15,
           "cost_bps": 19.28, "hold": 20, "currency": "CNY"},
    "US": {"initial_capital": 100000.0,  "top_k": 20, "max_weight": 0.15,
           "cost_bps": 3.28,  "hold": 20, "currency": "USD"},
    "HK": {"initial_capital": 500000.0,  "top_k": 20, "max_weight": 0.15,
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
DEFAULTS: Dict[str, Dict] = {
    "CN": {"initial_capital": 1_000_000.0, "top_k": 20, "max_weight": 0.15,
           "cost_bps": 19.28, "hold": 20, "currency": "CNY"},
    "US": {"initial_capital": 100_000.0, "top_k": 20, "max_weight": 0.15,
           "cost_bps": 3.28, "hold": 20, "currency": "USD"},
    "HK": {"initial_capital": 500_000.0, "top_k": 20, "max_weight": 0.15,
           "cost_bps": 38.17, "hold": 20, "currency": "HKD"},
}

# 字段 -> (下限, 上限)。initial_capital 按市场另设（见 CAPITAL_LIMITS）
LIMITS: Dict[str, Tuple[float, float]] = {
    "top_k": (1, 100),
    "max_weight": (0.0001, 1.0),
    "cost_bps": (0.0, 500.0),
    "hold": (1, 120),
}
CAPITAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "CN": (10_000.0, 1_000_000_000.0),
    "US": (1_000.0, 100_000_000.0),
    "HK": (10_000.0, 1_000_000_000.0),
}
INT_FIELDS = {"top_k", "hold"}


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
                out["markets"][m].update(
                    {k: v for k, v in raw["markets"][m].items() if k in DEFAULTS[m]})
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
    cfg["markets"][m].update(clean)
    cfg["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(STATE, exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PATH)
    return load(m), None


def summary() -> Dict:
    """三个市场一览，给 UI 用。"""
    cfg = load()
    return {"ok": True, "default_market": cfg["default_market"],
            "markets": {m: {**cfg["markets"][m], "market": m} for m in MARKETS},
            "updated_at": cfg.get("updated_at")}


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
