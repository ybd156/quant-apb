"""
模拟账户配置 —— 让用户能设「初始资金、持仓数、单票上限、成本」。

## 为什么要有这个

之前 `PaperConfig.initial` 写死 100 万，界面上无法修改。
用户要的是「我放 50 万进去，看它怎么跑」，这是最基本的需求。

## 存储

`data/state/account.json`：

```json
{
  "initial_capital": 500000.0,
  "top_k": 10,
  "max_weight": 0.15,
  "cost_bps": 10.0,
  "hold": 5,
  "updated_at": "2026-11-08T19:45:00+08:00"
}
```

## 校验

- `initial_capital` ∈ [10_000, 1_000_000_000]
- `top_k` ∈ [1, 50]
- `max_weight` ∈ (0, 1]
- `cost_bps` ∈ [0, 200]
- `hold` ∈ [1, 60]

越界直接拒绝，不做"自动纠正"——悄悄改用户的钱数是不可接受的。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
PATH = os.path.join(STATE_DIR, "account.json")

DEFAULTS: Dict = {
    "initial_capital": 1_000_000.0,
    "top_k": 10,
    "max_weight": 0.15,
    "cost_bps": 10.0,
    "hold": 5,
}

LIMITS: Dict[str, Tuple[float, float]] = {
    "initial_capital": (10_000.0, 1_000_000_000.0),
    "top_k": (1, 50),
    "max_weight": (0.0001, 1.0),
    "cost_bps": (0.0, 200.0),
    "hold": (1, 60),
}

INT_FIELDS = {"top_k", "hold"}


def load() -> Dict:
    """读配置；文件缺失或损坏时回退默认值（不抛异常）。"""
    try:
        with open(PATH, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return dict(DEFAULTS)
    out = dict(DEFAULTS)
    for k, (lo, hi) in LIMITS.items():
        v = d.get(k)
        if isinstance(v, (int, float)) and lo <= float(v) <= hi:
            out[k] = int(v) if k in INT_FIELDS else float(v)
    out["updated_at"] = d.get("updated_at")
    return out


def validate(payload: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """
    校验用户提交的配置。返回 (clean, error)。
    只接受白名单字段；越界返回明确错误，不做静默纠正。
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"
    clean: Dict = {}
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


def save(payload: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """校验后合并写入。返回 (新配置, error)。"""
    clean, err = validate(payload)
    if err:
        return None, err
    cur = load()
    cur.update(clean)
    cur["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PATH)
    return cur, None


if __name__ == "__main__":
    print(json.dumps(load(), ensure_ascii=False, indent=1))
