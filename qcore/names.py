"""
代码 → 名称映射 —— 界面上不能只显示 600519，要显示「贵州茅台」。

## 为什么要单独落盘

UI 层不做计算（INV-8），也不该自己去查名称。
流水线每次运行把当前股票池的 code/name/industry 写成 `data/state/names.json`，
前端拿来做展示映射。

## 结构

```json
{
  "as_of": "2026-11-08T19:40:00+08:00",
  "n": 300,
  "names": {
    "600519": {"name": "贵州茅台", "industry": "白酒"},
    "000001": {"name": "平安银行", "industry": "银行"}
  }
}
```

港股/美股代码原样保留（如 `00700.HK`、`AAPL`）。
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")


def build_map(pool_df=None, extra: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
    """从股票池 DataFrame 构造映射。pool_df 需含 code/name 列。"""
    out: Dict[str, Dict] = {}
    if pool_df is not None and len(pool_df):
        cols = set(pool_df.columns)
        for row in pool_df.itertuples(index=False):
            code = str(getattr(row, "code", "") or "")
            if not code:
                continue
            rec = {"name": str(getattr(row, "name", "") or code)}
            if "industry" in cols:
                ind = getattr(row, "industry", None)
                if isinstance(ind, str) and ind:
                    rec["industry"] = ind
            out[code] = rec
    if extra:
        for k, v in extra.items():
            out.setdefault(str(k), {"name": str(v)})
    return out


def write(names: Dict[str, Dict], path: Optional[str] = None) -> str:
    import json
    if path is None:
        path = os.path.join(STATE_DIR, "names.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
               "n": len(names), "names": names}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def export_from_pool(pool_df=None) -> str:
    """流水线调用：把当前池的名称写进 state。"""
    return write(build_map(pool_df))


def load(path: Optional[str] = None) -> Dict[str, Dict]:
    import json
    if path is None:
        path = os.path.join(STATE_DIR, "names.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("names") or {}
    except Exception:
        return {}


if __name__ == "__main__":
    from qcore.data import get_pool
    p = get_pool(top_n=300)
    print(export_from_pool(p))
