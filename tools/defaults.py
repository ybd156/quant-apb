"""
默认配置 —— NAS 地址、周期、树数。训练机套件里有一个 apb_defaults.json 可覆盖。

优先级：
    命令行参数  >  apb_defaults.json  >  这里的兜底常量

这样别人拿到套件，只要改 json 就能指向自己的 NAS，不用动代码。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

# 兜底默认值（本项目官方部署地址）
DEFAULT_NAS = "https://apbcdn.683868.xyz"
DEFAULT_HORIZONS: List[int] = [1, 3, 5, 10, 15]
DEFAULT_TREES = 250
DEFAULT_STOCKS = 300

# 配置文件位置：项目根目录 / 套件根目录
_ROOTS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.getcwd(),
]


def _load_file() -> Dict:
    for root in _ROOTS:
        p = os.path.join(root, "apb_defaults.json")
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                return d if isinstance(d, dict) else {}
            except Exception:
                continue
    return {}


_cache: Dict = {}


def get() -> Dict:
    """合并后的默认配置。"""
    if _cache:
        return _cache
    f = _load_file()
    out = {
        "nas_url": str(f.get("nas_url") or DEFAULT_NAS).rstrip("/"),
        "horizons": list(f.get("default_horizons") or DEFAULT_HORIZONS),
        "trees": int(f.get("default_trees") or DEFAULT_TREES),
        "stocks": int(f.get("default_stocks") or DEFAULT_STOCKS),
        "source": "apb_defaults.json" if f else "builtin",
    }
    _cache.update(out)
    return out


def nas(default: str = "") -> str:
    return (default or get()["nas_url"]).rstrip("/")


def horizons(default: str = "") -> List[int]:
    if default:
        try:
            return [int(x) for x in str(default).split(",") if x.strip()]
        except Exception:
            pass
    return list(get()["horizons"])


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(get(), ensure_ascii=False, indent=1))
