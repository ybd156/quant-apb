"""qcore —— 自学习参数体（Adaptive Parameter Body）核心包。"""
import os as _os

# 版本只有一个真源：仓库根目录的 VERSION 文件。
# 旧写法在这里硬编码 "0.27.8"，与 VERSION 文件并存 → 改版本必然漏一处；
# 而 kit_update 正是靠版本号判断训练机要不要更新（版本漂移 = 训练机永远不更新）。
def _read_version() -> str:
    try:
        _p = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "VERSION")
        with open(_p, encoding="utf-8") as f:
            v = f.read().strip()
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    return "0.0.0"


__version__ = _read_version()
