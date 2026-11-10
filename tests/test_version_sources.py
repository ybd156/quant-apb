"""
回归测试：版本号只能有一个真源。

实测事故（2026-11-10）：容器跑着 `ghcr.io/...:0.28.3`，`/api/version` 却报 **0.27.8**。
根因是项目里同时存在三份版本号：

  1. 仓库根 `VERSION`            ← 我改的是这个（CI 校验、kit 打包也用它）
  2. `webui/VERSION`             ← **停在 0.27.8**，而 server.py 读的是它
  3. `qcore/__init__.py` 硬编码   ← 已改为读根 VERSION

于是 NAS 的 `/api/version` 与 `/api/train/kit` 清单都报 0.27.8，
训练机 `--auto-update` 永远认为「已是最新」。

本测试锁三件事：
  1. 根 VERSION 是合法 semver
  2. `qcore.__version__` 与 `webui.server._read_version()` 都与根 VERSION 一致
  3. `webui/VERSION` 不再存在（防止第二份真源复活）

跑法：python -m tests.test_version_sources
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    root_file = os.path.join(ROOT, "VERSION")
    webui_file = os.path.join(ROOT, "webui", "VERSION")

    check("仓库根 VERSION 存在", os.path.isfile(root_file))
    if not os.path.isfile(root_file):
        print(f"\nRESULT: {PASS} passed, {FAIL} failed")
        return 1
    root_ver = open(root_file, encoding="utf-8").read().strip()

    check("根 VERSION 是合法 semver", bool(re.match(r"^\d+\.\d+\.\d+$", root_ver)), root_ver)

    import qcore
    check("qcore.__version__ == 根 VERSION", qcore.__version__ == root_ver,
          f"{qcore.__version__} vs {root_ver}")

    try:
        from webui import server as WS
        sv = WS._read_version()
        check("server._read_version() == 根 VERSION", sv == root_ver, f"{sv} vs {root_ver}")
    except Exception as e:  # noqa: BLE001
        check("可导入 webui.server 并读版本", False, f"{type(e).__name__}: {e}")

    check("webui/VERSION 不再存在（第二份真源）", not os.path.isfile(webui_file),
          "存在的话会覆盖根 VERSION，导致对外报旧版本" if os.path.isfile(webui_file) else "")

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
