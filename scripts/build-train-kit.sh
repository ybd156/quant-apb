#!/usr/bin/env bash
# 打包「训练机套件」—— 拷到任意一台电脑，解压即可训练。
#
# 产物： apb-train-kit_<version>.zip
#
# 里面只有训练需要的东西（不含 Web UI、不含 fnOS 打包文件），
# 目标是「解压 → 双击 → 训练 → 得到可上传的模型 zip」。
set -euo pipefail

VERSION="${1:-0.0.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/apb-train-kit_${VERSION}.zip"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[kit] version=$VERSION"
KIT="$WORK/apb-train-kit"
mkdir -p "$KIT"

# ---------- 1. 训练需要的代码（**只拷闭包**）----------
# 原来整个 qcore/ 都拷（126 个模块 / 42,222 行），但训练机只需要
# 训练 + 外包 pipeline/prewarm 真正用到的 62 个 / 24,424 行。
# 闭包由 scripts/kit_manifest.py 用 AST 算出来，可复现、可复核。
mkdir -p "$KIT/tools" "$KIT/scripts"
python -m scripts.kit_manifest --json > "$KIT/kit_manifest.json" 2>/dev/null || true
if [ -s "$KIT/kit_manifest.json" ]; then
  SLIM_OK=1
  # 逐行读出 files 数组，按相对路径拷贝
  python - "$ROOT" "$KIT" <<'PYEOF_SLIM' || SLIM_OK=0
import json, os, shutil, sys
root, kit = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(kit, "kit_manifest.json"), encoding="utf-8"))
n = 0
for rel in man.get("files") or []:
    src = os.path.join(root, rel)
    if not os.path.isfile(src):
        continue
    dst = os.path.join(kit, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    n += 1
print(f"[kit] slim: copied {n} files")
sys.exit(0 if n > 40 else 1)
PYEOF_SLIM
  if [ "$SLIM_OK" = "1" ]; then
    echo "[kit] slim qcore (closure only)"
  else
    echo "[kit] slim failed -> fallback to full qcore"
    rm -rf "$KIT/qcore"
    cp -a "$ROOT/qcore" "$KIT/"
  fi
else
  echo "[kit] no manifest -> full qcore"
  cp -a "$ROOT/qcore" "$KIT/"
fi
cp "$ROOT/tools/train_one_click.py" "$KIT/tools/"
cp "$ROOT/tools/sync_agent.py" "$KIT/tools/"
cp "$ROOT/tools/train_missing.py" "$KIT/tools/"
cp "$ROOT/tools/defaults.py" "$KIT/tools/"
cp "$ROOT/tools/bootstrap.ps1" "$KIT/tools/"
cp "$ROOT/tools/kit_selftest.py" "$KIT/tools/"
cp "$ROOT/tools/agent_daemon.py" "$KIT/tools/"
cp "$ROOT/tools/kit_setup.py" "$KIT/tools/"
cp "$ROOT/tools/kit_status.py" "$KIT/tools/"
cp "$ROOT/tools/kit_menu.py" "$KIT/tools/"
# 根目录只留**一个**入口 APB.bat（菜单式，含一屏状态）。
# 旧的多入口（start_agent.bat / start_agent_hidden.vbs / setup.bat / daemon.bat）
# 全部不再打包 —— 用户原话：「那么多文件，都不知点哪个」。
cp "$ROOT/APB.bat" "$KIT/" 2>/dev/null || true
cp "$ROOT/scripts/kit_manifest.py" "$KIT/scripts/"
cp "$ROOT/scripts/benchmark_alpha158.py" "$KIT/scripts/"
cp "$ROOT/requirements.txt" "$KIT/"
cp "$ROOT/VERSION" "$KIT/"
cp "$ROOT/LICENSE" "$KIT/" 2>/dev/null || true
# 只带 train.bat（Windows 引导）。train.sh 是 Linux 用的，
# 而这个套件是 Windows 专用的（wheels 是 win_amd64）——
# 多一个用不上的脚本只会增加「不知点哪个」的困惑。
cp "$ROOT/train.bat" "$KIT/"
# daemon.bat 是 NAS/常驻侧的东西，训练机不需要（拷过去只会让人困惑）
chmod +x "$KIT/train.bat" 2>/dev/null || true

# ---------- 1.5 构建后校验：这个套件能不能承接所有任务 ----------
if [ -f "$KIT/tools/kit_selftest.py" ]; then
  if python tools/kit_selftest.py --json >/dev/null 2>&1; then :; fi
  ( cd "$KIT" && python tools/kit_selftest.py >/dev/null 2>&1 ) &&     echo "[kit] selftest OK (train / pipeline / prewarm / sync)" ||     echo "[kit] WARN: selftest failed in built kit -- check before shipping"
fi

# ---------- 2. 清掉不需要的 ----------
find "$KIT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$KIT" -name '*.pyc' -delete 2>/dev/null || true
# 数据与模型不进包（数据 545MB，模型由训练产出）
rm -rf "$KIT/data" 2>/dev/null || true
# 便携运行时由 bootstrap.ps1 首次运行时生成，不进包
rm -rf "$KIT/runtime" 2>/dev/null || true

# ---------- 2.5 预下载依赖 wheel（关键：避免用户端联网装 numpy/pandas）----------
# 国内装 numpy+pandas 常要 5-20 分钟，且可能失败。
# 把 wheel 打进套件后，用户端是纯本地安装，几秒完成。
# 版本锁死到和开发环境一致（numpy 2.3.5 / pandas 3.0.1），保证行为一致。
#
# **必须同时带 pip / setuptools / wheel**：Python 3.12 移除了 distutils，
# 而镜像上的 get-pip.py 是老版本、仍 import distutils —— 实测
# `ModuleNotFoundError: No module named 'distutils'`，整个 bootstrap 失败。
# pip 本身是纯 Python，把它的 wheel 直接铺进 site-packages 就能用，
# 完全不需要联网、也不需要 get-pip。
WHEEL_CACHE="$ROOT/.wheel-cache"
mkdir -p "$WHEEL_CACHE"
if ! ls "$WHEEL_CACHE"/numpy-*.whl >/dev/null 2>&1; then
  echo "[kit] downloading dependency wheels (one-time, cached in .wheel-cache) ..."
  PYBIN=""
  for cand in python3 python; do
    command -v "$cand" >/dev/null 2>&1 && { PYBIN="$cand"; break; }
  done
  if [ -n "$PYBIN" ]; then
    "$PYBIN" -m pip download --only-binary=:all: \
      --python-version 3.12 --platform win_amd64 --implementation cp --abi cp312 \
      --dest "$WHEEL_CACHE" \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      "numpy==2.3.5" "pandas==3.0.1" "pip" "setuptools" "wheel" >/dev/null 2>&1 || \
    "$PYBIN" -m pip download --only-binary=:all: \
      --python-version 3.12 --platform win_amd64 --implementation cp --abi cp312 \
      --dest "$WHEEL_CACHE" "numpy==2.3.5" "pandas==3.0.1" "pip" "setuptools" "wheel" >/dev/null 2>&1 || true
  fi
fi

if ls "$WHEEL_CACHE"/*.whl >/dev/null 2>&1; then
  mkdir -p "$KIT/wheels"
  cp "$WHEEL_CACHE"/*.whl "$KIT/wheels/"
  WHL_N=$(ls "$KIT/wheels"/*.whl | wc -l | tr -d ' ')
  WHL_MB=$(du -sm "$KIT/wheels" | awk '{print $1}')
  echo "[kit] bundled $WHL_N wheels (${WHL_MB} MB) -> offline install"
else
  echo "[kit] WARN: no wheels downloaded, users will need internet for numpy/pandas"
fi

# ---------- 3. 默认配置（含默认 NAS 域名）----------
cat > "$KIT/apb_defaults.json" <<'JSON'
{
  "nas_url": "https://apbcdn.683868.xyz",
  "note": "训练完把模型推到这个地址。改这里或命令行 --nas 覆盖。",
  "token": "",
  "password": "",
  "auth_note": "上传模型需要鉴权。填 token（NAS 的 APB_TOKEN）或 password（NAS 登录密码）任意一个即可；两个都空会报 login_required —— 模型训好了却传不回来。",
  "default_horizons": [1, 3, 5, 10, 15],
  "default_trees": 250,
  "default_stocks": 300
}
JSON

# ---------- 4. 说明文档 ----------
cat > "$KIT/README-训练机.md" <<'MD'
# APB 训练机套件

解压后即可训练。**不需要装数据库、不需要 Docker、不需要联网训练**（只有下载数据时联网）。

---

## 只有一个入口：APB.bat

双击 **`APB.bat`**，它会先显示这台机器的状态，再给你一个菜单：

```
==============================================================
  APB 训练机状态
==============================================================
  版本         0.32.1
  依赖         numpy 2.3.5 / pandas 3.0.1
  鉴权         已配令牌
  数据         2026-11-10
  守护进程     运行中（PID 1234，心跳 5s 前）
  ────────────────────────────────────────────────────────────
   1  配置（填 NAS 地址 / 令牌）           setup
   2  启动后台运行（无窗口，推荐）          start
   3  停止                                  stop
   4  刷新状态                              status
   5  立即训练一次（前台，能看到过程）       train
   6  自检（能否承接所有任务）              selftest
   7  检查更新                              update
   0  退出
```

### 第一次用，只做两步

1. 双击 `APB.bat` → 选 **1**，填令牌（从 NAS「设置 → 访问令牌」复制）
   - 窗口里点「测试连接」，通了再保存
2. 再双击 `APB.bat` → 选 **2**，后台跑起来

之后想看在干嘛，随时双击 `APB.bat`，最上面就是状态。

### 也可以用命令行

```
APB.bat status     只看状态
APB.bat start      后台启动
APB.bat stop       停止
APB.bat setup      配置
APB.bat train      前台训练一次
APB.bat selftest   自检能不能承接所有任务
```

### 后台运行是什么意思

选 **2** 之后没有任何窗口，守护进程会：

- 每 60 秒问一次 NAS 有没有新任务，有就领
- 训练时**逐周期回报进度**（NAS 网页上能看到「h5 · 1/2 · 约2分」）
- 训完自动上传，不用手动打包
- 服务端有新版本就自动更新并重启自己
- 崩了按 10s→300s 退避重拉
- 日志在 `logs/agent.log`，超过 5MB 自动轮转

> 这是 Windows 套件（`wheels/` 是 win_amd64）。Linux/macOS 请直接用仓库里的
> `python tools/train_one_click.py`。

---

## 训练完怎么给 NAS

**方法一：网页上传（最简单）**

1. 打开 https://apbcdn.683868.xyz
2. 进「设置 → 模型导入」
3. 把 `dist/models/apb_models_*.zip` 拖进去

**方法二：自动推送（一条命令）**

```bash
python tools/sync_agent.py --nas https://apbcdn.683868.xyz --push-only --horizons 1,5,15
```

**方法三：接 NAS 下发的任务（推荐常驻）**

```bash
python tools/sync_agent.py --nas https://apbcdn.683868.xyz --loop 60
```

然后在 NAS 网页「设置 → 训练任务」排任务，这台机器会自动领活、训练、上传。

---

## 常用参数

```bash
python tools/train_one_click.py --horizons 1,5      # 只训两个周期
python tools/train_one_click.py --trees 150         # 慢机器降树数
python tools/train_one_click.py --skip-data         # 数据已有，跳过检查
python tools/train_one_click.py --dry-run           # 只检查环境
```

---

## 硬件与耗时参考

实测（2026-11）：

| 设备 | 250 树一个周期 | 说明 |
|---|---|---|
| Intel N100（NAS） | **5.8 分钟** | GBDT 是内存受限，核显帮不上 |
| AMD Ryzen 16 线程 | 10 分钟 | 当时还在跑别的任务 |
| RTX 3060 | 直方图快 6.2× | 但整体受内存带宽限制 |

**6 个周期大约 35-60 分钟**，夜里跑完全够。

> 核显（Intel UHD）**不要用来加速**：实测带宽和 CPU 打平、算力只有 37%，
> GBDT 直方图反而更慢。

---

## 目录结构

```
apb-train-kit/
├─ APB.bat                     **唯一入口**（状态 + 菜单）
├─ train.bat                   首次引导（生成便携运行时）
├─ tools/
│   ├─ train_one_click.py      训练主程序
│   └─ sync_agent.py           自动领任务 + 上传
├─ qcore/                      核心库（GBDT / Alpha158 / 数据同步）
├─ scripts/benchmark_alpha158.py
├─ apb_defaults.json           默认 NAS 地址（可改）
└─ requirements.txt
```

---

## 出问题看这里

| 现象 | 原因 | 处理 |
|---|---|---|
| `找不到 python` | 没装或没进 PATH | 装 Python 3.10+，勾选 Add to PATH |
| `缺少 numpy` | 依赖没装 | `pip install numpy pandas` |
| 数据下载失败 | 网络/代理 | 脚本会自动切直连/代理；也可手动下 `qlib_bin.tar.gz` 放 `data/qlib/` |
| 训练很慢 | 树太多 | `--trees 150` |
| 上传 401/403 | token 不对 | `--token <NAS 的 APB_TOKEN>` |

---

生成版本：见同目录 `VERSION` 文件。
MD

# ---------- 5. 打 zip ----------
cd "$WORK"
if command -v zip >/dev/null 2>&1; then
  zip -qr "$OUT" apb-train-kit
else
  # 没有 zip 就用 python（Windows 上是 python，Linux 上是 python3）
  PYBIN=""
  for cand in python3 python; do
    command -v "$cand" >/dev/null 2>&1 && { PYBIN="$cand"; break; }
  done
  [ -n "$PYBIN" ] || { echo "[kit] ERROR 找不到 python，无法打 zip"; exit 1; }
  "$PYBIN" -c "
import os, shutil, sys
out = sys.argv[1]
base = out[:-4] if out.endswith('.zip') else out
shutil.make_archive(base, 'zip', '.', 'apb-train-kit')
" "$OUT"
fi

SIZE=$(wc -c < "$OUT" | tr -d ' ')
echo "[kit] built $OUT (${SIZE} bytes)"
echo "[kit] 内容："
cd "$WORK" && find apb-train-kit -type f | sed 's/^/  /' | head -40
