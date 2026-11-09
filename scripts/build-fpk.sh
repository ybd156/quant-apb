#!/usr/bin/env bash
# 构建 fnOS .fpk 安装包
#
# ⚠️ 本脚本**只在 CI 里运行**（GitHub Actions ubuntu-latest）。
#    本地不构建、不产出任何二进制——产物一律由 Release 页面下载。
#
# 用法： ./scripts/build-fpk.sh <version> <platform>
#   version   如 0.0.1
#   platform  x86 | arm
#
# 产物： apb_<version>_<platform>.fpk
#
# fpk 内部结构（参照 fnOS 第三方应用打包规范）：
#   manifest            元数据（含 app.tgz 的 md5）
#   app.tgz             应用载荷
#   cmd/service-setup   服务控制脚本
#   config/privilege    运行权限
#   config/resource     端口/共享目录
#   ui/config           桌面入口
#   ui/images/256.png   图标
#   ICON.PNG ICON_256.PNG
set -euo pipefail

VERSION="${1:?usage: build-fpk.sh <version> <platform>}"
PLATFORM="${2:-x86}"
APPNAME="apb"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/deploy/fnos/fpk"
WORK="$(mktemp -d)"
PKG="$WORK/package"

echo "[fpk] version=$VERSION platform=$PLATFORM"

# ---------- 1. 校验源文件 ----------
for f in manifest config/privilege config/resource cmd/service-setup ui/config \
         ICON.PNG ICON_256.PNG; do
  [ -e "$SRC/$f" ] || { echo "[fpk] ERROR missing $SRC/$f"; exit 1; }
done

# 图标：优先用仓库里已提交的；缺失才报错（不依赖生成步骤）
if [ ! -f "$SRC/ICON.PNG" ] || [ ! -f "$SRC/ICON_256.PNG" ]; then
  echo "[fpk] icons missing, generating..."
  python "$ROOT/scripts/make_icons.py" || true
fi
[ -f "$SRC/ICON.PNG" ] || { echo "[fpk] ERROR ICON.PNG missing"; exit 1; }

# ---------- 2. 打应用载荷 app.tgz ----------
mkdir -p "$WORK/app"
cp -a "$ROOT/qcore" "$WORK/app/"
cp -a "$ROOT/webui" "$WORK/app/"
cp "$ROOT/requirements.txt" "$ROOT/VERSION" "$ROOT/LICENSE" "$WORK/app/"
[ -f "$ROOT/README.md" ] && cp "$ROOT/README.md" "$WORK/app/"
find "$WORK/app" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$WORK/app" -name '*.pyc' -delete 2>/dev/null || true

# ui/ 必须打进 app.tgz —— fnOS 从 **target 目录** 读 <target>/ui/config
# （lemon-music 的 ui/ 就在 /vol1/@appcenter/lemon-music/ui/）。
# 只放 fpk 根目录的话，它不会出现在 target 里，app_service 就没有记录，
# 应用中心只显示「停用」而没有「打开」，桌面也不会出现图标。
mkdir -p "$WORK/app/ui/images"
if [ -f "$SRC/ICON_256.PNG" ]; then
  cp "$SRC/ICON_256.PNG" "$WORK/app/ui/images/icon_256.png"
  cp "$SRC/ICON_256.PNG" "$WORK/app/ui/images/icon-256.png"
fi
if [ -f "$SRC/ICON.PNG" ]; then
  cp "$SRC/ICON.PNG" "$WORK/app/ui/images/icon_64.png"
  cp "$SRC/ICON.PNG" "$WORK/app/ui/images/icon-64.png"
  cp "$SRC/ICON.PNG" "$WORK/app/ui/images/icon_128.png"
  cp "$SRC/ICON.PNG" "$WORK/app/ui/images/icon-128.png"
fi
[ -f "$SRC/ui/config" ] && cp "$SRC/ui/config" "$WORK/app/ui/config"
echo "[fpk] app.tgz 内含 ui/: $(ls "$WORK/app/ui/images" 2>/dev/null | tr '\n' ' ')"

tar -czf "$WORK/app.tgz" -C "$WORK/app" .
APP_TGZ_SIZE=$(wc -c < "$WORK/app.tgz" | tr -d " ")
echo "[fpk] app.tgz = $APP_TGZ_SIZE bytes"
[ "$APP_TGZ_SIZE" -ge 10240 ] || { echo "[fpk] ERROR app.tgz too small"; exit 1; }

# ---------- 3. 组装 package ----------
mkdir -p "$PKG/cmd" "$PKG/config" "$PKG/ui/images"
cp "$WORK/app.tgz" "$PKG/app.tgz"
cp -a "$SRC/config/." "$PKG/config/"
cp -a "$SRC/ui/." "$PKG/ui/"
# fnOS 桌面入口写的是 icon: "images/icon_{0}.png"，{0} 会被替换成 64/256 等尺寸。
# 所以 ui/images 下必须有 icon_256.png / icon_64.png，否则桌面不显示图标。
mkdir -p "$PKG/ui/images"
if [ -f "$SRC/ICON_256.PNG" ]; then
  cp "$SRC/ICON_256.PNG" "$PKG/ui/images/icon_256.png"
  cp "$SRC/ICON_256.PNG" "$PKG/ui/images/icon-256.png"
fi
if [ -f "$SRC/ICON.PNG" ]; then
  cp "$SRC/ICON.PNG" "$PKG/ui/images/icon_64.png"
  cp "$SRC/ICON.PNG" "$PKG/ui/images/icon-64.png"
  cp "$SRC/ICON.PNG" "$PKG/ui/images/icon_128.png"
  cp "$SRC/ICON.PNG" "$PKG/ui/images/icon-128.png"
fi
echo "[fpk] ui/images: $(ls "$PKG/ui/images" 2>/dev/null | tr '\n' ' ')"
# fnOS 要求完整的生命周期脚本集：main/installer/service-setup/common + 各 callback
cp -a "$SRC/cmd/." "$PKG/cmd/"
chmod +x "$PKG"/cmd/* 2>/dev/null || true
# APB.sc 已移除：改用 ui/config 的 type=url + port 定义服务（同 lemon-music）
cp "$SRC/ICON.PNG" "$SRC/ICON_256.PNG" "$PKG/"
cp "$SRC/ICON_256.PNG" "$PKG/ui/images/256.png"
cp "$SRC/ICON.PNG" "$PKG/ui/images/64.png"

# ---------- 4. manifest（写入 version/platform/checksum）----------
CHECKSUM=$(md5sum "$WORK/app.tgz" | cut -d' ' -f1)
sed -e "s/^version .*/version         = ${VERSION}/" \
    -e "s/^platform .*/platform        = ${PLATFORM}/" \
    -e "s/^checksum .*/checksum        = ${CHECKSUM}/" \
    "$SRC/manifest" > "$PKG/manifest"

echo "[fpk] checksum=$CHECKSUM"
grep -E '^(appname|version|platform|checksum|service_port)' "$PKG/manifest"

# ---------- 5. 打包 ----------
OUT="$ROOT/${APPNAME}_${VERSION}_${PLATFORM}.fpk"
tar -czf "$OUT" -C "$PKG" .
echo "[fpk] built $OUT ($(wc -c < "$OUT" | tr -d " ") bytes)"

# ---------- 6. 自检：能否解开、关键文件是否在 ----------
tar tzf "$OUT" >/dev/null || { echo "[fpk] ERROR corrupted"; exit 1; }
LIST="$(tar tzf "$OUT")"
echo "[fpk] contents:"; echo "$LIST" | head -20
for f in manifest app.tgz cmd/main cmd/installer cmd/service-setup cmd/common          cmd/install_callback cmd/uninstall_callback cmd/upgrade_callback          cmd/config_callback config/privilege config/resource ICON.PNG; do
  echo "$LIST" | grep -q "$f" || { echo "[fpk] ERROR missing $f in fpk"; exit 1; }
done
echo "[fpk] OK"
