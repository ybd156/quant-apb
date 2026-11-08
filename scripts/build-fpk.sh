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
#   APB.sc              端口转发
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
for f in manifest config/privilege config/resource APB.sc cmd/service-setup ui/config \
         ICON.PNG ICON_256.PNG; do
  [ -e "$SRC/$f" ] || { echo "[fpk] ERROR missing $SRC/$f"; exit 1; }
done

# ---------- 2. 打应用载荷 app.tgz ----------
mkdir -p "$WORK/app"
cp -a "$ROOT/qcore" "$WORK/app/"
cp -a "$ROOT/webui" "$WORK/app/"
cp "$ROOT/requirements.txt" "$ROOT/VERSION" "$ROOT/LICENSE" "$WORK/app/"
[ -f "$ROOT/README.md" ] && cp "$ROOT/README.md" "$WORK/app/"
find "$WORK/app" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$WORK/app" -name '*.pyc' -delete 2>/dev/null || true

tar -czf "$WORK/app.tgz" -C "$WORK/app" .
APP_TGZ_SIZE=$(stat -c%s "$WORK/app.tgz")
echo "[fpk] app.tgz = $APP_TGZ_SIZE bytes"
[ "$APP_TGZ_SIZE" -ge 10240 ] || { echo "[fpk] ERROR app.tgz too small"; exit 1; }

# ---------- 3. 组装 package ----------
mkdir -p "$PKG/cmd" "$PKG/config" "$PKG/ui/images"
cp "$WORK/app.tgz" "$PKG/app.tgz"
cp -a "$SRC/config/." "$PKG/config/"
cp -a "$SRC/ui/." "$PKG/ui/"
cp "$SRC/cmd/service-setup" "$PKG/cmd/"
chmod +x "$PKG/cmd/service-setup"
cp "$SRC/APB.sc" "$PKG/"
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
echo "[fpk] built $OUT ($(stat -c%s "$OUT") bytes)"

# ---------- 6. 自检：能否解开、关键文件是否在 ----------
tar tzf "$OUT" >/dev/null || { echo "[fpk] ERROR corrupted"; exit 1; }
for f in manifest app.tgz cmd/service-setup config/privilege config/resource ICON.PNG; do
  tar tzf "$OUT" | grep -qx "./$f" || { echo "[fpk] ERROR missing $f in fpk"; exit 1; }
done
echo "[fpk] OK"
