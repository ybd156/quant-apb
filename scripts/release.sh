#!/usr/bin/env bash
# 发版：只推 HEAD + 本次的 tag。
#
# ## 为什么必须有这个脚本
#
# 2026-11-12 的事故：用 `git push origin HEAD --follow-tags` 发版，
# `--follow-tags` 会推送**所有可达的 annotated tag** —— 仓库里积压了 97 个
# 本地 tag，于是一次推送触发 **97 次 Release workflow**（每次：测试 → 构建
# 镜像 → 推 GHCR → 建 GitHub Release）。CI 配额、镜像仓库、Release 列表全被灌满。
#
# 正确的发版只有两条命令：
#     git push origin HEAD
#     git push origin "v<VERSION>"
#
# 本脚本把这两条固定下来，并**显式拒绝**任何 --follow-tags / --tags。
#
# 用法：
#     bash scripts/release.sh                 # 推 HEAD + 当前 VERSION 对应的 tag
#     bash scripts/release.sh --dry-run       # 只打印将执行的命令
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --follow-tags|--tags|--all|--mirror)
      echo "[X] 不许用 $a —— 它会把积压的旧 tag 一起推上去（见脚本顶部的 97 次事故）" >&2
      exit 2 ;;
    *) echo "[X] 未知参数：$a" >&2; exit 2 ;;
  esac
done

V="$(cat VERSION | tr -d '[:space:]')"
TAG="v$V"

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "[X] 本地没有 tag $TAG。先建："
  echo "    git tag -a $TAG -m \"$TAG\""
  echo "  并确认 tag 指向 HEAD（tag 与 VERSION 不一致时 release.yml 会拒绝出镜像）。"
  exit 3
fi

HEAD_TAG="$(git tag --points-at HEAD | tr '\n' ' ')"
if ! echo "$HEAD_TAG" | grep -q "$TAG"; then
  echo "[X] $TAG 不在 HEAD 上（HEAD 的 tag: ${HEAD_TAG:-无}）"
  echo "    版本与代码对不上就发版 = 镜像版本号和代码内容错位。"
  exit 3
fi

echo "将执行："
echo "  git push origin HEAD"
echo "  git push origin $TAG"
if [ "$DRY" = "1" ]; then
  echo "（--dry-run，不执行）"
  exit 0
fi

git push origin HEAD
git push origin "$TAG"

echo
echo "已推送 $TAG。远端 tag 现状（只列 v0.33.x，避免刷屏）："
git ls-remote --tags origin 'v0.33.*' | sed 's#refs/tags/##' || true
echo
echo "注意：Release workflow 只应有一次新运行。若看到多条，说明又推了别的 tag。"
