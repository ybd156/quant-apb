# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# APB 自学习参数体 —— 多阶段构建
#   目标：镜像尽量小、可跑在 amd64 / arm64 的 NAS 上（飞牛 fnOS 等）
#   原则：只装 numpy + pandas，不引入编译链
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12

# ---------- stage 1: 依赖 ----------
FROM python:${PYTHON_VERSION}-slim AS deps
WORKDIR /build

# 只装运行时依赖，构建完把 wheel 缓存清掉
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- stage 2: 运行时 ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="APB Adaptive Parameter Body" \
      org.opencontainers.image.description="Self-learning parameter system for stock analysis" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/ybd156/quant-apb"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    APB_HOST=0.0.0.0 \
    APB_PORT=8765 \
    APB_STATE_DIR=/app/data/state

COPY --from=deps /install /usr/local

WORKDIR /app

# 先拷代码（数据目录由 volume 挂载，不烤进镜像）
COPY qcore/ ./qcore/
COPY webui/ ./webui/
COPY docker/start.sh ./start.sh
COPY VERSION README.md LICENSE ./

RUN mkdir -p /app/data/state /app/data/raw /app/data/pred_ledger /app/logs \
 && useradd -m -u 10001 -s /usr/sbin/nologin apb \
 && chown -R apb:apb /app

RUN chmod +x /app/start.sh

USER apb

EXPOSE 8765

# 健康检查：/api/health 返回 200 即视为存活（缺数据时也应返回 200 + ok:false）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=4).status==200 else 1)"

ENTRYPOINT ["/app/start.sh"]
