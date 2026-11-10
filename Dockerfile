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
# scripts/ 里 benchmark_alpha158.py 是 live_model 和 watch 的依赖
# （build_features 从这里加载），漏了它自选/训练都会 FileNotFoundError
COPY scripts/ ./scripts/
# 只带不变量检查需要的静态扫描器（15KB），不带测试目录
COPY audit/leak_scan.py ./audit/
# INV-1 无未来函数检验需要它（28KB）。夹具不进镜像 ——
# 该检验优先用真实 qlib 数据，没有夹具时才退回夹具。
COPY tests/test_no_lookahead.py ./tests/
COPY tools/ ./tools/
COPY docker/start.sh ./start.sh
# 构建期就校验入口脚本语法。实测事故：python -c "..." 块里的反引号被 sh
# 当成命令替换，容器起来直接崩溃循环（Bad function name），而镜像构建是绿的。
RUN sh -n /app/start.sh
COPY VERSION README.md LICENSE ./

RUN mkdir -p /app/data/state /app/data/raw /app/data/pred_ledger /app/logs \
 && useradd -m -u 10001 -s /usr/sbin/nologin apb \
 && chown -R apb:apb /app

RUN chmod +x /app/start.sh

USER apb

EXPOSE 8765

# 健康检查必须探 /api/healthz（公开端点）。
# /api/health 需要登录 —— 设了 APB_PASSWORD 之后它返回 401，
# 容器会一直显示 unhealthy（实测踩过：加了密码后健康检查全红）。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/healthz',timeout=4).status==200 else 1)"

ENTRYPOINT ["/app/start.sh"]
