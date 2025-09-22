# 使用官方 Python 3.10 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置 PaddleOCR 3.1 模型存储目录环境变量
ENV PADDLEHUB_HOME=/app/models
ENV PADDLE_HOME=/app/models
ENV HOME=/app/models

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    # 图像处理依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PDF 处理依赖
    poppler-utils \
    # 网络工具
    wget \
    curl \
    # 清理缓存
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements*.txt ./

# 升级 pip 并安装依赖（适配 PaddleOCR 3.1）
RUN pip install --upgrade pip && \
    # 先安装 PaddlePaddle GPU 版本
    pip install paddlepaddle-gpu==3.1.0 && \
    # 然后安装 PaddleOCR
    pip install paddleocr && \
    # 最后安装其他依赖
    pip install -r requirements-linux.txt

# 复制应用代码
COPY app.py .

# 创建模型存储目录（用于持久化模型文件）
RUN mkdir -p /app/models/.paddleocr

# 创建临时文件目录
RUN mkdir -p /app/picture

# 设置权限
RUN chmod -R 755 /app

# 暴露端口（从5103改为5104以匹配app.py中的配置）
EXPOSE 5104

# 健康检查（更新端口号）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5104/ || exit 1

# 启动命令
CMD ["python", "app.py"]
