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
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y \
    # OpenGL 完整依赖（适配 Debian Trixie）
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgomp1 \
    libglu1-mesa \
    # 网络工具
    wget \
    curl \
    # PDF 处理
    poppler-utils \
    # 清理缓存
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements*.txt ./

# 升级 pip 并安装依赖（适配 PaddleOCR 3.1）
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    # 安装 PaddlePaddle GPU 3.0.0 版本
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    # 然后安装 PaddleOCR
    pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    # 最后安装其他依赖
    pip install flask==2.3.3 flask-cors==4.0.0 flask-restx==1.1.0 numpy opencv-python Pillow pdf2image PyMuPDF requests Werkzeug==2.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple

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
