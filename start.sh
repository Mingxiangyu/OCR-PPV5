#!/bin/bash

# OCR-PPV5 GPU 启动脚本
# 适配 PaddleOCR 3.1 版本（仅 GPU 模式）

echo "🚀 OCR-PPV5 GPU 服务启动脚本"
echo "适配 PaddleOCR 3.1 版本（GPU 加速）"
echo "================================="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 检查 NVIDIA Docker 支持
if ! command -v nvidia-docker &> /dev/null; then
    echo "⚠️  警告：未检测到 nvidia-docker，GPU 模式可能无法正常工作"
    echo "请确保已安装 NVIDIA Docker Runtime"
    read -p "是否继续？(y/n): " continue_choice
    if [ "$continue_choice" != "y" ]; then
        echo "❌ 用户取消操作"
        exit 1
    fi
fi

# 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p ./models
mkdir -p ./logs

# 设置目录权限
chmod 755 ./models
chmod 755 ./logs

echo "✅ 目录创建完成"

echo ""
echo "🎮 启动 GPU 模式..."
echo "注意：确保您的系统支持 NVIDIA GPU 和 CUDA"

# 启动服务
docker-compose up --build