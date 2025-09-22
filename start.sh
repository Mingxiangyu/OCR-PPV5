#!/bin/bash

# OCR-PPV5 启动脚本
# 适配 PaddleOCR 3.1 版本

echo "🚀 OCR-PPV5 服务启动脚本"
echo "适配 PaddleOCR 3.1 版本"
echo "========================="

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

# 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p ./models
mkdir -p ./logs

# 设置目录权限
chmod 755 ./models
chmod 755 ./logs

echo "✅ 目录创建完成"

# 询问用户选择运行模式
echo ""
echo "请选择运行模式："
echo "1) CPU 模式 (推荐，兼容性好)"
echo "2) GPU 模式 (需要 NVIDIA Docker 支持)"
read -p "请输入选择 (1 或 2): " choice

case $choice in
    1)
        echo "🖥️  启动 CPU 模式..."
        docker-compose up --build
        ;;
    2)
        echo "🎮 启动 GPU 模式..."
        if ! command -v nvidia-docker &> /dev/null; then
            echo "⚠️  警告：未检测到 nvidia-docker，GPU 模式可能无法正常工作"
            read -p "是否继续？(y/n): " continue_choice
            if [ "$continue_choice" != "y" ]; then
                echo "❌ 用户取消操作"
                exit 1
            fi
        fi
        docker-compose -f docker-compose.gpu.yml up --build
        ;;
    *)
        echo "❌ 无效选择，请重新运行脚本"
        exit 1
        ;;
esac