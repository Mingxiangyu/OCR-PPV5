@echo off
REM OCR-PPV5 Windows GPU 启动脚本
REM 适配 PaddleOCR 3.1 版本（仅 GPU 模式）

echo 🚀 OCR-PPV5 GPU 服务启动脚本
echo 适配 PaddleOCR 3.1 版本（GPU 加速）
echo =================================

REM 检查 Docker 是否安装
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker 未安装，请先安装 Docker Desktop
    pause
    exit /b 1
)

REM 检查 Docker Compose 是否安装
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose 未安装，请先安装 Docker Compose
    pause
    exit /b 1
)

REM 创建必要的目录
echo 📁 创建必要目录...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
echo ✅ 目录创建完成

echo.
echo 🎮 启动 GPU 模式...
echo ⚠️  注意：GPU 模式需要 NVIDIA GPU 和 NVIDIA Docker 支持
echo 请确保您的系统已安装 NVIDIA GPU 驱动和 CUDA
pause

REM 启动服务
docker-compose up --build

pause