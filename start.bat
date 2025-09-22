@echo off
REM OCR-PPV5 Windows 启动脚本
REM 适配 PaddleOCR 3.1 版本

echo 🚀 OCR-PPV5 服务启动脚本
echo 适配 PaddleOCR 3.1 版本
echo =========================

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
echo 请选择运行模式：
echo 1) CPU 模式 (推荐，兼容性好)
echo 2) GPU 模式 (需要 NVIDIA Docker 支持)
set /p choice=请输入选择 (1 或 2): 

if "%choice%"=="1" (
    echo 🖥️  启动 CPU 模式...
    docker-compose up --build
) else if "%choice%"=="2" (
    echo 🎮 启动 GPU 模式...
    echo ⚠️  注意：GPU 模式需要 NVIDIA Docker 支持
    pause
    docker-compose -f docker-compose.gpu.yml up --build
) else (
    echo ❌ 无效选择，请重新运行脚本
    pause
    exit /b 1
)

pause