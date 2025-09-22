# OCR-PPV5 GPU 部署指南

基于 PaddleOCR 3.1 的 GPU 加速容器化部署方案

## 📋 系统要求

### GPU 加速要求
- **NVIDIA GPU**（支持 CUDA 11.8+）
- **NVIDIA Docker Runtime**
- **NVIDIA Driver 470+**
- **至少 6GB GPU 显存**
- **至少 8GB 系统内存**
- **至少 15GB 可用磁盘空间**（用于模型文件和 CUDA 环境）

### 基本环境
- Docker 20.10+
- Docker Compose 1.29+
- NVIDIA Container Toolkit

## 🚀 快速部署

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd OCR-PPV5
```

### 2. 选择部署方式

#### 方式 A：使用启动脚本（推荐）
```bash
# Linux/Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

#### 方式 B：手动部署
```bash
# GPU 模式（唯一支持的模式）
docker-compose up --build -d
```

### 3. 验证部署
```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 测试 API
curl http://localhost:5104/health
```

## 📁 目录结构

部署后的目录结构：
```
OCR-PPV5/
├── app.py                    # 主应用
├── Dockerfile               # GPU 版本镜像
├── docker-compose.yml       # GPU 版本编排
├── requirements.txt         # Python 依赖
├── requirements-linux.txt   # Linux 特定依赖
├── start.sh                 # Linux 启动脚本
├── start.bat                # Windows 启动脚本
├── models/                  # 模型文件存储（持久化）
│   └── .paddleocr/         # PaddleOCR 模型缓存
└── logs/                   # 日志文件存储
```

## ⚙️ 配置说明

### 环境变量
| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PADDLEHUB_HOME` | `/app/models` | PaddleHub 模型缓存目录 |
| `PADDLE_HOME` | `/app/models` | Paddle 框架主目录 |
| `HOME` | `/app/models` | 系统 HOME 目录（用于模型下载） |
| `PYTHONUNBUFFERED` | `1` | Python 输出不缓冲 |

### 端口配置
- **API 服务**: `5104`
- **健康检查**: `http://localhost:5104/health`
- **API 文档**: `http://localhost:5104/swagger`

### 资源配置

#### GPU 模式（唯一支持）
- **内存限制**: 6GB
- **CPU 限制**: 4 核心
- **内存保留**: 3GB
- **CPU 保留**: 2 核心
- **GPU**: 1 个 NVIDIA GPU

## 🔧 故障排除

### 常见问题

#### 1. 容器无法启动
```bash
# 查看详细日志
docker-compose logs ocr-api

# 检查镜像构建
docker-compose build --no-cache
```

#### 2. 模型下载失败
```bash
# 确保网络连接正常
ping paddlepaddle.org.cn

# 检查磁盘空间
df -h

# 清理并重新下载
rm -rf ./models/*
docker-compose restart
```

#### 3. GPU 模式无法使用
```bash
# 检查 NVIDIA Docker
nvidia-docker --version

# 检查 GPU 状态
nvidia-smi

# 验证 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### 4. 端口冲突
```bash
# 查看端口占用
netstat -tlnp | grep 5104

# 修改端口（在 docker-compose.yml 中）
ports:
  - "5105:5104"  # 映射到本地 5105 端口
```

### 日志分析

#### 应用日志
```bash
# 实时查看日志
docker-compose logs -f ocr-api

# 查看最近 100 行日志
docker-compose logs --tail=100 ocr-api
```

#### 系统资源监控
```bash
# 查看容器资源使用
docker stats

# 查看容器内部进程
docker exec -it ocr-ppv5-service top
```

## 🔄 更新部署

### 更新代码
```bash
# 拉取最新代码
git pull origin main

# 重新构建并启动
docker-compose down
docker-compose up --build -d
```

### 更新模型
```bash
# 清理旧模型
rm -rf ./models/.paddleocr/*

# 重启服务（会自动下载最新模型）
docker-compose restart
```

## 📊 性能优化

### CPU 优化
- 调整 worker 进程数量
- 优化 OMP_NUM_THREADS 环境变量
- 使用 SSD 存储模型文件

### GPU 优化
- 确保 CUDA 版本兼容
- 调整 batch_size 参数
- 监控 GPU 内存使用

### 内存优化
- 定期清理临时文件
- 调整引擎池大小
- 监控内存泄漏

## 🔒 安全配置

### 网络安全
- 使用反向代理（Nginx）
- 配置 HTTPS
- 限制访问 IP

### 容器安全
- 非 root 用户运行
- 只读文件系统
- 资源限制

### 数据安全
- 定期备份模型文件
- 日志轮转和清理
- 敏感数据加密

## 📈 监控告警

### 健康检查
- HTTP 健康检查端点
- 容器状态监控
- 自动重启策略

### 性能监控
- API 响应时间
- 内存使用率
- CPU 使用率
- 磁盘空间

### 日志告警
- 错误日志监控
- 异常模式识别
- 自动通知机制

## 🆘 支持与维护

### 联系信息
- 项目仓库: [GitHub/GitLab URL]
- 技术支持: [Support Email]
- 文档中心: [Documentation URL]

### 维护计划
- 定期更新 PaddleOCR 版本
- 安全补丁升级
- 性能优化调整