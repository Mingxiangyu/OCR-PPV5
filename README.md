# OCR-PPV5

基于 PaddleOCR 3.1 的高性能光学字符识别（OCR）服务系统，提供高效、准确的文本识别能力，支持图像和PDF文档中的文字提取。

## 🚀 项目特性

- **高精度识别**：基于 PaddleOCR 3.1 实现业界领先的文字识别精度
- **GPU 加速**：支持 NVIDIA GPU 加速，显著提升处理速度
- **容器化部署**：基于 Docker 的一键部署，环境一致性保障
- **RESTful API**：提供标准化的 HTTP 接口，易于集成
- **模型持久化**：本地缓存模型文件，避免重复下载
- **高并发支持**：引擎池设计，支持多线程并发处理

## 📋 系统要求

### 硬件要求
- **CPU**：4核心及以上推荐
- **内存**：6GB 及以上（推荐 8GB+）
- **GPU**：NVIDIA GPU（可选，用于加速）
- **存储**：2GB 可用空间（用于模型文件）

### 软件环境
- **Docker**：20.10+ 版本
- **Docker Compose**：2.0+ 版本
- **NVIDIA Docker**：GPU 加速时需要

## 🛠️ 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd OCR-PPV5
```

### 2. 构建并启动服务

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 3. 验证服务

服务启动后，访问 http://localhost:5104 验证服务状态。

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PADDLEHUB_HOME` | PaddleOCR 模型存储目录 | `/app/models` |
| `PADDLE_HOME` | PaddlePaddle 框架目录 | `/app/models` |
| `HOME` | 用户主目录 | `/app/models` |
| `NVIDIA_VISIBLE_DEVICES` | 可见GPU设备 | `all` |

### 端口配置

- **服务端口**：5104（HTTP API 服务）

### 存储卷

```yaml
volumes:
  - ./models:/app/models  # 模型文件持久化
  - ./logs:/app/logs      # 日志文件（可选）
```

## 📚 API 文档

### 健康检查

```bash
GET /
```

响应：服务状态信息

### OCR 识别接口

```bash
POST /ocr
Content-Type: multipart/form-data

参数：
- file: 上传的图像文件或PDF文件
```

响应示例：
```json
{
  "status": "success",
  "data": {
    "text": "识别出的文字内容",
    "confidence": 0.95,
    "boxes": [...]
  }
}
```

## 🐳 Docker 部署

### 标准部署

```bash
# 使用 docker-compose（推荐）
docker-compose up -d

# 或直接使用 Docker
docker build -t ocr-ppv5 .
docker run -d -p 5104:5104 -v ./models:/app/models ocr-ppv5
```

### GPU 加速部署

确保已安装 NVIDIA Docker 运行时：

```bash
# 安装 nvidia-container-toolkit
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# 验证 GPU 可用性
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 资源限制

默认配置：
- **内存限制**：6GB
- **CPU 限制**：4 核心
- **内存预留**：3GB
- **CPU 预留**：2 核心

## 🏗️ 开发环境

### 本地开发

```bash
# 1. 安装 Python 3.10
python --version  # 确保版本为 3.10+

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行服务
python app.py
```

### 依赖管理

- `requirements.txt`：通用依赖
- `requirements-linux.txt`：Linux 专用依赖

### 技术栈

- **Python**：3.10
- **PaddleOCR**：3.1
- **PaddlePaddle**：3.0.0 (GPU版本)
- **Flask**：2.3.3
- **OpenCV**：最新版
- **Docker**：容器化部署

## 📊 性能优化

### 模型缓存

首次启动时会自动下载模型文件到 `./models` 目录，后续启动将直接使用缓存模型。

### 并发处理

系统采用引擎池设计，支持多请求并发处理。可通过以下方式调整：

```python
# 在 app.py 中调整引擎池大小
ENGINE_POOL_SIZE = 3  # 根据硬件配置调整
```

### GPU 加速

启用 GPU 加速可显著提升处理速度，特别是批量处理场景。

## 🐛 故障排除

### 常见问题

1. **GPU 相关错误**
   ```
   libcuda.so.1: cannot open shared object file
   ```
   **解决方案**：安装 NVIDIA Docker 运行时或使用 CPU 版本

2. **内存不足**
   ```
   OOM (Out of Memory)
   ```
   **解决方案**：增加系统内存或减少并发数

3. **模型下载失败**
   ```
   Model download failed
   ```
   **解决方案**：检查网络连接，或手动下载模型文件

### 日志查看

```bash
# 查看服务日志
docker-compose logs -f ocr-api

# 查看详细日志
docker-compose logs --tail=100 ocr-api
```

## 📝 更新日志

### v1.0.0
- 基于 PaddleOCR 3.1 的全新架构
- 支持 GPU 加速
- 模型本地化缓存
- Docker 容器化部署
- RESTful API 接口

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🆘 技术支持

如有问题，请通过以下方式获取支持：

- 提交 Issue
- 查看故障排除章节
- 参考 PaddleOCR 官方文档

---

**注意**：首次启动可能需要较长时间下载模型文件，请确保网络连接稳定。