# PaddleOCR 3.1 模型本地化配置说明

## 📋 修改概述

针对 PaddleOCR 3.1 版本，我们对 `app.py` 进行了以下关键修改，确保模型优先从同级目录加载，如果没有则下载到同级目录。

## 🔧 主要修改内容

### 1. 环境变量配置（第10-17行）

```python
# 配置日志和模型目录
ROOT_DIR = os.getcwd()
# 设置模型存储目录为当前工作目录下的models文件夹
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 设置PaddleOCR模型下载目录的环境变量
# PaddleOCR 3.1版本通过环境变量控制模型存储位置
os.environ['PADDLEHUB_HOME'] = MODEL_DIR  # PaddleHub模型缓存目录
os.environ['PADDLE_HOME'] = MODEL_DIR      # Paddle框架主目录
# 创建.paddleocr子目录用于存储模型
paddle_cache_dir = os.path.join(MODEL_DIR, '.paddleocr')
os.makedirs(paddle_cache_dir, exist_ok=True)
os.environ['HOME'] = MODEL_DIR             # 某些情况下PaddleOCR会使用HOME/.paddleocr
```

### 2. 模型检查函数更新（第19-40行）

```python
def check_and_prepare_models():
    """检查并准备模型文件"""
    logger.info(f"检查模型文件，模型目录: {MODEL_DIR}")
    
    # 检查模型目录是否存在模型文件
    model_files = []
    paddleocr_dir = os.path.join(MODEL_DIR, '.paddleocr')
    if os.path.exists(paddleocr_dir):
        for root, dirs, files in os.walk(paddleocr_dir):
            for file in files:
                if file.endswith(('.pdmodel', '.pdiparams', '.pdopt')):
                    model_files.append(os.path.join(root, file))
    
    if model_files:
        logger.info(f"在模型目录中找到 {len(model_files)} 个模型文件")
        logger.info(f"模型文件示例: {model_files[:3]}")
    else:
        logger.info("模型目录中暂无模型文件，首次运行时将自动下载到该目录")
    
    # 确保目录存在且有写权限
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(paddleocr_dir, exist_ok=True)
        # 测试写权限...
        return True
    except Exception as e:
        logger.error(f"模型目录权限检查失败: {e}")
        return False
```

### 3. PaddleOCR 初始化适配 3.1 版本（第90-140行）

```python
def _initialize_pools(self):
    """预创建所有语言的引擎实例 - 适配PaddleOCR 3.1"""
    try:
        logger.info(f"开始初始化PaddleOCR引擎池（PaddleOCR 3.1），模型存储目录: {MODEL_DIR}")
        
        # 中文引擎池初始化（默认）
        for i in range(3):
            engine = PaddleOCR(
                use_angle_cls=False,     # PaddleOCR 3.1 使用 use_angle_cls 替代 use_doc_orientation_classify
                lang='ch',
                use_gpu=False,           # 使用CPU，如需GPU请设置True
                show_log=False           # 关闭详细日志
            )
            self.pools['ch'].put(engine)
        
        # 其他语言引擎类似配置...
        
        logger.info(f"PaddleOCR引擎池初始化成功，支持中英日韩多语言识别，模型存储在: {MODEL_DIR}")
```

### 4. 紧急引擎创建适配（第190-210行）

```python
def _create_emergency_engine(self, lang):
    """紧急情况下创建新的引擎实例 - 适配PaddleOCR 3.1"""
    if lang == 'server':
        return PaddleOCR(
            use_angle_cls=False,
            lang='ch',
            use_gpu=False,
            show_log=False,
            det_model_dir=None,     # 让PaddleOCR自动处理
            rec_model_dir=None
        )
    else:
        return PaddleOCR(
            use_angle_cls=False,
            lang=lang,
            use_gpu=False,
            show_log=False
        )
```

## 📦 依赖文件更新

### requirements.txt 和 requirements-linux.txt

```
paddlepaddle-gpu==3.1.0
paddleocr
flask==2.3.3
flask-cors==4.0.0
flask-restx==1.1.0
numpy
opencv-python
Pillow
pdf2image
PyMuPDF
requests
Werkzeug==2.3.7
```

## 🗂️ 目录结构

修改后的项目目录结构：

```
OCR-PPV5/
├── app.py                    # 主应用（已适配PaddleOCR 3.1）
├── models/                   # 模型存储目录（新增）
│   └── .paddleocr/          # PaddleOCR模型缓存（自动创建）
│       ├── whl/             # 模型文件（自动下载）
│       └── ...
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # 已更新为PaddleOCR 3.1
├── requirements-linux.txt    # 已更新为PaddleOCR 3.1
├── test_paddleocr31.py      # PaddleOCR 3.1测试脚本（新增）
└── demo_model_config.py     # 配置演示脚本
```

## ✨ 修改效果

1. **✅ 本地优先加载**: 如果 `models/` 目录中已有模型文件，优先使用本地模型
2. **✅ 自动下载管理**: 首次运行时自动下载模型到指定目录
3. **✅ 版本兼容**: 完全适配 PaddleOCR 3.1 的 API 变化
4. **✅ 离线部署**: 支持提前下载模型用于离线环境
5. **✅ 统一管理**: 所有模型文件集中存储在 `models/` 目录

## 🚀 运行流程

1. **应用启动**: 检查 `models/.paddleocr/` 目录中的模型文件
2. **环境设置**: 通过环境变量指导 PaddleOCR 使用指定目录
3. **模型加载**: 
   - 如有本地模型文件 → 直接加载使用
   - 如无本地模型文件 → 自动下载到指定目录
4. **后续运行**: 直接使用本地模型文件，无需重复下载

## ⚠️ 重要说明

1. **版本升级**: 代码已从 PaddleOCR 2.7.0 升级适配到 3.1 版本
2. **参数变化**: `use_doc_orientation_classify` 等参数在 3.1 版本中已更改为 `use_angle_cls`
3. **GPU支持**: 如需使用GPU，请将 `use_gpu=False` 改为 `use_gpu=True`
4. **首次运行**: 首次运行时需要下载模型，请确保网络连接正常

## 🐳 Docker 部署

### 快速启动

1. **使用启动脚本（推荐）**：
   ```bash
   # Linux/Mac
   chmod +x start.sh
   ./start.sh
   
   # Windows
   start.bat
   ```

2. **手动启动**：
   ```bash
   # CPU 模式
   docker-compose up --build
   
   # GPU 模式（需要 NVIDIA Docker）
   docker-compose -f docker-compose.gpu.yml up --build
   ```

### 容器配置特性

- **模型持久化**: 通过 `./models:/app/models` 卷挂载确保模型文件持久化
- **日志挂载**: 通过 `./logs:/app/logs` 卷挂载便于查看日志
- **环境变量**: 自动设置 PaddleOCR 3.1 所需的环境变量
- **健康检查**: 内置健康检查确保服务正常运行
- **资源限制**: 合理的 CPU 和内存限制配置

### 端口说明

- **服务端口**: `5104` (已从原来的 5103 更新)
- **API 文档**: `http://localhost:5104/swagger`
- **健康检查**: `http://localhost:5104/health`

## 🎯 与原版本的主要区别

| 项目 | 原版本 (2.7.0) | 新版本 (3.1) |
|------|----------------|---------------|
| 参数名称 | `use_doc_orientation_classify` | `use_angle_cls` |
| 模型控制 | `model_dir`, `download_dir` | 环境变量控制 |
| 高精度模式 | `text_detection_model_name` | `det_model_dir`, `rec_model_dir` |
| 日志控制 | 内置参数较少 | `show_log` 参数 |

现在您的应用已经完全适配 PaddleOCR 3.1，并且实现了模型文件的本地化管理！