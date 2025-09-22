import logging
import os
import tempfile
import threading
import traceback
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Queue

import fitz  # PyMuPDF
import numpy as np
import requests
from flask import Flask, redirect
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from werkzeug.datastructures import FileStorage

# 强制CPU模式，避免GPU相关的线程安全问题（可选择启用GPU）
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 设置OpenMP线程数为1，避免OpenBlas多线程冲突
os.environ['OMP_NUM_THREADS'] = '1'

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


def check_and_prepare_models():
    """检查并准备模型文件"""
    logger.info(f"检查模型文件，模型目录: {MODEL_DIR}")
    
    # 检查模型目录是否存在模型文件
    model_files = []
    if os.path.exists(MODEL_DIR):
        for root, dirs, files in os.walk(MODEL_DIR):
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
        # 测试写权限
        test_file = os.path.join(MODEL_DIR, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"模型目录权限检查通过: {MODEL_DIR}")
        return True
    except Exception as e:
        logger.error(f"模型目录权限检查失败: {e}")
        return False


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 文件日志处理
file_handler = RotatingFileHandler(
    os.path.join(ROOT_DIR, 'paddleocr.log'),
    maxBytes=1024 * 1024,
    backupCount=5
)
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 初始化Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 10  # 160 MB
CORS(app)


# 添加根路由重定向（必须在API创建之前）
@app.route('/')
def index():
    """根路径重定向到Swagger文档"""
    return redirect('/swagger')


# 创建带Swagger文档的API
api = Api(app,
          version='1.0',
          title='PaddleOCR V5文字识别API',
          description='基于PaddleOCR V5的图像文字识别API服务',
          doc='/swagger'
          )


# PaddleOCR引擎池类 - 解决线程安全问题
class PaddleOCREnginePool:
    """线程安全的PaddleOCR引擎池"""

    def __init__(self):
        self.pools = {
            'ch': Queue(maxsize=3),      # 中文引擎池
            'en': Queue(maxsize=2),      # 英文引擎池
            'japan': Queue(maxsize=2),   # 日文引擎池
            'korean': Queue(maxsize=2),  # 韩文引擎池
            'server': Queue(maxsize=2),  # Server版本引擎池
        }
        self.pool_locks = {
            'ch': threading.Lock(),
            'en': threading.Lock(),
            'japan': threading.Lock(),
            'korean': threading.Lock(),
            'server': threading.Lock(),
        }
        self._initialize_pools()

    def _initialize_pools(self):
        """预创建所有语言的引擎实例 - 适配PaddleOCR 3.1最极简API"""
        try:
            logger.info(f"开始初始化PaddleOCR引擎池（PaddleOCR 3.1），模型存储目录: {MODEL_DIR}")
            
            # 中文引擎池初始化（默认）
            logger.info("初始化中文引擎池...")
            for i in range(3):
                logger.info(f"创建第{i+1}个中文引擎实例")
                engine = PaddleOCR(lang='ch')  # 只使用lang参数
                self.pools['ch'].put(engine)
                logger.info(f"中文引擎实例{i+1}创建完成")

            # 英文引擎池初始化
            logger.info("初始化英文引擎池...")
            for i in range(2):
                logger.info(f"创建第{i+1}个英文引擎实例")
                engine = PaddleOCR(lang='en')
                self.pools['en'].put(engine)
                logger.info(f"英文引擎实例{i+1}创建完成")

            # 日文引擎池初始化
            logger.info("初始化日文引擎池...")
            for i in range(2):
                logger.info(f"创建第{i+1}个日文引擎实例")
                engine = PaddleOCR(lang='japan')
                self.pools['japan'].put(engine)
                logger.info(f"日文引擎实例{i+1}创建完成")

            # 韩文引擎池初始化
            logger.info("初始化韩文引擎池...")
            for i in range(2):
                logger.info(f"创建第{i+1}个韩文引擎实例")
                engine = PaddleOCR(lang='korean')
                self.pools['korean'].put(engine)
                logger.info(f"韩文引擎实例{i+1}创建完成")

            # Server版本引擎池初始化（高精度模型）
            logger.info("初始化高精度Server引擎池...")
            for i in range(2):
                logger.info(f"创建第{i+1}个Server引擎实例")
                engine = PaddleOCR(lang='ch')  # server也只使用lang参数
                self.pools['server'].put(engine)
                logger.info(f"Server引擎实例{i+1}创建完成")

            logger.info(f"PaddleOCR引擎池初始化成功，支持中英日韩多语言识别，模型存储在: {MODEL_DIR}")

        except Exception as e:
            logger.error(f"PaddleOCR引擎池初始化失败: {e}")
            raise e

    def get_engine(self, lang='ch'):
        """获取指定语言的引擎实例"""
        if lang not in self.pools:
            raise ValueError(f"不支持的语言: {lang}")

        try:
            # 从池中获取引擎实例，超时30秒
            engine = self.pools[lang].get(timeout=30)
            return engine
        except Exception as e:
            logger.error(f"获取{lang}引擎失败: {e}")
            # 如果池为空，创建新的引擎实例
            return self._create_emergency_engine(lang)

    def return_engine(self, lang, engine):
        """归还引擎实例到池中"""
        if lang in self.pools and engine is not None:
            try:
                self.pools[lang].put_nowait(engine)
            except:
                # 如果池已满，丢弃引擎实例
                pass

    def _create_emergency_engine(self, lang):
        """紧急情况下创建新的引擎实例 - 适配PaddleOCR 3.1最极简API"""
        logger.warning(f"创建紧急{lang}引擎实例，使用模型目录: {MODEL_DIR}")
        return PaddleOCR(lang='ch' if lang == 'server' else lang)

    def get_pool_status(self):
        """获取引擎池状态"""
        status = {}
        for lang, pool in self.pools.items():
            status[lang] = {
                'available': pool.qsize(),
                'max_size': pool.maxsize
            }
        return status


# OCR相关命名空间
ocr_ns = api.namespace('ocr', description='PaddleOCR V5文字识别操作')

# 初始化PaddleOCR引擎池
try:
    # 检查和准备模型文件
    if not check_and_prepare_models():
        logger.error("模型目录权限检查失败，可能影响模型下载")
    
    logger.info("开始创建PaddleOCR引擎池...")
    ocr_engine_pool = PaddleOCREnginePool()
    logger.info("PaddleOCR引擎池创建成功")
except Exception as e:
    logger.error(f"PaddleOCR引擎池初始化失败: {e}")
    ocr_engine_pool = None

# 确保临时文件目录存在
TMP_DIR = os.path.join(os.getcwd(), 'picture')
os.makedirs(TMP_DIR, exist_ok=True)

# 定义文件上传解析器
file_parser = api.parser()
file_parser.add_argument('file', location='files',
                         type=FileStorage,
                         required=True,
                         help='要识别的文件（支持图像：jpg/png/bmp/tiff，文档：pdf）')
file_parser.add_argument('lang', location='form',
                         type=str,
                         required=False,
                         default='ch',
                         choices=['ch', 'en', 'japan', 'korean', 'server'],
                         help='识别语言类型：ch(中文), en(英文), japan(日文), korean(韩文), server(高精度中文)')

# URL识别的解析器
url_parser = api.parser()
url_parser.add_argument('url',
                        required=True,
                        help='图像文件的URL')
url_parser.add_argument('lang',
                        required=False,
                        default='ch',
                        choices=['ch', 'en', 'japan', 'korean', 'server'],
                        help='识别语言类型：ch(中文), en(英文), japan(日文), korean(韩文), server(高精度中文)')

# OCR结果响应模型
ocr_model = api.model('OCRResult', {
    'message': fields.Raw(description='OCR识别结果或错误信息', required=True),
    'error_type': fields.String(description='错误类型：HTTP_PARSE|FILE_PROCESS|OCR_ENGINE|SYSTEM_ERROR', required=False),
    'error_details': fields.String(description='详细错误信息', required=False),
    'suggestions': fields.List(fields.String, description='解决建议列表', required=False)
})


def extract_filename_from_url(url):
    """从URL提取文件名"""
    return url.split('/')[-1]


def safe_filename_handler(original_filename):
    """处理中文文件名编码问题，生成安全的临时文件名"""
    try:
        if not original_filename:
            return str(uuid.uuid4()) + '.tmp'

        # 获取文件扩展名
        file_path = Path(original_filename)
        file_ext = file_path.suffix.lower()

        # 生成UUID文件名，保留原始扩展名
        safe_name = str(uuid.uuid4()) + file_ext

        logger.info(f"文件名处理: '{original_filename}' -> '{safe_name}'")
        return safe_name

    except Exception as e:
        logger.warning(f"文件名处理失败: {e}")
        return str(uuid.uuid4()) + '.tmp'


def download_image(url, file_path):
    """从URL下载图像文件"""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def safe_remove_file(file_path):
    """安全删除文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"删除文件失败 {file_path}: {e}")


def diagnose_paddleocr_error(error, file_path, filename):
    """分析PaddleOCR错误类型，提供详细诊断信息"""
    error_str = str(error).lower()
    error_type = "OCR_ENGINE"
    suggestions = []

    # 分析PaddleOCR具体错误类型
    if "cuda out of memory" in error_str or "gpu memory" in error_str:
        error_details = "GPU内存不足，PaddleOCR处理失败"
        suggestions.extend([
            "尝试使用CPU模式",
            "减小batch_size参数",
            "重启服务释放GPU内存"
        ])
    elif "no module named" in error_str or "import" in error_str:
        error_details = "PaddleOCR依赖模块缺失"
        suggestions.extend([
            "检查PaddleOCR是否正确安装",
            "验证Python环境和依赖包",
            "尝试重新安装PaddleOCR: pip install paddleocr"
        ])
    elif "model" in error_str and ("download" in error_str or "load" in error_str):
        error_details = "PaddleOCR模型文件下载或加载失败"
        suggestions.extend([
            "检查网络连接，首次使用需要下载模型",
            "验证模型存储目录权限",
            "清理模型缓存后重试"
        ])
    elif "image format" in error_str or "cannot identify image" in error_str:
        error_details = "图像格式不支持或文件损坏"
        suggestions.extend([
            "检查图像文件是否完整",
            "确认图像格式是否受支持",
            "尝试转换为PNG或JPG格式"
        ])
    elif "memory" in error_str or "out of memory" in error_str:
        error_details = "内存不足导致PaddleOCR处理失败"
        suggestions.extend([
            "尝试处理较小的图像文件",
            "重启服务释放内存",
            "检查系统可用内存"
        ])
    else:
        error_details = f"未知PaddleOCR错误: {error}"
        suggestions.extend([
            "检查图像文件是否完整",
            "验证PaddleOCR安装是否正确",
            "查看详细错误日志"
        ])

    # 记录详细诊断信息
    logger.error(f"PaddleOCR错误诊断 - 文件: {filename}")
    logger.error(f"错误类型: {error_type}")
    logger.error(f"错误详情: {error_details}")
    logger.error(f"建议措施: {'; '.join(suggestions)}")
    logger.error(f"完整错误堆栈: {traceback.format_exc()}")

    return {
        "error_type": error_type,
        "error_details": error_details,
        "suggestions": suggestions
    }


def validate_image_file(file_path):
    """验证图像文件完整性和格式兼容性，并进行必要的预处理"""
    try:
        from PIL import Image
        import tempfile

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False, "文件不存在", file_path

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "文件为空", file_path

        if file_size > 50 * 1024 * 1024:  # 50MB限制
            return False, f"文件过大: {file_size / 1024 / 1024:.1f}MB", file_path

        # 尝试使用PIL打开图像
        try:
            with Image.open(file_path) as img:
                # 验证图像基本属性
                width, height = img.size
                mode = img.mode
                format_name = img.format

                logger.info(f"图像验证成功: {width}x{height}, 模式: {mode}, 格式: {format_name}")

                # 检查图像尺寸是否合理
                if width < 10 or height < 10:
                    return False, f"图像尺寸过小: {width}x{height}", file_path

                if width > 10000 or height > 10000:
                    return False, f"图像尺寸过大: {width}x{height}", file_path
                
                # **重要的预处理**: 处理RGBA模式的图像
                if mode == 'RGBA':
                    logger.info("检测到RGBA模式，转换为RGB以提高OCR识别效果")
                    
                    # 创建RGB图像（白色背景）
                    rgb_img = Image.new('RGB', (width, height), (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    
                    # 保存为新的临时文件
                    converted_path = tempfile.mktemp(suffix='_rgb.jpg')
                    rgb_img.save(converted_path, 'JPEG', quality=95)
                    
                    logger.info(f"RGBA图像已转换为RGB: {converted_path}")
                    return True, "图像文件验证通过（已转换RGBA为RGB）", converted_path
                
                # 处理其他潜在问题模式
                elif mode in ['P', 'L']:
                    logger.info(f"检测到{mode}模式，转换为RGB以确保兼容性")
                    rgb_img = img.convert('RGB')
                    converted_path = tempfile.mktemp(suffix='_rgb.jpg')
                    rgb_img.save(converted_path, 'JPEG', quality=95)
                    
                    logger.info(f"{mode}图像已转换为RGB: {converted_path}")
                    return True, f"图像文件验证通过（已转换{mode}为RGB）", converted_path

                return True, "图像文件验证通过", file_path

        except Exception as img_error:
            return False, f"图像格式错误: {img_error}", file_path

    except Exception as e:
        logger.error(f"图像验证过程出错: {e}")
        return False, f"验证过程出错: {e}", file_path


def pdf_to_images(pdf_path):
    """将PDF转换为图像列表"""
    images = []
    temp_files = []

    try:
        # 方法1: 使用pdf2image
        try:
            pages = convert_from_path(pdf_path, dpi=200)
            for i, page in enumerate(pages):
                # 保存为临时文件
                temp_img_path = tempfile.mktemp(suffix=f'_page_{i}.png')
                page.save(temp_img_path, 'PNG')
                images.append(temp_img_path)
                temp_files.append(temp_img_path)
            logger.info(f"使用pdf2image成功转换PDF，共{len(images)}页")
            return images, temp_files
        except Exception as e:
            logger.warning(f"pdf2image转换失败: {e}")

        # 方法2: 使用PyMuPDF作为备选
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放提高质量
            temp_img_path = tempfile.mktemp(suffix=f'_page_{page_num}.png')
            pix.save(temp_img_path)
            images.append(temp_img_path)
            temp_files.append(temp_img_path)
        doc.close()
        logger.info(f"使用PyMuPDF成功转换PDF，共{len(images)}页")
        return images, temp_files

    except Exception as e:
        logger.error(f"PDF转换失败: {e}")
        # 清理已创建的临时文件
        for temp_file in temp_files:
            safe_remove_file(temp_file)
        return [], []


def convert_paddleocr_to_standard_format(paddleocr_result):
    """将PaddleOCR输出转换为标准格式"""
    if not paddleocr_result:
        logger.warning("PaddleOCR结果为空")
        return []

    results = []
    logger.info(f"开始转换PaddleOCR结果，结果数量: {len(paddleocr_result)}")
    
    # 遍历每个结果
    for i, result_obj in enumerate(paddleocr_result):
        logger.info(f"处理第{i+1}个结果，类型: {type(result_obj)}")
        
        try:
            # 检查是否为新版PaddleOCR详细结果格式
            if isinstance(result_obj, dict) and 'rec_texts' in result_obj and 'rec_polys' in result_obj:
                logger.info("检测到新版PaddleOCR详细结果格式")
                
                rec_texts = result_obj['rec_texts']
                rec_polys = result_obj['rec_polys']
                rec_scores = result_obj.get('rec_scores', [])
                
                logger.info(f"识别到 {len(rec_texts)} 个文本区域")
                
                for j, (text, poly) in enumerate(zip(rec_texts, rec_polys)):
                    try:
                        # 获取置信度
                        confidence = rec_scores[j] if j < len(rec_scores) else 1.0
                        
                        # 转换坐标格式
                        if hasattr(poly, 'tolist'):
                            coords = poly.tolist()
                        else:
                            coords = poly
                        
                        # 确保坐标格式正确
                        if len(coords) == 4 and all(len(point) == 2 for point in coords):
                            # 转换为浮点数
                            formatted_coords = [[float(point[0]), float(point[1])] for point in coords]
                            
                            # 过滤空文本
                            if text and text.strip():
                                results.append([formatted_coords, str(text), float(confidence)])
                                logger.info(f"添加文本: '{text}', 置信度: {confidence:.3f}")
                            else:
                                logger.warning(f"第{j+1}个文本为空: '{text}'")
                        else:
                            logger.warning(f"第{j+1}个坐标格式不正确: {coords}")
                            
                    except Exception as text_error:
                        logger.error(f"处理第{j+1}个文本时出错: {text_error}")
                        continue
                        
            # 检查是否为结构化结果
            elif hasattr(result_obj, 'structure_result'):
                logger.info("检测到结构化结果")
                structure_data = result_obj.structure_result
                if 'texts' in structure_data:
                    for text_item in structure_data['texts']:
                        bbox = text_item.get('bbox', [])
                        text = text_item.get('text', '')
                        confidence = text_item.get('confidence', 1.0)
                        
                        if len(bbox) >= 4 and text and text.strip():
                            coords = [
                                [float(bbox[0]), float(bbox[1])],  # 左上
                                [float(bbox[2]), float(bbox[1])],  # 右上
                                [float(bbox[2]), float(bbox[3])],  # 右下
                                [float(bbox[0]), float(bbox[3])]   # 左下
                            ]
                            results.append([coords, str(text), float(confidence)])
                            logger.info(f"添加结构化文本: {text}")
                            
            # 传统的PaddleOCR格式
            elif isinstance(result_obj, (list, tuple)) and len(result_obj) >= 2:
                logger.info(f"处理传统格式OCR结果: {result_obj}")
                
                bbox = result_obj[0]
                text_info = result_obj[1]
                
                # 提取文本和置信度
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0])
                    confidence = float(text_info[1])
                elif isinstance(text_info, str):
                    text = text_info
                    confidence = 1.0
                else:
                    text = str(text_info)
                    confidence = 1.0
                
                # 过滤空文本
                if text and text.strip():
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        coords = [
                            [float(bbox[0][0]), float(bbox[0][1])],  # 左上
                            [float(bbox[1][0]), float(bbox[1][1])],  # 右上
                            [float(bbox[2][0]), float(bbox[2][1])],  # 右下
                            [float(bbox[3][0]), float(bbox[3][1])]   # 左下
                        ]
                        results.append([coords, text, confidence])
                        logger.info(f"添加传统文本: {text}, 置信度: {confidence}")
                    else:
                        logger.warning(f"bbox格式不正确: {bbox}")
                else:
                    logger.warning(f"文本为空或只包含空白字符: '{text}'")
            else:
                logger.warning(f"未知的结果格式: {type(result_obj)}")
                logger.warning(f"结果内容示例: {str(result_obj)[:500]}...")  # 显示前500个字符
                    
        except Exception as e:
            logger.error(f"处理第{i+1}个结果时出错: {e}")
            logger.error(f"结果内容: {str(result_obj)[:200]}...")  # 显示前200个字符
            continue

    logger.info(f"转换完成，有效结果数量: {len(results)}")
    return results


def process_file_ocr(file_path, filename, lang='ch'):
    """处理文件OCR识别（支持图片和PDF）- 使用PaddleOCR引擎池"""
    if not ocr_engine_pool:
        raise Exception("PaddleOCR引擎池未初始化")

    # 验证语言支持
    if lang not in ['ch', 'en', 'japan', 'korean', 'server']:
        raise Exception(f"不支持的语言: {lang}")

    file_ext = os.path.splitext(filename)[1].lower()
    temp_files_to_clean = [file_path]  # 需要清理的临时文件列表
    all_results = []
    engine = None

    try:
        # 从引擎池获取引擎实例
        engine = ocr_engine_pool.get_engine(lang)
        logger.info(f"获取{lang}引擎成功")

        # 对于图像文件，先进行预处理验证
        if file_ext != '.pdf':
            is_valid, validation_msg, processed_file_path = validate_image_file(file_path)
            if not is_valid:
                raise Exception(f"图像文件验证失败: {validation_msg}")
            logger.info(f"图像文件验证通过: {validation_msg}")
            
            # 如果文件被转换，更新文件路径并添加到清理列表
            if processed_file_path != file_path:
                file_path = processed_file_path
                temp_files_to_clean.append(processed_file_path)

        if file_ext == '.pdf':
            # PDF文件处理
            logger.info(f"处理PDF文件: {filename} (语言: {lang})")
            image_paths, pdf_temp_files = pdf_to_images(file_path)
            temp_files_to_clean.extend(pdf_temp_files)

            if not image_paths:
                raise Exception("PDF转换为图像失败")

            # 对每一页进行OCR识别
            for i, img_path in enumerate(image_paths):
                try:
                    # 验证转换后的图像
                    is_valid, validation_msg, processed_img_path = validate_image_file(img_path)
                    if not is_valid:
                        logger.warning(f"PDF第{i + 1}页图像验证失败: {validation_msg}")
                        continue
                    
                    # 如果图像被转换，更新路径并添加到清理列表
                    if processed_img_path != img_path:
                        img_path = processed_img_path
                        temp_files_to_clean.append(processed_img_path)

                    # 使用PaddleOCR进行识别
                    logger.info(f"调用PaddleOCR识别PDF第{i+1}页: {img_path}")
                    ocr_output = engine.predict(img_path)
                    
                    # 处理PaddleOCR结果
                    page_results = []
                    if ocr_output:
                        logger.info(f"PDF第{i+1}页OCR输出: {ocr_output[:2] if len(ocr_output) > 2 else ocr_output}")
                        # 直接处理PaddleOCR返回的结果列表
                        converted = convert_paddleocr_to_standard_format(ocr_output)
                        page_results.extend(converted)
                    else:
                        logger.warning(f"PDF第{i+1}页OCR返回空结果")

                    # 为每页结果添加页码信息
                    for item in page_results:
                        if len(item) >= 3:
                            coords = item[0]
                            text = item[1]
                            confidence = item[2]
                            all_results.append(
                                [coords, f"[第{i + 1}页] {text}", confidence])

                    logger.info(f"PDF第{i + 1}页识别完成，识别到 {len(page_results)} 个文本区域")
                except Exception as e:
                    logger.error(f"PDF第{i + 1}页识别失败: {e}")
                    continue
        else:
            # 图像文件处理
            logger.info(f"处理图像文件: {filename} (语言: {lang})")
            try:
                # 使用PaddleOCR进行识别
                logger.info(f"调用PaddleOCR引擎识别文件: {file_path}")
                ocr_output = engine.predict(file_path)
                
                # 详细记录OCR输出结果
                logger.info(f"PaddleOCR原始输出类型: {type(ocr_output)}")
                logger.info(f"PaddleOCR原始输出长度: {len(ocr_output) if ocr_output else 0}")
                if ocr_output:
                    logger.info(f"PaddleOCR原始输出内容: {ocr_output[:2] if len(ocr_output) > 2 else ocr_output}")
                
                # 处理PaddleOCR结果
                if ocr_output:
                    # 直接处理PaddleOCR返回的结果列表
                    converted_results = convert_paddleocr_to_standard_format(ocr_output)
                    all_results.extend(converted_results)
                    logger.info(f"转换后的结果数量: {len(converted_results)}")
                else:
                    logger.warning("PaddleOCR返回空结果")
                
                logger.info(f"图像OCR处理完成，识别到 {len(all_results)} 个文本区域")
            except Exception as ocr_error:
                # 详细的OCR错误诊断
                diagnosis = diagnose_paddleocr_error(ocr_error, file_path, filename)
                raise Exception(f"OCR处理失败: {diagnosis['error_details']}")

    except Exception as e:
        # 记录详细错误信息
        logger.error(f"文件OCR处理失败 - 文件: {filename}, 错误: {e}")
        raise e
    finally:
        # 归还引擎实例到池中
        if engine and ocr_engine_pool:
            ocr_engine_pool.return_engine(lang, engine)
            logger.info(f"归还{lang}引擎到池中")

        # 清理所有临时文件
        for temp_file in temp_files_to_clean:
            safe_remove_file(temp_file)

    return all_results


def convert_np_float32(result):
    """将numpy.float32转换为Python原生float"""
    if not result:
        return result
    for inner_array in result:
        # 修复 isinstance 检查
        if len(inner_array) >= 3 and hasattr(inner_array[2], 'dtype'):
            # 棄用isinstance检查，直接转换
            try:
                inner_array[2] = float(inner_array[2])
            except (ValueError, TypeError):
                pass  # 如果转换失败，保持原值
    return result


def log_ocr_performance(lang, processing_time, success, result_count=0):
    """记录OCR性能指标"""
    status = "成功" if success else "失败"
    logger.info(f"OCR性能统计 - 语言: {lang}, 耗时: {processing_time:.2f}s, 状态: {status}, 识别数量: {result_count}")

    # 记录引擎池状态
    if ocr_engine_pool:
        pool_status = ocr_engine_pool.get_pool_status()
        logger.info(f"引擎池状态: {pool_status}")


def get_engine_pool_health():
    """获取引擎池健康状态"""
    if not ocr_engine_pool:
        return {"status": "error", "message": "引擎池未初始化"}

    try:
        pool_status = ocr_engine_pool.get_pool_status()
        total_engines = sum(status['max_size'] for status in pool_status.values())
        available_engines = sum(status['available'] for status in pool_status.values())

        return {
            "status": "healthy",
            "total_engines": total_engines,
            "available_engines": available_engines,
            "pool_details": pool_status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@ocr_ns.route('/file')
class OCRFromFile(Resource):
    @api.expect(file_parser)
    @api.marshal_with(ocr_model)
    def post(self):
        """
        从上传的文件进行OCR识别 - 使用PaddleOCR V5引擎
        支持图像文件（jpg/png/bmp/tiff）和PDF文档
        支持中英日韩多语言识别，线程安全，高精度识别
        """
        temp_file_path = None
        original_filename = None

        try:
            # HTTP层错误处理 - 捕获请求解析错误
            try:
                args = file_parser.parse_args()
                uploaded_file = args['file']
                lang = args.get('lang', 'ch')
            except Exception as parse_error:
                logger.error(f"HTTP请求解析失败: {parse_error}")
                return {
                    "message": "请求格式错误",
                    "error_type": "HTTP_PARSE",
                    "error_details": f"无法解析上传请求: {str(parse_error)}",
                    "suggestions": [
                        "确认使用multipart/form-data格式上传文件",
                        "检查文件字段名是否为'file'",
                        "验证请求头Content-Type是否正确"
                    ]
                }, 400

            if not uploaded_file or not uploaded_file.filename:
                return {
                    "message": "未提供有效文件",
                    "error_type": "HTTP_PARSE",
                    "error_details": "文件字段为空或文件名缺失",
                    "suggestions": ["确认已选择文件进行上传", "检查文件字段名是否正确"]
                }, 400

            # 文件处理层错误处理
            try:
                original_filename = uploaded_file.filename
                logger.info(f"接收到文件上传请求: {original_filename}")

                # 检查文件类型
                file_ext = os.path.splitext(original_filename)[1].lower()
                supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']

                if file_ext not in supported_formats:
                    return {
                        "message": f"不支持的文件格式: {file_ext}",
                        "error_type": "FILE_PROCESS",
                        "error_details": f"文件格式 {file_ext} 不在支持列表中",
                        "suggestions": [f"支持的格式: {', '.join(supported_formats)}", "请转换文件格式后重试"]
                    }, 400

                # 使用安全的文件名处理
                safe_filename = safe_filename_handler(original_filename)
                temp_file_path = tempfile.mktemp(suffix=file_ext)

                # 保存上传文件
                uploaded_file.save(temp_file_path)
                logger.info(f"文件已保存到临时路径: {temp_file_path}")

            except Exception as file_error:
                logger.error(f"文件处理失败: {file_error}")
                return {
                    "message": "文件处理失败",
                    "error_type": "FILE_PROCESS",
                    "error_details": f"文件保存或处理过程出错: {str(file_error)}",
                    "suggestions": [
                        "检查文件是否损坏",
                        "确认文件大小是否超出限制",
                        "尝试重新上传文件"
                    ]
                }, 500

            logger.info(f"开始OCR处理: {original_filename} ({file_ext}, 语言: {lang})")

            # OCR处理层错误处理
            try:
                import time
                start_time = time.time()

                result = process_file_ocr(temp_file_path, original_filename, lang)

                processing_time = time.time() - start_time

                if not result:
                    log_ocr_performance(lang, processing_time, False, 0)
                    return {
                        "message": "未识别到任何文字内容",
                        "error_type": "OCR_ENGINE",
                        "error_details": "OCR处理完成但未检测到文字",
                        "suggestions": [
                            "确认图像中包含清晰的文字内容",
                            "尝试提高图像质量或分辨率",
                            "检查图像是否为正确的方向"
                        ]
                    }, 200

                # 转换numpy类型以确保JSON序列化
                result = convert_np_float32(result)
                log_ocr_performance(lang, processing_time, True, len(result))
                logger.info(f"PaddleOCR处理成功完成: {original_filename}, 耗时: {processing_time:.2f}s")
                return {"message": result}, 200

            except Exception as ocr_error:
                # 记录失败的性能统计
                processing_time = 0
                if 'start_time' in locals():
                    import time
                    processing_time = time.time() - start_time
                log_ocr_performance(lang, processing_time, False, 0)

                # 使用详细的OCR错误诊断
                diagnosis = diagnose_paddleocr_error(ocr_error, temp_file_path, original_filename)
                logger.error(f"PaddleOCR处理失败: {original_filename}")

                return {
                    "message": "OCR识别失败",
                    "error_type": diagnosis["error_type"],
                    "error_details": diagnosis["error_details"],
                    "suggestions": diagnosis["suggestions"]
                }, 500

        except Exception as unexpected_error:
            # 捕获所有未预期的错误
            logger.error(f'[OCR文件]未预期错误: {unexpected_error}')
            logger.error(f'完整错误堆栈: {traceback.format_exc()}')
            return {
                "message": "服务器内部错误",
                "error_type": "SYSTEM_ERROR",
                "error_details": f"未预期的系统错误: {str(unexpected_error)}",
                "suggestions": [
                    "请稍后重试",
                    "如问题持续存在，请联系技术支持",
                    "检查服务器日志获取更多信息"
                ]
            }, 500
        finally:
            # 确保临时文件被清理
            if temp_file_path:
                safe_remove_file(temp_file_path)


@ocr_ns.route('/url')
class OCRFromURL(Resource):
    @api.expect(url_parser)
    @api.marshal_with(ocr_model)
    def post(self):
        """
        从URL识别图像文字 - 使用PaddleOCR V5引擎
        提供图像文件的URL并获取文字识别结果
        支持中英日韩多语言识别，线程安全，高精度识别
        """
        temp_file_path = None
        try:
            args = url_parser.parse_args()
            url = args['url']
            lang = args.get('lang', 'ch')

            # 提取文件名并下载到临时文件
            file_name = extract_filename_from_url(url)
            file_ext = os.path.splitext(file_name)[1].lower()
            temp_file_path = tempfile.mktemp(suffix=file_ext)

            download_image(url, temp_file_path)
            logger.info(f"从URL下载文件: {url} (语言: {lang})")

            # 处理文件OCR识别
            import time
            start_time = time.time()

            result = process_file_ocr(temp_file_path, file_name, lang)

            processing_time = time.time() - start_time

            if not result:
                log_ocr_performance(lang, processing_time, False, 0)
                return {"message": "未识别到任何文字内容"}, 200

            # 转换numpy类型以确保JSON序列化
            result = convert_np_float32(result)
            log_ocr_performance(lang, processing_time, True, len(result))
            logger.info(f"PaddleOCR URL处理成功: {url}, 耗时: {processing_time:.2f}s")
            return {"message": result}, 200

        except Exception as e:
            # 记录失败的性能统计
            processing_time = 0
            lang = args.get('lang', 'ch') if 'args' in locals() else 'ch'
            if 'start_time' in locals():
                import time
                processing_time = time.time() - start_time
            log_ocr_performance(lang, processing_time, False, 0)

            logger.error(f'[PaddleOCR URL]错误: {e}')
            return {'message': f'识别失败: {str(e)}'}, 500
        finally:
            # 确保临时文件被清理
            if temp_file_path:
                safe_remove_file(temp_file_path)


@ocr_ns.route('/debug')
class OCRDebug(Resource):
    @api.expect(file_parser)
    def post(self):
        """
        调试模式OCR识别 - 提供详细的诊断信息
        帮助分析为什么OCR无法识别文字
        """
        temp_file_path = None
        original_filename = None

        try:
            args = file_parser.parse_args()
            uploaded_file = args['file']
            lang = args.get('lang', 'ch')

            if not uploaded_file or not uploaded_file.filename:
                return {"error": "未提供有效文件"}, 400

            original_filename = uploaded_file.filename
            file_ext = os.path.splitext(original_filename)[1].lower()
            temp_file_path = tempfile.mktemp(suffix=file_ext)
            uploaded_file.save(temp_file_path)

            debug_info = {
                "file_info": {
                    "filename": original_filename,
                    "file_size": os.path.getsize(temp_file_path),
                    "file_extension": file_ext
                },
                "image_validation": {},
                "ocr_result": {},
                "engine_status": {}
            }

            # 1. 图像验证
            try:
                from PIL import Image
                with Image.open(temp_file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format
                    
                    debug_info["image_validation"] = {
                        "is_valid": True,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "format": format_name,
                        "size_mb": round(os.path.getsize(temp_file_path) / 1024 / 1024, 2)
                    }
                    
                    # 检查潜在问题
                    issues = []
                    if width < 50 or height < 50:
                        issues.append("图像尺寸可能太小")
                    if width > 8000 or height > 8000:
                        issues.append("图像尺寸可能太大")
                    if mode == 'RGBA':
                        issues.append("RGBA模式可能影响识别，建议转换为RGB")
                    
                    debug_info["image_validation"]["potential_issues"] = issues
                    
            except Exception as img_error:
                debug_info["image_validation"] = {
                    "is_valid": False,
                    "error": str(img_error)
                }

            # 2. 引擎状态检查
            if ocr_engine_pool:
                debug_info["engine_status"] = ocr_engine_pool.get_pool_status()
            else:
                debug_info["engine_status"] = {"error": "引擎池未初始化"}

            # 3. OCR识别测试
            if ocr_engine_pool and debug_info["image_validation"].get("is_valid", False):
                try:
                    engine = ocr_engine_pool.get_engine(lang)
                    
                    # 记录引擎调用前状态
                    logger.info(f"[调试模式] 开始调用{lang}引擎")
                    
                    # 调用OCR引擎
                    ocr_output = engine.predict(temp_file_path)
                    
                    # 详细记录输出
                    debug_info["ocr_result"] = {
                        "engine_called": True,
                        "raw_output_type": str(type(ocr_output)),
                        "raw_output_length": len(ocr_output) if ocr_output else 0,
                        "raw_output_sample": str(ocr_output[:1]) if ocr_output else "空结果"
                    }
                    
                    if ocr_output:
                        # 尝试处理结果
                        try:
                            converted_results = convert_paddleocr_to_standard_format(ocr_output)
                            debug_info["ocr_result"]["converted_results_count"] = len(converted_results)
                            debug_info["ocr_result"]["sample_results"] = converted_results[:3] if converted_results else []
                            
                            # 提取所有识别到的文字
                            all_texts = []
                            for result in converted_results:
                                if len(result) >= 2:
                                    all_texts.append(result[1])
                            debug_info["ocr_result"]["all_texts"] = all_texts
                            
                        except Exception as convert_error:
                            debug_info["ocr_result"]["conversion_error"] = str(convert_error)
                    else:
                        debug_info["ocr_result"]["issue"] = "PaddleOCR返回空结果"
                    
                    # 归还引擎
                    ocr_engine_pool.return_engine(lang, engine)
                    
                except Exception as ocr_error:
                    debug_info["ocr_result"] = {
                        "engine_called": False,
                        "error": str(ocr_error)
                    }
            else:
                debug_info["ocr_result"] = {
                    "engine_called": False,
                    "reason": "引擎池未初始化或图像验证失败"
                }

            return {"debug_info": debug_info}, 200

        except Exception as e:
            return {"error": f"调试过程出错: {str(e)}"}, 500
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                safe_remove_file(temp_file_path)


@ocr_ns.route('/health')
class OCRHealth(Resource):
    def get(self):
        """
        获取PaddleOCR引擎池健康状态
        返回引擎池的详细状态信息，用于监控和诊断
        """
        try:
            health_status = get_engine_pool_health()
            return health_status, 200
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "error",
                "message": f"健康检查失败: {str(e)}"
            }, 500


@ocr_ns.route('/models')
class OCRModels(Resource):
    def get(self):
        """
        获取支持的模型和语言信息
        """
        try:
            models_info = {
                "supported_languages": {
                    "ch": "中文（默认PP-OCRv5模型）",
                    "en": "英文",
                    "japan": "日文",
                    "korean": "韩文",
                    "server": "中文高精度（PP-OCRv5 Server模型）"
                },
                "supported_formats": [
                    "jpg", "jpeg", "png", "bmp", "tiff", "tif", "pdf"
                ],
                "model_versions": {
                    "default": "PP-OCRv5",
                    "server": "PP-OCRv5 Server (高精度版本)"
                },
                "features": [
                    "多语言支持",
                    "PDF文档识别",
                    "线程安全",
                    "引擎池管理",
                    "高精度识别"
                ]
            }
            return models_info, 200
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {"error": f"获取模型信息失败: {str(e)}"}, 500


if __name__ == '__main__':
    try:
        web_address = '0.0.0.0:5104'
        host, port = web_address.split(':')

        # 启动前检查引擎池状态
        health_status = get_engine_pool_health()
        logger.info(f"PaddleOCR引擎池健康检查: {health_status}")

        if health_status['status'] != 'healthy':
            logger.warning("引擎池状态异常，但继续启动服务")

        logger.info(f"启动PaddleOCR V5 API服务器 - 地址: http://{web_address}")
        logger.info("API文档地址: http://{}/swagger".format(web_address))
        logger.info("支持的语言: 中文(ch), 英文(en), 日文(japan), 韩文(korean), 高精度中文(server)")
        logger.info(f"模型存储目录: {MODEL_DIR}")

        app.run(host=host, port=int(port), debug=True)

    except Exception as e:
        logger.error(f"[PaddleOCR App]启动错误: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")