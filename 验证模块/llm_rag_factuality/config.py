"""
全局配置文件
包含模型路径、参数设置、验证阈值等配置信息
"""
import os

# 模拟dotenv功能
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有dotenv，手动加载环境变量
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# ====================== 基础路径配置 ======================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录
RAW_DOC_DIR = os.path.join(PROJECT_ROOT, "data", "raw_docs")
PROCESSED_DOC_DIR = os.path.join(PROJECT_ROOT, "data", "processed_docs")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# 实验结果目录
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ====================== 模型配置 ======================
# 嵌入模型
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# LLM配置
LLM_MODEL = "deepseek-chat"
LLM_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ====================== 文本处理配置 ======================
# 文本分块参数
CHUNK_SIZE = 500  # 分块大小（词数）
CHUNK_OVERLAP = 50  # 分块重叠长度（词数）

# 检索参数
TOP_K_RETRIEVAL = 5  # 检索返回的文档块数量
SIMILARITY_THRESHOLD = 0.7  # 相似度阈值

# ====================== 验证配置 ======================
# 幻觉检测阈值
HALLUCINATION_THRESHOLD = 0.7  # 置信度低于此值认为可能存在幻觉

# 验证级别
VERIFICATION_LEVELS = {
    "basic": "基础验证（规则检查）",
    "semantic": "语义验证（LLM检查）", 
    "comprehensive": "综合验证（基础+语义）"
}

# 默认验证级别
DEFAULT_VERIFICATION_LEVEL = "comprehensive"

# ====================== 纠正配置 ======================
# 纠正轮次
MAX_CORRECTION_ROUNDS = 2  # 最大纠正轮次

# 纠正策略
CORRECTION_STRATEGIES = [
    "基于证据重写",
    "部分纠正",
    "完全重写"
]

# ====================== 实验配置 ======================
# 实验数据集
DATASETS = {
    "truthfulqa": "TruthfulQA数据集",
    "faithdial": "FaithDial数据集",
    "custom": "自定义数据集"
}

# 评估指标
EVALUATION_METRICS = [
    "准确率 (Accuracy)",
    "幻觉率 (Hallucination Rate)", 
    "精确率 (Precision)",
    "召回率 (Recall)",
    "F1分数 (F1 Score)"
]

# ====================== API配置 ======================
# 请求配置
REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数

# 批处理配置
BATCH_SIZE = 10  # 批处理大小
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数

# ====================== 日志配置 ======================
# 日志级别
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 日志文件
LOG_FILE = os.path.join(LOGS_DIR, "rag_system.log")

# ====================== 缓存配置 ======================
# 嵌入缓存
ENABLE_EMBEDDING_CACHE = True
EMBEDDING_CACHE_SIZE = 1000

# 检索缓存
ENABLE_RETRIEVAL_CACHE = True
RETRIEVAL_CACHE_SIZE = 500

# ====================== 性能配置 ======================
# 内存限制
MAX_MEMORY_USAGE = "4GB"  # 最大内存使用

# 磁盘限制
MAX_DISK_USAGE = "10GB"  # 最大磁盘使用

# ====================== 安全配置 ======================
# 内容过滤
ENABLE_CONTENT_FILTER = True
BLOCKED_KEYWORDS = [
    "暴力", "色情", "赌博", "毒品", "诈骗"
]

# ====================== 开发配置 ======================
# 调试模式
DEBUG_MODE = False

# 详细日志
VERBOSE_LOGGING = False

# 测试模式
TEST_MODE = False

# ====================== 目录创建函数 ======================
def create_directories():
    """创建必要的目录结构"""
    directories = [
        RAW_DOC_DIR,
        PROCESSED_DOC_DIR, 
        CHROMA_DB_DIR,
        EXPERIMENTS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        FIGURES_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

# ====================== 配置验证函数 ======================
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查API密钥
    if not DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_API_KEY未设置")
    
    # 检查目录权限
    for directory in [RAW_DOC_DIR, PROCESSED_DOC_DIR, CHROMA_DB_DIR]:
        if not os.access(os.path.dirname(directory), os.W_OK):
            errors.append(f"目录无写权限: {directory}")
    
    # 检查数值参数
    if CHUNK_SIZE <= 0:
        errors.append("CHUNK_SIZE必须大于0")
    
    if TOP_K_RETRIEVAL <= 0:
        errors.append("TOP_K_RETRIEVAL必须大于0")
    
    if HALLUCINATION_THRESHOLD < 0 or HALLUCINATION_THRESHOLD > 1:
        errors.append("HALLUCINATION_THRESHOLD必须在0-1之间")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    print("配置验证通过")

if __name__ == "__main__":
    # 创建目录
    create_directories()
    
    # 验证配置
    validate_config()
    
    print("配置初始化完成")