import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件中的敏感信息（API密钥等）
load_dotenv()

# ========================= 基础路径配置 =========================
# 项目根目录（自动适配不同运行环境）
PROJECT_ROOT = Path(__file__).parent.resolve()

# 数据存储路径
RAW_DOCS_PATH = PROJECT_ROOT / "data" / "raw_docs"  # 原始文档路径
PROCESSED_DOCS_PATH = PROJECT_ROOT / "data" / "processed_docs"  # 处理后文档路径
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"  # Chroma向量库持久化路径
DATASETS_PATH = PROJECT_ROOT / "experiments" / "datasets"  # 实验数据集路径

# 结果输出路径
LOGS_PATH = PROJECT_ROOT / "results" / "logs"  # 日志文件路径
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # 可视化图表路径

# 创建必要目录（不存在则自动生成）
for path in [
    RAW_DOCS_PATH,
    PROCESSED_DOCS_PATH,
    CHROMA_DB_PATH,
    DATASETS_PATH,
    LOGS_PATH,
    FIGURES_PATH,
]:
    path.mkdir(parents=True, exist_ok=True)

# ========================= 模型与API配置 =========================
# LLM（DeepseekV3）配置
DEEPSEEK_API_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),  # 从.env读取API密钥
    "base_url": "https://api.deepseek.com/v1/chat/completions",  # Deepseek API基础URL
    "model_name": "deepseek-chat",  # 模型名称（DeepseekV3对应型号）
    "temperature": 0.3,  # 生成温度（越低越稳定，减少幻觉）
    "max_tokens": 2048,  # 最大生成长度
    "top_p": 0.9,  # 采样Top-P参数
    "timeout": 30,  # API请求超时时间（秒）
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-base-en-v1.5",  # 嵌入模型名称
    "embedding_dim": 768,  # 模型输出向量维度（bge-base-en-v1.5默认768）
    "device": "auto",  # 运行设备（auto自动识别GPU/CPU）
    "normalize_embeddings": True,  # 是否归一化嵌入向量（提升检索精度）
}

# ========================= 核心流程参数 =========================
# 文本分块配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,  # 每个分块的词数（默认500词）
    "chunk_overlap": 50,  # 分块重叠词数（默认50词，保证上下文连贯性）
    "separators": ["\n\n", "\n", ". ", " ", ""],  # 分块分隔符优先级
}

# 检索模块配置
RETRIEVAL_CONFIG = {
    "top_k": 5,  # 检索返回的Top-K相关文档块（默认5）
    "similarity_threshold": 0.5,  # 相似度阈值（低于此值的结果将被过滤，可选）
    "search_type": "similarity",  # 检索类型（similarity/MMR，MMR可减少结果冗余）
}

# 验证-纠正流程配置
FACT_CHECK_CONFIG = {
    "verification_rounds": 2,  # 验证-纠正轮数（默认2轮）
    "hallucination_threshold": 0.6,  # 幻觉判定阈值（0-1，越高越严格）
    "use_rule_based_check": True,  # 是否启用规则式校验（辅助LLM验证，提升效率）
}

# ========================= 实验与日志配置 =========================
# 实验配置
EXPERIMENT_CONFIG = {
    "default_dataset": "TruthfulQA",  # 默认测试数据集
    "test_sample_count": None,  # 测试样本数量（None表示全量测试）
    "evaluation_metrics": [
        "hallucination_rate",  # 幻觉率（核心指标）
        "fact_accuracy",  # 事实准确率
        "response_relevance",  # 回答相关性
    ],  # 需计算的评估指标
    "save_results": True,  # 是否保存实验结果到results目录
}

# 日志配置
LOG_CONFIG = {
    "log_level": "INFO",  # 日志级别（DEBUG/INFO/WARNING/ERROR，开发时用DEBUG）
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 日志格式
    "log_file_name": "system.log",  # 日志文件名
    "log_rotation": "daily",  # 日志轮转策略（daily/weekly/monthly）
}

# ========================= 通用配置 =========================
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")  # 环境标识（dev/test/prod）
DEBUG_MODE = ENVIRONMENT == "dev"  # 调试模式（开发环境自动开启）
PRINT_PROCESS_LOG = True  # 是否打印流程日志（交互式运行时启用）