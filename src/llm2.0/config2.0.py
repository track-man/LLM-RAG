import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件中的敏感信息
load_dotenv()

# ========================= 基础路径配置 =========================
PROJECT_ROOT = Path(__file__).parent.resolve()

# 数据存储路径
RAW_DOCS_PATH = PROJECT_ROOT / "data" / "cleaned_data"
PROCESSED_DOCS_PATH = PROJECT_ROOT / "data" / "processed_docs"
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
DATASETS_PATH = PROJECT_ROOT / "experiments" / "datasets"

# 结果输出路径
LOGS_PATH = PROJECT_ROOT / "results" / "logs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

# 创建必要目录
for path in [
    RAW_DOCS_PATH, PROCESSED_DOCS_PATH, CHROMA_DB_PATH,
    DATASETS_PATH, LOGS_PATH, FIGURES_PATH
]:
    path.mkdir(parents=True, exist_ok=True)

# ========================= 加载YAML配置 =========================
def load_yaml_config():
    """从YAML文件加载配置"""
    yaml_config_path = PROJECT_ROOT / "config.yaml"
    if yaml_config_path.exists():
        with open(yaml_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

# 加载YAML配置
YAML_CONFIG = load_yaml_config()

# ========================= 模型与API配置 =========================
# LLM配置（优先使用YAML配置）
LLM_CONFIG = {
    "provider": YAML_CONFIG.get('llm', {}).get('provider', 'deepseek'),
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "base_url": YAML_CONFIG.get('llm', {}).get('base_url', 'https://api.deepseek.com/v1'),
    "model_name": YAML_CONFIG.get('llm', {}).get('model', 'deepseek-chat'),
    "temperature": YAML_CONFIG.get('llm', {}).get('temperature', 0.1),
    "max_tokens": YAML_CONFIG.get('llm', {}).get('max_tokens', 1000),
    "top_p": 0.9,
    "timeout": 30,
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_name": YAML_CONFIG.get('vector_db', {}).get('embedding_model', 'BAAI/bge-base-en-v1.5'),
    "embedding_dim": 768,
    "device": "auto",
    "normalize_embeddings": True,
}

# 向量数据库配置
VECTOR_DB_CONFIG = {
    "embedding_model": EMBEDDING_CONFIG["model_name"],
    "db_path": YAML_CONFIG.get('vector_db', {}).get('db_path', str(CHROMA_DB_PATH)),
    "collection_name": YAML_CONFIG.get('vector_db', {}).get('collection_name', 'knowledge_base'),
    "top_k": YAML_CONFIG.get('vector_db', {}).get('top_k', 5),
}

# ========================= 核心流程参数 =========================
# 文本分块配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", " ", ""],
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": VECTOR_DB_CONFIG["top_k"],
    "similarity_threshold": YAML_CONFIG.get('retrieval', {}).get('similarity_threshold', 0.7),
    "max_retrieved_docs": YAML_CONFIG.get('retrieval', {}).get('max_retrieved_docs', 10),
    "search_type": "similarity",
}

# 验证-纠正流程配置
FACT_CHECK_CONFIG = {
    "verification_rounds": 2,
    "hallucination_threshold": 0.7,
    "confidence_threshold": YAML_CONFIG.get('verification', {}).get('confidence_threshold', 0.8),
    "max_verification_attempts": YAML_CONFIG.get('verification', {}).get('max_verification_attempts', 3),
    "use_rule_based_check": True,
}

# 意图分类配置
INTENT_CONFIG = {
    "supported_intents": YAML_CONFIG.get('intent', {}).get('supported_intents', ["事实查询", "比较查询", "方法查询", "观点查询"]),
    "default_intent": YAML_CONFIG.get('intent', {}).get('default_intent', "事实查询"),
}

# ========================= 实验与日志配置 =========================
EXPERIMENT_CONFIG = {
    "default_dataset": "TruthfulQA",
    "test_sample_count": None,
    "evaluation_metrics": [
        "hallucination_rate", "fact_accuracy", "response_relevance"
    ],
    "save_results": True,
}

LOG_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file_name": "system.log",
    "log_rotation": "daily",
}

# ========================= 通用配置 =========================
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
DEBUG_MODE = ENVIRONMENT == "dev"
PRINT_PROCESS_LOG = True

# ========================= 配置验证 =========================
def validate_config():
    """验证关键配置"""
    # 验证LLM配置
    if not LLM_CONFIG["api_key"]:
        raise ValueError("DEEPSEEK_API_KEY环境变量未设置")
    
    # 验证向量数据库路径
    db_path = Path(VECTOR_DB_CONFIG["db_path"])
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("✓ 配置验证通过")

# 启动时验证配置
if __name__ == "__main__":
    validate_config()