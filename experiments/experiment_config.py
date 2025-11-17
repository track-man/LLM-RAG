# experiments/experiment_config.py
import os
from pathlib import Path

# 实验配置
EXPERIMENT_CONFIG = {
    "datasets": {
        "test": "experiments/datasets/processed/test.json",
        "val": "experiments/datasets/processed/val.json"
    },
    "models": {
        "baseline": "deepseek_v3",  # 原始模型
        "rag_system": "rag_with_fact_checking"  # 我们的RAG系统
    },
    "evaluation_metrics": [
        "hallucination_rate",
        "fact_accuracy",
        "response_relevance",
        "answer_length",
        "processing_time"
    ],
    "output_paths": {
        "results": "results/experiment_results",
        "comparison": "results/comparison_analysis",
        "logs": "results/logs/experiments"
    },
    "experiment_params": {
        "max_samples": None,  # None表示使用全部测试集
        "random_seed": 42,
        "batch_size": 10
    }
}

# 创建输出目录
for path in EXPERIMENT_CONFIG["output_paths"].values():
    Path(path).mkdir(parents=True, exist_ok=True)