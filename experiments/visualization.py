"""
实验结果可视化
"""
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

def plot_experiment_results(results_dir: str = "results"):
    """绘制实验结果图表"""
    
    # 加载结果
    results_file = Path(results_dir) / "results.csv"
    summary_file = Path(results_dir) / "experiment_summary.json"
    
    if not results_file.exists():
        print("结果文件不存在")
        return
    
    # 读取数据
    df = pd.read_csv(results_file)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 幻觉率分布
    hallucination_counts = df['has_hallucination'].value_counts()
    axes[0, 0].pie(hallucination_counts.values, labels=['无幻觉', '有幻觉'], autopct='%1.1f%%')
    axes[0, 0].set_title('幻觉分布')
    
    # 2. 相似度分布
    axes[0, 1].hist(df['similarity_score'], bins=20, alpha=0.7)
    axes[0, 1].set_xlabel('相似度分数')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('相似度分布')
    
    # 3. 按类别统计幻觉率
    if 'category' in df.columns:
        category_stats = df.groupby('category')['has_hallucination'].mean()
        axes[1, 0].bar(category_stats.index, category_stats.values)
        axes[1, 0].set_title('按类别幻觉率')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 纠正轮次分布
    if 'correction_rounds' in df.columns:
        rounds_counts = df['correction_rounds'].value_counts().sort_index()
        axes[1, 1].bar(rounds_counts.index, rounds_counts.values)
        axes[1, 1].set_xlabel('纠正轮次')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('纠正轮次分布')
    
    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_experiment_results()