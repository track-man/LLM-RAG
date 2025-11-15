"""
实验自动化脚本 - 运行批量测试并计算指标
"""
import os
import sys
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.rag_pipeline import rag_with_fact_checking
from experiments.evaluation_metrics import HallucinationEvaluator
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("experiment_runner")

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, chroma_path: str, results_dir: str = "results"):
        """
        初始化实验运行器
        
        Args:
            chroma_path: Chroma向量库路径
            results_dir: 结果保存目录
        """
        self.chroma_path = chroma_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化评估器
        self.evaluator = HallucinationEvaluator()
        
        logger.info(f"实验运行器初始化完成，结果目录: {results_dir}")
    
    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        加载测试数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            测试数据列表
        """
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            logger.warning(f"数据集文件不存在: {dataset_path}")
            # 创建示例测试数据
            return self._create_sample_dataset()
        
        try:
            if dataset_file.suffix == '.json':
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif dataset_file.suffix == '.csv':
                data = pd.read_csv(dataset_file).to_dict('records')
            else:
                raise ValueError(f"不支持的格式: {dataset_file.suffix}")
            
            logger.info(f"成功加载数据集: {len(data)} 条测试数据")
            return data
            
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[Dict[str, Any]]:
        """创建示例测试数据集"""
        sample_queries = [
            {
                "query": "BAAI/bge-base-en-v1.5 嵌入模型的输出向量维度是多少？",
                "expected_answer": "768",
                "category": "事实查询"
            },
            {
                "query": "比较一下BERT和GPT模型的区别",
                "expected_answer": "BERT是双向编码器，GPT是自回归生成模型",
                "category": "比较查询"
            },
            {
                "query": "如何安装sentence-transformers库？",
                "expected_answer": "使用pip install sentence-transformers",
                "category": "方法查询"
            },
            {
                "query": "深度学习在自然语言处理中的应用前景",
                "expected_answer": "深度学习在NLP中有广泛应用前景",
                "category": "观点查询"
            }
        ]
        logger.info("使用示例测试数据集")
        return sample_queries
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            测试结果
        """
        query = test_case["query"]
        expected_answer = test_case.get("expected_answer", "")
        category = test_case.get("category", "未知")
        
        logger.info(f"运行测试: {query}")
        
        try:
            # 运行RAG流程
            result = rag_with_fact_checking(
                query=query,
                chroma_path=self.chroma_path,
                max_correction_rounds=config.MAX_CORRECTION_ROUNDS
            )
            
            # 评估结果
            evaluation = self.evaluator.evaluate_response(
                query=query,
                generated_answer=result["final_answer"],
                expected_answer=expected_answer,
                retrieved_chunks=result["retrieved_chunks"],
                has_hallucination=result["has_幻觉"]
            )
            
            # 构建测试结果
            test_result = {
                "query": query,
                "expected_answer": expected_answer,
                "category": category,
                "generated_answer": result["final_answer"],
                "initial_answer": result["initial_answer"],
                "has_hallucination": result["has_幻觉"],
                "retrieved_chunks_count": len(result["retrieved_chunks"]),
                "correction_rounds": len(result["correction_history"]),
                "evaluation": evaluation,
                "process_log": result["process_log"],
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"测试完成: 幻觉={result['has_幻觉']}, 相似度={evaluation['similarity_score']:.3f}")
            return test_result
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            return {
                "query": query,
                "error": str(e),
                "has_hallucination": True,
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def run_batch_tests(self, test_cases: List[Dict[str, Any]], 
                       save_results: bool = True) -> Dict[str, Any]:
        """
        运行批量测试
        
        Args:
            test_cases: 测试用例列表
            save_results: 是否保存结果
            
        Returns:
            批量测试结果
        """
        logger.info(f"开始批量测试，共 {len(test_cases)} 个测试用例")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"进度: {i}/{len(test_cases)}")
            result = self.run_single_test(test_case)
            results.append(result)
        
        # 计算总体指标
        summary = self.evaluator.calculate_batch_metrics(results)
        
        if save_results:
            self.save_results(results, summary)
        
        logger.info(f"批量测试完成，幻觉率: {summary['hallucination_rate']:.3f}")
        return {"results": results, "summary": summary}
    
    def save_results(self, results: List[Dict], summary: Dict):
        """保存测试结果"""
        # 保存详细结果
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存摘要
        summary_file = self.results_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV
        df = pd.DataFrame([
            {
                "query": r["query"],
                "category": r.get("category", "未知"),
                "has_hallucination": r["has_hallucination"],
                "similarity_score": r.get("evaluation", {}).get("similarity_score", 0),
                "correction_rounds": r.get("correction_rounds", 0)
            }
            for r in results
        ])
        csv_file = self.results_dir / "results.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"结果已保存到: {self.results_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行RAG幻觉评估实验")
    parser.add_argument("--dataset", type=str, default="experiments/datasets/sample.json",
                       help="测试数据集路径")
    parser.add_argument("--chroma_path", type=str, default=config.CHROMA_PATH,
                       help="Chroma向量库路径")
    parser.add_argument("--results_dir", type=str, default="results/experiments",
                       help="结果保存目录")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="测试样本数量（None表示全部）")
    
    args = parser.parse_args()
    
    # 初始化实验运行器
    runner = ExperimentRunner(
        chroma_path=args.chroma_path,
        results_dir=args.results_dir
    )
    
    # 加载数据集
    test_cases = runner.load_test_dataset(args.dataset)
    
    if args.sample_size and args.sample_size < len(test_cases):
        test_cases = test_cases[:args.sample_size]
        logger.info(f"使用前 {args.sample_size} 个测试用例")
    
    # 运行批量测试
    results = runner.run_batch_tests(test_cases, save_results=True)
    
    # 打印摘要
    summary = results["summary"]
    print("\n" + "="*50)
    print("实验摘要")
    print("="*50)
    print(f"总测试用例: {summary['total_cases']}")
    print(f"幻觉率: {summary['hallucination_rate']:.3f}")
    print(f"平均相似度: {summary['avg_similarity']:.3f}")
    print(f"平均检索文档数: {summary['avg_retrieved_chunks']:.2f}")
    print(f"平均纠正轮次: {summary['avg_correction_rounds']:.2f}")
    
    # 按类别统计
    if 'metrics_by_category' in summary:
        print("\n按类别统计:")
        for category, metrics in summary['metrics_by_category'].items():
            print(f"  {category}: 幻觉率={metrics['hallucination_rate']:.3f}, "
                  f"样本数={metrics['count']}")

if __name__ == "__main__":
    main()