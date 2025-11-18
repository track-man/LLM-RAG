import os
import json
import time
import logging
import sys
from datetime import datetime
from typing import List, Dict

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("experiments")

try:
    from src.core.rag_pipeline import rag_with_fact_checking
    from experiments.evaluation_metrics import ExperimentEvaluator
    from experiments.experiment_config import EXPERIMENT_CONFIG
    import config
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error("请确保所有模块文件都存在")
    sys.exit(1)


class ExperimentRunner:
    """实验运行器（仅保留RAG相关功能）"""

    def __init__(self):
        self.evaluator = ExperimentEvaluator()
        self.config = EXPERIMENT_CONFIG
        # 创建中间结果和报告保存目录
        self._create_intermediate_dirs()

    def _create_intermediate_dirs(self):
        """创建中间结果和报告的保存目录"""
        intermediate_dirs = [
            os.path.join(project_root, self.config["output_paths"]["results"], "intermediate"),
            os.path.join(project_root, self.config["output_paths"]["comparison"], "intermediate")
        ]
        for dir_path in intermediate_dirs:
            os.makedirs(dir_path, exist_ok=True)

    def load_test_dataset(self) -> List[Dict]:
        """加载测试数据集"""
        test_path = self.config["datasets"]["test"]
        logger.info(f"加载测试数据集: {test_path}")

        # 使用绝对路径
        full_test_path = os.path.join(project_root, test_path)

        try:
            with open(full_test_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 限制样本数量（用于快速测试）
            max_samples = self.config["experiment_params"]["max_samples"]
            if max_samples and max_samples < len(data):
                data = data[:max_samples]

            logger.info(f"成功加载 {len(data)} 个测试样本")
            return data
        except Exception as e:
            logger.error(f"加载测试数据集失败: {e}")
            return []

    def save_results(self, results: List[Dict], filename: str, is_intermediate=False):
        """保存结果到文件"""
        output_dir = self.config["output_paths"]["results"]
        if is_intermediate:
            output_dir = os.path.join(output_dir, "intermediate")
            
        full_output_dir = os.path.join(project_root, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        filepath = os.path.join(full_output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {filepath}")
        return filepath

    def run_rag_system(self, test_data: List[Dict]) -> List[Dict]:
        """运行RAG系统"""
        logger.info("开始运行RAG系统...")

        results = []
        total_samples = len(test_data)  # 记录总样本数，方便进度展示
        batch_size = 10  # 每10个样本生成一次中间报告
        
        for i, sample in enumerate(test_data):
            try:
                start_time = time.time()

                # 使用RAG系统生成回答
                question = sample['question']
                rag_result = rag_with_fact_checking(
                    query=question,
                    chroma_path=config.CHROMA_DB_PATH,
                    max_correction_rounds=config.FACT_CHECK_CONFIG["max_verification_attempts"]
                )

                processing_time = time.time() - start_time

                result = {
                    'sample_id': i,
                    'question': question,
                    'ground_truth': sample.get('ground_truth', ''),
                    'context': sample.get('context', ''),
                    'generated_answer': rag_result.get('final_answer', ''),
                    'processing_time': processing_time,
                    'model': 'rag_with_fact_checking',
                    'has_hallucination': rag_result.get('has_幻觉', False),
                    'verification_rounds': rag_result.get('verification_rounds', 0)
                }

                results.append(result)

                # 每处理10个样本或最后一批样本时生成中间报告
                if (i + 1) % batch_size == 0 or (i + 1) == total_samples:
                    # 保存中间结果
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save_results(
                        results, 
                        f'rag_intermediate_results_{i+1}_samples_{timestamp}.json',
                        is_intermediate=True
                    )
                    
                    # 生成中间评估报告
                    intermediate_evaluation = self.evaluate_rag_performance(results)
                    self.generate_rag_report(
                        intermediate_evaluation, 
                        is_intermediate=True,
                        processed_count=i+1
                    )
                    
                    logger.info(f"RAG系统已完成 {i + 1}/{total_samples} 个样本 "
                                f"({(i + 1) / total_samples * 100:.1f}%)")

            except Exception as e:
                logger.error(f"处理样本 {i} 时出错: {e}")
                # 创建一个错误结果
                result = {
                    'sample_id': i,
                    'question': sample['question'],
                    'ground_truth': sample.get('ground_truth', ''),
                    'context': sample.get('context', ''),
                    'generated_answer': f"RAG系统错误: {str(e)}",
                    'processing_time': 0,
                    'model': 'rag_with_fact_checking',
                    'error': True
                }
                results.append(result)
                continue

        return results

    
    def evaluate_rag_performance(self, rag_results: List[Dict]) -> Dict:
        """评估RAG系统性能"""
        logger.info("开始评估RAG系统性能...")

        # 过滤掉错误结果
        valid_rag = [r for r in rag_results if not r.get('error', False)]
        rag_metrics = self.evaluator.evaluate_model_performance(valid_rag)

        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'rag_system': rag_metrics,
            'error_stats': {
                'rag_errors': len(rag_results) - len(valid_rag)
            }
        }

        return evaluation

    def generate_rag_report(self, evaluation: Dict, is_intermediate=False, processed_count=0):
        """生成RAG系统评估报告"""
        logger.info("生成RAG系统评估报告...")

        # 报告标题和标识
        report_type = "中间" if is_intermediate else "最终"
        report_title = f"{report_type} RAG系统评估报告"
        
        report = {
            '实验信息': {
                '报告类型': report_type,
                '已处理样本数': processed_count,
                '实验时间': evaluation['timestamp'],
                '测试样本数量': evaluation['rag_system'].get('total_samples', 0),
                '有效样本统计': {
                    'RAG系统有效样本': evaluation['rag_system'].get('total_samples', 0),
                    'RAG系统错误数': evaluation['error_stats']['rag_errors']
                },
                '评估模型': 'RAG with Fact Checking'
            },
            '性能指标': {
                '幻觉率 (Hallucination Rate)': f"{evaluation['rag_system'].get('hallucination_rate', 0):.4f}",
                '事实准确率 (Fact Accuracy)': f"{evaluation['rag_system'].get('fact_accuracy', 0):.4f}",
                '回答相关性 (Response Relevance)': f"{evaluation['rag_system'].get('response_relevance', 0):.4f}",
                '平均回答长度': f"{evaluation['rag_system'].get('avg_answer_length', 0):.2f}",
                '平均处理时间 (秒)': f"{evaluation['rag_system'].get('avg_processing_time', 0):.2f}"
            },
            '结论分析': self._generate_rag_conclusion(evaluation)
        }

        # 保存报告
        output_path = self.config["output_paths"]["comparison"]
        if is_intermediate:
            output_path = os.path.join(output_path, "intermediate")
            
        full_output_path = os.path.join(project_root, output_path)
        os.makedirs(full_output_path, exist_ok=True)

        # 生成带时间戳和处理数量的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'rag_{report_type.lower()}_report_{processed_count}_samples_{timestamp}.json'
        report_path = os.path.join(full_output_path, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成Markdown格式报告
        self._generate_rag_markdown_report(report, full_output_path, report_type, processed_count, timestamp)

        logger.info(f"{report_type} RAG评估报告已生成: {report_path}")

    @staticmethod
    def _generate_rag_conclusion(evaluation: Dict) -> str:
        """生成RAG系统结论分析"""
        hallucination_rate = evaluation['rag_system'].get('hallucination_rate', 0)
        accuracy = evaluation['rag_system'].get('fact_accuracy', 0)

        conclusion = f"""
        RAG系统评估结果：
        - 幻觉率: {hallucination_rate:.2%}
        - 事实准确率: {accuracy:.2%}
        
        错误统计:
        - RAG系统错误数: {evaluation['error_stats']['rag_errors']}
        """

        if hallucination_rate < 0.2:
            conclusion += "\n\nRAG系统表现优秀，幻觉率较低。"
        elif hallucination_rate < 0.4:
            conclusion += "\n\nRAG系统表现良好，但仍有优化空间。"
        else:
            conclusion += "\n\n需要进一步优化RAG系统的检索和验证机制以降低幻觉率。"

        return conclusion

    @staticmethod
    def _generate_rag_markdown_report(report: Dict, output_path: str, report_type: str, processed_count: int, timestamp: str):
        """生成RAG系统Markdown格式报告"""
        md_content = f"# {report_type} RAG系统降低大模型幻觉评估报告\n\n"
        md_content += f"**生成时间**: {timestamp}\n"
        md_content += f"**已处理样本数**: {processed_count}\n\n"

        md_content += "## 实验信息\n"
        md_content += f"- **实验时间**: {report['实验信息']['实验时间']}\n"
        md_content += f"- **测试样本数量**: {report['实验信息']['测试样本数量']}\n"
        md_content += "- **有效样本统计**:\n"
        md_content += f"  - RAG系统有效样本: {report['实验信息']['有效样本统计']['RAG系统有效样本']}\n"
        md_content += f"  - RAG系统错误数: {report['实验信息']['有效样本统计']['RAG系统错误数']}\n"
        md_content += f"- **评估模型**: {report['实验信息']['评估模型']}\n\n"

        md_content += "## 性能指标\n\n"
        md_content += "| 指标 | 数值 |\n"
        md_content += "|------|------|\n"

        metrics = report['性能指标']
        for metric_name, value in metrics.items():
            md_content += f"| {metric_name} | {value} |\n"

        md_content += f"\n## 结论分析\n\n{report['结论分析']}"

        md_filename = f'rag_{report_type.lower()}_report_{processed_count}_samples_{timestamp}.md'
        md_path = os.path.join(output_path, md_filename)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"{report_type} RAG系统Markdown报告已保存: {md_path}")


def main():
    """RAG系统测试主流程"""
    logger.info("开始RAG系统自动化测试...")

    runner = ExperimentRunner()

    # 1. 加载测试数据
    test_data = runner.load_test_dataset()
    if not test_data:
        logger.error("无法加载测试数据，测试终止")
        return

    print(f"将使用 {len(test_data)} 个样本进行RAG系统测试")

    # 2. 运行RAG系统（会自动生成中间报告）
    rag_results = runner.run_rag_system(test_data)
    runner.save_results(rag_results, 'rag_final_results.json')

    # 3. 生成最终评估报告
    final_evaluation = runner.evaluate_rag_performance(rag_results)
    runner.generate_rag_report(final_evaluation, is_intermediate=False, processed_count=len(test_data))

    logger.info("RAG系统自动化测试完成！")

    # 打印简要结果
    print("\n" + "=" * 50)
    print("RAG系统测试简要结果:")
    print(f"RAG系统幻觉率: {final_evaluation['rag_system'].get('hallucination_rate', 0):.2%}")
    print(f"RAG系统事实准确率: {final_evaluation['rag_system'].get('fact_accuracy', 0):.2%}")
    print(f"平均处理时间: {final_evaluation['rag_system'].get('avg_processing_time', 0):.2f}秒")
    print("=" * 50)


if __name__ == "__main__":
    main()