# experiments/run_experiments.py
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
    # 修改导入路径，使用您现有的 deepseek_client
    from src.llm2.deepseek_client import llm_inference
    from experiments.evaluation_metrics import ExperimentEvaluator
    from experiments.experiment_config import EXPERIMENT_CONFIG
    import config
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error("请确保所有模块文件都存在")
    sys.exit(1)


class ExperimentRunner:
    """实验运行器"""

    def __init__(self):
        self.evaluator = ExperimentEvaluator()
        self.config = EXPERIMENT_CONFIG
        # 初始化中间结果存储目录
        self._init_intermediate_dir()

    def _init_intermediate_dir(self):
        """初始化中间结果存储目录"""
        self.intermediate_dir = os.path.join(
            project_root, self.config["output_paths"]["results"], "intermediate"
        )
        os.makedirs(self.intermediate_dir, exist_ok=True)
        logger.info(f"中间结果将保存至: {self.intermediate_dir}")

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

    def run_baseline_model(self, test_data: List[Dict]) -> List[Dict]:
        """运行原始DeepSeek-V3模型 - 使用现有的 llm_inference 函数"""
        logger.info("开始运行原始DeepSeek-V3模型...")

        results = []
        total_samples = len(test_data)
        
        # 检查是否有中间结果
        latest_batch = self._get_latest_batch("baseline")
        if latest_batch > 0:
            logger.info(f"发现中间结果，从第 {latest_batch} 个样本继续运行")
            results = self._load_intermediate_results("baseline", latest_batch)
            start_idx = latest_batch
        else:
            start_idx = 0

        for i in range(start_idx, total_samples):
            sample = test_data[i]
            try:
                start_time = time.time()

                # 使用现有的 llm_inference 函数生成回答
                question = sample['question']
                response = llm_inference(question)

                processing_time = time.time() - start_time

                result = {
                    'sample_id': i,
                    'question': question,
                    'ground_truth': sample.get('ground_truth', ''),
                    'context': sample.get('context', ''),
                    'generated_answer': response,
                    'processing_time': processing_time,
                    'model': 'deepseek_v3_baseline'
                }

                results.append(result)

                # 每10个样本保存一次中间结果
                if (i + 1) % 10 == 0:
                    logger.info(f"基线模型已完成 {i + 1}/{total_samples} 个样本，保存中间结果...")
                    self._save_intermediate_results("baseline", results, i + 1)

                # 每5个样本输出一次进度（保持原有进度提示）
                if (i + 1) % 5 == 0:
                    logger.info(f"基线模型已完成 {i + 1}/{total_samples} 个样本")

            except Exception as e:
                logger.error(f"处理样本 {i} 时出错: {e}")
                # 创建一个错误结果，避免中断实验
                result = {
                    'sample_id': i,
                    'question': sample['question'],
                    'ground_truth': sample.get('ground_truth', ''),
                    'context': sample.get('context', ''),
                    'generated_answer': f"错误: {str(e)}",
                    'processing_time': 0,
                    'model': 'deepseek_v3_baseline',
                    'error': True
                }
                results.append(result)
                # 出错时也保存一次中间结果
                self._save_intermediate_results("baseline", results, i + 1)
                continue

        # 完成后保存最终结果
        self._save_intermediate_results("baseline", results, total_samples, is_final=True)
        return results

    @staticmethod
    def run_rag_system(test_data: List[Dict]) -> List[Dict]:
        """运行RAG系统（静态方法）"""
        logger.info("开始运行RAG系统...")

        results = []
        total_samples = len(test_data)
        
        # 检查是否有中间结果
        latest_batch = ExperimentRunner._get_latest_batch("rag", project_root, EXPERIMENT_CONFIG)
        if latest_batch > 0:
            logger.info(f"发现中间结果，从第 {latest_batch} 个样本继续运行")
            results = ExperimentRunner._load_intermediate_results(
                "rag", latest_batch, project_root, EXPERIMENT_CONFIG
            )
            start_idx = latest_batch
        else:
            start_idx = 0

        for i in range(start_idx, total_samples):
            sample = test_data[i]
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

                # 每10个样本保存一次中间结果
                if (i + 1) % 10 == 0:
                    logger.info(f"RAG系统已完成 {i + 1}/{total_samples} 个样本，保存中间结果...")
                    ExperimentRunner._save_intermediate_results(
                        "rag", results, i + 1, project_root, EXPERIMENT_CONFIG
                    )

                # 每5个样本输出一次进度（保持原有进度提示）
                if (i + 1) % 5 == 0:
                    logger.info(f"RAG系统已完成 {i + 1}/{total_samples} 个样本")

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
                # 出错时也保存一次中间结果
                ExperimentRunner._save_intermediate_results(
                    "rag", results, i + 1, project_root, EXPERIMENT_CONFIG
                )
                continue

        # 完成后保存最终结果
        ExperimentRunner._save_intermediate_results(
            "rag", results, total_samples, project_root, EXPERIMENT_CONFIG, is_final=True
        )
        return results

    def save_results(self, results: List[Dict], filename: str):
        """保存结果到文件"""
        output_dir = self.config["output_paths"]["results"]
        full_output_dir = os.path.join(project_root, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        filepath = os.path.join(full_output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {filepath}")

    def compare_models(self, baseline_results: List[Dict], rag_results: List[Dict]) -> Dict:
        """对比两个模型的性能"""
        logger.info("开始对比模型性能...")

        # 过滤掉错误结果
        valid_baseline = [r for r in baseline_results if not r.get('error', False)]
        valid_rag = [r for r in rag_results if not r.get('error', False)]

        baseline_metrics = self.evaluator.evaluate_model_performance(valid_baseline)
        rag_metrics = self.evaluator.evaluate_model_performance(valid_rag)

        comparison = {
            'timestamp': datetime.now().isoformat(),
            'baseline_model': baseline_metrics,
            'rag_system': rag_metrics,
            'improvement': {},
            'error_stats': {
                'baseline_errors': len(baseline_results) - len(valid_baseline),
                'rag_errors': len(rag_results) - len(valid_rag)
            }
        }

        # 计算改进程度
        for metric in ['hallucination_rate', 'fact_accuracy', 'response_relevance']:
            if metric in baseline_metrics and metric in rag_metrics:
                baseline_val = baseline_metrics[metric]
                rag_val = rag_metrics[metric]

                if baseline_val == 0:  # 避免除零
                    comparison['improvement'][metric] = 0.0
                else:
                    if metric == 'hallucination_rate':
                        # 幻觉率越低越好
                        improvement = (baseline_val - rag_val) / baseline_val * 100
                    else:
                        # 其他指标越高越好
                        improvement = (rag_val - baseline_val) / baseline_val * 100

                    comparison['improvement'][metric] = improvement

        return comparison

    def generate_report(self, comparison: Dict):
        """生成对比报告"""
        logger.info("生成实验报告...")

        report = {
            '实验信息': {
                '实验时间': comparison['timestamp'],
                '测试样本数量': comparison['baseline_model'].get('total_samples', 0),
                '有效样本统计': {
                    '基线模型有效样本': comparison['baseline_model'].get('total_samples', 0),
                    'RAG系统有效样本': comparison['rag_system'].get('total_samples', 0),
                    '基线模型错误数': comparison['error_stats']['baseline_errors'],
                    'RAG系统错误数': comparison['error_stats']['rag_errors']
                },
                '对比模型': ['DeepSeek-V3 Baseline', 'RAG with Fact Checking']
            },
            '性能指标对比': {
                '幻觉率 (Hallucination Rate)': {
                    'DeepSeek-V3 Baseline': f"{comparison['baseline_model'].get('hallucination_rate', 0):.4f}",
                    'RAG System': f"{comparison['rag_system'].get('hallucination_rate', 0):.4f}",
                    '改进程度': f"{comparison['improvement'].get('hallucination_rate', 0):.2f}%"
                },
                '事实准确率 (Fact Accuracy)': {
                    'DeepSeek-V3 Baseline': f"{comparison['baseline_model'].get('fact_accuracy', 0):.4f}",
                    'RAG System': f"{comparison['rag_system'].get('fact_accuracy', 0):.4f}",
                    '改进程度': f"{comparison['improvement'].get('fact_accuracy', 0):.2f}%"
                },
                '回答相关性 (Response Relevance)': {
                    'DeepSeek-V3 Baseline': f"{comparison['baseline_model'].get('response_relevance', 0):.4f}",
                    'RAG System': f"{comparison['rag_system'].get('response_relevance', 0):.4f}",
                    '改进程度': f"{comparison['improvement'].get('response_relevance', 0):.2f}%"
                },
                '平均处理时间 (秒)': {
                    'DeepSeek-V3 Baseline': f"{comparison['baseline_model'].get('avg_processing_time', 0):.2f}",
                    'RAG System': f"{comparison['rag_system'].get('avg_processing_time', 0):.2f}"
                }
            },
            '结论分析': self._generate_conclusion(comparison)
        }

        # 保存报告
        output_path = self.config["output_paths"]["comparison"]
        full_output_path = os.path.join(project_root, output_path)
        os.makedirs(full_output_path, exist_ok=True)

        report_path = os.path.join(full_output_path, 'experiment_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成Markdown格式报告
        self._generate_markdown_report(report, full_output_path)

        logger.info(f"实验报告已生成: {report_path}")

    @staticmethod
    def _generate_conclusion(comparison: Dict) -> str:
        """生成结论分析（静态方法）"""
        hallucination_improvement = comparison['improvement'].get('hallucination_rate', 0)
        accuracy_improvement = comparison['improvement'].get('fact_accuracy', 0)

        conclusion = f"""
        实验结果表明，RAG系统相比原始DeepSeek-V3模型在降低幻觉方面取得了{'显著' if hallucination_improvement > 0 else '有限'}效果：

        - 幻觉率 {'降低' if hallucination_improvement > 0 else '增加'}了 {abs(hallucination_improvement):.2f}%
        - 事实准确率 {'提高' if accuracy_improvement > 0 else '降低'}了 {abs(accuracy_improvement):.2f}%

        错误统计:
        - 基线模型错误数: {comparison['error_stats']['baseline_errors']}
        - RAG系统错误数: {comparison['error_stats']['rag_errors']}
        """

        if hallucination_improvement > 10:
            conclusion += "\n\nRAG系统通过检索增强和事实核查机制，显著减少了模型幻觉的产生。"
        elif hallucination_improvement > 0:
            conclusion += "\n\nRAG系统在一定程度上减少了模型幻觉，但仍有优化空间。"
        else:
            conclusion += "\n\n需要进一步优化RAG系统的检索和验证机制。"

        return conclusion

    @staticmethod
    def _generate_markdown_report(report: Dict, output_path: str):
        """生成Markdown格式报告（静态方法）"""
        md_content = f"""# RAG系统降低大模型幻觉实验报告

## 实验信息
- **实验时间**: {report['实验信息']['实验时间']}
- **测试样本数量**: {report['实验信息']['测试样本数量']}
- **有效样本统计**:
  - 基线模型有效样本: {report['实验信息']['有效样本统计']['基线模型有效样本']}
  - RAG系统有效样本: {report['实验信息']['有效样本统计']['RAG系统有效样本']}
  - 基线模型错误数: {report['实验信息']['有效样本统计']['基线模型错误数']}
  - RAG系统错误数: {report['实验信息']['有效样本统计']['RAG系统错误数']}
- **对比模型**: {', '.join(report['实验信息']['对比模型'])}

## 性能指标对比

| 指标 | DeepSeek-V3 Baseline | RAG System | 改进程度 |
|------|---------------------|-------------|----------|
"""

        metrics = report['性能指标对比']
        for metric_name, values in metrics.items():
            md_content += f"| {metric_name} | {values['DeepSeek-V3 Baseline']} | {values['RAG System']} | {values.get('改进程度', 'N/A')} |\n"

        md_content += f"\n## 结论分析\n\n{report['结论分析']}"

        md_path = os.path.join(output_path, 'experiment_report.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Markdown报告已保存: {md_path}")

    # 中间结果处理相关方法
    def _get_latest_batch(self, model_type: str) -> int:
        """获取最新已完成的批次编号"""
        return ExperimentRunner._get_latest_batch_static(
            model_type, self.intermediate_dir
        )

    @staticmethod
    def _get_latest_batch(model_type: str, project_root: str, config: Dict) -> int:
        """静态方法：获取最新已完成的批次编号"""
        intermediate_dir = os.path.join(
            project_root, config["output_paths"]["results"], "intermediate"
        )
        return ExperimentRunner._get_latest_batch_static(model_type, intermediate_dir)

    @staticmethod
    def _get_latest_batch_static(model_type: str, intermediate_dir: str) -> int:
        """静态辅助方法：获取最新已完成的批次编号"""
        if not os.path.exists(intermediate_dir):
            return 0

        max_batch = 0
        for filename in os.listdir(intermediate_dir):
            if filename.startswith(f"{model_type}_results_batch_") and filename.endswith(".json"):
                try:
                    # 提取批次编号（如 "baseline_results_batch_10.json" -> 10）
                    batch_num = int(filename.split("_")[-1].split(".")[0])
                    if batch_num > max_batch:
                        max_batch = batch_num
                except (ValueError, IndexError):
                    continue
        return max_batch

    def _load_intermediate_results(self, model_type: str, batch_num: int) -> List[Dict]:
        """加载中间结果"""
        return ExperimentRunner._load_intermediate_results_static(
            model_type, batch_num, self.intermediate_dir
        )

    @staticmethod
    def _load_intermediate_results(
        model_type: str, batch_num: int, project_root: str, config: Dict
    ) -> List[Dict]:
        """静态方法：加载中间结果"""
        intermediate_dir = os.path.join(
            project_root, config["output_paths"]["results"], "intermediate"
        )
        return ExperimentRunner._load_intermediate_results_static(
            model_type, batch_num, intermediate_dir
        )

    @staticmethod
    def _load_intermediate_results_static(
        model_type: str, batch_num: int, intermediate_dir: str
    ) -> List[Dict]:
        """静态辅助方法：加载中间结果"""
        filepath = os.path.join(intermediate_dir, f"{model_type}_results_batch_{batch_num}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_intermediate_results(
        self, model_type: str, results: List[Dict], batch_num: int, is_final: bool = False
    ):
        """保存中间结果"""
        ExperimentRunner._save_intermediate_results_static(
            model_type, results, batch_num, self.intermediate_dir, is_final
        )

    @staticmethod
    def _save_intermediate_results(
        model_type: str, results: List[Dict], batch_num: int, project_root: str, config: Dict,
        is_final: bool = False
    ):
        """静态方法：保存中间结果"""
        intermediate_dir = os.path.join(
            project_root, config["output_paths"]["results"], "intermediate"
        )
        ExperimentRunner._save_intermediate_results_static(
            model_type, results, batch_num, intermediate_dir, is_final
        )

    @staticmethod
    def _save_intermediate_results_static(
        model_type: str, results: List[Dict], batch_num: int, intermediate_dir: str,
        is_final: bool = False
    ):
        """静态辅助方法：保存中间结果"""
        filename = f"{model_type}_results_batch_{batch_num}.json"
        filepath = os.path.join(intermediate_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 如果是最终结果，同时创建一个标识文件
        if is_final:
            final_flag = os.path.join(intermediate_dir, f"{model_type}_results_final.txt")
            with open(final_flag, 'w', encoding='utf-8') as f:
                f.write(f"Completed at: {datetime.now().isoformat()}\n")
                f.write(f"Total samples: {len(results)}")


def main():
    """主实验流程"""
    logger.info("开始自动化实验...")

    runner = ExperimentRunner()

    # 1. 加载测试数据
    test_data = runner.load_test_dataset()
    if not test_data:
        logger.error("无法加载测试数据，实验终止")
        return

    print(f"将使用 {len(test_data)} 个样本进行测试")

    # 2. 运行原始DeepSeek-V3模型
    baseline_results = runner.run_baseline_model(test_data)
    runner.save_results(baseline_results, 'baseline_results.json')

    # 3. 运行RAG系统
    rag_results = runner.run_rag_system(test_data)
    runner.save_results(rag_results, 'rag_results.json')

    # 4. 对比分析
    comparison = runner.compare_models(baseline_results, rag_results)

    # 5. 生成报告
    runner.generate_report(comparison)

    logger.info("自动化实验完成！")

    # 打印简要结果
    print("\n" + "=" * 50)
    print("实验简要结果:")
    print(f"基线模型幻觉率: {comparison['baseline_model'].get('hallucination_rate', 0):.2%}")
    print(f"RAG系统幻觉率: {comparison['rag_system'].get('hallucination_rate', 0):.2%}")
    print(f"幻觉率改进: {comparison['improvement'].get('hallucination_rate', 0):.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()