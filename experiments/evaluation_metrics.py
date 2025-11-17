# experiments/evaluation_metrics.py
import json
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class ExperimentEvaluator:
    """实验评估器"""

    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_hallucination_rate(self, generated_answers: List[str],
                                     ground_truths: List[str],
                                     contexts: List[str] = None) -> float:
        """
        计算幻觉率
        基于生成回答与真实答案的语义相似度
        """
        if not generated_answers or not ground_truths:
            return 0.0

        # 编码文本
        gen_embeddings = self.similarity_model.encode(generated_answers)
        gt_embeddings = self.similarity_model.encode(ground_truths)

        # 计算余弦相似度
        similarities = []
        for i in range(len(gen_embeddings)):
            sim = cosine_similarity(
                [gen_embeddings[i]],
                [gt_embeddings[i]]
            )[0][0]
            similarities.append(sim)

        # 相似度低于阈值视为幻觉
        hallucination_threshold = 0.6
        hallucination_count = sum(1 for sim in similarities if sim < hallucination_threshold)

        return hallucination_count / len(similarities)

    def calculate_fact_accuracy(self, generated_answers: List[str],
                                ground_truths: List[str]) -> float:
        """
        计算事实准确率
        基于关键信息匹配
        """
        if not generated_answers:
            return 0.0

        correct_count = 0
        for gen, truth in zip(generated_answers, ground_truths):
            # 简单的关键词匹配（可以替换为更复杂的NLP方法）
            gen_lower = gen.lower()
            truth_lower = truth.lower()

            # 计算重叠关键词比例
            gen_words = set(gen_lower.split())
            truth_words = set(truth_lower.split())

            if truth_words:
                overlap = len(gen_words.intersection(truth_words)) / len(truth_words)
                if overlap > 0.3:  # 30%重叠视为正确
                    correct_count += 1

        return correct_count / len(generated_answers)

    def calculate_response_relevance(self, generated_answers: List[str],
                                     questions: List[str]) -> float:
        """
        计算回答相关性
        """
        if not generated_answers or not questions:
            return 0.0

        question_embeddings = self.similarity_model.encode(questions)
        answer_embeddings = self.similarity_model.encode(generated_answers)

        relevances = []
        for i in range(len(question_embeddings)):
            rel = cosine_similarity(
                [question_embeddings[i]],
                [answer_embeddings[i]]
            )[0][0]
            relevances.append(rel)

        return np.mean(relevances)

    def evaluate_model_performance(self, results: List[Dict]) -> Dict[str, float]:
        """评估模型整体性能"""
        if not results:
            return {}

        questions = [r.get('question', '') for r in results]
        generated_answers = [r.get('generated_answer', '') for r in results]
        ground_truths = [r.get('ground_truth', '') for r in results]
        contexts = [r.get('context', '') for r in results]
        processing_times = [r.get('processing_time', 0) for r in results]

        metrics = {
            "hallucination_rate": self.calculate_hallucination_rate(
                generated_answers, ground_truths, contexts
            ),
            "fact_accuracy": self.calculate_fact_accuracy(
                generated_answers, ground_truths
            ),
            "response_relevance": self.calculate_response_relevance(
                generated_answers, questions
            ),
            "avg_answer_length": np.mean([len(ans) for ans in generated_answers]),
            "avg_processing_time": np.mean(processing_times),
            "total_samples": len(results)
        }

        return metrics