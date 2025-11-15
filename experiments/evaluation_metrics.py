"""
评估指标计算模块 - 幻觉率、准确率等指标
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import json

logger = logging.getLogger(__name__)

class HallucinationEvaluator:
    """幻觉评估器"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        初始化评估器
        
        Args:
            model_name: 用于计算相似度的模型
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"初始化评估器，使用模型: {model_name}")
    
    def evaluate_response(self, 
                         query: str,
                         generated_answer: str,
                         expected_answer: str,
                         retrieved_chunks: List[Dict] = None,
                         has_hallucination: bool = None) -> Dict[str, Any]:
        """
        评估单个回答
        
        Args:
            query: 用户查询
            generated_answer: 生成的回答
            expected_answer: 期望回答
            retrieved_chunks: 检索到的文档块
            has_hallucination: 系统检测到的幻觉标记
            
        Returns:
            评估结果字典
        """
        evaluation = {
            "similarity_score": 0.0,
            "factual_consistency": 0.0,
            "retrieval_relevance": 0.0,
            "hallucination_detected": False,
            "detailed_analysis": {}
        }
        
        try:
            # 1. 计算语义相似度
            similarity_score = self._calculate_similarity(
                generated_answer, expected_answer
            )
            evaluation["similarity_score"] = similarity_score
            
            # 2. 计算事实一致性
            if retrieved_chunks:
                factual_consistency = self._calculate_factual_consistency(
                    generated_answer, retrieved_chunks
                )
                evaluation["factual_consistency"] = factual_consistency
            
            # 3. 计算检索相关性
            if retrieved_chunks and query:
                retrieval_relevance = self._calculate_retrieval_relevance(
                    query, retrieved_chunks
                )
                evaluation["retrieval_relevance"] = retrieval_relevance
            
            # 4. 综合判断是否存在幻觉
            evaluation["hallucination_detected"] = self._detect_hallucination(
                similarity_score,
                evaluation.get("factual_consistency", 0),
                has_hallucination
            )
            
            # 5. 详细分析
            evaluation["detailed_analysis"] = self._detailed_analysis(
                generated_answer, expected_answer, retrieved_chunks
            )
            
        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            evaluation["error"] = str(e)
        
        return evaluation
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        if not text1 or not text2:
            return 0.0
        
        try:
            # 生成嵌入向量
            embeddings = self.model.encode([text1, text2])
            
            # 计算余弦相似度
            similarity = cosine_similarity(
                [embeddings[0]], 
                [embeddings[1]]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            return self._fallback_similarity(text1, text2)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """回退相似度计算方法（基于词重叠）"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_factual_consistency(self, answer: str, retrieved_chunks: List[Dict]) -> float:
        """计算事实一致性"""
        if not answer or not retrieved_chunks:
            return 0.0
        
        try:
            # 将检索文档合并为参考文本
            reference_text = " ".join([chunk.get("text", "") for chunk in retrieved_chunks])
            
            if not reference_text.strip():
                return 0.0
            
            # 计算回答与参考文本的相似度
            return self._calculate_similarity(answer, reference_text)
            
        except Exception as e:
            logger.warning(f"事实一致性计算失败: {e}")
            return 0.0
    
    def _calculate_retrieval_relevance(self, query: str, retrieved_chunks: List[Dict]) -> float:
        """计算检索相关性"""
        if not query or not retrieved_chunks:
            return 0.0
        
        try:
            # 计算查询与每个检索文档的相似度
            similarities = []
            for chunk in retrieved_chunks:
                chunk_text = chunk.get("text", "")
                if chunk_text:
                    similarity = self._calculate_similarity(query, chunk_text)
                    similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"检索相关性计算失败: {e}")
            return 0.0
    
    def _detect_hallucination(self, 
                            similarity_score: float, 
                            factual_consistency: float,
                            system_detection: bool = None) -> bool:
        """检测是否存在幻觉"""
        # 基于多个指标综合判断
        thresholds = {
            'similarity': 0.6,
            'consistency': 0.5
        }
        
        # 如果系统已经检测到幻觉，直接返回True
        if system_detection is True:
            return True
        
        # 基于相似度和一致性判断
        if similarity_score < thresholds['similarity']:
            return True
        
        if factual_consistency < thresholds['consistency']:
            return True
        
        return False
    
    def _detailed_analysis(self, generated_answer: str, expected_answer: str, 
                          retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """详细分析"""
        analysis = {
            "answer_length": len(generated_answer),
            "expected_length": len(expected_answer),
            "length_ratio": len(generated_answer) / max(len(expected_answer), 1),
            "key_entities_missing": [],
            "contradictions_found": []
        }
        
        # 简单的实体匹配分析
        expected_entities = self._extract_entities(expected_answer)
        generated_entities = self._extract_entities(generated_answer)
        
        analysis["key_entities_missing"] = list(expected_entities - generated_entities)
        
        return analysis
    
    def _extract_entities(self, text: str) -> set:
        """提取文本中的实体（简单实现）"""
        if not text:
            return set()
        
        # 提取大写字母开头的词（简单实体识别）
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # 提取数字
        numbers = re.findall(r'\b\d+\b', text)
        
        return set(entities + numbers)
    
    def calculate_batch_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算批量测试的总体指标
        
        Args:
            results: 测试结果列表
            
        Returns:
            总体指标字典
        """
        if not results:
            return {}
        
        # 基础统计
        total_cases = len(results)
        hallucination_cases = sum(1 for r in results if r.get("has_hallucination", False))
        hallucination_rate = hallucination_cases / total_cases
        
        # 相似度统计
        similarity_scores = [
            r.get("evaluation", {}).get("similarity_score", 0) 
            for r in results 
            if "evaluation" in r
        ]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # 检索相关统计
        retrieved_counts = [r.get("retrieved_chunks_count", 0) for r in results]
        avg_retrieved_chunks = np.mean(retrieved_counts) if retrieved_counts else 0
        
        # 纠正轮次统计
        correction_rounds = [r.get("correction_rounds", 0) for r in results]
        avg_correction_rounds = np.mean(correction_rounds) if correction_rounds else 0
        
        # 按类别统计
        metrics_by_category = {}
        for result in results:
            category = result.get("category", "未知")
            if category not in metrics_by_category:
                metrics_by_category[category] = {
                    "count": 0,
                    "hallucination_count": 0,
                    "similarity_scores": []
                }
            
            metrics = metrics_by_category[category]
            metrics["count"] += 1
            if result.get("has_hallucination", False):
                metrics["hallucination_count"] += 1
            
            similarity = result.get("evaluation", {}).get("similarity_score", 0)
            metrics["similarity_scores"].append(similarity)
        
        # 计算每个类别的指标
        for category, metrics in metrics_by_category.items():
            metrics["hallucination_rate"] = (
                metrics["hallucination_count"] / metrics["count"]
            )
            metrics["avg_similarity"] = np.mean(metrics["similarity_scores"])
        
        summary = {
            "total_cases": total_cases,
            "hallucination_cases": hallucination_cases,
            "hallucination_rate": hallucination_rate,
            "avg_similarity": avg_similarity,
            "avg_retrieved_chunks": avg_retrieved_chunks,
            "avg_correction_rounds": avg_correction_rounds,
            "similarity_std": np.std(similarity_scores) if similarity_scores else 0,
            "metrics_by_category": metrics_by_category
        }
        
        logger.info(f"批量指标计算完成: 幻觉率={hallucination_rate:.3f}")
        return summary

def calculate_additional_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    计算额外评估指标
    
    Args:
        results: 测试结果列表
        
    Returns:
        额外指标字典
    """
    if not results:
        return {}
    
    # 计算精确率、召回率、F1分数（如果需要）
    # 这里可以根据具体需求实现更复杂的指标
    
    additional_metrics = {
        "total_queries_processed": len(results),
        "successful_queries": sum(1 for r in results if "error" not in r),
        "failed_queries": sum(1 for r in results if "error" in r),
        "avg_processing_time": 0,  # 需要在实际运行时记录时间
    }
    
    return additional_metrics