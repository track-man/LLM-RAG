"""
验证模块：回答与文档一致性验证
负责检测LLM回答中的幻觉内容，提供详细的验证结果
"""
import re
import logging
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    import config
except ImportError:
    # 如果导入失败，创建模拟config
    class MockConfig:
        HALLUCINATION_THRESHOLD = 0.7
    config = MockConfig()

# 配置日志
logger = logging.getLogger(__name__)

class VerificationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"           # 基础验证（规则检查）
    SEMANTIC = "semantic"     # 语义验证（LLM检查）
    COMPREHENSIVE = "comprehensive"  # 综合验证（基础+语义）

@dataclass
class VerificationResult:
    """验证结果数据结构"""
    has_hallucination: bool           # 是否存在幻觉
    confidence_score: float           # 置信度评分 (0-1)
    error_descriptions: List[str]     # 错误描述列表
    verification_details: Dict[str, Any]  # 详细验证信息
    evidence_chunks: List[Dict]       # 支持证据文档块
    verification_level: str          # 验证级别

class FactChecker:
    """事实检查器类"""
    
    def __init__(self, verification_level: VerificationLevel = VerificationLevel.COMPREHENSIVE):
        """
        初始化事实检查器
        
        Args:
            verification_level: 验证级别
        """
        self.verification_level = verification_level
        self.logger = logging.getLogger(__name__)
        
        # 预定义验证规则
        self._init_verification_rules()
    
    def _init_verification_rules(self):
        """初始化验证规则"""
        # 数字验证规则
        self.number_patterns = [
            r'\b\d+\b',  # 整数
            r'\b\d+\.\d+\b',  # 小数
            r'\b\d+%',  # 百分比
            r'\b\d+年\b',  # 年份
            r'\b\d+月\b',  # 月份
        ]
        
        # 实体验证规则
        self.entity_patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # 人名/地名
            r'\b[A-Z]{2,}\b',  # 缩写
            r'\b\w+@\w+\.\w+\b',  # 邮箱
            r'https?://[^\s]+',  # URL
        ]
        
        # 否定词和不确定性词
        self.uncertainty_words = [
            '可能', '也许', '大概', '似乎', '或许', '应该', '估计', '猜测'
        ]
        
        self.negative_indicators = [
            '不对', '错误', '虚假', '错误', '不正确', '不是', '没有'
        ]
    
    def verify_answer(self, 
                     answer: str, 
                     retrieved_chunks: List[Dict],
                     query: str = None) -> VerificationResult:
        """
        验证回答与文档的一致性
        
        Args:
            answer: LLM生成的回答
            retrieved_chunks: 检索到的相关文档块
            query: 原始查询（可选）
            
        Returns:
            VerificationResult: 验证结果
        """
        self.logger.info(f"开始验证回答，检索文档块数量: {len(retrieved_chunks)}")
        
        # 提取关键信息
        key_info = self._extract_key_information(answer)
        
        # 基础验证
        basic_result = self._basic_verification(answer, key_info, retrieved_chunks)
        
        # 语义验证（如果需要）
        semantic_result = None
        if self.verification_level in [VerificationLevel.SEMANTIC, VerificationLevel.COMPREHENSIVE]:
            semantic_result = self._semantic_verification(answer, key_info, retrieved_chunks, query)
        
        # 综合验证结果
        final_result = self._combine_verification_results(
            basic_result, semantic_result, retrieved_chunks
        )
        
        self.logger.info(f"验证完成，幻觉检测: {final_result.has_hallucination}, "
                        f"置信度: {final_result.confidence_score:.3f}")
        
        return final_result
    
    def _extract_key_information(self, text: str) -> Dict[str, List[str]]:
        """提取文本中的关键信息"""
        key_info = {
            'numbers': [],
            'entities': [],
            'dates': [],
            'claims': []
        }
        
        # 提取数字
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text)
            key_info['numbers'].extend(matches)
        
        # 提取实体
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            key_info['entities'].extend(matches)
        
        # 提取日期
        date_patterns = [
            r'\b\d{4}年\d{1,2}月\d{1,2}日\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b\d{4}年\b'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            key_info['dates'].extend(matches)
        
        # 提取声明性句子
        sentences = re.split(r'[。！？]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤过短的句子
                key_info['claims'].append(sentence)
        
        return key_info
    
    def _basic_verification(self, 
                          answer: str, 
                          key_info: Dict[str, List[str]], 
                          retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """基础验证：规则基础的验证"""
        verification_details = {
            'level': 'basic',
            'checks_performed': [],
            'issues_found': [],
            'supporting_evidence': 0
        }
        
        # 检查1：数字一致性验证
        number_issues = self._verify_numbers(key_info['numbers'], retrieved_chunks)
        if number_issues:
            verification_details['issues_found'].extend(number_issues)
        verification_details['checks_performed'].append('number_consistency')
        
        # 检查2：实体存在性验证
        entity_issues = self._verify_entities(key_info['entities'], retrieved_chunks)
        if entity_issues:
            verification_details['issues_found'].extend(entity_issues)
        verification_details['checks_performed'].append('entity_existence')
        
        # 检查3：声明支持性验证
        claim_issues = self._verify_claims(key_info['claims'], retrieved_chunks)
        if claim_issues:
            verification_details['issues_found'].extend(claim_issues)
        verification_details['checks_performed'].append('claim_support')
        
        # 计算基础验证置信度
        total_checks = len(verification_details['checks_performed'])
        issues_count = len(verification_details['issues_found'])
        basic_confidence = max(0.0, 1.0 - (issues_count / max(total_checks, 1)) * 0.3)
        
        verification_details['confidence'] = basic_confidence
        
        return verification_details
    
    def _verify_numbers(self, numbers: List[str], retrieved_chunks: List[Dict]) -> List[str]:
        """验证数字的一致性"""
        issues = []
        
        for number in numbers:
            found_in_chunks = 0
            for chunk in retrieved_chunks:
                if number in chunk['text']:
                    found_in_chunks += 1
            
            # 如果数字在多个文档块中都出现，认为是支持的
            if found_in_chunks == 0:
                issues.append(f"数字 '{number}' 在检索文档中未找到支持")
        
        return issues
    
    def _verify_entities(self, entities: List[str], retrieved_chunks: List[Dict]) -> List[str]:
        """验证实体的存在性"""
        issues = []
        
        for entity in entities:
            if len(entity) < 2:  # 过滤过短的实体
                continue
                
            found_in_chunks = 0
            for chunk in retrieved_chunks:
                if entity.lower() in chunk['text'].lower():
                    found_in_chunks += 1
            
            # 如果实体在检索文档中未找到，可能存在问题
            if found_in_chunks == 0:
                issues.append(f"实体 '{entity}' 在检索文档中未找到")
        
        return issues
    
    def _verify_claims(self, claims: List[str], retrieved_chunks: List[Dict]) -> List[str]:
        """验证声明的支持性"""
        issues = []
        
        for claim in claims:
            if len(claim.strip()) < 10:  # 过滤过短的声明
                continue
            
            # 检查声明中的关键词是否在文档中找到
            claim_words = claim.split()
            support_score = 0
            total_words = len(claim_words)
            
            for word in claim_words:
                if len(word) < 2:  # 过滤过短的词
                    continue
                    
                for chunk in retrieved_chunks:
                    if word.lower() in chunk['text'].lower():
                        support_score += 1
                        break
            
            # 如果支持分数过低，可能存在问题
            support_ratio = support_score / max(total_words, 1)
            if support_ratio < 0.3:  # 支持率低于30%
                issues.append(f"声明 '{claim[:50]}...' 支持度不足 ({support_ratio:.2f})")
        
        return issues
    
    def _semantic_verification(self, 
                             answer: str, 
                             key_info: Dict[str, List[str]], 
                             retrieved_chunks: List[Dict],
                             query: str = None) -> Dict[str, Any]:
        """语义验证：基于LLM的语义一致性检查"""
        verification_details = {
            'level': 'semantic',
            'llm_verification': None,
            'semantic_issues': [],
            'confidence': 0.0
        }
        
        try:
            # 构建语义验证prompt
            verification_prompt = self._build_semantic_verification_prompt(
                answer, retrieved_chunks, query
            )
            
            # 调用LLM进行语义验证（这里需要集成LLM客户端）
            # 由于还没有实现LLM客户端，这里先返回模拟结果
            verification_details['llm_verification'] = {
                'is_consistent': True,
                'confidence': 0.85,
                'reasoning': '语义验证通过（模拟结果）'
            }
            
            verification_details['confidence'] = 0.85
            
        except Exception as e:
            self.logger.warning(f"语义验证失败: {e}")
            verification_details['semantic_issues'].append(f"语义验证执行失败: {str(e)}")
            verification_details['confidence'] = 0.5
        
        return verification_details
    
    def _build_semantic_verification_prompt(self, 
                                          answer: str, 
                                          retrieved_chunks: List[Dict], 
                                          query: str = None) -> str:
        """构建语义验证prompt"""
        prompt = f"""作为一个事实检查专家，请验证以下回答与提供的参考文档的一致性。

原始查询: {query if query else '未提供'}

LLM回答: 
{answer}

参考文档:
"""
        
        for i, chunk in enumerate(retrieved_chunks):
            prompt += f"\n文档片段 {i+1}:\n{chunk['text']}\n"
        
        prompt += """
请从以下几个方面进行验证：
1. 回答中的事实陈述是否与参考文档一致
2. 是否存在参考文档中未提及的信息
3. 数字、日期、实体等关键信息是否准确
4. 回答的逻辑是否合理

请提供：
- 一致性评分 (0-1)
- 发现的问题列表
- 支持或反驳的证据
"""
        
        return prompt
    
    def _combine_verification_results(self, 
                                    basic_result: Dict[str, Any], 
                                    semantic_result: Dict[str, Any],
                                    retrieved_chunks: List[Dict]) -> VerificationResult:
        """综合验证结果"""
        
        # 计算综合置信度
        basic_confidence = basic_result.get('confidence', 0.5)
        
        if semantic_result:
            semantic_confidence = semantic_result.get('confidence', 0.5)
            # 综合置信度 = 基础验证 * 0.6 + 语义验证 * 0.4
            final_confidence = basic_confidence * 0.6 + semantic_confidence * 0.4
        else:
            final_confidence = basic_confidence
        
        # 收集所有问题
        all_issues = []
        all_issues.extend(basic_result.get('issues_found', []))
        
        if semantic_result:
            all_issues.extend(semantic_result.get('semantic_issues', []))
        
        # 判断是否存在幻觉
        has_hallucination = (
            len(all_issues) > 0 or 
            final_confidence < config.HALLUCINATION_THRESHOLD
        )
        
        # 收集支持证据
        evidence_chunks = []
        for chunk in retrieved_chunks:
            evidence_chunks.append({
                'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'metadata': chunk.get('metadata', {}),
                'relevance_score': chunk.get('distance', 0.0)
            })
        
        return VerificationResult(
            has_hallucination=has_hallucination,
            confidence_score=final_confidence,
            error_descriptions=all_issues,
            verification_details={
                'basic': basic_result,
                'semantic': semantic_result
            },
            evidence_chunks=evidence_chunks,
            verification_level=self.verification_level.value
        )

def verify_answer(answer: str, 
                 retrieved_chunks: List[Dict], 
                 query: str = None,
                 verification_level: str = "comprehensive") -> VerificationResult:
    """
    验证回答的便捷函数
    
    Args:
        answer: LLM生成的回答
        retrieved_chunks: 检索到的相关文档块
        query: 原始查询
        verification_level: 验证级别 ("basic", "semantic", "comprehensive")
        
    Returns:
        VerificationResult: 验证结果
    """
    level_map = {
        "basic": VerificationLevel.BASIC,
        "semantic": VerificationLevel.SEMANTIC,
        "comprehensive": VerificationLevel.COMPREHENSIVE
    }
    
    level = level_map.get(verification_level, VerificationLevel.COMPREHENSIVE)
    checker = FactChecker(verification_level=level)
    
    return checker.verify_answer(answer, retrieved_chunks, query)