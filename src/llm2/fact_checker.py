"""
事实验证模块
验证LLM生成回答的准确性
"""
from typing import List, Dict, Tuple
from src.llm.deepseek_client import llm_inference
from src.llm.prompt_templates import get_verification_prompt
import logging
import re

logger = logging.getLogger("fact_checker")

def verify_answer(answer: str, chunks: List[Dict]) -> Tuple[bool, List[str]]:
    """
    验证回答的准确性
    
    参数:
        answer: 待验证的回答
        chunks: 检索到的相关文档块
        
    返回:
        Tuple[是否有幻觉: bool, 错误描述列表: List[str]]
    """
    try:
        # 如果回答为空或太短，直接返回无幻觉
        if not answer or len(answer.strip()) < 10:
            return False, []
            
        # 生成验证提示
        verification_prompt = get_verification_prompt(
            query="",  # 问题在验证中不是必需的
            answer=answer,
            chunks=chunks
        )
        
        # 调用LLM进行验证
        verification_result = llm_inference(
            prompt=verification_prompt,
            temperature=0.1,
            max_tokens=500
        )
        
        # 解析验证结果
        has_hallucination, errors = parse_verification_result(verification_result)
        
        logger.info(f"验证完成，是否有幻觉: {has_hallucination}, 错误数量: {len(errors)}")
        return has_hallucination, errors
        
    except Exception as e:
        error_msg = f"验证过程出错: {str(e)}"
        logger.error(error_msg)
        # 在验证过程出错时，保守地认为存在幻觉
        return True, [f"验证过程异常: {str(e)}"]

def parse_verification_result(result: str) -> Tuple[bool, List[str]]:
    """
    解析验证结果
    
    参数:
        result: LLM返回的验证结果文本
        
    返回:
        Tuple[是否有幻觉: bool, 错误描述列表: List[str]]
    """
    has_hallucination = False
    errors = []
    
    # 使用正则表达式提取关键信息
    problem_match = re.search(r'是否存在问题：\s*(是|否)', result)
    description_match = re.search(r'问题描述：\s*(.+?)(?=\n\n|\n[A-Z]|$)', result, re.DOTALL)
    
    if problem_match:
        has_hallucination = (problem_match.group(1) == '是')
    
    if has_hallucination and description_match:
        description = description_match.group(1).strip()
        if description and description != "无":
            # 分割多个错误描述
            error_list = [err.strip() for err in description.split(';') if err.strip()]
            errors.extend(error_list)
    
    # 如果没有明确匹配到格式，但结果中包含问题关键词，则认为是存在幻觉
    if not has_hallucination and any(keyword in result.lower() for keyword in ['错误', '不符', '矛盾', '虚构', '缺乏证据']):
        has_hallucination = True
        errors.append("回答中存在不准确或缺乏证据支持的内容")
    
    return has_hallucination, errors

def simple_rule_based_check(answer: str, chunks: List[Dict]) -> Tuple[bool, List[str]]:
    """
    简单的基于规则的检查（备用方案）
    """
    errors = []
    
    # 检查回答是否包含明显的虚构内容
    fiction_indicators = [
        "根据最新研究", "最近发现", "最新数据显示", 
        "权威专家指出", "研究表明", "实验证明"
    ]
    
    for indicator in fiction_indicators:
        if indicator in answer:
            # 检查检索的文档中是否包含相关证据
            has_evidence = any(indicator in chunk.get('text', '') for chunk in chunks)
            if not has_evidence:
                errors.append(f"回答中包含未经证实的表述: '{indicator}'")
    
    has_hallucination = len(errors) > 0
    return has_hallucination, errors