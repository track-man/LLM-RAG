"""
答案纠正模块
基于验证结果修正LLM生成的回答
"""
from typing import List, Dict
from src.llm.deepseek_client import llm_inference
from src.llm.prompt_templates import get_correction_prompt
import logging

logger = logging.getLogger("answer_corrector")

def correct_answer(answer: str, chunks: List[Dict], errors: List[str]) -> str:
    """
    基于验证结果纠正回答
    
    参数:
        answer: 当前待纠正的回答
        chunks: 检索到的相关文档块
        errors: 验证发现的错误列表
        
    返回:
        纠正后的回答
    """
    try:
        # 生成纠正提示
        correction_prompt = get_correction_prompt(
            answer=answer,
            chunks=chunks,
            errors=errors
        )
        
        # 调用LLM进行纠正
        corrected_answer = llm_inference(
            prompt=correction_prompt,
            temperature=0.1,  # 使用较低温度确保准确性
            max_tokens=1500   # 允许稍长的输出以包含修正内容
        )
        
        logger.info(f"答案纠正完成，原始回答长度: {len(answer)}，纠正后长度: {len(corrected_answer)}")
        return corrected_answer
        
    except Exception as e:
        error_msg = f"答案纠正过程出错: {str(e)}"
        logger.error(error_msg)
        # 返回原始回答作为fallback
        return f"{answer}\n\n[注意：纠正过程中出现错误，以上为原始回答]"

def batch_correct_answers(answers_with_errors: List[Dict]) -> List[str]:
    """
    批量纠正多个回答
    
    参数:
        answers_with_errors: 包含回答、文档块和错误的字典列表
            [{"answer": "...", "chunks": [...], "errors": [...]}, ...]
            
    返回:
        纠正后的回答列表
    """
    corrected_answers = []
    
    for item in answers_with_errors:
        corrected = correct_answer(
            answer=item["answer"],
            chunks=item["chunks"],
            errors=item["errors"]
        )
        corrected_answers.append(corrected)
    
    return corrected_answers