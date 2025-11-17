"""
Prompt模板管理模块
适配rag_pipeline接口要求
"""
from typing import List, Dict, Optional

# ==================== 初始回答生成模板 ====================
INITIAL_ANSWER_TEMPLATE = """
请基于以下参考文档回答问题，不需要进行事实核查或验证，提供最合适的答案。

问题: {query}

参考文档:
{chunks_text}

请提供详细、全面的回答，包括所有相关信息和背景知识：
"""

# ==================== 答案纠正模板 ====================
CORRECTION_TEMPLATE = """
作为专业的内容修正专家，请根据验证结果和参考文档重新生成准确、客观的答案。

初始答案：{answer}

参考文档片段：
{chunks}

验证发现的问题：
{errors}

## 修正要求
1. 严格基于参考文档修正错误，移除未经证实的信息
2. 对无法验证的内容明确标注不确定性
3. 保持回答的完整性、逻辑性和可读性
4. 客观中立，避免主观偏向

修正后的答案：
"""

# ==================== 事实验证模板 ====================
VERIFICATION_TEMPLATE = """
请基于提供的参考文档验证以下回答的准确性：

问题：{query}
待验证的回答：{answer}

参考文档：
{chunks_text}

请分析回答中是否存在以下问题：
1. 与参考文档不符的事实性错误
2. 缺乏证据支持的主观断言
3. 逻辑不一致或矛盾的内容
4. 过度扩展或虚构的信息

请按以下格式输出：
是否存在问题：是/否
问题描述：[具体的问题描述，如果没有问题则写"无"]
"""

def get_initial_prompt(query: str, chunks: List[Dict]) -> str:
    """
    获取初始回答生成提示词
    适配rag_pipeline的调用参数要求：接收query和chunks参数
    """
    # 格式化文档片段为字符串
    chunks_text = "\n".join([
        f"文档片段 {i+1}:\n{chunk.get('text', chunk.get('content', ''))}" 
        for i, chunk in enumerate(chunks)
    ])
    return INITIAL_ANSWER_TEMPLATE.format(
        query=query,
        chunks_text=chunks_text
    )

def get_correction_prompt(answer: str, chunks: List[Dict], errors: List[str]) -> str:
    """
    获取答案纠正提示词
    适配rag_pipeline的调用参数要求：接收answer, chunks, errors参数
    """
    # 格式化文档片段为字符串
    chunks_text = "\n".join([
        f"文档片段 {i+1}:\n{chunk.get('text', chunk.get('content', ''))}" 
        for i, chunk in enumerate(chunks)
    ])
    
    # 格式化错误信息
    errors_text = "\n".join([f"- {error}" for error in errors])
    
    return CORRECTION_TEMPLATE.format(
        answer=answer,
        chunks=chunks_text,
        errors=errors_text
    )

def get_verification_prompt(query: str, answer: str, chunks: List[Dict]) -> str:
    """
    获取事实验证提示词
    用于验证回答的准确性
    """
    chunks_text = "\n".join([
        f"文档片段 {i+1}:\n{chunk.get('text', chunk.get('content', ''))}" 
        for i, chunk in enumerate(chunks)
    ])
    
    return VERIFICATION_TEMPLATE.format(
        query=query,
        answer=answer,
        chunks_text=chunks_text
    )