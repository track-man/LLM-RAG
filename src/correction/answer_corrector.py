
from typing import List, Dict, Any
from src.llm.deepseek_client import LLMAdapter

from src.llm.prompt_templates import CORRECTION_TEMPLATES, get_correction_prompt

    
def __init__(self, llm_adapter: LLMAdapter):
    self.llm = llm_adapter
    
    # 意图特定的纠正模板
    self.correction_templates = {
        "事实查询": self._get_factual_correction_template(),
        "比较查询": self._get_comparison_correction_template(),
        "方法查询": self._get_method_correction_template(),
        "观点查询": self._get_opinion_correction_template()
    }

# 修改后的 correct_answer 函数（适配输入输出规范，移除意图分析）
def correct_answer(answer: str, chunks: List[Dict], errors: List[str]) -> str:#ffffffffffffffffffffffffffffffffffffffffffffk
    """
    基于验证错误和文档块纠正回答
    输入：
        answer: 当前回答字符串
        chunks: 检索到的文档块列表
        errors: 错误描述列表
    输出：
        纠正后的回答字符串
    """
    from src.llm.prompt_templates import CORRECTION_TEMPLATES
    from src.llm.deepseek_client import llm_inference

    # 1. 提取原始查询（从文档块元数据）
    query = chunks[0].get("metadata", {}).get("query", "未知查询") if chunks else "未知查询"
    
    # 2. 生成验证摘要（基于错误和文档块）
    def _prepare_verification_summary(errors: List[str], chunks: List[Dict]) -> str:
        if not errors:
            return "未发现错误"
        
        summary_parts = []
        for i, error in enumerate(errors, 1):
            # 匹配相关证据文档块
            related_chunks = [c for c in chunks if error in c.get('text', '')]
            evidence_text = "\n".join([f"文档{i+1}: {c['text'][:100]}..." for i, c in enumerate(related_chunks[:2])])
            
            summary_parts.append(f"错误{i}: {error}")
            summary_parts.append(f"相关证据: {evidence_text if evidence_text else '无'}")
            summary_parts.append("---")
        
        return "\n".join(summary_parts)
    
    verification_summary = _prepare_verification_summary(errors, chunks)
    
    # 3. 使用默认事实查询模板（移除意图分析）
    template = CORRECTION_TEMPLATES["事实查询"]
    prompt = template.format(
        query=query,
        intent="事实查询",  # 固定为事实查询，移除意图分析
        initial_answer=answer,
        verification_summary=verification_summary
    )
    
    # 4. 调用LLM生成纠正结果
    response = llm_inference(
        prompt=prompt,
        temperature=0.3,
        max_tokens=1200
    )
    
    return response

def _prepare_verification_summary(self, verifications: List[Dict]) -> str:
    """准备验证结果摘要"""
    if not verifications:
        return "没有进行声明验证或验证全部失败。"
    
    summary_parts = []
    for i, verification in enumerate(verifications, 1):
        claim = verification.get('claim', f'声明{i}')
        verdict = verification.get('verdict', 'UNVERIFIED')
        confidence = verification.get('confidence', 0)
        reasoning = verification.get('reasoning', '无详细推理')
        
        summary_parts.append(f"声明{i}: {claim}")
        summary_parts.append(f"验证结果: {verdict} (置信度: {confidence:.2f})")
        
        # 添加证据摘要
        supporting_evidence = verification.get('supporting_evidence', [])
        contradicting_evidence = verification.get('contradicting_evidence', [])
        
        if supporting_evidence:
            summary_parts.append(f"支持证据: {len(supporting_evidence)}条")
        if contradicting_evidence:
            summary_parts.append(f"矛盾证据: {len(contradicting_evidence)}条")
        
        summary_parts.append(f"推理: {reasoning[:100]}..." if len(reasoning) > 100 else f"推理: {reasoning}")
        summary_parts.append("---")
    
    return "\n".join(summary_parts)
    
    def _get_factual_correction_template(self) -> str:
        """事实查询的纠正模板"""
        return """
        作为事实核查专家，请根据验证结果重新生成一个准确的事实性答案。

        查询意图：{intent} - 事实查询
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的答案，要求：
        1. 严格基于验证证据，不添加未经证实的信息
        2. 保持客观、准确、简洁的专业风格
        3. 如果某些声明无法验证或存在矛盾，请明确说明局限性
        4. 优先使用支持度高的证据，对矛盾证据进行说明
        5. 保持答案的完整性和连贯性

        修正后的答案应直接回答原始查询，同时体现验证过程的严谨性。

        修正后的答案：
        """
    
    def _get_comparison_correction_template(self) -> str:
        """比较查询的纠正模板"""
        return """
        作为比较分析专家，请根据验证结果重新生成一个全面准确的比较性答案。

        查询意图：{intent} - 比较查询  
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的比较分析，要求：
        1. 确保比较维度的全面性和平衡性，避免偏颇
        2. 基于证据提供具体的对比点和数据支持
        3. 客观呈现各方的优势和劣势
        4. 如果某些比较点缺乏充分证据，请谨慎表述并说明不确定性
        5. 提供有依据的结论或建议

        修正后的比较分析应具有结构化的对比框架和基于证据的判断。

        修正后的比较分析：
        """
    
    def _get_method_correction_template(self) -> str:
        """方法查询的纠正模板"""
        return """
        作为方法指导专家，请根据验证结果重新生成一个可操作的方法指南。

        查询意图：{intent} - 方法查询
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的方法指南，要求：
        1. 确保步骤的可行性、正确性和安全性
        2. 提供清晰、有序的操作指引
        3. 基于验证证据调整或完善有问题的步骤
        4. 包含必要的注意事项和常见问题解决方案
        5. 如果某些方法步骤缺乏充分验证，请注明其不确定性

        修正后的方法指南应具有实用性和可操作性。

        修正后的方法指南：
        """
    
    def _get_opinion_correction_template(self) -> str:
        """观点查询的纠正模板"""
        return """
        作为观点综述专家，请根据验证结果重新生成一个平衡客观的观点综述。

        查询意图：{intent} - 观点查询
        原始查询："{query}"
        原始答案：{original_answer}

        验证结果摘要：
        {verification_summary}

        请生成修正后的观点综述，要求：
        1. 全面、平衡地呈现不同的观点立场和论据
        2. 基于证据客观表述各方观点，避免主观偏向
        3. 明确区分事实性内容和观点性内容
        4. 如果某些观点缺乏充分证据支持，请说明其争议性
        5. 提供基于证据的综合分析或趋势判断

        修正后的观点综述应体现多元视角和客观分析。

        修正后的观点综述：

        """
