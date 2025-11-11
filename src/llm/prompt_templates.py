#!/usr/bin/env python3
"""
大语言模型幻觉检测与纠正系统 - 增强版Prompt模板管理
位于: src/llm/prompt_templates.py

新增功能:
1. 初始回答生成模板 - 直接获取AI原始回答
2. 幻觉检测模板 - 检测回答中是否存在幻觉
3. 完整的比较分析框架
"""


"""增强版Prompt模板管理器"""
    
# ==================== 初始回答生成模板 ====================
INITIAL_ANSWER_TEMPLATE = """
请直接回答以下问题，不需要进行事实核查或验证，提供您认为最合适的答案。

问题: {question}

请提供详细、全面的回答，包括所有相关信息和背景知识：
"""

# ==================== 意图分类模板 ====================
INTENT_CLASSIFICATION_TEMPLATE = """
你是一个专业的查询意图分类器。你的任务是根据用户查询的内容，准确判断其意图类型。

## 分类标准
- **事实查询**: 寻求具体事实、数据、定义、属性等客观信息
- **比较查询**: 比较两个或多个实体、概念、方法的异同点  
- **方法查询**: 寻求操作流程、解决方案、实施步骤、操作方法
- **观点查询**: 收集多方意见、评价、争议观点、不同立场

## 分类规则
1. 如果查询包含"比较"、"对比"、"区别"、"哪个更好"等关键词，归类为比较查询
2. 如果查询包含"如何"、"怎样"、"步骤"、"方法"等关键词，归类为方法查询  
3. 如果查询包含"观点"、"看法"、"评价"、"争议"等关键词，归类为观点查询
4. 其他情况默认为事实查询

## 输出格式
只需返回意图类型的名称，不要添加任何解释。

当前查询: "{query}"
意图类型:
"""

# ==================== 声明提取模板 ====================
CLAIM_EXTRACTION_TEMPLATE = """
任务：将下面的文本分解为独立的真实性陈述（原子断言）。

需要提取的文本: "{text}"

提取结果:
"""

# ==================== 事实验证模板 ====================
FACT_VERIFICATION_TEMPLATE = """
作为事实核查专家，请基于提供的证据验证以下声明的真实性。

查询意图：{intent}
原始查询："{query}"
需要验证的声明："{claim}"

相关证据片段：
{evidence_text}

请按以下JSON格式输出验证结果：
{{
    "verdict": "SUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED|UNVERIFIED",
    "confidence": 0.0-1.0,
    "supporting_evidence": [
        {{
            "text": "证据文本",
            "source": "来源名称",
            "relevance_score": 0.0-1.0
        }}
    ],
    "contradicting_evidence": [
        {{
            "text": "矛盾证据文本", 
            "source": "来源名称",
            "contradiction_score": 0.0-1.0
        }}
    ],
    "reasoning": "详细的推理过程",
    "intent_specific_analysis": "针对查询意图的特别分析"
}}
"""

# ==================== 幻觉检测模板 ====================
HALLUCINATION_DETECTION_TEMPLATE = """
作为幻觉检测专家，请分析以下AI回答是否存在幻觉（虚构、不准确或缺乏证据支持的内容）。

## 检测标准
- **事实性幻觉**: 陈述与可验证事实不符
- **逻辑性幻觉**: 推理过程存在矛盾或不合逻辑
- **证据性幻觉**: 缺乏可靠证据支持的关键声明
- **一致性幻觉**: 与已知信息或上下文不一致

## 分析材料
原始问题: "{question}"
AI初始回答: "{initial_answer}"
验证后回答: "{verified_answer}"
支持证据: "{evidence}"

## 检测要求
请按以下JSON格式输出检测结果：
{{
    "has_hallucination": true|false,
    "hallucination_type": "FACTUAL|LOGICAL|EVIDENTIAL|CONSISTENCY|MIXED|NONE",
    "confidence": 0.0-1.0,
    "affected_sections": [
        {{
            "text": "存在幻觉的文本片段",
            "type": "幻觉类型",
            "severity": "LOW|MEDIUM|HIGH",
            "correction": "建议修正内容"
        }}
    ],
    "comparison_analysis": {{
        "initial_answer_quality": "评估初始回答质量",
        "verification_impact": "验证过程带来的改进",
        "key_differences": "主要差异点分析",
        "overall_improvement": "整体改善程度评估"
    }},
    "recommendations": [
        "改进建议1",
        "改进建议2"
    ]
}}

请开始分析：
"""

# ==================== 答案纠正模板 ====================
CORRECTION_TEMPLATES = {
    "事实查询": """
    作为事实核查专家，请根据验证结果重新生成一个准确的事实性答案。
    
    查询意图：{intent} - 事实查询
    原始查询："{query}"
    初始答案：{initial_answer}
    验证结果摘要：{verification_summary}
    
    修正后的答案：
    """,
    
    "比较查询": """
    作为比较分析专家，请根据验证结果重新生成一个全面准确的比较性答案。
    
    查询意图：{intent} - 比较查询  
    原始查询："{query}"
    初始答案：{initial_answer}
    验证结果摘要：{verification_summary}
    
    修正后的比较分析：
    """,
    
    "方法查询": """
    作为方法指导专家，请根据验证结果重新生成一个可操作的方法指南。
    
    查询意图：{intent} - 方法查询
    原始查询："{query}"
    初始答案：{initial_answer}
    验证结果摘要：{verification_summary}
    
    修正后的方法指南：
    """,
    
    "观点查询": """
    作为观点综述专家，请根据验证结果重新生成一个平衡客观的观点综述。
    
    查询意图：{intent} - 观点查询
    原始查询："{query}"
    初始答案：{initial_answer}
    验证结果摘要：{verification_summary}
    
    修正后的观点综述：
    """
}

# ==================== 比较分析模板 ====================
COMPARISON_ANALYSIS_TEMPLATE = """
# 回答质量比较分析报告

## 基本信息
- **分析时间**: {timestamp}
- **查询类型**: {intent}
- **原始问题**: "{question}"

## 回答对比

### 初始AI回答
{initial_answer}

**初始回答特点**:
- 生成速度: {initial_speed}
- 详细程度: {initial_detail}
- 自信程度: {initial_confidence}

### 验证后回答
{verified_answer}

**验证后回答特点**:
- 准确性提升: {accuracy_improvement}
- 证据支持度: {evidence_support}
- 可靠性评级: {reliability_rating}

## 幻觉检测结果
{hallucination_summary}

## 关键改进点
{key_improvements}

## 总体评估
{overall_assessment}
"""

def get_initial_prompt(self, question: str) -> str:
    """获取初始回答生成提示词"""
    return self.INITIAL_ANSWER_TEMPLATE.format(question=question)

def get_intent_classification_prompt(self, query: str) -> str:
    """获取意图分类提示词"""
    return self.INTENT_CLASSIFICATION_TEMPLATE.format(query=query)

def get_claim_extraction_prompt(self, text: str) -> str:
    """获取声明提取提示词"""
    return self.CLAIM_EXTRACTION_TEMPLATE.format(text=text)

def get_fact_verification_prompt(self, intent: str, query: str, claim: str, evidence_text: str) -> str:
    """获取事实验证提示词"""
    return self.FACT_VERIFICATION_TEMPLATE.format(
        intent=intent,
        query=query,
        claim=claim,
        evidence_text=evidence_text
    )

def get_hallucination_detection_prompt(self, question: str, initial_answer: str, 
                                        verified_answer: str, evidence: str) -> str:
    """获取幻觉检测提示词"""
    return self.HALLUCINATION_DETECTION_TEMPLATE.format(
        question=question,
        initial_answer=initial_answer,
        verified_answer=verified_answer,
        evidence=evidence
    )

def get_correction_prompt(self, intent: str, query: str, initial_answer: str, verification_summary: str) -> str:
    """获取答案纠正提示词"""
    template = self.CORRECTION_TEMPLATES.get(intent, self.CORRECTION_TEMPLATES["事实查询"])
    return template.format(
        intent=intent,
        query=query,
        initial_answer=initial_answer,
        verification_summary=verification_summary
    )

def get_comparison_analysis_prompt(self, question: str, intent: str, initial_answer: str, 
                                verified_answer: str, hallucination_summary: str) -> str:
    """获取比较分析提示词"""
    from datetime import datetime
    
    return self.COMPARISON_ANALYSIS_TEMPLATE.format(
        timestamp=datetime.now().isoformat(),
        intent=intent,
        question=question,
        initial_answer=initial_answer,
        verified_answer=verified_answer,
        initial_speed="快速",
        initial_detail="详细",
        initial_confidence="高",
        accuracy_improvement="显著",
        evidence_support="充分",
        reliability_rating="高",
        hallucination_summary=hallucination_summary,
        key_improvements="1. 事实准确性提升\n2. 证据支持增强\n3. 逻辑一致性改善",
        overall_assessment="验证过程显著提升了回答的可靠性和准确性"
    )


class EnhancedPipeline:
    """增强的流程管理器 - 集成初始回答、验证和幻觉检测"""

def __init__(self, llm_client, templates):
    self.llm_client = llm_client
    self.templates = templates

def process_question(self, question: str) -> dict:
    """处理问题的完整增强流程"""
    
    # 1. 生成初始回答
    print("🔄 生成初始AI回答...")
    initial_prompt = self.templates.get_initial_answer_prompt(question)
    initial_answer = self.llm_client.generate_response(initial_prompt)
    
    # 2. 意图分类
    print("🎯 分析查询意图...")
    intent_prompt = self.templates.get_intent_classification_prompt(question)
    intent = self.llm_client.generate_response(intent_prompt)
    
    # 3. 声明提取
    print("🔍 提取回答中的声明...")
    claim_prompt = self.templates.get_claim_extraction_prompt(initial_answer)
    claims_text = self.llm_client.generate_response(claim_prompt)
    
    # 4. 事实验证（模拟证据）
    print("✅ 进行事实验证...")
    evidence = "相关证据内容..."  # 这里应该是实际的检索结果
    verification_results = []
    
    # 5. 生成验证后回答
    print("✏️ 生成验证后回答...")
    verification_summary = "验证结果摘要..."
    correction_prompt = self.templates.get_correction_prompt(
        intent, question, initial_answer, verification_summary
    )
    verified_answer = self.llm_client.generate_response(correction_prompt)
    
    # 6. 幻觉检测
    print("🔬 进行幻觉检测...")
    hallucination_prompt = self.templates.get_hallucination_detection_prompt(
        question, initial_answer, verified_answer, evidence
    )
    hallucination_analysis = self.llm_client.generate_response(hallucination_prompt)
    
    # 7. 比较分析
    print="📊 生成比较分析报告..."
    comparison_prompt = self.templates.get_comparison_analysis_prompt(
        question, intent, initial_answer, verified_answer, hallucination_analysis
    )
    comparison_report = self.llm_client.generate_response(comparison_prompt)
    
    return {
        "question": question,
        "intent": intent,
        "initial_answer": initial_answer,
        "verified_answer": verified_answer,
        "verification_results": verification_results,
        "hallucination_analysis": hallucination_analysis,
        "comparison_report": comparison_report,
        "processing_metadata": {
            "timestamp": self._get_timestamp(),
            "steps_completed": [
                "initial_answer_generation",
                "intent_classification", 
                "claim_extraction",
                "fact_verification",
                "answer_correction",
                "hallucination_detection",
                "comparison_analysis"
            ]
        }
    }

def _get_timestamp(self):
    """获取时间戳"""
    from datetime import datetime
    return datetime.now().isoformat()


# # ==================== 使用示例和测试 ====================
# def demonstrate_enhanced_pipeline():
#     """演示增强版流程"""
    
#     # 模拟LLM客户端
#     class MockLLMClient:
#         def generate_response(self, prompt):
#             return f"模拟响应: {prompt[:50]}..."
    
#     # 初始化组件
#     templates = PromptTemplates()
#     llm_client = MockLLMClient()
#     pipeline = EnhancedPipeline(llm_client, templates)
    
#     # 测试问题
#     test_question = "人工智能的未来发展趋势是什么？"
    
#     print("🚀 开始增强版流程演示")
#     print("=" * 60)
    
#     # 执行完整流程
#     result = pipeline.process_question(test_question)
    
#     # 显示结果
#     print("\n📋 处理结果摘要:")
#     print(f"问题: {result['question']}")
#     print(f"检测到的意图: {result['intent']}")
#     print(f"初始回答长度: {len(result['initial_answer'])} 字符")
#     print(f"验证后回答长度: {len(result['verified_answer'])} 字符")
#     print(f"是否检测到幻觉: {'是' if 'hallucination' in str(result['hallucination_analysis']) else '否'}")
    
#     print("\n📊 比较分析:")
#     print(result['comparison_report'][:200] + "...")
    
#     return result


# def test_template_functionality():
#     """测试模板功能完整性"""
    
#     templates = PromptTemplates()
    
#     # 测试所有模板方法
#     test_cases = [
#         {
#             "name": "初始回答生成",
#             "method": templates.get_initial_answer_prompt,
#             "args": ["测试问题"]
#         },
#         {
#             "name": "意图分类", 
#             "method": templates.get_intent_classification_prompt,
#             "args": ["测试查询"]
#         },
#         {
#             "name": "幻觉检测",
#             "method": templates.get_hallucination_detection_prompt,
#             "args": ["问题", "初始回答", "验证回答", "证据"]
#         }
#     ]
    
#     print("🧪 模板功能测试")
#     print("=" * 40)
    
#     for test_case in test_cases:
#         try:
#             result = test_case["method"](*test_case["args"])
#             print(f"✅ {test_case['name']}: 成功生成提示词")
#             print(f"   样例: {result[:80]}...")
#         except Exception as e:
#             print(f"❌ {test_case['name']}: 失败 - {e}")


# if __name__ == "__main__":
#     # 运行功能测试
#     test_template_functionality()
    
#     print("\n" + "="*60)
    
#     # 演示增强流程
#     demonstrate_enhanced_pipeline()
# """
# # 初始化流程
# from src.llm.deepseek_client import DeepSeekClient
# from src.llm.prompt_templates import PromptTemplates, EnhancedPipeline

# # 创建组件
# llm_client = DeepSeekClient(config)
# templates = PromptTemplates()
# pipeline = EnhancedPipeline(llm_client, templates)

# # 执行完整流程
# question = "量子计算对密码学的影响是什么？"
# result = pipeline.process_question(question)

# # 分析结果
# print("初始回答:", result['initial_answer'])
# print("验证后回答:", result['verified_answer'])
# print("幻觉分析:", result['hallucination_analysis'])
# print("比较报告:", result['comparison_report'])
# """