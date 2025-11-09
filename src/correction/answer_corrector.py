from typing import List, Dict, Any
from .deepseek_client import LLMAdapter

class IntentAwareCorrector:
    """意图感知的答案纠正器"""
    
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter
        
        # 意图特定的纠正模板
        self.correction_templates = {
            "事实查询": self._get_factual_correction_template(),
            "比较查询": self._get_comparison_correction_template(),
            "方法查询": self._get_method_correction_template(),
            "观点查询": self._get_opinion_correction_template()
        }
    
    def correct_answer(self, original_answer: str, verifications: List[Dict], 
                      query: str, intent: str) -> Dict[str, Any]:
        """生成纠正后的答案"""
        
        template = self.correction_templates.get(intent, self._get_factual_correction_template())
        verification_summary = self._prepare_verification_summary(verifications)
        
        prompt = template.format(
            query=query,
            intent=intent,
            original_answer=original_answer,
            verification_summary=verification_summary
        )
        
        response = self.llm.call_with_retry(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.3
        )
        
        if response.get('error'):
            corrected_text = f"纠正过程出错: {response.get('error_message', '未知错误')}"
        else:
            corrected_text = response['text']
        
        return {
            "corrected_answer": corrected_text,
            "original_answer": original_answer,
            "query": query,
            "intent": intent,
            "verifications_count": len(verifications),
            "supported_claims": len([v for v in verifications if v.get('verdict') == 'SUPPORTED']),
            "contradicted_claims": len([v for v in verifications if v.get('verdict') == 'CONTRADICTED']),
            "correction_metadata": response.get('usage', {})
        }
    
    def _prepare_verification_summary(self, verifications: List[Dict]) -> str:
        """准备验证结果摘要"""
        if not verifications:
            return "没有进行声明验证或验证全部失败。"
        
        summary_parts = []
        for i, verification in enumerate(verifications, 1):
            claim = verification