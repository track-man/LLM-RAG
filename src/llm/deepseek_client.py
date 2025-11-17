"""
DeepSeek客户端模块 - 封装LLMAdapter供RAG流程调用
适配rag_pipeline接口要求
"""
import os
import sys
from typing import Dict, Any, Optional

# 添加父目录到路径，确保可以导入LLMAdapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm.llm_adapter import LLMAdapter
import config  # 导入统一配置

def llm_inference(prompt: str, 
                 temperature: float = 0.1,
                 system_message: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs) -> str:
    """
    LLM推理函数 - 供RAG流程调用
    
    参数:
        prompt: 用户输入的提示文本
        temperature: 生成温度，控制随机性
        system_message: 系统角色消息
        max_tokens: 最大生成token数
        **kwargs: 其他LLM参数
        
    返回:
        LLM生成的文本内容
    """
    try:
        # 使用统一配置
        llm_config = config.LLM_CONFIG
        
        # 创建LLM适配器实例
        llm_adapter = LLMAdapter(
            provider=llm_config["provider"],
            config=llm_config
        )
        
        # 调用LLM
        result = llm_adapter.call(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 检查错误
        if result.get('error'):
            error_msg = f"LLM调用错误: {result.get('error_message', '未知错误')}"
            print(f"错误: {error_msg}")
            return f"抱歉，生成回答时出现错误: {error_msg}"
            
        return result['text']
        
    except Exception as e:
        error_msg = f"LLM推理过程异常: {str(e)}"
        print(f"异常: {error_msg}")
        return f"系统错误: {error_msg}"

def llm_inference_with_retry(prompt: str, 
                           max_retries: int = 3,
                           **kwargs) -> str:
    """
    带重试机制的LLM推理函数
    
    参数:
        prompt: 用户输入的提示文本
        max_retries: 最大重试次数
        **kwargs: 其他LLM参数
        
    返回:
        LLM生成的文本内容
    """
    try:
        llm_config = config.LLM_CONFIG
        llm_adapter = LLMAdapter(
            provider=llm_config["provider"],
            config=llm_config
        )
        
        result = llm_adapter.call_with_retry(
            prompt=prompt,
            max_retries=max_retries,
            **kwargs
        )
        
        if result.get('error'):
            return f"重试后仍失败: {result.get('error_message', '未知错误')}"
            
        return result['text']
        
    except Exception as e:
        return f"带重试的LLM调用异常: {str(e)}"

# 测试函数
if __name__ == "__main__":
    # 简单测试
    test_prompt = "请用中文简单介绍一下你自己。"
    result = llm_inference(test_prompt, temperature=0.7)
    print("测试结果:")
    print(result)