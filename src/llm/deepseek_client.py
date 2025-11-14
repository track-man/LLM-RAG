"""
DeepSeek客户端模块 - 封装LLMAdapter供RAG流程调用
"""
import os
import sys
from typing import Dict, Any, Optional
import yaml

# 添加父目录到路径，确保可以导入LLMAdapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm.llm_adapter import LLMAdapter  # 假设LLMAdapter在这个路径

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "llm_config.yaml")

def load_llm_config() -> Dict[str, Any]:
    """
    加载LLM配置
    """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('deepseek', {})
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        # 返回默认配置
        return {
            'provider': 'deepseek',
            'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
            'base_url': 'https://api.deepseek.com/v1',
            'model': 'deepseek-chat',
            'max_tokens': 2000,
            'temperature': 0.1
        }

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
        # 加载配置
        config = load_llm_config()
        
        # 更新配置参数
        if temperature is not None:
            config['temperature'] = temperature
        if max_tokens is not None:
            config['max_tokens'] = max_tokens
            
        # 创建LLM适配器实例
        llm_adapter = LLMAdapter(
            provider=config['provider'],
            config=config
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
        config = load_llm_config()
        llm_adapter = LLMAdapter(
            provider=config['provider'],
            config=config
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