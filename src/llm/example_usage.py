#!/usr/bin/env python3
"""
配置系统使用
"""

from config_parser import ConfigManager, LLMClient
import logging

def demonstrate_advanced_usage():
    """演示高级用法"""
    
    # 初始化配置管理器
    config_manager = ConfigManager("config.yaml", ".env")
    
    # 获取完整配置
    full_config = config_manager.get_config()
    print("完整配置结构:")
    for section, settings in full_config.items():
        print(f"  {section}: {list(settings.keys())}")
    
    # 获取特定配置段
    llm_config = config_manager.get_llm_config()
    vector_config = config_manager.get_vector_db_config()
    
    print(f"\nLLM提供商: {llm_config.get('provider')}")
    print(f"向量数据库: {vector_config.get('db_path')}")
    
    # 使用LLM客户端
    llm_client = LLMClient(config_manager)
    client_config = llm_client.get_client_config()
    
    print(f"\n客户端配置:")
    for key, value in client_config.items():
        if key == 'api_key':
            value = f"{value[:10]}..."  # 安全显示
        print(f"  {key}: {value}")
    
    # 验证连接
    if llm_client.validate_connection():
        print("\n✅ 系统就绪，可以开始处理请求")
        
        # 这里可以添加实际的API调用代码
        # 例如使用requests调用DeepSeek API
        demonstrate_api_call(llm_config)
    else:
        print("\n❌ 配置验证失败")

def demonstrate_api_call(llm_config: dict):
    """演示API调用（示例）"""
    import requests
    import json
    
    api_key = llm_config.get('api_key')
    base_url = llm_config.get('base_url')
    
    if not api_key or api_key.startswith('your_'):
        print("请配置有效的API密钥")
        return
    
    # 示例API调用（需要根据实际API调整）
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": llm_config.get('model'),
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": llm_config.get('temperature'),
        "max_tokens": llm_config.get('max_tokens', 1000)
    }
    
    print(f"\n尝试调用API: {base_url}/chat/completions")
    # 注意：实际调用需要根据API文档调整
    # response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
    print("API调用代码已准备就绪（实际调用需要取消注释）")

if __name__ == "__main__":
    demonstrate_advanced_usage()