#!/usr/bin/env python3
"""
配置自动解析系统
解决环境变量自动替换问题
"""

import os
import yaml
import logging
import re
from typing import Dict, Any, Optional
from pathlib import Path

class EnvironmentVariableResolver:
    """环境变量解析器"""
    
    @staticmethod
    def resolve_string(value: str) -> str:
        """
        解析字符串中的环境变量占位符
        支持格式: ${VAR_NAME} 和 $VAR_NAME
        """
        if not isinstance(value, str):
            return value
            
        def replace_match(match):
            var_name = match.group(1) or match.group(2)
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            else:
                # 如果环境变量不存在，保持原占位符
                logging.warning(f"环境变量未设置: {var_name}")
                return match.group(0)
        
        # 匹配 ${VAR} 和 $VAR 格式
        pattern = r'\$\{([^}]+)\}|\$([a-zA-Z_][a-zA-Z0-9_]*)'
        return re.sub(pattern, replace_match, value)
    
    @staticmethod
    def resolve_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """递归解析配置中的所有环境变量"""
        
        def resolve_value(obj):
            if isinstance(obj, str):
                return EnvironmentVariableResolver.resolve_string(obj)
            elif isinstance(obj, dict):
                return {k: resolve_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_value(item) for item in obj]
            else:
                return obj
        
        return resolve_value(config_data)


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_llm_config(config: Dict[str, Any]) -> bool:
        """验证LLM配置完整性"""
        required_fields = ['provider', 'api_key', 'model']
        llm_config = config.get('llm', {})
        
        for field in required_fields:
            if field not in llm_config or not llm_config[field]:
                logging.error(f"LLM配置缺少必要字段: {field}")
                return False
        
        # 检查API密钥是否已正确解析（不应包含${}占位符）
        api_key = llm_config['api_key']
        if api_key.startswith('${') and api_key.endswith('}'):
            logging.error("API密钥环境变量未正确解析")
            return False
            
        return True
    
    @staticmethod
    def validate_vector_db_config(config: Dict[str, Any]) -> bool:
        """验证向量数据库配置"""
        vector_config = config.get('vector_db', {})
        required_fields = ['embedding_model', 'db_path']
        
        for field in required_fields:
            if field not in vector_config:
                logging.warning(f"向量数据库配置缺少字段: {field}")
                
        return True
    
    @staticmethod
    def validate_complete_config(config: Dict[str, Any]) -> bool:
        """验证完整配置"""
        if not config:
            logging.error("配置为空")
            return False
            
        if not ConfigValidator.validate_llm_config(config):
            return False
            
        if not ConfigValidator.validate_vector_db_config(config):
            return False
            
        logging.info("配置验证通过")
        return True


class ConfigManager:
    """配置管理器 - 主类"""
    
    def __init__(self, config_path: str = "config.yaml", env_file: str = ".env"):
        self.config_path = Path(config_path)
        self.env_file = Path(env_file)
        self._config = None
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('config_manager.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_environment_variables(self) -> bool:
        """加载环境变量"""
        try:
            # 优先加载.env文件
            if self.env_file.exists():
                with open(self.env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                self.logger.info(f"已加载环境变量文件: {self.env_file}")
            
            # 检查必需的环境变量
            required_vars = ['DEEPSEEK_API_KEY']
            for var in required_vars:
                if var not in os.environ or not os.environ[var]:
                    self.logger.warning(f"环境变量未设置: {var}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"加载环境变量失败: {str(e)}")
            return False
    
    def load_config_file(self) -> Optional[Dict[str, Any]]:
        """加载YAML配置文件"""
        try:
            if not self.config_path.exists():
                self.logger.error(f"配置文件不存在: {self.config_path}")
                return None
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            self.logger.info(f"配置文件加载成功: {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML解析错误: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"读取配置文件失败: {str(e)}")
            return None
    
    def resolve_config(self) -> Optional[Dict[str, Any]]:
        """解析配置中的环境变量"""
        try:
            # 加载环境变量
            if not self.load_environment_variables():
                return None
                
            # 加载配置文件
            raw_config = self.load_config_file()
            if raw_config is None:
                return None
            
            # 解析环境变量
            resolved_config = EnvironmentVariableResolver.resolve_config(raw_config)
            
            # 验证配置
            if not ConfigValidator.validate_complete_config(resolved_config):
                return None
                
            self._config = resolved_config
            return resolved_config
            
        except Exception as e:
            self.logger.error(f"解析配置失败: {str(e)}")
            return None
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置（单例模式）"""
        if self._config is None:
            self.resolve_config()
        return self._config or {}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.get_config().get('llm', {})
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """获取向量数据库配置"""
        return self.get_config().get('vector_db', {})
    
    def create_env_template(self) -> bool:
        """创建环境变量模板"""
        try:
            template = """# DeepSeek API配置
DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here

# 可选配置
LOG_LEVEL=INFO
DEBUG=false
"""
            with open('.env.template', 'w', encoding='utf-8') as f:
                f.write(template)
            self.logger.info("已创建环境变量模板: .env.template")
            return True
        except Exception as e:
            self.logger.error(f"创建环境变量模板失败: {str(e)}")
            return False


class LLMClient:
    """LLM客户端示例（使用解析后的配置）"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_llm_config()
        self.logger = logging.getLogger(__name__)
    
    def validate_connection(self) -> bool:
        """验证API连接"""
        api_key = self.config.get('api_key', '')
        provider = self.config.get('provider', '')
        
        if not api_key or api_key.startswith('your_'):
            self.logger.error("API密钥未正确配置")
            return False
            
        self.logger.info(f"✅ {provider.upper()} 配置验证通过")
        self.logger.info(f"   模型: {self.config.get('model')}")
        self.logger.info(f"   Base URL: {self.config.get('base_url')}")
        
        return True
    
    def get_client_config(self) -> Dict[str, Any]:
        """获取客户端配置"""
        return {
            'api_key': self.config.get('api_key'),
            'model': self.config.get('model'),
            'base_url': self.config.get('base_url'),
            'temperature': self.config.get('temperature', 0.1),
            'max_tokens': self.config.get('max_tokens', 1000)
        }


def setup_environment_interactive():
    """交互式环境设置"""
    print("=== 环境变量设置向导 ===")
    print()
    
    deepseek_key = input("请输入您的DeepSeek API密钥: ").strip()
    
    if not deepseek_key:
        print("❌ API密钥不能为空")
        return False
    
    env_content = f"""DEEPSEEK_API_KEY={deepseek_key}
LOG_LEVEL=INFO
DEBUG=false
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ 环境变量已保存到 .env 文件")
        return True
    except Exception as e:
        print(f"❌ 保存环境变量文件失败: {e}")
        return False


def main():
    """主函数示例"""
    print("=== 配置自动解析系统 ===")
    print()
    
    # 检查环境变量文件是否存在
    if not Path('.env').exists():
        print("检测到缺少 .env 文件")
        setup = input("是否立即设置环境变量? (y/n): ")
        if setup.lower() in ['y', 'yes']:
            if not setup_environment_interactive():
                return
        else:
            print("请手动创建 .env 文件或设置环境变量")
            return
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 加载并解析配置
    config = config_manager.resolve_config()
    
    if config is None:
        print("❌ 配置加载失败，请检查日志文件")
        return
    
    print("✅ 配置加载成功")
    print()
    
    # 显示配置信息
    llm_config = config_manager.get_llm_config()
    print("LLM配置信息:")
    print(f"  提供商: {llm_config.get('provider')}")
    print(f"  模型: {llm_config.get('model')}")
    print(f"  API密钥: {llm_config.get('api_key')[:10]}...")  # 安全显示
    print(f"  温度: {llm_config.get('temperature')}")
    print()
    
    # 验证LLM连接
    llm_client = LLMClient(config_manager)
    if llm_client.validate_connection():
        print("🎉 系统配置完成，可以正常使用！")
    else:
        print("❌ LLM配置验证失败，请检查API密钥")


if __name__ == "__main__":
    main()