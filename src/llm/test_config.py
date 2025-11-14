#!/usr/bin/env python3
"""
配置系统测试
"""

import unittest
import tempfile
import os
from config_parser import EnvironmentVariableResolver, ConfigValidator

class TestConfigSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前设置环境变量"""
        os.environ['TEST_API_KEY'] = 'test_key_123'
        os.environ['TEST_URL'] = 'https://api.test.com'
    
    def test_environment_variable_resolution(self):
        """测试环境变量解析"""
        
        # 测试 ${VAR} 格式
        result = EnvironmentVariableResolver.resolve_string('api_key: ${TEST_API_KEY}')
        self.assertEqual(result, 'api_key: test_key_123')
        
        # 测试嵌套配置
        config = {
            'llm': {
                'api_key': '${TEST_API_KEY}',
                'url': '${TEST_URL}'
            }
        }
        
        resolved = EnvironmentVariableResolver.resolve_config(config)
        self.assertEqual(resolved['llm']['api_key'], 'test_key_123')
        self.assertEqual(resolved['llm']['url'], 'https://api.test.com')
    
    def test_config_validation(self):
        """测试配置验证"""
        
        # 有效配置
        valid_config = {
            'llm': {
                'provider': 'deepseek',
                'api_key': 'valid_key',
                'model': 'deepseek-chat'
            }
        }
        
        self.assertTrue(ConfigValidator.validate_llm_config(valid_config))
        
        # 无效配置（缺少api_key）
        invalid_config = {
            'llm': {
                'provider': 'deepseek',
                'model': 'deepseek-chat'
                # 缺少 api_key
            }
        }
        
        self.assertFalse(ConfigValidator.validate_llm_config(invalid_config))

def run_tests():
    """运行测试"""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()