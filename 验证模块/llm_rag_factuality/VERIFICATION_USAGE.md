# 验证模块使用指南

## 🚀 快速开始

### 1. 安装依赖
```bash
cd llm_rag_factuality
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
# 复制环境变量模板
cp .env.template .env

# 编辑 .env 文件，填入您的API密钥
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 运行测试
```bash
python test_verification.py
```

## 📖 基本用法

### 导入模块
```python
from src.verification.fact_checker import verify_answer, FactChecker
```

### 简单验证
```python
# 准备数据
answer = "BAAI/bge-base-en-v1.5嵌入模型的输出向量维度是768维。"
retrieved_chunks = [
    {
        "text": "BAAI/bge-base-en-v1.5是一个嵌入模型，输出维度为768维。",
        "metadata": {"source": "model_info.txt"},
        "distance": 0.1
    }
]

# 执行验证
result = verify_answer(answer, retrieved_chunks, "嵌入模型维度")

# 检查结果
if result.has_hallucination:
    print(f"检测到幻觉，置信度: {result.confidence_score:.3f}")
    print(f"错误描述: {result.error_descriptions}")
else:
    print(f"验证通过，置信度: {result.confidence_score:.3f}")
```

### 高级用法
```python
from src.verification.fact_checker import FactChecker, VerificationLevel

# 创建自定义验证器
checker = FactChecker(verification_level=VerificationLevel.COMPREHENSIVE)

# 执行详细验证
result = checker.verify_answer(
    answer=answer,
    retrieved_chunks=retrieved_chunks,
    query="嵌入模型的技术规格"
)

# 访问详细结果
print(f"验证级别: {result.verification_level}")
print(f"支持证据数量: {len(result.evidence_chunks)}")
print(f"基础验证详情: {result.verification_details['basic']}")
print(f"语义验证详情: {result.verification_details['semantic']}")
```

## 🔧 配置选项

### 验证级别
- **"basic"**: 基础验证（规则检查）- 快速，适合实时应用
- **"semantic"**: 语义验证（LLM检查）- 深度分析，适合重要场景
- **"comprehensive"**: 综合验证（基础+语义）- 最全面，适合生产环境

### 关键配置参数
在 `config.py` 中调整：
```python
# 幻觉检测阈值
HALLUCINATION_THRESHOLD = 0.7

# 验证级别
DEFAULT_VERIFICATION_LEVEL = "comprehensive"
```

## 🧪 测试用例

运行完整测试套件：
```bash
python test_verification.py
```

测试包含：
- ✅ 基础验证功能测试
- ✅ 综合验证功能测试  
- ✅ 不同验证级别测试
- ✅ 边界情况处理测试

## 📊 结果解读

### VerificationResult 对象
```python
{
    "has_hallucination": bool,          # 是否存在幻觉
    "confidence_score": float,          # 置信度评分 (0-1)
    "error_descriptions": List[str],    # 错误描述列表
    "verification_details": Dict,       # 详细验证信息
    "evidence_chunks": List[Dict],      # 支持证据文档块
    "verification_level": str          # 验证级别
}
```

### 置信度评分说明
- **0.9-1.0**: 高度可信，基本无幻觉
- **0.7-0.9**: 较为可信，可能存在小问题
- **0.5-0.7**: 中等可信，存在明显问题
- **0.0-0.5**: 低可信度，存在严重幻觉

### 错误描述类型
- **数字不一致**: "数字 '1024' 在检索文档中未找到支持"
- **实体不存在**: "实体 'XXX' 在检索文档中未找到"
- **支持度不足**: "声明 '...' 支持度不足 (0.23)"

## ⚡ 性能优化

### 1. 选择合适的验证级别
```python
# 实时应用使用基础验证
result = verify_answer(answer, chunks, query, "basic")

# 重要决策使用综合验证
result = verify_answer(answer, chunks, query, "comprehensive")
```

### 2. 缓存策略
```python
# 启用验证结果缓存
checker = FactChecker()
checker.enable_cache = True
```

### 3. 批处理
```python
# 批量验证多个回答
def batch_verify(answers, chunks_list, queries):
    results = []
    for answer, chunks, query in zip(answers, chunks_list, queries):
        result = verify_answer(answer, chunks, query)
        results.append(result)
    return results
```

## 🛠️ 故障排除

### 常见问题

**1. LLM调用失败**
```
错误: LLMAPIError: API调用失败
解决: 检查API密钥和网络连接，启用降级到基础验证
```

**2. 内存不足**
```
错误: MemoryError: 内存不足
解决: 减少文档块数量，启用分批处理
```

**3. 依赖包缺失**
```
错误: ModuleNotFoundError: No module named 'xxx'
解决: pip install -r requirements.txt
```

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
checker = FactChecker()
checker.verbose = True
```

## 🔗 集成指南

### 与检索模块集成
```python
from src.retrieval.chroma_retriever import retrieve_relevant_chunks

# 检索相关文档
retrieved_chunks = retrieve_relevant_chunks(query, chroma_path)

# 验证检索结果
result = verify_answer(answer, retrieved_chunks, query)
```

### 与LLM模块集成
```python
from src.llm.deepseek_client import generate_answer

# 生成初步回答
answer = generate_answer(query, retrieved_chunks)

# 验证回答
result = verify_answer(answer, retrieved_chunks, query)

# 如果存在幻觉，触发纠正
if result.has_hallucination:
    corrected_answer = correct_answer(answer, result, retrieved_chunks)
```

### 与纠正模块集成
```python
from src.correction.answer_corrector import correct_answer

# 使用验证结果进行纠正
if result.has_hallucination:
    corrected_result = correct_answer(
        original_answer=answer,
        verification_result=result,
        retrieved_chunks=retrieved_chunks
    )
```

## 📈 最佳实践

### 1. 验证前准备
- 确保检索文档质量高
- 预处理文本，去除噪声
- 设置合适的验证级别

### 2. 结果处理
- 关注置信度评分趋势
- 仔细分析错误描述
- 结合证据文档判断

### 3. 性能调优
- 根据场景选择验证级别
- 合理设置缓存策略
- 监控验证延迟

## 📚 更多资源

- [设计文档](docs/verification_module_design.md) - 详细的技术设计说明
- [完成报告](docs/verification_module_completion_report.md) - 开发完成情况总结
- [API文档](docs/api_reference.md) - 完整的API参考（待完善）

---

💡 **提示**: 验证模块是RAG系统的核心组件，建议在生产环境中使用"comprehensive"验证级别以获得最佳效果。