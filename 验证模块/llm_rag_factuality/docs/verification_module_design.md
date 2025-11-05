# 验证模块设计与实现说明

## 📋 概述

验证模块是RAG减弱大模型幻觉系统的核心组件之一，负责检测LLM生成回答中的幻觉内容，确保回答与检索文档的一致性。本模块采用多层次验证策略，结合规则基础验证和语义验证，提供详细的验证结果和纠正建议。

## 🏗️ 架构设计

### 核心组件

```
验证模块 (src/verification/)
├── fact_checker.py          # 核心验证逻辑
├── config.py               # 验证相关配置
└── test_verification.py    # 测试用例
```

### 类层次结构

```
FactChecker (核心验证器)
├── __init__()              # 初始化验证器
├── verify_answer()         # 主要验证接口
├── _extract_key_information()  # 关键信息提取
├── _basic_verification()   # 基础验证
├── _semantic_verification()    # 语义验证
└── _combine_verification_results()  # 结果综合

VerificationResult (验证结果)
├── has_hallucination       # 是否存在幻觉
├── confidence_score        # 置信度评分
├── error_descriptions      # 错误描述列表
├── verification_details    # 详细验证信息
├── evidence_chunks         # 支持证据
└── verification_level      # 验证级别
```

## 🔍 验证策略

### 1. 基础验证 (Basic Verification)

**目标**: 通过规则检查快速发现明显的事实错误

**验证维度**:
- **数字一致性**: 检查回答中的数字是否在检索文档中找到支持
- **实体存在性**: 验证人名、地名、机构名等实体是否在文档中存在
- **声明支持性**: 检查声明性句子的关键词在文档中的支持度

**实现方法**:
```python
def _basic_verification(self, answer, key_info, retrieved_chunks):
    # 1. 数字验证
    number_issues = self._verify_numbers(key_info['numbers'], retrieved_chunks)
    
    # 2. 实体验证  
    entity_issues = self._verify_entities(key_info['entities'], retrieved_chunks)
    
    # 3. 声明验证
    claim_issues = self._verify_claims(key_info['claims'], retrieved_chunks)
    
    return {
        'confidence': 计算置信度,
        'issues_found': 合并所有问题,
        'checks_performed': 记录执行的检查
    }
```

### 2. 语义验证 (Semantic Verification)

**目标**: 通过LLM进行深层次的语义一致性检查

**验证维度**:
- **事实一致性**: 回答中的事实陈述是否与参考文档一致
- **逻辑合理性**: 回答的推理逻辑是否合理
- **信息完整性**: 重要信息是否被正确包含或排除

**实现方法**:
```python
def _semantic_verification(self, answer, key_info, retrieved_chunks, query):
    # 构建语义验证prompt
    verification_prompt = self._build_semantic_verification_prompt(
        answer, retrieved_chunks, query
    )
    
    # 调用LLM进行验证
    llm_result = self._call_llm_verification(verification_prompt)
    
    return {
        'is_consistent': llm_result['consistency'],
        'confidence': llm_result['confidence'],
        'reasoning': llm_result['reasoning']
    }
```

### 3. 综合验证 (Comprehensive Verification)

**目标**: 结合基础验证和语义验证，提供最全面的验证结果

**策略**:
- 基础验证权重: 60%
- 语义验证权重: 40%
- 综合置信度计算
- 幻觉判定阈值

## 📊 数据结构

### VerificationResult 详解

```python
@dataclass
class VerificationResult:
    has_hallucination: bool           # 是否存在幻觉
    confidence_score: float           # 置信度评分 (0-1)
    error_descriptions: List[str]     # 错误描述列表
    verification_details: Dict[str, Any]  # 详细验证信息
    evidence_chunks: List[Dict]       # 支持证据文档块
    verification_level: str          # 验证级别
```

**字段说明**:
- `has_hallucination`: 基于置信度和错误数量判断是否存在幻觉
- `confidence_score`: 综合验证结果的置信度评分
- `error_descriptions`: 具体的错误描述，用于后续纠正
- `verification_details`: 包含基础验证和语义验证的详细信息
- `evidence_chunks`: 提供支持的文档证据
- `verification_level`: 执行的验证级别

## 🔧 使用方法

### 基本使用

```python
from src.verification.fact_checker import verify_answer

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

# 处理结果
if result.has_hallucination:
    print(f"检测到幻觉，置信度: {result.confidence_score:.3f}")
    print(f"错误描述: {result.error_descriptions}")
else:
    print(f"验证通过，置信度: {result.confidence_score:.3f}")
```

### 高级使用

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
basic_details = result.verification_details['basic']
semantic_details = result.verification_details['semantic']
evidence = result.evidence_chunks
```

### 不同验证级别

```python
# 快速基础验证
result_basic = verify_answer(answer, chunks, "query", "basic")

# 语义验证
result_semantic = verify_answer(answer, chunks, "query", "semantic") 

# 综合验证
result_comprehensive = verify_answer(answer, chunks, "query", "comprehensive")
```

## ⚙️ 配置参数

### config.py 中的关键配置

```python
# 幻觉检测阈值
HALLUCINATION_THRESHOLD = 0.7  # 置信度低于此值认为可能存在幻觉

# 验证级别
VERIFICATION_LEVELS = {
    "basic": "基础验证（规则检查）",
    "semantic": "语义验证（LLM检查）", 
    "comprehensive": "综合验证（基础+语义）"
}

# 默认验证级别
DEFAULT_VERIFICATION_LEVEL = "comprehensive"
```

## 🧪 测试验证

### 运行测试

```bash
cd llm_rag_factuality
python test_verification.py
```

### 测试覆盖

1. **基础验证测试**: 测试数字、实体、声明验证
2. **综合验证测试**: 测试多层次验证组合
3. **不同级别测试**: 测试三种验证级别的差异
4. **边界情况测试**: 测试极端输入的处理

## 🚀 性能优化

### 1. 缓存机制
- 嵌入向量缓存
- 检索结果缓存
- 验证结果缓存

### 2. 并行处理
- 多文档块并行验证
- 异步LLM调用
- 批量验证支持

### 3. 内存优化
- 文档块分页加载
- 及时释放大型对象
- 限制并发请求数

## 🔮 扩展性

### 1. 新增验证策略
```python
class CustomFactChecker(FactChecker):
    def _custom_verification(self, answer, chunks):
        # 实现自定义验证逻辑
        pass
    
    def verify_answer(self, answer, chunks, query):
        # 调用自定义验证
        custom_result = self._custom_verification(answer, chunks)
        # 与现有验证结果合并
        return self._combine_results(custom_result)
```

### 2. 集成新的LLM
```python
def _call_llm_verification(self, prompt):
    # 支持多种LLM提供商
    if self.llm_provider == "openai":
        return self._call_openai(prompt)
    elif self.llm_provider == "deepseek":
        return self._call_deepseek(prompt)
```

### 3. 自定义验证规则
```python
# 在初始化时添加自定义规则
self.custom_rules = [
    (pattern, validation_function),
    (pattern2, validation_function2),
]
```

## 📈 评估指标

### 验证准确性
- **准确率**: 正确识别的幻觉/非幻觉比例
- **精确率**: 识别为幻觉中真正是幻觉的比例
- **召回率**: 实际幻觉中被正确识别的比例

### 性能指标
- **验证延迟**: 单次验证的平均时间
- **吞吐量**: 每秒处理的验证请求数
- **资源使用**: CPU和内存使用情况

## 🛡️ 错误处理

### 常见错误类型
1. **LLM调用失败**: 网络问题、API限制
2. **文档解析错误**: 编码问题、格式错误
3. **内存不足**: 大文档处理时的内存溢出

### 错误处理策略
```python
try:
    result = checker.verify_answer(answer, chunks, query)
except LLMAPIError as e:
    # 降级到基础验证
    result = fallback_basic_verification(answer, chunks)
except MemoryError:
    # 分批处理
    result = batch_verification(answer, chunks)
except Exception as e:
    # 记录错误并返回保守结果
    logger.error(f"验证失败: {e}")
    return conservative_result()
```

## 📝 最佳实践

### 1. 验证前准备
- 确保检索文档质量高
- 预处理文本，去除噪声
- 设置合适的验证级别

### 2. 结果解读
- 关注置信度评分趋势
- 仔细分析错误描述
- 结合证据文档判断

### 3. 性能调优
- 根据场景选择验证级别
- 合理设置缓存策略
- 监控验证延迟

## 🔗 集成说明

### 与其他模块的接口

**输入接口**:
- `answer`: LLM生成的回答文本
- `retrieved_chunks`: 检索模块返回的文档块
- `query`: 原始查询（可选）

**输出接口**:
- `VerificationResult`: 结构化的验证结果
- 支持后续纠正模块使用

**依赖关系**:
- 依赖检索模块提供文档块
- 依赖LLM模块进行语义验证
- 为纠正模块提供验证结果

这个验证模块为整个RAG系统提供了坚实的事实基础，确保生成的回答具有高度的可信度和准确性。