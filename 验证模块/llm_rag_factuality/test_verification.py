"""
验证模块测试用例
演示如何使用fact_checker进行回答验证
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verification.fact_checker import verify_answer, VerificationLevel
import config

def test_basic_verification():
    """测试基础验证功能"""
    print("🧪 测试基础验证功能")
    print("=" * 50)
    
    # 模拟检索到的文档块
    retrieved_chunks = [
        {
            "text": "BAAI/bge-base-en-v1.5是一个由北京人工智能研究院开发的嵌入模型，输出向量维度为768维。",
            "metadata": {"source": "embedding_model_info.txt"},
            "distance": 0.1
        },
        {
            "text": "该模型在多个自然语言处理任务中表现优异，包括文本相似度计算和文档检索。",
            "metadata": {"source": "model_performance.txt"},
            "distance": 0.2
        }
    ]
    
    # 测试案例1：正确回答
    answer1 = "BAAI/bge-base-en-v1.5嵌入模型的输出向量维度是768维。"
    result1 = verify_answer(answer1, retrieved_chunks, "嵌入模型维度", "basic")
    
    print(f"测试案例1 - 正确回答:")
    print(f"回答: {answer1}")
    print(f"幻觉检测: {'是' if result1.has_hallucination else '否'}")
    print(f"置信度: {result1.confidence_score:.3f}")
    print(f"错误描述: {result1.error_descriptions}")
    print()
    
    # 测试案例2：包含幻觉的回答
    answer2 = "BAAI/bge-base-en-v1.5嵌入模型的输出向量维度是1024维，并且该模型可以处理图像输入。"
    result2 = verify_answer(answer2, retrieved_chunks, "嵌入模型维度", "basic")
    
    print(f"测试案例2 - 包含幻觉的回答:")
    print(f"回答: {answer2}")
    print(f"幻觉检测: {'是' if result2.has_hallucination else '否'}")
    print(f"置信度: {result2.confidence_score:.3f}")
    print(f"错误描述: {result2.error_descriptions}")
    print()

def test_comprehensive_verification():
    """测试综合验证功能"""
    print("🧪 测试综合验证功能")
    print("=" * 50)
    
    # 模拟检索到的文档块
    retrieved_chunks = [
        {
            "text": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。",
            "metadata": {"source": "deep_learning_intro.txt"},
            "distance": 0.1
        },
        {
            "text": "卷积神经网络（CNN）特别适用于图像处理任务，而循环神经网络（RNN）适用于序列数据。",
            "metadata": {"source": "neural_networks.txt"},
            "distance": 0.15
        }
    ]
    
    # 测试案例：部分正确的回答
    answer = "深度学习使用多层神经网络，CNN主要用于图像处理，但RNN也可以处理图像任务。"
    result = verify_answer(answer, retrieved_chunks, "深度学习网络类型", "comprehensive")
    
    print(f"测试案例 - 部分正确的回答:")
    print(f"回答: {answer}")
    print(f"幻觉检测: {'是' if result.has_hallucination else '否'}")
    print(f"置信度: {result.confidence_score:.3f}")
    print(f"验证级别: {result.verification_level}")
    print(f"错误描述: {result.error_descriptions}")
    print(f"支持证据数量: {len(result.evidence_chunks)}")
    print()

def test_verification_levels():
    """测试不同验证级别"""
    print("🧪 测试不同验证级别")
    print("=" * 50)
    
    retrieved_chunks = [
        {
            "text": "Python是一种解释型、面向对象、动态数据类型的高级程序设计语言。",
            "metadata": {"source": "python_info.txt"},
            "distance": 0.1
        }
    ]
    
    answer = "Python是一种编译型编程语言，主要用于Web开发。"
    
    # 测试不同验证级别
    levels = ["basic", "semantic", "comprehensive"]
    
    for level in levels:
        result = verify_answer(answer, retrieved_chunks, "Python语言特点", level)
        print(f"验证级别: {level}")
        print(f"  幻觉检测: {'是' if result.has_hallucination else '否'}")
        print(f"  置信度: {result.confidence_score:.3f}")
        print(f"  错误数量: {len(result.error_descriptions)}")
        print()

def create_sample_documents():
    """创建示例文档用于测试"""
    print("📝 创建示例文档")
    print("=" * 50)
    
    # 创建raw_docs目录
    os.makedirs(config.RAW_DOC_DIR, exist_ok=True)
    
    # 创建示例文档
    sample_docs = {
        "embedding_models.txt": """
BAAI/bge-base-en-v1.5是由北京人工智能研究院开发的嵌入模型。
该模型在MTEB基准测试中表现优异，特别是在检索任务上。
模型输出向量维度为768维，支持最大序列长度512。
""",
        
        "deep_learning.txt": """
深度学习是机器学习的一个重要分支。
它使用多层神经网络来学习数据的复杂表示。
卷积神经网络（CNN）特别适用于图像处理任务。
循环神经网络（RNN）适用于处理序列数据，如文本和时间序列。
Transformer架构在自然语言处理任务中取得了突破性进展。
""",
        
        "python_language.txt": """
Python是一种解释型、面向对象、动态数据类型的高级程序设计语言。
Python语言具有简洁的语法和强大的功能。
Python广泛应用于Web开发、数据科学、人工智能等领域。
Python拥有丰富的第三方库和框架。
"""
    }
    
    for filename, content in sample_docs.items():
        filepath = os.path.join(config.RAW_DOC_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建文档: {filepath}")
    
    print(f"示例文档创建完成，共{len(sample_docs)}个文档")
    print()

def main():
    """主测试函数"""
    print("🔍 验证模块测试")
    print("=" * 60)
    
    # 创建示例文档
    create_sample_documents()
    
    # 运行测试
    test_basic_verification()
    test_comprehensive_verification()
    test_verification_levels()
    
    print("✅ 所有测试完成")

if __name__ == "__main__":
    main()