 各文件详细说明
1. 原始数据文件 (raw/)
TruthfulQA.csv
作用: TruthfulQA基准测试集的原始数据
内容: 790个问题，包含问题、多个正确答案、错误答案、类别等信息
用途: 构建事实准确性评估的基础数据

cmu.csv, topical.csv, wow.csv
作用: FaithDial Gold数据集的三个子集
内容:
cmu.csv: 201个CMU对话样本
topical.csv: 997个主题对话样本
wow.csv: 200个Wizard of Wikipedia样本
用途: 提供对话场景下的幻觉标注数据

2. 中间处理文件 (interim/)
truthfulqa_processed.json
作用: 处理后的TruthfulQA数据集
内容: 3,556个样本，每个包含：
用途: 标准化格式，便于后续处理

faithdial_gold_processed.json
作用: 处理后的FaithDial Gold数据集
内容: 589个样本，每个包含：
用途: 提供高质量的幻觉检测数据

3. 最终数据集文件 (processed/)
train.json (2,901样本)
作用: 模型训练数据集
内容: 70%的总数据，用于训练RAG系统
分布: TruthfulQA(85.3%) + FaithDial Gold(14.7%)

val.json (415样本)
作用: 超参数调优和模型选择
内容: 10%的总数据，用于开发集评估
分布: TruthfulQA(84.8%) + FaithDial Gold(15.2%)

test.json (829样本)
作用: 最终性能评估
内容: 20%的总数据，用于测试集评估
分布: TruthfulQA(87.8%) + FaithDial Gold(12.2%)

4. 处理脚本文件 (scripts/)
inspect_data_files.py
作用: 检查原始数据文件结构
功能: 显示列名、数据形状、样本示例
用途: 数据探索和问题诊断

process_truthfulqa.py
作用: TruthfulQA数据预处理
功能: 数据清洗、格式转换、多答案提取
输出: truthfulqa_processed.json

process_faithdial_gold.py
作用: FaithDial Gold数据预处理
功能: 列名映射、数据验证、类别推断
输出: faithdial_gold_processed.json

merge_and_split_datasets.py
作用: 数据集合并与划分
功能: 合并两个数据集、随机划分、统计分析
输出: train.json, val.json, test.json

verify_final_datasets.py
作用: 最终数据集验证
功能: 检查文件完整性、样本质量、数据分布
用途: 质量保证