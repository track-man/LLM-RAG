注：
gold/ 
内容：人工修正的忠实回复
用途：作为ground truth评估RAG系统的效果

CTRL/, GPT2/, DoHA/ 
内容：不同模型生成的有幻觉的回复
用途：分析不同模型的幻觉模式（可作为对比分析）

1. faithdial_batch_classification_report.json - 详细分类报告
文件内容结构：
这个文件包含12个详细的数据集分类报告，每个报告对应一个处理文件。
字段详细说明：
dataset: 数据集标识（格式：FaithDial_模型_数据集）
total_samples: 该文件的样本总数
intent_distribution: 意图分布（具体数量）
percentage_distribution: 百分比分布

2. faithdial_batch_summary.json - 批量处理汇总
文件内容结构：
这个文件包含12个处理结果的汇总信息，比详细报告多了文件路径和模型信息。
额外字段说明：
model: 模型名称（CTRL、DoHA、GPT2、gold）
dataset: 数据集名称（cmu、topical、wow）
file: 原始文件名

3. 分类后的数据文件 (12个CSV文件)
具体文件列表：
faithdial_CTRL_cmu_classified.csv
faithdial_CTRL_topical_classified.csv  
faithdial_CTRL_wow_classified.csv
faithdial_DoHA_cmu_classified.csv
faithdial_DoHA_topical_classified.csv
faithdial_DoHA_wow_classified.csv
faithdial_GPT2_cmu_classified.csv
faithdial_GPT2_topical_classified.csv
faithdial_GPT2_wow_classified.csv
faithdial_gold_cmu_classified.csv
faithdial_gold_topical_classified.csv
faithdial_gold_wow_classified.csv
文件内容：
原始数据列：knowledge, history, 模型回复列, begin_label, vrm_label
新增列：intent - 意图分类结果