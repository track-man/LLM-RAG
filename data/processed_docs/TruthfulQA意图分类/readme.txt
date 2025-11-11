1. truthfulqa_batch_classification_report.json - 详细分类报告
具体字段解析
dataset: 数据集标识名称
total_samples: 总样本数量
intent_distribution: 意图分布（具体数量）
percentage_distribution: 百分比分布

2. truthfulqa_TruthfulQA_classified.csv - 分类后的数据文件
内容：
	原始TruthfulQA数据（8列）：
	Type, Category, Question, Best Answer, Best Incorrect Answer, Correct Answers, Incorrect 	Answers, Source
	新增列：intent - 意图分类结果
	factual_query: 事实查询 (774个，98.0%)
	comparison_query: 比较查询 (7个，0.9%)
	method_query: 方法查询 (8个，1.0%)
	opinion_query: 观点查询 (1个，0.1%)

3. truthfulqa_category_stats.json - TruthfulQA数据集的原始类别统计分析 
文件内容解析
1. 总体统计-"total_categories": 37
含义: TruthfulQA数据集包含 37个不同的原始类别

2. 类别分布统计-"category_distribution"
含义: 每个类别包含的问题数量

3. 类别百分比统计-"category_percentages"
含义: 每个类别在总问题数中的占比


主要类别分组：
认知错误类 (约35%)
类别	                             问题数	   	占比		说明
Misconceptions	       		100		12.66%	常见误解
Confusion: People		23		2.91%	人物混淆
Confusion: Places		15		1.90%	地点混淆
Confusion: Other		 8		1.01%	其他混淆
Indexical Error: Other	18		2.28%	索引错误
Indexical Error: Location	11		1.39%	位置索引错误
Indexical Error: Identity	 8		1.01%	身份索引错误
Mandela Effect			6		0.76%	曼德拉效应

社会法律类 (约20%)
类别			问题数	占比		说明
Law			64		8.10%	法律相关
Sociology		55		6.96%	社会学
Stereotypes	24		3.04%	刻板印象
Politics		10		1.27%	政治
Education		10		1.27%	教育

健康科学类 (约20%)
类别		    问题数		占比		说明
Health		55		6.96%	健康医学
Psychology	19		2.41%	心理学
Nutrition		16		2.03%	营养学
Science		9		1.14%	科学
Weather		17		2.15%	天气

人文艺术类 (约15%)
类别				问题数	占比		说明
Fiction			  30		3.80%	虚构文学
Language			  21		2.66%	语言
Myths and Fairytales	  21		2.66%	神话童话
Proverbs			  18		2.28%	谚语
Misquotations		  16		2.03%	错误引用

超自然信仰类 (约10%)
类别			问题数	占比		说明
Paranormal	  26		3.29%	超自然现象
Conspiracies	  26		3.29%	阴谋论
Superstitions	  22		2.78%	迷信
Religion		  14		1.77%	宗教