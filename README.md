# RAG减弱大模型幻觉系统 - README

## 项目介绍
本项目旨在通过**检索增强生成（RAG）+ 多轮验证纠正**的技术方案，降低大模型（DeepseekV3）输出的幻觉率，提升回答的事实准确性。系统采用模块化设计，支持分工开发与灵活扩展，同时具备实验友好的特性，可基于TruthfulQA、FaithDial等数据集进行自动化测试与指标评估。


## 系统架构
```
llm_rag_factuality/
├── main.py                 # 主程序入口（初始化+用户交互）
├── config.py               # 全局配置（模型路径、参数等）
├── requirements.txt        # 依赖库列表
├── .env                    # 敏感信息（API密钥等）
├── src/
│   ├── core/
│   │   └── rag_pipeline.py # 主流程控制代码（串联各模块）
│   ├── data_processing/    # 数据处理与索引模块
│   │   ├── document_loader.py  # 文档加载
│   │   ├── text_splitter.py    # 文本分块
│   │   └── embedding_handler.py # 嵌入生成与存储
│   ├── retrieval/          # 检索模块
│   │   └── chroma_retriever.py # 基于Chroma的检索
│   ├── llm/                # LLM交互模块
│   │   ├── deepseek_client.py  # DeepseekV3调用封装
│   │   └── prompt_templates.py # 各类prompt模板
│   ├── verification/       # 验证模块
│   │   └── fact_checker.py     # 回答与文档一致性验证
│   └── correction/         # 纠正模块
│       └── answer_corrector.py # 基于验证结果修正回答
├── data/
│   ├── raw_docs/           # 原始文档
│   ├── processed_docs/     # 处理后的文档
│   └── chroma_db/          # Chroma向量库持久化数据
├── experiments/            # 实验相关
│   ├── datasets/           # TruthfulQA、FaithDial等数据集
│   ├── run_experiments.py  # 实验自动化脚本
│   └── evaluation_metrics.py # 评估指标计算（幻觉率、准确率等）
└── results/                # 实验结果
    ├── logs/               # 日志文件
    └── figures/            # 可视化图表
```


## 系统流程图
<img width="2946" height="5044" alt="exported_image" src="https://github.com/user-attachments/assets/89771425-d7da-4839-ac90-e7ea4957bf0c" />


## 快速开始

### 环境准备
1. 安装依赖：
```bash
pip install -r requirements.txt
```
2. 配置敏感信息：在 `.env` 文件中填入DeepseekV3的API密钥：
```
DEEPSEEK_API_KEY=your_api_key_here
```


### 系统初始化（构建向量库）
首次运行需加载文档并构建Chroma向量库：
```python
# 执行main.py中的init_system()函数
python main.py --init
```


### 清理缓存
如果对多个实验要清理缓存：
```python
python main.py --clear-cache
```


### 启动交互式查询
运行主程序，输入查询即可体验RAG+幻觉纠正流程：
```bash
python main.py
```
示例查询：`BAAI/bge-base-en-v1.5 嵌入模型的输出向量维度是多少？`


### 实验模式（批量测试）
通过自动化脚本运行数据集测试，评估幻觉率、准确率等指标：
```bash
python experiments/run_experiments.py
```


## 核心模块功能

### 1. 数据处理与索引模块（`src/data_processing/`）
- `document_loader.py`: 加载原始文档，输出含文本和来源元数据的文档列表。
- `text_splitter.py`: 按指定大小（默认500词）和重叠度（默认50词）分割文档。
- `embedding_handler.py`: 生成文本嵌入向量（基于`BAAI/bge-base-en-v1.5`），并将分块与嵌入存入Chroma向量库。


### 2. 检索模块（`src/retrieval/chroma_retriever.py`）
- `retrieve_relevant_chunks()`: 对用户查询进行检索，返回Top-K（默认5）相关文档块，含内容、元数据和与查询的距离。


### 3. LLM交互模块（`src/llm/`）
- `deepseek_client.py`: 封装DeepseekV3的API调用，支持生成式推理。
- `prompt_templates.py`: 提供三类Prompt模板：生成初步回答、验证回答一致性、纠正幻觉内容。


### 4. 验证模块（`src/verification/fact_checker.py`）
- `verify_answer()`: 验证初步回答与检索文档的一致性，返回是否存在幻觉及错误描述列表。


### 5. 纠正模块（`src/correction/answer_corrector.py`）
- `correct_answer()`: 基于验证结果修正回答，输出纠正后的内容。


### 6. 主流程控制（`src/core/rag_pipeline.py`）
- `rag_with_fact_checking()`: 串联“检索→生成→验证→纠正”全流程，返回最终回答及过程信息。


## 技术特点
- **模块解耦**：各模块通过明确的输入输出接口交互，支持分工开发与单独测试。
- **可扩展性**：支持更换嵌入模型、LLM或向量库（只需替换对应模块实现）。
- **实验友好**：内置自动化实验脚本，可批量运行数据集测试并输出可视化结果。
- **幻觉控制**：多轮验证-纠正机制（默认2轮）逐步降低幻觉率，验证环节结合规则与LLM实现精准校验。


## 依赖库
- 向量数据库：`chromadb`
- 嵌入模型：`sentence-transformers`（用于加载`BAAI/bge-base-en-v1.5`）
- LLM调用：`requests`（用于DeepseekV3 API交互）
- 数据处理：`langchain`（可选，用于文档加载与分块）
- 实验与评估：`pandas`、`matplotlib`


## 团队分工
- **项目经理**：管理`main.py`、`config.py`与整体进度，协调模块集成。
- **主程**：负责`src/core/rag_pipeline.py`的流程控制代码，确保模块间数据流转。
- **索引与验证模块开发**（3人）：分别负责`data_processing`、`retrieval`、`verification`模块的实现与优化。
- **LLM交互与纠正模块开发**（1人）：负责`llm`、`correction`模块的Prompt设计与LLM调用封装。
- **实验设计与数据负责**（2人）：处理`experiments/datasets`，开发`run_experiments.py`自动化脚本。
- **数据分析与可视化**（2人）：基于`results/logs`计算评估指标，生成`figures`可视化图表。


## 许可证
本项目采用MIT许可证，详情见`LICENSE`文件。
