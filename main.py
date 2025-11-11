"""
主程序入口：负责系统初始化、交互式查询与实验模式调度
严格遵循模块输入输出规范，协调数据预处理与在线交互流程
"""
import os
import argparse
import logging
from dotenv import load_dotenv
from src.core.rag_pipeline import rag_with_fact_checking
from src.data_processing.document_loader import load_documents
from src.data_processing.text_splitter import split_text
from src.data_processing.embedding_handler import generate_embeddings, index_documents
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def init_system() -> None:
    """
    系统初始化流程：加载文档→文本分块→生成嵌入→构建Chroma向量库
    输入：无（依赖config.py配置的路径）
    输出：无（向量库持久化到本地路径）
    """
    logger.info("开始系统初始化...")

    # ====================== 步骤1：加载原始文档 ======================
    logger.info(f"加载原始文档（路径：{config.RAW_DOCS_PATH}）...")
    # 调用数据处理模块：src/data_processing/document_loader.py
    # 输入规范：
    #   doc_dir: 原始文档目录路径（从config读取）
    # 输出规范：
    #   文档列表，每个元素为{"text": 文本内容, "metadata": {"source": 来源路径}}
    docs = load_documents(config.RAW_DOCS_PATH)
    if not docs:
        raise ValueError(f"未在{config.RAW_DOCS_PATH}找到任何文档，请检查路径或添加文档")
    logger.info(f"原始文档加载完成，共{len(docs)}个文档")

    # ====================== 步骤2：文本分块 ======================
    logger.info(f"文本分块（大小：{config.TEXT_SPLITTER_CONFIG['chunk_size']}，重叠：{config.TEXT_SPLITTER_CONFIG['chunk_overlap']}）...")
    # 调用数据处理模块：src/data_processing/text_splitter.py
    # 输入规范：
    #   documents: 原始文档列表（同load_documents输出）
    #   chunk_size: 分块大小（从config读取）
    #   chunk_overlap: 分块重叠长度（从config读取）
    # 输出规范：
    #   分块列表，每个元素为{"text": 块内容, "metadata": 原文档元数据+块索引}
    chunks = split_text(
        documents=docs,
        chunk_size=config.TEXT_SPLITTER_CONFIG['chunk_size'],
        chunk_overlap=config.TEXT_SPLITTER_CONFIG['chunk_overlap']
    )
    logger.info(f"文本分块完成，共生成{len(chunks)}个文本块")

    # ====================== 步骤3：生成嵌入向量 ======================
    logger.info(f"生成嵌入向量（模型：{config.EMBEDDING_CONFIG['model_name']}）...")
    # 提取分块文本
    chunk_texts = [chunk["text"] for chunk in chunks]
    # 调用数据处理模块：src/data_processing/embedding_handler.py
    # 输入规范：
    #   texts: 分块文本列表
    # 输出规范：
    #   嵌入向量列表，每个向量为List[float]
    embeddings = generate_embeddings(chunk_texts)
    logger.info("嵌入向量生成完成")

    # ====================== 步骤4：构建Chroma向量库 ======================
    logger.info(f"将分块与嵌入存入向量库（路径：{config.CHROMA_DB_PATH}）...")
    # 调用数据处理模块：src/data_processing/embedding_handler.py
    # 输入规范：
    #   chunks: 分块列表（同split_text输出）
    #   embeddings: 嵌入向量列表（同generate_embeddings输出）
    #   CHROMA_DB_PATH: Chroma向量库存储路径（从config读取）
    # 输出规范：无（向量库持久化到本地）
    index_documents(chunks, embeddings, config.CHROMA_DB_PATH)
    logger.info("系统初始化完成，向量库构建成功！")


def interactive_query() -> None:
    """
    交互式查询模式：接收用户输入→调用RAG流程→输出结果
    输入：用户从命令行输入的查询字符串
    输出：命令行打印的最终回答与流程信息
    """
    print("RAG减弱大模型幻觉系统已启动（输入'quit'退出）")
    while True:
        query = input("\n请输入查询：").strip()
        if query.lower() == "quit":
            print("系统退出")
            break
        if not query:
            continue

        logger.info(f"开始处理查询：{query}")
        # 调用主流程模块：src/core/rag_pipeline.py
        # 输入规范：
        #   query: 用户查询字符串
        #   CHROMA_DB_PATH: Chroma向量库路径（从config读取）
        #   max_correction_rounds: 最大纠正轮次（从config读取）
        # 输出规范：
        #   包含全流程信息的字典（同rag_with_fact_checking输出）
        result = rag_with_fact_checking(
            query=query,
            chroma_path=config.CHROMA_DB_PATH,
            max_correction_rounds=config.MAX_CORRECTION_ROUNDS
        )

        # 打印最终结果
        print("\n===== 最终回答 =====")
        print(result["final_answer"])
        
        if result["has_幻觉"]:
            print("\n⚠️ 警告：回答可能仍存在未验证的信息")
        
        # 可选：打印流程日志（用于调试）
        # print("\n===== 流程日志 =====")
        # for log in result["process_log"]:
        #     print(log)


def main() -> None:
    # 加载环境变量（如API密钥）
    load_dotenv()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RAG减弱大模型幻觉系统")
    parser.add_argument(
        "--init", 
        action="store_true", 
        help="初始化系统（构建向量库，首次运行需执行）"
    )
    parser.add_argument(
        "--experiment", 
        action="store_true", 
        help="实验模式（批量运行数据集测试，需额外实现run_experiments.py）"
    )
    args = parser.parse_args()

    # 执行初始化
    if args.init:
        init_system()
    # 执行实验模式（需补充experiments/run_experiments.py逻辑）
    elif args.experiment:
        logger.info("启动实验模式，开始批量数据集测试...")
        # 此处需调用experiments/run_experiments.py的自动化逻辑
        # 示例：
        # from experiments.run_experiments import run_batch_tests
        # run_batch_tests()
    # 默认启动交互式查询
    else:
        interactive_query()


if __name__ == "__main__":
    main()