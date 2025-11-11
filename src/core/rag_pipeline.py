"""
主流程控制模块：串联检索、生成、验证、纠正全链路
负责协调各模块数据流转，严格遵循输入输出规范
"""
from typing import List, Dict, Tuple, Optional
import logging
from src.retrieval.chroma_retriever import ChromaRetriever  
from src.llm.llm.prompt_templates import (
    get_initial_prompt,
    get_verification_prompt,
    get_correction_prompt
)
from src.llm.llm.deepseek_client import llm_inference
from src.verification.fact_checker import verify_answer
from src.correction.answer_corrector import correct_answer
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_pipeline")


def rag_with_fact_checking(
    query: str,
    chroma_path: str,
    max_correction_rounds: int = 2
) -> Dict[str, any]:
    """
    完整RAG+幻觉纠正流程控制器
    输入：
        query: 用户查询字符串（例如："BAAI/bge模型的向量维度是多少？"）
        chroma_path: Chroma向量库本地存储路径（例如："./data/chroma_db/"）
        max_correction_rounds: 最大纠正轮次（默认2轮）
    输出：
        包含全流程信息的字典，结构如下：
        {
            "final_answer": 最终回答字符串,
            "initial_answer": 初步回答字符串,
            "retrieved_chunks": 检索到的文档块列表（同retrieve_relevant_chunks输出）,
            "correction_history": 纠正过程记录列表,
            "has_幻觉": 布尔值（最终是否仍存在幻觉）,
            "process_log": 流程日志列表
        }
    """
    # 初始化返回结果结构
    result: Dict[str, any] = {
        "final_answer": "",
        "initial_answer": "",
        "retrieved_chunks": [],
        "correction_history": [],
        "has_幻觉": False,
        "process_log": []
    }
    process_log: List[str] = []

    try:
        # ====================== 步骤1：检索相关文档块 ======================
        process_log.append("开始检索相关文档块...")
        # 调用检索模块：src/retrieval/chroma_retriever.py
        # 输入规范：
        #   query: 用户查询字符串
        #   top_k: 检索返回数量（从config读取，默认5）
        #   chroma_path: Chroma向量库路径
        # 输出规范：
        #   列表，每个元素为{"text": 块内容, "metadata": 元数据, "distance": 距离}
        # 实例化Chroma检索器
        retriever = ChromaRetriever(
            db_path=chroma_path,  # 指定向量库路径
            collection_name="documents"  # 集合名称（与初始化时一致）
        )
        # 调用检索方法
        retrieved_chunks: List[Dict[str, any]] = retriever.retrieve_similar_chunks(
            query=query,
            top_k=config.TOP_K
        )
        result["retrieved_chunks"] = retrieved_chunks
        process_log.append(f"检索完成，返回{len(retrieved_chunks)}个相关文档块")

        # 检索失败处理
        if not retrieved_chunks:
            process_log.append("未检索到任何相关文档，无法生成回答")
            result["final_answer"] = "抱歉，未找到相关信息，无法回答您的问题。"
            result["process_log"] = process_log
            return result

        # ====================== 步骤2：生成初步回答 ======================
        process_log.append("开始生成初步回答...")
        # 调用Prompt模板模块：src/llm/prompt_templates.py
        # 输入规范：
        #   query: 用户查询
        #   chunks: 检索到的文档块列表（同retrieved_chunks）
        # 输出规范：
        #   格式化的生成用Prompt字符串
        initial_prompt: str = get_initial_prompt(
            query=query,
            chunks=retrieved_chunks
        )
        process_log.append("初步回答Prompt生成完成")

        # 调用LLM交互模块：src/llm/deepseek_client.py
        # 输入规范：
        #   prompt: 生成用Prompt字符串
        #   temperature: 温度参数（默认0.1，降低随机性）
        # 输出规范：
        #   LLM生成的初步回答字符串
        initial_answer: str = llm_inference(
            prompt=initial_prompt,
            temperature=0.1
        )
        result["initial_answer"] = initial_answer
        current_answer: str = initial_answer  # 当前回答（后续可能被纠正）
        process_log.append("初步回答生成完成")

        # ====================== 步骤3：多轮验证与纠正 ======================
        process_log.append(f"开始多轮验证与纠正（最大{max_correction_rounds}轮）...")
        correction_round: int = 0  # 当前纠正轮次
        has_幻觉: bool = False     # 是否存在幻觉
        correction_history: List[Dict[str, any]] = []  # 纠正历史记录

        while correction_round < max_correction_rounds:
            # -------------------- 子步骤3.1：验证当前回答 --------------------
            process_log.append(f"第{correction_round+1}轮验证开始...")
            # 调用验证模块：src/verification/fact_checker.py
            # 输入规范：
            #   answer: 当前待验证的回答字符串
            #   chunks: 检索到的文档块列表
            # 输出规范：
            #   元组(是否有幻觉: bool, 错误描述列表: List[str])
            verification_result: Tuple[bool, List[str]] = verify_answer(
                answer=current_answer,
                chunks=retrieved_chunks
            )
            has_幻觉, errors = verification_result
            process_log.append(f"第{correction_round+1}轮验证完成，是否有幻觉：{has_幻觉}")

            if not has_幻觉:
                process_log.append("回答无幻觉，终止纠正流程")
                break  # 无幻觉则退出循环

            # 记录本轮错误信息
            correction_history.append({
                "round": correction_round + 1,
                "errors": errors,
                "answer_before_correction": current_answer
            })

            # -------------------- 子步骤3.2：纠正当前回答 --------------------
            process_log.append(f"第{correction_round+1}轮纠正开始...")
            # 调用Prompt模板模块：生成纠正用Prompt
            # 输入规范：
            #   answer: 当前回答字符串
            #   chunks: 检索到的文档块列表
            #   errors: 错误描述列表
            # 输出规范：
            #   格式化的纠正用Prompt字符串
            correction_prompt: str = get_correction_prompt(
                answer=current_answer,
                chunks=retrieved_chunks,
                errors=errors
            )

            # 调用纠正模块：src/correction/answer_corrector.py
            # （内部已封装LLM调用，此处直接使用纠正模块接口）
            # 输入规范：
            #   answer: 当前回答字符串
            #   chunks: 检索到的文档块列表
            #   errors: 错误描述列表
            # 输出规范：
            #   纠正后的回答字符串
            corrected_answer: str = correct_answer(
                answer=current_answer,
                chunks=retrieved_chunks,
                errors=errors
            )

            current_answer = corrected_answer
            process_log.append(f"第{correction_round+1}轮纠正完成")
            correction_round += 1

        # 若达到最大轮次仍有幻觉，记录状态
        if correction_round >= max_correction_rounds and has_幻觉:
            process_log.append(f"已达最大纠正轮次（{max_correction_rounds}轮），仍存在幻觉")

        # ====================== 步骤4：整理最终结果 ======================
        result["final_answer"] = current_answer
        result["correction_history"] = correction_history
        result["has_幻觉"] = has_幻觉
        result["process_log"] = process_log
        logger.info("RAG流程执行完成")

    except Exception as e:
        # 异常处理
        error_msg = f"流程执行出错：{str(e)}"
        process_log.append(error_msg)
        result["final_answer"] = f"系统出错：{error_msg}"
        result["process_log"] = process_log
        logger.error(error_msg, exc_info=True)

    return result