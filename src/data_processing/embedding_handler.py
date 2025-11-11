"""
嵌入生成与存储模块
负责生成文本嵌入向量并存储到Chroma向量数据库中
"""
import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import config
from chromadb.api import ClientAPI

# 配置日志
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    """嵌入处理器类"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        初始化嵌入处理器
        
        Args:
            model_name: 嵌入模型名称，默认为config.EMBEDDING_CONFIG['model_name']
            device: 计算设备，'cuda'或'cpu'，自动检测
        """
        self.model_name = model_name or config.EMBEDDING_CONFIG['model_name']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载嵌入模型
        logger.info(f"正在加载嵌入模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"嵌入模型加载完成，使用设备: {self.device}")
        
        # 获取模型维度
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"嵌入向量维度: {self.embedding_dimension}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 待编码的文本列表
            
        Returns:
            嵌入向量列表，每个向量为List[float]
        """
        if not texts:
            logger.warning("输入文本列表为空")
            return []
        
        logger.info(f"开始生成 {len(texts)} 个文本的嵌入向量")
        
        try:
            # 生成嵌入向量
            embeddings = self.model.encode(
                texts,
                batch_size=32,  # 批处理大小，平衡速度和内存
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化向量，便于后续距离计算
            )
            
            # 转换为Python列表格式
            embeddings_list = embeddings.tolist()
            
            logger.info(f"成功生成 {len(embeddings_list)} 个嵌入向量")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"生成嵌入向量时发生错误: {str(e)}")
            raise

    def create_chroma_client(self, chroma_path: str) -> ClientAPI:
        """
        创建Chroma客户端
        
        Args:
            chroma_path: Chroma数据库存储路径
            
        Returns:
            Chroma客户端实例
        """
        # 确保路径存在
        os.makedirs(chroma_path, exist_ok=True)
        
        # 创建Chroma客户端
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,  # 关闭匿名遥测
                allow_reset=True,  # 允许重置数据库
            )
        )
        
        logger.info(f"Chroma客户端创建完成，数据库路径: {chroma_path}")
        return client

    def index_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]], 
        chroma_path: str,
        collection_name: str = "documents"
    ) -> None:
        """
        将文档分块与嵌入向量存储到Chroma向量数据库
        
        Args:
            chunks: 文档分块列表，每个元素包含text和metadata
            embeddings: 嵌入向量列表
            chroma_path: Chroma数据库存储路径
            collection_name: 集合名称
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"分块数量 ({len(chunks)}) 与嵌入向量数量 ({len(embeddings)}) 不匹配")
        
        if not chunks:
            logger.warning("没有文档分块需要索引")
            return
        
        logger.info(f"开始索引 {len(chunks)} 个文档分块到集合: {collection_name}")
        
        try:
            # 创建Chroma客户端
            client = self.create_chroma_client(chroma_path)
            
            # 获取或创建集合
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "RAG系统文档分块集合"}
            )
            
            # 准备数据
            ids = []
            documents = []
            metadatas = []
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
            
            # 批量添加到集合
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_array,
                metadatas=metadatas
            )
            
            logger.info(f"成功索引 {len(chunks)} 个文档分块")
            logger.info(f"集合 '{collection_name}' 当前包含 {collection.count()} 个文档")
            
        except Exception as e:
            logger.error(f"索引文档时发生错误: {str(e)}")
            raise

    def load_existing_collection(
        self, 
        chroma_path: str, 
        collection_name: str = "documents"
    ) -> chromadb.Collection:
        """
        加载已存在的集合
        
        Args:
            chroma_path: Chroma数据库存储路径
            collection_name: 集合名称
            
        Returns:
            Chroma集合实例
        """
        try:
            client = self.create_chroma_client(chroma_path)
            collection = client.get_collection(name=collection_name)
            logger.info(f"成功加载集合 '{collection_name}'，包含 {collection.count()} 个文档")
            return collection
        except Exception as e:
            logger.error(f"加载集合时发生错误: {str(e)}")
            raise

    def get_collection_info(self, chroma_path: str, collection_name: str = "documents") -> Dict[str, Any]:
        """
        获取集合信息
        
        Args:
            chroma_path: Chroma数据库存储路径
            collection_name: 集合名称
            
        Returns:
            集合信息字典
        """
        try:
            collection = self.load_existing_collection(chroma_path, collection_name)
            
            info = {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
            
            logger.info(f"集合信息: {info}")
            return info
            
        except Exception as e:
            logger.error(f"获取集合信息时发生错误: {str(e)}")
            raise


# 全局嵌入处理器实例
_embedding_handler = None

def get_embedding_handler() -> EmbeddingHandler:
    """获取全局嵌入处理器实例（单例模式）"""
    global _embedding_handler
    if _embedding_handler is None:
        _embedding_handler = EmbeddingHandler()
    return _embedding_handler

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    生成文本嵌入向量的便捷函数
    
    Args:
        texts: 待编码的文本列表
        
    Returns:
        嵌入向量列表，每个向量为List[float]
    """
    handler = get_embedding_handler()
    return handler.generate_embeddings(texts)

def index_documents(
    chunks: List[Dict[str, Any]], 
    embeddings: List[List[float]], 
    chroma_path: str
) -> None:
    """
    将文档分块与嵌入向量存储到Chroma向量数据库的便捷函数
    
    Args:
        chunks: 文档分块列表，每个元素包含text和metadata
        embeddings: 嵌入向量列表
        chroma_path: Chroma数据库存储路径
    """
    handler = get_embedding_handler()
    handler.index_documents(chunks, embeddings, chroma_path)

def load_existing_collection(chroma_path: str, collection_name: str = "documents") -> chromadb.Collection:
    """
    加载已存在的集合的便捷函数
    
    Args:
        chroma_path: Chroma数据库存储路径
        collection_name: 集合名称
        
    Returns:
        Chroma集合实例
    """
    handler = get_embedding_handler()
    return handler.load_existing_collection(chroma_path, collection_name)

def get_collection_info(chroma_path: str, collection_name: str = "documents") -> Dict[str, Any]:
    """
    获取集合信息的便捷函数
    
    Args:
        chroma_path: Chroma数据库存储路径
        collection_name: 集合名称
        
    Returns:
        集合信息字典
    """
    handler = get_embedding_handler()
    return handler.get_collection_info(chroma_path, collection_name)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 测试嵌入生成
    test_texts = [
        "这是一个测试文本。",
        "这是另一个测试文本。",
        "RAG系统用于检索增强生成。"
    ]
    
    try:
        print("正在测试嵌入生成...")
        embeddings = generate_embeddings(test_texts)
        print(f"成功生成 {len(embeddings)} 个嵌入向量")
        print(f"第一个向量维度: {len(embeddings[0])}")
        print(f"第一个向量前5个元素: {embeddings[0][:5]}")
        
        # 测试集合信息获取
        print("\n正在测试集合信息获取...")
        info = get_collection_info(config.CHROMA_PATH)
        print(f"集合信息: {info}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")