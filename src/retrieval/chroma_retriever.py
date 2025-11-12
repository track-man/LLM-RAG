"""
基于ChromaDB的检索模块 - 使用bge-base-en-v1.5嵌入模型
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path



logger = logging.getLogger(__name__)

class ChromaRetriever:
    """ChromaDB检索器 - 使用bge-base-en-v1.5嵌入模型"""
    
    def __init__(self, 
                 db_path: str = "data/chroma_db",
                 collection_name: str = "documents",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 reset_collection: bool = False):  # 新增参数  11/12修改增加
        """
        初始化检索器
        
        Args:
            db_path: ChromaDB数据库路径
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            reset_collection: 是否重置集合
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.model_name = embedding_model
        self.reset_collection = reset_collection  # 新增参数  11/12修改增加
        
        # 检索参数
        self.default_top_k = 5
        self.similarity_threshold = 0.7
        
        # 组件
        self.client = None
        self.collection = None
        self.embedder = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化ChromaDB客户端和嵌入模型"""
        try:
            logger.info("初始化ChromaDB检索器...")
            
            # 创建数据库目录
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            logger.info(f"ChromaDB客户端初始化: {self.db_path}")
            

            # 初始化嵌入模型
            logger.info(f"加载嵌入模型: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)

            # 验证模型维度
            test_embedding = self.embedder.encode(["test"])
            logger.info(f"嵌入维度: {len(test_embedding[0])}")
            logger.info("嵌入模型加载成功")

            # 处理集合创建/重置逻辑
            if self.reset_collection:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"已删除现有集合: {self.collection_name}")
                except Exception as e:
                    logger.info(f"删除集合时忽略错误（可能集合不存在）: {e}")

                # 创建新集合    
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={"description": "Document chunks for RAG system"}
                )
                logger.info(f"✅ 创建新集合成功: {self.collection_name}")
            else:
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"使用现有集合: {self.collection_name}")
                except Exception:
                    # 集合不存在，创建新集合
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._get_embedding_function(),
                        metadata={"description": "Document chunks for RAG system"}
                )
                logger.info(f"创建新集合: {self.collection_name}")

                 # 验证集合是否成功设置
            if self.collection is None:
                logger.error("集合初始化失败：collection为None")
                raise RuntimeError("集合初始化失败")
        
            logger.info("✅ 检索器初始化成功")
            
        except Exception as e:
            logger.error(f"检索器初始化失败: {e}")
            self.collection = None
            raise
    
    def _get_embedding_function(self):
    # """获取自定义嵌入函数"""
    # 定义符合ChromaDB新接口要求的嵌入函数类
        class CustomEmbeddingFunction:
            def __init__(self, embedder):
                self.embedder = embedder
            
            def __call__(self, input):
            # """ChromaDB要求的嵌入函数签名"""
                if isinstance(input, str):
                    texts = [input]
                else:
                    texts = input

                if not texts:
                    return []
            
            # 使用bge模型生成嵌入
                embeddings = self.embedder.encode(texts)
                return embeddings.tolist()
    
        return CustomEmbeddingFunction(self.embedder)
    
    def is_ready(self) -> bool:
        """检查检索器是否就绪"""
        if not all([self.client, self.collection, self.embedder]):
            logger.warning(f"组件未就绪: client={self.client is not None}, "
                      f"collection={self.collection is not None}, "
                      f"embedder={self.embedder is not None}")
            return False
        return True
    
    def retrieve_similar_chunks(self, 
                               query: str,
                               top_k: Optional[int] = None,
                               similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        基于语义相似度检索相关文档块
        """
        if not self.is_ready():
            logger.error("检索器未就绪")
            return []
            
        top_k = top_k or self.default_top_k
        threshold = similarity_threshold or self.similarity_threshold
        
        try:
            logger.info(f"执行语义检索: '{query}'")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            processed_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                ids = results['ids'][0] if results['ids'] else [f"result_{i}" for i in range(len(documents))]
                
                for i, (doc, metadata, distance, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                    similarity = self._distance_to_similarity(distance)
                    
                    if similarity >= threshold:
                        result = {
                            'id': doc_id,
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity,
                            'distance': distance,
                            'rank': i + 1
                        }
                        processed_results.append(result)
            
            logger.info(f"语义检索完成: {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            return []
    
    def retrieve_by_metadata(self, 
                           filters: Dict[str, Any],
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """基于元数据过滤检索文档"""
        if not self.is_ready():
            logger.error("检索器未就绪")
            return []
            
        try:
            logger.info(f"执行元数据检索: {filters}")
            
            results = self.collection.query(
                query_texts=[""],  # 空查询，只使用元数据过滤
                n_results=limit or self.default_top_k,
                where=filters,
                include=["metadatas", "documents"]
            )
            
            processed_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                ids = results['ids'][0] if results['ids'] else [f"meta_{i}" for i in range(len(documents))]
                
                for i, (doc, metadata, doc_id) in enumerate(zip(documents, metadatas, ids)):
                    processed_results.append({
                        'id': doc_id,
                        'content': doc,
                        'metadata': metadata,
                        'match_type': 'metadata_filter',
                        'rank': i + 1
                    })
            
            logger.info(f"元数据检索完成: {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"元数据检索失败: {e}")
            return []
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """添加文档到集合"""
        if not self.is_ready():
            logger.error("检索器未就绪")
            return False
            
        try:
            if metadatas is None:
                metadatas = [{} for _ in documents]
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
                
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功添加 {len(documents)} 个文档")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        为文本列表生成嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not self.is_ready():
            logger.error("检索器未就绪")
            return []
            
        try:
            embeddings = self.embedder.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            return []
    
    def _distance_to_similarity(self, distance: float) -> float:
        """将ChromaDB距离转换为相似度分数"""
        similarity = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, similarity))
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.is_ready():
            return {"status": "not_ready"}
            
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.model_name,
                "database_path": str(self.db_path)
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"error": str(e)}


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 测试使用bge-base-en-v1.5的ChromaRetriever...")
    
    try:
        # 创建检索器实例并重置集合  11/12修改增加重置参数
        retriever = ChromaRetriever(
            db_path="test_chroma_db",
        reset_collection=True  # 重置现有集合
)
        # 创建检索器实例
        # 11/12修改删除  retriever = ChromaRetriever(db_path="test_chroma_db")
        
        # 检查是否就绪
        print(f"检索器就绪状态: {retriever.is_ready()}")
        
        # 获取集合信息
        stats = retriever.get_collection_stats()
        print(f"集合信息: {stats}")
        
        # 测试嵌入生成
        test_texts = ["这是一个测试文档", "机器学习是人工智能的重要分支"]
        embeddings = retriever.generate_embeddings(test_texts)
        print(f"嵌入向量维度: {len(embeddings[0]) if embeddings else 0}")
        
        print("✅ ChromaRetriever 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
