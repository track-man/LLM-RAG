"""
文本分块模块
负责将长文档分割成较小的文本块，优化检索效果
"""
import os
import logging
import re
from typing import List, Dict, Any, Tuple,Optional
import config

# 配置日志
logger = logging.getLogger(__name__)

class TextSplitter:
    """文本分块器类"""
    
    def __init__(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠长度（字符数）
            separators: 文本分割符优先级列表
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # 文本分割符（按优先级排序）
        self.separators = separators or [
            '\n\n',  # 段落分隔
            '\n',    # 行分隔
            '。',    # 句子分隔（中文）
            '.',     # 句子分隔（英文）
            ';',     # 分号
            '，',    # 逗号（中文）
            ',',     # 逗号（英文）
            ' ',     # 空格
            ''       # 字符级分割
        ]
        
        # 验证参数
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"重叠长度 ({self.chunk_overlap}) 不能大于等于分块大小 ({self.chunk_size})")
        
        logger.info(f"文本分块器初始化完成: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分割文档列表
        
        Args:
            documents: 文档列表，每个元素包含text和metadata
            
        Returns:
            分块列表，每个元素包含text和metadata（包含原文档元数据+块索引）
        """
        if not documents:
            logger.warning("输入文档列表为空")
            return []
        
        logger.info(f"开始分割 {len(documents)} 个文档")
        
        chunks = []
        for doc_idx, document in enumerate(documents):
            doc_chunks = self.split_single_document(document, doc_idx)
            chunks.extend(doc_chunks)
        
        logger.info(f"文档分割完成，共生成 {len(chunks)} 个文本块")
        return chunks

    def split_single_document(self, document: Dict[str, Any], doc_index: int) -> List[Dict[str, Any]]:
        """
        分割单个文档
        
        Args:
            document: 文档对象，包含text和metadata
            doc_index: 文档索引
            
        Returns:
            该文档的分块列表
        """
        text = document["text"]
        original_metadata = document["metadata"]
        
        if not text.strip():
            logger.warning(f"文档 {original_metadata.get('filename', 'unknown')} 为空，跳过分割")
            return []
        
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 分割文本
        text_chunks = self._split_text_by_size(processed_text)
        
        # 构建分块对象
        chunks = []
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                **original_metadata,  # 保留原始元数据
                "chunk_index": chunk_idx,
                "chunk_size": len(chunk_text),
                "chunk_char_count": len(chunk_text),
                "chunk_word_count": len(chunk_text.split()),
                "document_index": doc_index
            }
            
            chunk = {
                "text": chunk_text,
                "metadata": chunk_metadata
            }
            chunks.append(chunk)
        
        logger.debug(f"文档 {original_metadata.get('filename', 'unknown')} 分割为 {len(chunks)} 个块")
        return chunks

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\(\)\[\]\{\}\-\"\'\u4e00-\u9fff]', ' ', text)
        
        # 清理多余空格
        text = re.sub(r' +', ' ', text).strip()
        
        return text

    def _split_text_by_size(self, text: str) -> List[str]:
        """
        按大小分割文本
        
        Args:
            text: 待分割的文本
            
        Returns:
            文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在合适的位置断开
            if end < len(text):
                end = self._find_break_point(text, start, end)
            
            # 提取文本块
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 移动起始位置（考虑重叠）
            start = end - self.chunk_overlap
            
            # 避免无限循环
            if start >= len(text):
                break
        
        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """
        在指定范围内寻找合适的断点
        
        Args:
            text: 完整文本
            start: 起始位置
            end: 结束位置
            
        Returns:
            断点位置
        """
        # 优先在句子边界断开
        for separator in self.separators:
            if separator:
                # 在范围内搜索分隔符
                search_start = max(start, end - 100)  # 在结束位置附近搜索
                search_end = end
                
                pos = text.rfind(separator, search_start, search_end)
                if pos > start + self.chunk_size // 2:  # 确保断点不会太靠前
                    return pos + len(separator)
        
        # 如果没有找到合适的分隔符，在空格处断开
        search_start = max(start, end - 50)
        search_end = end
        
        pos = text.rfind(' ', search_start, search_end)
        if pos > start + self.chunk_size // 2:
            return pos
        
        # 最后手段：在指定位置断开
        return end

    def split_by_paragraphs(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按段落分割文档（保持段落完整性）
        
        Args:
            documents: 文档列表
            
        Returns:
            分块列表
        """
        chunks = []
        
        for doc_idx, document in enumerate(documents):
            text = document["text"]
            original_metadata = document["metadata"]
            
            # 按段落分割
            paragraphs = text.split('\n\n')
            
            current_chunk = ""
            chunk_idx = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # 检查是否需要创建新块
                if len(current_chunk) + len(paragraph) + 2 > self.chunk_size and current_chunk:
                    # 保存当前块
                    chunk_metadata = {
                        **original_metadata,
                        "chunk_index": chunk_idx,
                        "chunk_size": len(current_chunk),
                        "chunk_char_count": len(current_chunk),
                        "chunk_word_count": len(current_chunk.split()),
                        "document_index": doc_idx,
                        "split_method": "paragraph"
                    }
                    
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    
                    # 开始新块（考虑重叠）
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + paragraph
                    chunk_idx += 1
                else:
                    # 添加到当前块
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # 保存最后一个块
            if current_chunk:
                chunk_metadata = {
                    **original_metadata,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(current_chunk),
                    "chunk_char_count": len(current_chunk),
                    "chunk_word_count": len(current_chunk.split()),
                    "document_index": doc_idx,
                    "split_method": "paragraph"
                }
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
        
        logger.info(f"按段落分割完成，共生成 {len(chunks)} 个文本块")
        return chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取分块统计信息
        
        Args:
            chunks: 分块列表
            
        Returns:
            统计信息字典
        """
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [len(chunk["text"]) for chunk in chunks]
        chunk_word_counts = [chunk["metadata"]["chunk_word_count"] for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_word_count": sum(chunk_word_counts) / len(chunk_word_counts),
            "min_word_count": min(chunk_word_counts),
            "max_word_count": max(chunk_word_counts),
            "total_characters": sum(chunk_sizes),
            "total_words": sum(chunk_word_counts)
        }
        
        logger.info(f"分块统计: {stats}")
        return stats


# 全局文本分块器实例
_text_splitter = None

def get_text_splitter(
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None
) -> TextSplitter:
    """获取全局文本分块器实例"""
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = TextSplitter(chunk_size, chunk_overlap, separators)
    return _text_splitter

def split_text(
    documents: List[Dict[str, Any]], 
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    分割文本的便捷函数
    
    Args:
        documents: 文档列表
        chunk_size: 分块大小
        chunk_overlap: 分块重叠长度
        
    Returns:
        分块列表
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)

def split_by_paragraphs(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按段落分割文本的便捷函数
    
    Args:
        documents: 文档列表
        
    Returns:
        分块列表
    """
    splitter = get_text_splitter()
    return splitter.split_by_paragraphs(documents)

def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取分块统计信息的便捷函数
    
    Args:
        chunks: 分块列表
        
    Returns:
        统计信息字典
    """
    splitter = get_text_splitter()
    return splitter.get_chunk_statistics(chunks)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        # 创建测试文档
        test_documents = [
            {
                "text": "这是第一段测试文本。包含多个句子。用于测试文本分块功能。\n\n这是第二段文本。也包含多个句子。用于验证分块效果。",
                "metadata": {"source": "test1.txt", "filename": "test1.txt"}
            },
            {
                "text": "第三段测试文本。\n\n第四段测试文本。\n\n第五段测试文本。",
                "metadata": {"source": "test2.txt", "filename": "test2.txt"}
            }
        ]
        
        # 测试文本分割
        print("正在测试文本分割...")
        chunks = split_text(test_documents, chunk_size=50, chunk_overlap=10)
        print(f"成功分割为 {len(chunks)} 个文本块")
        
        # 显示统计信息
        stats = get_chunk_statistics(chunks)
        print(f"分块统计: {stats}")
        
        # 显示前3个分块
        for i, chunk in enumerate(chunks[:3]):
            print(f"块 {i+1}: {chunk['text']} (长度: {len(chunk['text'])})")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")