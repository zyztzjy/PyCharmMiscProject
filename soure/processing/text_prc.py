# soure/text_processing/text_processor.py

import re
from typing import List, Dict, Optional
import logging
from datetime import datetime
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_raw_text(text: str, chunk_size: int = 512, overlap: int = 50, metadata: Optional[Dict] = None) -> List[
    Dict]:
    """
    处理原始文本，进行清洗、分块，并添加元数据。

    Args:
        text (str): 输入的原始文本。
        chunk_size (int): 每个文本块的最大字符数。默认为 512。
        overlap (int): 相邻文本块之间的重叠字符数。默认为 50。
        metadata (Optional[Dict]): 与文本关联的元数据字典。默认为 None。

    Returns:
        List[Dict]: 包含处理后的文本块及其元数据的列表。
                    每个元素是一个字典，格式为:
                    {
                        "content": "文本块内容",
                        "metadata": { ... },
                        "id": "文本块唯一ID (可选)"
                    }
    """
    if not text or not isinstance(text, str):
        logger.warning("输入的文本为空或非字符串类型，返回空列表。")
        return []

    # 1. 清洗文本
    cleaned_text = _clean_text(text)

    # 2. 分块
    chunks = _chunk_text(cleaned_text, chunk_size, overlap)

    # 3. 构建结果列表
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "processed_at": datetime.now().isoformat(),
            "source_type": "raw_text"
        })

        chunk_id = hashlib.md5((chunk + str(chunk_metadata)).encode('utf-8')).hexdigest()

        processed_chunk = {
            "content": chunk,
            "metadata": chunk_metadata,
            "id": chunk_id  # 添加一个基于内容和元数据的ID
        }
        processed_chunks.append(processed_chunk)

    logger.info(f"原始文本处理完成，共生成 {len(processed_chunks)} 个文本块。")
    return processed_chunks


def _clean_text(text: str) -> str:
    """
    清洗文本，去除多余空白、特殊字符等。

    Args:
        text (str): 原始文本。

    Returns:
        str: 清洗后的文本。
    """
    # 去除首尾空白
    text = text.strip()
    # 将多个换行符替换为单个换行符
    text = re.sub(r'\n+', '\n', text)
    # 将多个空格替换为单个空格
    text = re.sub(r' +', ' ', text)
    # 去除一些常见的非打印字符或特殊符号（可根据需要调整）
    # text = re.sub(r'[^\w\s\n\u4e00-\u9fff.,!?;:()\-\'\"]', ' ', text) # 示例：保留中文、英文、数字、基本标点
    return text


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    将文本分割成指定大小的块。

    Args:
        text (str): 待分割的文本。
        chunk_size (int): 每个块的最大字符数。
        overlap (int): 块之间的重叠字符数。

    Returns:
        List[str]: 分割后的文本块列表。
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句子边界处分割，避免在单词中间切断
        # 这里简单地按字符数分割，如果需要更智能的分割，可以使用 nltk 或 spaCy
        if end < len(text):  # 如果不是最后一块
            # 查找最后一个合适的分割点（例如，句号、问号、感叹号后）
            last_punct_index = -1
            for punct in '.!?。！？':
                idx = chunk.rfind(punct)
                if idx > last_punct_index:
                    last_punct_index = idx
            if last_punct_index != -1 and last_punct_index > chunk_size // 2:  # 避免切得太短
                end = start + last_punct_index + 1
                chunk = text[start:end]

        chunks.append(chunk)

        # 计算下一个块的起始位置，考虑重叠
        start = end - overlap
        if start >= len(text):
            break
        # 确保不会无限循环（例如，当overlap >= chunk_size 且 text 长度大于 chunk_size 时）
        if start <= 0:
            start = end  # 强制向前移动，避免无限循环，此时重叠无效

    # 最后一个块可能长度小于 chunk_size，无需特殊处理，因为它已经是剩余部分
    # 但如果最后一个块与前一个块有重叠，且内容完全一样，则移除前一个块（或后一个）
    # 这里简单处理，如果最后两个块内容相同且重叠，则移除一个
    if len(chunks) > 1 and chunks[-1] == chunks[-2]:
        chunks = chunks[:-1]

    # 再次确保没有块长度为0
    chunks = [c for c in chunks if c]

    return chunks


# --- 可选：其他文本处理辅助函数 ---

def extract_entities(text: str) -> List[str]:
    """
    (示例) 提取文本中的实体（如公司名、人名等）。
    此函数需要 NER 模型支持，例如使用 jieba、spaCy 或 transformers。
    这里仅作示意，返回一个空列表。
    """
    # import jieba
    # import jieba.posseg as pseg
    # entities = []
    # for word, flag in pseg.cut(text):
    #     if flag in ['nr', 'ns', 'nt', 'nz']: # jieba 的命名实体标签示例
    #         entities.append(word)
    # return entities
    logger.warning("实体提取功能未实现，需要集成 NER 模型。")
    return []


def detect_language(text: str) -> str:
    """
    (示例) 检测文本语言。
    此函数需要 langdetect 或其他语言检测库。
    这里仅作示意，返回 'unknown'。
    """
    # from langdetect import detect
    # try:
    #     return detect(text)
    # except:
    #     return 'unknown'
    logger.warning("语言检测功能未实现，需要集成相关库。")
    return 'unknown'


# --- 主函数，用于测试 ---
if __name__ == "__main__":
    sample_text = """
    这是一段示例文本。它包含多个句子。
    这是第二个句子。我们希望通过 process_raw_text 函数来处理它。
    这段文本可能会很长，需要被分块。
    并且，它可能包含一些  换行符  和    空格。
    """
    metadata_example = {"source": "example_doc", "author": "test_user"}

    processed_result = process_raw_text(sample_text, chunk_size=100, overlap=20, metadata=metadata_example)
    print(f"处理结果包含 {len(processed_result)} 个块:")
    for i, chunk_info in enumerate(processed_result):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"ID: {chunk_info.get('id')}")
        print(f"Content: {repr(chunk_info['content'])}")
        print(f"Metadata: {chunk_info['metadata']}")