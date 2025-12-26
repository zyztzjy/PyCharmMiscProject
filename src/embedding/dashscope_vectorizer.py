# src/embedding/dashscope_vectorizer.py
import dashscope
from dashscope import TextEmbedding
import numpy as np
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DashScopeVectorizer:
    """使用阿里云DashScope文本向量化服务"""

    def __init__(self, api_key: str, config: Dict = None):
        self.api_key = api_key
        dashscope.api_key = api_key

        # 配置参数
        self.config = config or {}
        self.model = self.config.get('embedding', {}).get('model', 'text-embedding-v1')
        self.batch_size = self.config.get('embedding', {}).get('batch_size', 25)

        # 初始化向量数据库
        persist_dir = self.config.get('vector_store', {}).get('persist_directory', './data/vector_store')
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=self.config.get('vector_store', {}).get('collection_name', 'enterprise_docs'),
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"初始化DashScope向量化器，使用模型: {self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """创建文本向量"""
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                resp = TextEmbedding.call(
                    model=self.model,
                    input=batch
                )

                if resp.status_code == 200:
                    batch_embeddings = [item['embedding'] for item in resp.output['embeddings']]
                    embeddings.extend(batch_embeddings)
                else:
                    logger.error(f"向量API调用失败: {resp.code} - {resp.message}")
                    raise Exception(f"API调用失败: {resp.message}")

            except Exception as e:
                logger.error(f"向量化异常: {e}")
                # 重试机制已通过@retry装饰器处理
                raise

        return np.array(embeddings)

    def store_documents(self, documents: List[Dict], batch_size: int = 20):
        """存储文档到向量数据库"""
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]

            ids = []
            texts = []
            metadatas = []

            for doc in batch:
                doc_id = doc.get("doc_id", f"doc_{i}_{len(ids)}")
                ids.append(doc_id)
                texts.append(doc["content"][:3000])  # 限制长度

                metadata = doc.get("metadata", {}).copy()
                metadata.update({
                    "source": doc.get("source", "unknown"),
                    "doc_type": doc.get("doc_type", "general"),
                    "timestamp": doc.get("timestamp", ""),
                    "length": len(doc["content"])
                })
                metadatas.append(metadata)

            try:
                # 生成向量
                embeddings = self.create_embeddings(texts)

                # 存储到向量数据库
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

                logger.info(f"存储批次 {i // batch_size + 1}/{(total - 1) // batch_size + 1}: {len(batch)} 个文档")

            except Exception as e:
                logger.error(f"存储失败: {e}")

    def search_similar(self, query: str, top_k: int = 10,
                       filters: Dict = None) -> List[Dict]:
        """语义搜索相似文档"""
        try:
            # 生成查询向量
            query_embedding = self.create_embeddings([query])[0]

            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            # 格式化结果
            retrieved_docs = []
            for i in range(len(results["ids"][0])):
                retrieved_docs.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # 转换为相似度
                    "distance": results["distances"][0][i],
                    "id": results["ids"][0][i]
                })

            return retrieved_docs

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []