# src/embedding/vectorizer_qwen.py
import uuid
import dashscope
from dashscope import TextEmbedding
import numpy as np
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import logging

import os
from dotenv import load_dotenv
import yaml  # 添加yaml导入

load_dotenv()  # 加载环境变量

logger = logging.getLogger(__name__)


class QwenVectorizer:
    """基于通义千问的向量化处理器"""

    def __init__(self, config: Dict = None):
        if config is None:
            with open("config/config.yaml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        self.config = config.get('embedding', {})
        self.api_key = config.get('llm', {}).get('api_key')

        # 解析环境变量格式的API密钥
        if isinstance(self.api_key, str) and self.api_key.startswith('${') and self.api_key.endswith('}'):
            env_var = self.api_key[2:-1]  # 提取变量名
            self.api_key = os.getenv(env_var, self.api_key)

        # 设置dashscope API密钥
        if self.api_key:
            dashscope.api_key = self.api_key
            self.api_available = True
        else:
            logger.warning("API密钥未配置，将使用本地模型")
            self.api_available = False

        # 初始化本地模型

        # 初始化向量数据库
        persist_dir = self.config.get('vector_store', {}).get('persist_directory', './data/vector_store')
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.config.get('vector_store', {}).get('collection_name', 'enterprise_documents'),
            metadata={"hnsw:space": "cosine"}
        )

    def create_embeddings(self, texts: List[str], use_api: bool = True) -> np.ndarray:
        """创建文本向量，支持API和本地两种方式"""
        if use_api and self.api_available and self.api_key:
            try:
                return self._create_embeddings_api(texts)
            except Exception as e:
                logger.warning(f"API向量化失败，使用本地模型: {e}")
                return self._create_embeddings_local(texts)
        else:
            return self._create_embeddings_local(texts)


    def _create_embeddings_api(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = TextEmbedding.call(model=self.config.get('model', 'text-embedding-v1'), input=batch)
            if resp.status_code == 200:
                batch_embeddings = [item['embedding'] for item in resp.output['embeddings']]
                embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"API调用失败: {resp.message}")
        return np.array(embeddings)

    def _create_embeddings_local(self, texts: List[str]) -> np.ndarray:
        return self.fallback_model.encode(texts, normalize_embeddings=True, batch_size=32)

    def store_documents(self, documents: List[Dict], batch_size: int = 50):
        """存储文档到向量数据库（统一入口）"""
        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            ids = []
            contents = []
            metadatas = []

            # 检查ID唯一性
            used_ids = set()

            for j, doc in enumerate(batch):
                # 确保文档ID唯一
                original_doc_id = doc.get("doc_id")
                if original_doc_id:
                    # 如果文档本身有ID，确保其唯一性
                    doc_id = original_doc_id
                    counter = 1
                    while doc_id in used_ids:
                        doc_id = f"{original_doc_id}_{counter}"
                        counter += 1
                else:
                    # 使用UUID生成唯一ID
                    doc_id = f"doc_{uuid.uuid4().hex[:12]}_{i}_{j}"

                # 确保ID不重复
                while doc_id in used_ids:
                    doc_id = f"doc_{uuid.uuid4().hex[:12]}_{i}_{j}_{uuid.uuid4().hex[:4]}"

                used_ids.add(doc_id)
                ids.append(doc_id)
                contents.append(doc["content"][:5000])

                metadata = {
                    "source": doc.get("source", "unknown"),
                    "doc_type": doc.get("doc_type", "general"),
                    "company_code": doc.get("company_code", ""),
                    "timestamp": doc.get("timestamp", ""),
                    "file_name": doc.get("file_name", "")
                }
                # 合并用户自定义元数据
                metadata.update(doc.get("metadata", {}))
                metadatas.append(metadata)

            try:
                embeddings = self.create_embeddings(contents).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"存储批次 {i // batch_size + 1}: {len(batch)} 个文档")
            except Exception as e:
                logger.error(f"存储失败: {e}")
                raise

    # ... [保留原有 search_similar, hybrid_search 等方法] ...

    def search_similar(self, query: str, top_k: int = 10,
                       filters: Dict = None, threshold: float = 0.7) -> List[Dict]:
        """语义搜索相似文档"""
        try:
            # 生成查询向量
            query_embedding = self.create_embeddings([query])

            # 执行搜索
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k * 2,  # 获取更多结果用于筛选
                where=filters,
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            # 格式化结果并应用阈值
            retrieved_docs = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # 转换为相似度

                if similarity >= threshold:
                    retrieved_docs.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                        "distance": distance,
                        "id": results["ids"][0][i]
                    })

            # 按相似度排序
            retrieved_docs.sort(key=lambda x: x["similarity"], reverse=True)

            return retrieved_docs[:top_k]

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def hybrid_search(self, query: str, keyword: str = None,
                      scenario: str = None, top_k: int = 10) -> List[Dict]:
        """混合搜索：语义搜索 + 关键词过滤"""

        # 1. 语义搜索
        semantic_filters = None
        if scenario:
            semantic_filters = self._get_scenario_filters(scenario)

        semantic_results = self.search_similar(
            query,
            top_k=top_k * 2,
            filters=semantic_filters
        )

        # 2. 如果有关键词，进行过滤
        if keyword:
            keyword_filtered = []
            for doc in semantic_results:
                content_lower = doc["content"].lower()
                keyword_lower = keyword.lower()

                # 计算关键词匹配度
                keyword_count = content_lower.count(keyword_lower)
                if keyword_count > 0:
                    # 提升包含关键词的文档权重
                    doc["similarity"] = min(doc["similarity"] * (1 + keyword_count * 0.1), 1.0)
                    keyword_filtered.append(doc)

            # 如果关键词过滤后有结果，使用过滤后的
            if keyword_filtered:
                keyword_filtered.sort(key=lambda x: x["similarity"], reverse=True)
                return keyword_filtered[:top_k]

        return semantic_results[:top_k]

    def _get_scenario_filters(self, scenario: str) -> Dict:
        """根据场景获取过滤器"""
        scenario_mapping = {
            "撤否企业分析": {
                "doc_type": {"$in": ["inquiry_letter", "announcement", "financial", "ipo_process"]}
                # <-- 添加 "ipo_process"
            },
            "长期辅导企业分析": {"doc_type": {"$in": ["guidance", "news", "financial"]}},
            "新三板企业分析": {"doc_type": {"$in": ["ntb", "financial", "news"]}},
            "供应链分析": {"doc_type": {"$in": ["supply_chain", "industry", "regulatory"]}},
            "财务分析": {"doc_type": {"$in": ["financial", "annual_report", "audit"]}},
            "舆情分析": {"doc_type": {"$in": ["news", "social_media", "report"]}}
        }
        return scenario_mapping.get(scenario, {})

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"total_documents": 0, "status": "error"}