# soure/embedding/vectorizer_qwen.py (完整修复版)
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
import re
from datetime import datetime


class QwenVectorizer:
    """通义千问向量化器（完整修复版）"""

    def __init__(self, config: Dict):
        self.config = config
        self.embedding_config = config.get('embedding', {})

        # 初始化ChromaDB
        chroma_config = self.embedding_config.get('chroma', {})
        self.persist_directory = chroma_config.get('persist_directory', './chroma_db')
        self.collection_name = chroma_config.get('collection_name', 'enterprise_docs')

        # 创建ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"加载已有集合: {self.collection_name}")

            # 获取集合统计
            stats = self.get_collection_stats()
            print(f"集合中的文档数量: {stats.get('total_documents', 0)}")

        except Exception as e:
            print(f"集合不存在，创建新集合: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # 添加优化参数
        self.search_optimization_config = {
            "company_name_boost": 2.0,
            "scenario_relevance_boost": 1.5,
            "hybrid_search_ratio": 0.7,
            "min_similarity_threshold": 0.3,
            "rerank_top_k": 10,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "status": "可用",
                "collection_name": self.collection_name
            }
        except Exception as e:
            print(f"获取集合统计失败: {e}")
            return {
                "total_documents": 0,
                "status": "不可用",
                "collection_name": self.collection_name
            }

    def clear_collection(self) -> bool:
        """清空集合中的所有文档"""
        try:
            # 删除整个集合
            self.client.delete_collection(self.collection_name)
            
            # 重新创建集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"集合 {self.collection_name} 已清空并重新创建")
            return True
        except Exception as e:
            print(f"清空集合失败: {e}")
            return False

    def store_documents(self, documents: List[Dict]) -> int:
        """存储文档到向量数据库"""
        if not documents:
            return 0

        try:
            ids = []
            contents = []
            metadatas = []

            for i, doc in enumerate(documents):
                if isinstance(doc, dict) and 'content' in doc:
                    # 生成唯一ID
                    doc_id = f"doc_{int(datetime.now().timestamp())}_{i}_{hashlib.md5(doc['content'].encode()).hexdigest()[:8]}"
                    ids.append(doc_id)

                    # 获取内容
                    content = doc.get('content', '')
                    contents.append(content)

                    # 获取元数据
                    metadata = doc.get('metadata', {})
                    if not metadata:
                        metadata = {
                            "source": doc.get('source', 'unknown'),
                            "document_type": doc.get('document_type', 'unknown'),
                            "company": doc.get('company', 'unknown'),
                            "upload_time": datetime.now().isoformat()
                        }
                    metadatas.append(metadata)

            # 批量添加文档到向量数据库
            if ids and contents:
                # 首先检查集合是否存在，如果不存在则创建
                try:
                    self.collection = self.client.get_collection(self.collection_name)
                except:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )

                # 实际存储文档到向量数据库
                self.collection.add(
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )

                print(f"成功存储 {len(ids)} 个文档片段到向量数据库")
                return len(ids)
            else:
                print("没有有效的文档内容")
                return 0

        except Exception as e:
            print(f"存储文档失败: {e}")
            import traceback
            traceback.print_exc()
            return 0


    def search_by_company_name(self,
                               company_name: str,
                               query: str,
                               top_k: int = 15,
                               similarity_threshold: float = 0.3) -> List[Dict]:
        """按企业名称搜索"""
        try:
            # 1. 企业名称精确匹配
            exact_matches = self._search_exact_company_name(company_name, top_k)

            # 2. 企业名称模糊匹配
            fuzzy_matches = self._search_fuzzy_company_name(company_name, top_k)

            # 3. 查询语义搜索
            semantic_matches = self._semantic_search(query, top_k)

            # 4. 合并和去重
            all_results = exact_matches + fuzzy_matches + semantic_matches
            merged_results = self._merge_and_dedup(all_results)

            # 5. 企业相关度增强
            enhanced_results = []
            for result in merged_results[:top_k * 2]:
                # 计算企业相关度分数
                company_similarity = self._calculate_company_similarity(
                    result, company_name
                )

                if company_similarity >= similarity_threshold:
                    # 增强相似度分数
                    enhanced_similarity = result.get("similarity", 0)
                    enhanced_similarity = enhanced_similarity * 0.7 + company_similarity * 0.3

                    result["similarity"] = enhanced_similarity
                    result["company_relevance"] = company_similarity
                    result["matched_company"] = company_name

                    enhanced_results.append(result)

            # 6. 按增强后的相似度排序
            enhanced_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

            return enhanced_results[:top_k]

        except Exception as e:
            print(f"企业名称搜索失败: {e}")
            return self._semantic_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
        """语义搜索（基础方法）"""
        try:
            # 调用ChromaDB进行搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            documents = []
            if results and results.get("documents"):
                for i in range(len(results["documents"][0])):
                    doc_content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 1.0

                    # 将距离转换为相似度（ChromaDB使用余弦距离）
                    similarity = 1.0 - distance

                    documents.append({
                        "content": doc_content,
                        "metadata": metadata,
                        "similarity": similarity,
                        "search_type": "semantic"
                    })

            return documents

        except Exception as e:
            print(f"语义搜索失败: {e}")
            return []

    def _search_exact_company_name(self, company_name: str, top_k: int) -> List[Dict]:
        """精确企业名称搜索"""
        try:
            # 构建精确匹配的查询
            exact_queries = [
                company_name,
                f"{company_name}公司",
                f"{company_name}股份",
                company_name.replace("公司", "").replace("股份", "")
            ]

            all_results = []
            for q in exact_queries:
                if len(q) >= 2:
                    # 尝试在元数据中精确匹配
                    try:
                        results = self.collection.query(
                            query_texts=[q],
                            n_results=top_k,
                            where={"company": {"$eq": company_name}}
                        )

                        if results and results.get("documents"):
                            for i, doc in enumerate(results["documents"][0]):
                                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                                all_results.append({
                                    "content": doc,
                                    "metadata": metadata,
                                    "similarity": 0.9,
                                    "search_type": "exact_company"
                                })
                    except Exception as e:
                        print(f"精确查询失败 {q}: {e}")
                        continue

            return all_results

        except Exception as e:
            print(f"精确企业名称搜索失败: {e}")
            return []

    def _search_fuzzy_company_name(self, company_name: str, top_k: int) -> List[Dict]:
        """模糊企业名称搜索"""
        try:
            # 提取企业名称的关键部分
            name_parts = self._extract_company_name_parts(company_name)

            all_results = []
            for part in name_parts:
                if len(part) >= 2:
                    try:
                        # 搜索内容中包含企业名称部分
                        results = self.collection.query(
                            query_texts=[part],
                            n_results=top_k
                        )

                        if results and results.get("documents"):
                            for i, doc in enumerate(results["documents"][0]):
                                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                                # 检查元数据或内容中是否包含企业名称
                                content = doc.lower()
                                metadata_company = str(metadata.get("company", "")).lower()

                                if part.lower() in content or part.lower() in metadata_company:
                                    # 计算模糊匹配分数
                                    fuzzy_score = self._calculate_fuzzy_match(part, metadata.get("company", ""))

                                    all_results.append({
                                        "content": doc,
                                        "metadata": metadata,
                                        "similarity": 0.7 * fuzzy_score,
                                        "search_type": "fuzzy_company"
                                    })
                    except Exception as e:
                        print(f"模糊查询失败 {part}: {e}")
                        continue

            return all_results

        except Exception as e:
            print(f"模糊企业名称搜索失败: {e}")
            return []

    def _extract_company_name_parts(self, company_name: str) -> List[str]:
        """提取企业名称的关键部分"""
        parts = []

        # 移除常见后缀
        name_without_suffix = company_name
        suffixes = ["公司", "股份", "集团", "有限", "科技", "电子", "有限公司", "股份有限公司"]
        for suffix in suffixes:
            if name_without_suffix.endswith(suffix):
                name_without_suffix = name_without_suffix[:-len(suffix)]
                break

        # 按字符分割（中文字符）
        if re.search(r'[\u4e00-\u9fa5]', name_without_suffix):
            # 中文企业名，取2-4字的关键部分
            for i in range(len(name_without_suffix)):
                for j in range(i + 2, min(i + 5, len(name_without_suffix) + 1)):
                    part = name_without_suffix[i:j]
                    if len(part) >= 2:
                        parts.append(part)
        else:
            # 英文企业名，按空格分割
            words = name_without_suffix.split()
            parts.extend([w for w in words if len(w) >= 3])

        # 添加完整名称（无后缀）
        if name_without_suffix and len(name_without_suffix) >= 2:
            parts.append(name_without_suffix)

        # 去重
        return list(set(parts))

    def search_similar_documents(
            self,
            query: str,
            top_k: int = 10,
            filters: Optional[Dict] = None,
            company_name: Optional[str] = None,
            scenario: Optional[str] = None
    ) -> List[Dict]:
        """
        搜索相似文档
        """
        try:
            # 预处理查询
            enhanced_query = self._preprocess_search_query(query, scenario)
            
            # 执行语义搜索
            results = self._semantic_search(enhanced_query, top_k, filters)
            
            # 如果按指定过滤条件没有找到结果，尝试放宽过滤条件
            if not results and filters:
                # 尝试不使用过滤条件搜索
                results = self._semantic_search(enhanced_query, top_k, None)
                
                # 如果有场景特定需求，再应用场景过滤
                if scenario and results:
                    results = self._filter_by_scenario_relevance(results, scenario)

            # 如果仍然没有结果且提供了企业名称，尝试按企业名称搜索
            if not results and company_name:
                # 如果没找到结果，尝试搜索企业名称
                results = self._semantic_search(company_name, top_k, filters)

            # 去重和重排序
            results = self._rerank_and_dedup(results, query, top_k)

            return results

        except Exception as e:
            print(f"搜索文档失败: {e}")
            import traceback
            traceback.print_exc()
            return []


    def _deduplicate_and_rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """去重并重新排序结果"""
        if not results:
            return []

        # 去重：基于内容的哈希值
        seen_content = set()
        unique_results = []
        
        for result in results:
            content = result.get('content', '')
            if content:
                content_hash = hash(content[:200])  # 只使用前200个字符进行哈希
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)

        # 重新排序：根据相似度排序
        unique_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return unique_results

    # 在 vectorizer_qwen.py 的 QwenVectorizer 类中添加以下方法

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """从文本中提取关键词"""
        if not text:
            return []

        # 移除标点符号和特殊字符
        import re
        clean_text = re.sub(r'[^\w\u4e00-\u9fa5\s]', ' ', text)

        # 中文分词（简化版）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,}', clean_text)

        # 英文单词
        english_words = re.findall(r'[a-zA-Z]{3,}', clean_text)

        # 合并所有词
        all_words = chinese_words + english_words

        # 计算词频
        word_freq = {}
        for word in all_words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

        # 排序并返回前N个关键词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:max_keywords]]

        return keywords

    def _preprocess_search_query(self, query: str, scenario: Optional[str] = None) -> str:
        """预处理搜索查询，增强与文档的匹配度"""
        query_lower = query.lower()
        enhanced_query = query

        # 根据常见搜索意图增强查询词
        search_intent_patterns = {
            "撤否": ["撤否", "终止", "撤回", "ipo失败", "审核终止", "上市失败"],
            "高风险": ["风险", "问题", "违规", "处罚", "诉讼", "st"],
            "辅导": ["辅导", "备案", "辅导期", "上市辅导"],
            "关联": ["关联", "上下游", "供应商", "客户", "关联交易"]
        }

        # 根据场景增强查询
        if scenario:
            scenario_keywords = {
                "撤否企业分析": ["撤否", "终止审核", "撤回原因", "审核障碍"],
                "长期辅导企业分析": ["辅导备案", "辅导进展", "上市障碍", "整改"],
                "上下游企业分析": ["关联交易", "供应商", "客户", "业务往来"]
            }
            if scenario in scenario_keywords:
                enhanced_query += " " + " ".join(scenario_keywords[scenario])

        # 根据查询内容动态增强
        for intent, keywords in search_intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                enhanced_query += " " + " ".join(keywords)
                break

        return enhanced_query

    def _extract_company_names_from_query(self, query: str) -> List[str]:
        """从查询中提取企业名称"""
        import re

        # 常见企业名称模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,20}?(股份|科技|电子|集团|有限公司|公司|证券))有限公司',
            r'分析(.*?)(的|在|相关)',
            r'评估(.*?)(的|企业|公司)',
            r'(.*?)（.*?）',  # 中文括号内的企业名称
            r'(.*?)[\(（].*?[\)）]'  # 包含括号的企业名称
        ]

        companies = set()

        for pattern in patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    company_name = match[0].strip()
                else:
                    company_name = match.strip()

                # 过滤掉太短或常见词
                common_words = ["公司", "企业", "分析", "评估", "查询", "搜索"]
                if (len(company_name) >= 4 and
                        not any(word in company_name for word in common_words)):
                    companies.add(company_name)

        return list(companies)

    def _search_by_company_name_direct(self, company_name: str, top_k: int) -> List[Dict]:
        """直接按企业名称搜索"""
        results = []

        try:
            # 构建查询条件
            filters = {}
            if company_name:
                # 尝试模糊匹配企业名称
                filters = {
                    "$or": [
                        {"company": {"$contains": company_name}},
                        {"company_name": {"$contains": company_name}}
                    ]
                }

            # 使用语义搜索
            semantic_results = self._semantic_search(company_name, top_k, filters)

            for result in semantic_results:
                # 检查元数据中是否包含企业名称
                metadata = result.get("metadata", {})
                metadata_company = metadata.get("company", "") or metadata.get("company_name", "")

                if company_name in metadata_company or metadata_company in company_name:
                    result["similarity"] = min(result.get("similarity", 0) + 0.3, 1.0)
                    result["match_type"] = "company_name_direct"
                    results.append(result)
                elif company_name.lower() in str(result.get("content", "")).lower():
                    result["similarity"] = min(result.get("similarity", 0) + 0.2, 1.0)
                    result["match_type"] = "company_content_match"
                    results.append(result)

            return results

        except Exception as e:
            print(f"直接企业名称搜索失败: {e}")
            return []

    def _keyword_search(self, keyword: str, top_k: int) -> List[Dict]:
        """关键词搜索"""
        results = []

        try:
            # 构建关键词查询
            vector_results = self._semantic_search(keyword, top_k)

            for result in vector_results:
                # 检查内容中是否包含关键词
                content = result.get("content", "").lower()
                if keyword.lower() in content:
                    # 关键词匹配，提升相似度
                    result["similarity"] = min(result.get("similarity", 0) + 0.2, 1.0)
                    result["match_type"] = "keyword_match"
                    results.append(result)

        except Exception as e:
            print(f"关键词搜索失败: {e}")

        return results
    def _calculate_company_similarity(self, document: Dict, company_name: str) -> float:
        """计算文档与企业的相关度"""
        metadata = document.get("metadata", {})
        content = document.get("content", "")

        # 检查元数据中的企业字段
        metadata_company = metadata.get("company", "")
        if metadata_company:
            if metadata_company == company_name:
                return 1.0
            elif company_name in metadata_company or metadata_company in company_name:
                return 0.8

        # 检查内容中的企业名称
        content_lower = content.lower()
        company_lower = company_name.lower()

        if company_lower in content_lower:
            # 计算出现频率和位置
            occurrences = content_lower.count(company_lower)
            first_pos = content_lower.find(company_lower)

            # 综合评分
            frequency_score = min(occurrences / 5, 1.0) * 0.4
            position_score = 1.0 if first_pos < 100 else 0.5 if first_pos < 500 else 0.2

            return frequency_score + position_score * 0.6

        # 模糊匹配
        if self._calculate_fuzzy_match(company_name, content) > 0.6:
            return 0.5

        return 0.0

    def _calculate_fuzzy_match(self, str1: str, str2: str) -> float:
        """计算模糊匹配分数"""
        if not str1 or not str2:
            return 0.0

        str1_lower = str1.lower()
        str2_lower = str2.lower()

        # 完全包含
        if str1_lower in str2_lower or str2_lower in str1_lower:
            return 0.8

        # 简化字符重叠度计算
        set1 = set(str1_lower)
        set2 = set(str2_lower)
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0

    def _enhance_with_company_relevance(self, results: List[Dict], company_name: str, query: str) -> List[Dict]:
        """用企业相关度增强搜索结果"""
        enhanced_results = []

        for result in results:
            company_similarity = self._calculate_company_similarity(result, company_name)

            # 增强相似度
            original_similarity = result.get("similarity", 0)
            enhanced_similarity = original_similarity * 0.6 + company_similarity * 0.4

            result["similarity"] = enhanced_similarity
            result["company_relevance"] = company_similarity

            enhanced_results.append(result)

        # 重新排序
        enhanced_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return enhanced_results

    def _filter_by_scenario_relevance(self, results: List[Dict], scenario: str) -> List[Dict]:
        """根据场景相关度过滤"""
        scenario_keywords = {
            "撤否企业分析": ["撤否", "审核", "问询", "证监会", "现场检查", "财务造假", "内控"],
            "长期辅导企业分析": ["辅导", "备案", "上市", "IPO", "保荐", "中介"],
            "上下游企业分析": ["关联", "股权", "交易", "控制", "投资", "股东"]
        }

        keywords = scenario_keywords.get(scenario, [])
        if not keywords:
            return results

        scored_results = []
        for result in results:
            content = result.get("content", "").lower()

            # 计算关键词匹配分数
            keyword_score = 0
            for keyword in keywords:
                if keyword in content:
                    keyword_score += 1

            # 计算最终分数
            original_similarity = result.get("similarity", 0)
            scenario_boost = min(keyword_score / 3, 1.0) * 0.3
            final_similarity = original_similarity + scenario_boost

            result["similarity"] = final_similarity
            result["scenario_relevance"] = keyword_score

            scored_results.append(result)

        # 过滤低相关度结果
        filtered_results = [r for r in scored_results if r.get("scenario_relevance", 0) > 0]

        # 如果过滤后结果太少，返回原始结果
        if len(filtered_results) < len(results) * 0.3:
            return results[:max(5, len(results) // 2)]

        return filtered_results

    def _merge_and_dedup(self, results: List[Dict]) -> List[Dict]:
        """合并和去重结果"""
        if not results:
            return []

        unique_results = []
        seen_content_hashes = set()

        for result in results:
            content = result.get("content", "")
            if content:
                # 创建简单的内容哈希
                content_hash = hash(content[:200]) if len(content) > 200 else hash(content)

                if content_hash not in seen_content_hashes:
                    seen_content_hashes.add(content_hash)
                    unique_results.append(result)

        return unique_results

    def _rerank_and_dedup(self, results: List[Dict], query: str, top_k: int) -> List[Dict]:
        """重排序和去重"""
        if not results:
            return []

        # 去重（基于内容相似度）
        unique_results = []
        seen_contents = set()

        for result in results:
            content = result.get("content", "")
            if len(content) > 50:
                # 取内容前200字符作为指纹
                content_fingerprint = content[:200]
                if content_fingerprint not in seen_contents:
                    seen_contents.add(content_fingerprint)
                    unique_results.append(result)
            else:
                unique_results.append(result)

        # 计算查询相关度
        query_terms = re.findall(r'[\u4e00-\u9fa5\w]{2,}', query.lower())

        for result in unique_results:
            content = result.get("content", "").lower()

            # 计算查询词匹配
            query_match_score = 0
            for term in query_terms:
                if term in content:
                    query_match_score += 1

            # 增强分数
            original_similarity = result.get("similarity", 0)
            query_boost = min(query_match_score / max(len(query_terms), 1), 1.0) * 0.2
            enhanced_similarity = original_similarity + query_boost

            result["similarity"] = enhanced_similarity
            result["query_relevance"] = query_match_score

        # 最终排序
        unique_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return unique_results[:top_k]