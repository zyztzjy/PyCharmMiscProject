# soure/rag/qwen_rag_processor.py (更新版 - 集成实时数据)
import dashscope
from dashscope import Generation
from typing import List, Dict, Optional, Union
import json
import re
from datetime import datetime
import logging
import numpy as np

from soure.data_ingestion.data_collector import collect_data
from soure.llm.qwen_client import QwenClient # 导入QwenClient


logger = logging.getLogger(__name__)

class QwenRAGProcessor:
    """基于通义千问的RAG处理器 - 集成实时网络数据"""
    def __init__(self, vectorizer, api_key: str, model: str = "qwen-max"):
        self.vectorizer = vectorizer
        self.llm = QwenClient(api_key, model)
        # 场景模板
        self.scenario_templates = {
            "撤否企业分析": "分析{company_code}的撤否原因、审核问题、财务异常等",
            "供应链分析": "分析{company_code}的上下游关系、同行业对比",
            "财务分析": "分析{company_code}的财务状况、财务风险",
            "舆情分析": "分析{company_code}的舆情情况、媒体报道",
            "行业分析": "分析{company_code}的行业地位、竞争格局"
        }

    def _build_search_query(self, query: str, scenario: str, company_code: str) -> str:
        """构建搜索查询"""
        if company_code:
            return f"{company_code} {scenario} {query}"
        return f"{scenario} {query}"

    def _organize_documents_by_type(self, docs: List[Dict]) -> Dict[str, List[str]]:
        """按文档类型组织文档"""
        organized = {}
        for doc in docs:
            doc_type = doc.get("metadata", {}).get("doc_type", "general")
            if doc_type not in organized:
                organized[doc_type] = []
            content = doc.get("content", "")[:500]
            if content:
                organized[doc_type].append(content)
        return organized

    def _build_prompt(self, query: str, local_docs: List[Dict], web_docs: List[Dict], scenario: str, company_code: str) -> str:
        """构建提示词，整合本地和网络文档"""
        # 组织本地文档
        local_context_parts = []
        for i, doc in enumerate(local_docs[:3]): # 限制本地文档数量
            content = doc.get("content", "")[:300]
            source = doc.get("metadata", {}).get("source", "未知")
            local_context_parts.append(f"【本地文档{i + 1}，来源：{source}】\n{content}")

        # 组织网络文档
        web_context_parts = []
        for i, doc in enumerate(web_docs):
            content = doc.get("content", "")[:300]
            source = doc.get("metadata", {}).get("source", "网络抓取")
            status = doc.get("metadata", {}).get("status", "")
            web_context_parts.append(f"【网络数据{i + 1}，来源：{source}, 状态：{status}】\n{content}")

        # 合并上下文
        all_context_parts = web_context_parts + local_context_parts # 优先使用网络最新数据
        context = "\n\n".join(all_context_parts) if all_context_parts else "暂无相关文档"

        # 根据场景构建提示词
        if scenario in ["供应链分析", "关系网分析"]:
            prompt = f"""作为企业分析专家，请基于以下信息分析{company_code or '目标企业'}：
            最新网络信息：
            {context}
            分析要求：{query}
            请从以下方面分析：
            1. 核心业务关系和网络结构
            2. 关键合作伙伴和竞争对手
            3. 潜在风险和机会
            4. 具体建议和改进方向
            请提供专业的分析报告。"""
        else:
            prompt = f"""作为资深的投行分析师，请基于以下信息回答问题：
            相关信息（包含最新网络数据）：
            {context}
            问题：{query}
            请提供：
            1. 专业分析和判断
            2. 数据支撑和依据 (特别是来自网络的最新信息)
            3. 风险评估
            4. 具体建议
            请确保回答专业、严谨。"""
        return prompt

    def _process_response(self, response: Dict, scenario: str) -> Dict:
        """处理模型响应"""
        if response.get("error"):
            return {
                "summary": "分析过程出现错误",
                "analysis": [response.get("analysis", "未知错误")],
                "risks": ["分析服务暂时不可用"],
                "recommendations": ["请稍后重试"],
                "confidence": 0.3
            }
        content = response.get("analysis", "")

        # 简单解析响应
        return {
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "analysis": [content],
            "risks": self._extract_risks(content),
            "recommendations": self._extract_recommendations(content),
            "confidence": 0.8 # 联网信息通常较新，置信度设为中等偏高
        }

    def _extract_risks(self, text: str) -> List[str]:
        """从文本中提取风险点"""
        risks = []
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["风险", "危险", "问题", "不足", "弱点", "挑战"]):
                risks.append(line.strip())
        return risks[:3] if risks else ["未识别到明确风险"]

    def _extract_recommendations(self, text: str) -> List[str]:
        """从文本中提取建议"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["建议", "对策", "措施", "改进", "优化", "策略"]):
                recommendations.append(line.strip())
        return recommendations[:3] if recommendations else ["暂无具体建议"]

    def process_query(self, query: str, scenario: str = None, company_code: str = None, use_web_data: bool = True) -> Dict:
        """处理用户查询 (集成实时数据)"""
        try:
            start_time = datetime.now()

            # 1. 构建搜索查询
            search_query = self._build_search_query(query, scenario or "通用分析", company_code)

            # 2. 检索本地向量数据库
            local_retrieved_docs = self.vectorizer.search_similar(search_query, top_k=5) if self.vectorizer else []

            # 3. (可选) 获取最新网络数据
            web_retrieved_docs = []
            if use_web_data:
                logger.info("正在抓取最新网络数据...")
                web_docs_raw = collect_data() # 调用数据收集函数
                # 将原始网络数据转换为适合RAG的格式 (如果需要)
                # 这里假设 collect_data 已经返回了格式化的 docs
                web_retrieved_docs = [doc for doc in web_docs_raw if company_code in doc.get("content", "") or scenario in doc.get("content", "")] # 简单过滤
                logger.info(f"从网络获取了 {len(web_retrieved_docs)} 条相关记录。")

            # 4. 构建提示词
            prompt = self._build_prompt(query, local_retrieved_docs, web_retrieved_docs, scenario or "通用分析", company_code)

            # 5. 调用大模型
            response = self.llm.structured_completion(prompt, scenario)

            # 6. 处理响应
            processed_response = self._process_response(response, scenario)

            # 计算时间
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "query": query,
                "scenario": scenario,
                "company_code": company_code,
                "response": processed_response,
                "source_documents": self._format_source_docs(local_retrieved_docs + web_retrieved_docs), # 合并显示来源
                "retrieval_stats": {
                    "total_docs_retrieved": len(local_retrieved_docs) + len(web_retrieved_docs),
                    "local_docs_count": len(local_retrieved_docs),
                    "web_docs_count": len(web_retrieved_docs),
                    "processing_time_seconds": round(processing_time, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return {
                "query": query,
                "scenario": scenario,
                "company_code": company_code,
                "response": {
                    "summary": f"分析失败: {str(e)}",
                    "analysis": ["处理过程中出现错误"],
                    "risks": ["系统暂时不可用"],
                    "recommendations": ["请检查网络连接或稍后重试"],
                    "confidence": 0.1
                },
                "source_documents": [],
                "retrieval_stats": {"total_docs_retrieved": 0, "local_docs_count": 0, "web_docs_count": 0, "processing_time_seconds": 0},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _format_source_docs(self, docs: List[Dict]) -> List[Dict]:
        """格式化源文档信息"""
        formatted = []
        for doc in docs[:5]: # 限制显示数量
            formatted.append({
                "id": doc.get("id", ""),
                "content_preview": doc.get("content", "")[:100] + "...",
                "metadata": doc.get("metadata", {}),
                "similarity": round(doc.get("similarity", 0), 3), # 对于网络数据，相似度可能为0或不适用
                "source": doc.get("metadata", {}).get("source", "未知")
            })
        return formatted