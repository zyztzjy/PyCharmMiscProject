# soure/rag/qwen_rag_processor.py
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..embedding.vectorizer_qwen import QwenVectorizer
from ..llm.qwen_completer import QwenWebCompleter
from ..llm.scenario_config import ScenarioConfig, ScenarioRule, ScenarioType
from ..llm.web_search import QwenWebSearcher
import os

import dashscope


class QwenRAGProcessor:
    """统一RAG处理器（集成通义千问联网搜索）"""

    def __init__(self,
                 vectorizer: QwenVectorizer,
                 api_key: str,
                 model: str = "qwen-max",
                 config: Optional[Dict] = None):

        self.vectorizer = vectorizer
        self.api_key = api_key
        self.model = model

        # 设置API密钥
        dashscope.api_key = api_key

        # 默认配置
        self.default_retrieval_count = 15
        self.similarity_threshold = 0.5

        # 加载配置
        self.config = config or {}
        web_config = self.config.get('web_search', {})

        # 初始化通义千问联网搜索组件
        try:
            self.web_searcher = QwenWebSearcher(api_key)
            self.web_completer = QwenWebCompleter(self.web_searcher, web_config)

            # 测试连接
            connection_ok, message = self.web_searcher.test_connection()
            if connection_ok:
                print(f"✅ 通义千问联网搜索功能已启用: {message}")
                self.web_search_enabled = True
            else:
                print(f"⚠️ 通义千问API连接测试失败: {message}")
                print("联网搜索功能降级为模拟模式")
                self.web_search_enabled = False

        except Exception as e:
            print(f"⚠️ 联网搜索初始化失败: {e}")
            self.web_searcher = None
            self.web_completer = None
            self.web_search_enabled = False

    def intelligent_company_search(
            self,
            search_query: str,
            search_intent: str = "general",
            limit: int = 20,
            use_llm_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        智能企业搜索 - 使用大模型分析检索意图并提取企业信息
        """
        try:
            start_time = datetime.now()

            print(f"执行智能企业搜索: {search_query}")
            print(f"搜索意图: {search_intent}")

            # 1. 使用大模型分析搜索意图
            if use_llm_analysis:
                intent_analysis = self._analyze_search_intent(search_query)
                search_intent = intent_analysis.get("primary_intent", search_intent)
                company_filters = intent_analysis.get("company_filters", {})
                scenario_filters = intent_analysis.get("scenario_filters", {})
                risk_filters = intent_analysis.get("risk_filters", {})

                print(f"意图分析结果: {intent_analysis}")
            else:
                company_filters = {}
                scenario_filters = {}
                risk_filters = {}

            # 2. 从向量库中检索相关文档
            relevant_docs = self._retrieve_relevant_documents(
                search_query=search_query,
                search_intent=search_intent,
                limit=limit * 3  # 获取更多文档供分析
            )

            print(f"检索到 {len(relevant_docs)} 个相关文档")

            if not relevant_docs:
                # 如果没有找到相关文档，尝试更宽松的搜索
                print("未找到相关文档，尝试更宽松的搜索")
                
                # 根据搜索意图尝试不同的文档类型
                if search_intent == "撤否企业":
                    # 尝试搜索包含"撤否"关键词的文档
                    relevant_docs = self.vectorizer.search_similar_documents(
                        query="撤否企业列表",
                        top_k=limit * 2,
                        filters={"document_type": {"$in": ["撤否企业列表", "财务报告"]}},
                        scenario=None
                    )
                elif search_intent == "辅导企业":
                    relevant_docs = self.vectorizer.search_similar_documents(
                        query="辅导企业列表",
                        top_k=limit * 2,
                        filters={"document_type": {"$in": ["辅导企业列表", "辅导报告"]}},
                        scenario=None
                    )
                elif search_intent == "关联企业":
                    relevant_docs = self.vectorizer.search_similar_documents(
                        query="关联企业列表",
                        top_k=limit * 2,
                        filters={"document_type": {"$in": ["关联企业列表", "关联交易报告"]}},
                        scenario=None
                    )
                else:
                    # 一般搜索，查找任何包含企业列表的文档
                    relevant_docs = self.vectorizer.search_similar_documents(
                        query=search_query,
                        top_k=limit * 2,
                        filters={"document_type": {"$in": ["撤否企业列表", "辅导企业列表", "关联企业列表", "风险企业列表", "企业名单"]}},
                        scenario=None
                    )
                
                print(f"宽松搜索找到 {len(relevant_docs)} 个相关文档")

            if not relevant_docs:
                # 再次尝试没有任何过滤条件的搜索
                print("再次尝试无过滤条件的搜索")
                relevant_docs = self.vectorizer.search_similar_documents(
                    query=search_query,
                    top_k=limit * 2,
                    filters=None,
                    scenario=None
                )
                print(f"无过滤条件搜索找到 {len(relevant_docs)} 个相关文档")

            if not relevant_docs:
                return self._build_empty_search_result(search_query)

            # 3. 使用大模型从文档中提取企业信息
            extracted_companies = self._extract_companies_with_llm(
                documents=relevant_docs,
                search_query=search_query,
                search_intent=search_intent,
                company_filters=company_filters,
                scenario_filters=scenario_filters,
                risk_filters=risk_filters,
                limit=limit
            )

            # 4. 如果LLM提取结果不足，使用传统方法补充
            if len(extracted_companies) < limit // 2:
                print(f"LLM提取结果不足 ({len(extracted_companies)})，使用传统方法补充")
                traditional_companies = self._extract_companies_traditional(relevant_docs)

                # 合并结果（去重）
                seen_names = set(c["company_name"] for c in extracted_companies)
                for company in traditional_companies:
                    if company["company_name"] not in seen_names:
                        extracted_companies.append(company)
                        seen_names.add(company["company_name"])

            # 5. 过滤和排序结果
            filtered_companies = self._filter_and_sort_companies(
                companies=extracted_companies,
                search_intent=search_intent,
                company_filters=company_filters,
                scenario_filters=scenario_filters,
                risk_filters=risk_filters,
                limit=limit
            )

            # 6. 使用大模型生成企业摘要和分析
            if use_llm_analysis and filtered_companies:
                enriched_companies = self._enrich_companies_with_llm(
                    companies=filtered_companies,
                    search_query=search_query,
                    relevant_docs=relevant_docs
                )
            else:
                enriched_companies = filtered_companies

            # 7. 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()

            # 8. 构建最终结果
            result = self._build_intelligent_search_result(
                search_query=search_query,
                companies=enriched_companies,
                intent_analysis=intent_analysis if use_llm_analysis else {},
                search_intent=search_intent,
                relevant_docs_count=len(relevant_docs),
                processing_time=processing_time
            )

            print(f"智能搜索完成，找到 {len(enriched_companies)} 个相关企业")
            return result

        except Exception as e:
            print(f"智能企业搜索失败: {e}")
            import traceback
            traceback.print_exc()

            return self._build_error_search_result(search_query, str(e))

    def _analyze_search_intent(self, search_query: str) -> Dict[str, Any]:
        """使用大模型分析搜索意图"""
        prompt = f"""
    作为一名企业情报分析专家，请分析用户的搜索查询，提取搜索意图和过滤条件。

    搜索查询："{search_query}"

    请分析以下内容：
    1. 主要搜索意图（撤否企业/辅导企业/关联企业/高风险企业/一般企业搜索）
    2. 具体企业名称或关键词（如果有）
    3. 相关场景要求（撤否分析/辅导分析/关联分析）
    4. 风险级别要求（高/中/低/特定风险类型）
    5. 其他特殊要求

    重要区别：
    - "撤否企业"：查询中包含明确的撤否结果词汇（如"已撤否"、"撤否了"、"撤单了"、"终止审核"、"撤回申请"、"ipo失败"、"上市失败"、"撤否企业"、"已撤否企业"），表示已发生事件
    - "撤否风险评估"：查询中询问撤否概率或风险（如"撤否可能"、"可能撤否"、"撤否风险"、"撤否概率"、"撤否预测"、"撤否倾向"、"能否通过审核"、"通过率"、"风险评估"等），表示潜在可能性分析

    请以JSON格式返回分析结果，包含以下字段：
    - primary_intent: 主要意图
    - company_filters: 企业名称相关的过滤条件
    - scenario_filters: 场景相关的过滤条件  
    - risk_filters: 风险相关的过滤条件
    - additional_requirements: 其他特殊要求

    只返回JSON对象，不要有其他说明文字。

    示例查询："列出存在撤否可能的高风险企业"
    示例返回：
    {{
        "primary_intent": "撤否风险评估",
        "company_filters": {{"type": "高风险"}},
        "scenario_filters": {{"scenario": "撤否企业分析"}},
        "risk_filters": {{"min_level": "高", "types": ["财务风险", "合规风险"]}},
        "additional_requirements": "重点关注撤否可能性高的企业"
    }}
    """

        try:
            response = dashscope.Generation.call(
                model="qwen-turbo",  # 使用更稳定的模型
                prompt=prompt,
                temperature=0.1,
                top_p=0.9,
                result_format='message',
                max_tokens=500,
                stop=["```", "###"]  # 添加停止词
            )

            if response.status_code == 200:
                response_text = response.output.choices[0].message.content
                print(f"大模型返回的原始响应: {response_text}")  # 调试信息

                # 清理响应文本
                cleaned_text = response_text.strip()

                # 移除可能的Markdown标记
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]

                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]

                # 移除首尾空白
                cleaned_text = cleaned_text.strip()

                # 尝试解析JSON
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败，原始内容: {cleaned_text}")
                    print(f"错误详情: {e}")

                    # 尝试修复常见的JSON问题
                    cleaned_text = self._fix_json_format(cleaned_text)
                    try:
                        return json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        # 如果还是失败，使用规则匹配
                        print("JSON修复失败，使用规则匹配")
                        return self._analyze_search_intent_rule_based(search_query)
            else:
                print(f"大模型API调用失败: {response.code} - {response.message}")
                return self._analyze_search_intent_rule_based(search_query)

        except Exception as e:
            print(f"搜索意图分析失败: {e}")
            import traceback
            traceback.print_exc()
            return self._analyze_search_intent_rule_based(search_query)

    def _fix_json_format(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""
        if not json_str:
            return "{}"

        # 1. 确保以{开始，以}结束
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')

        if start_idx == -1 or end_idx == -1:
            return "{}"

        json_str = json_str[start_idx:end_idx + 1]

        # 2. 修复双引号问题
        json_str = json_str.replace("'", '"')

        # 3. 修复多余的逗号
        lines = json_str.split('\n')
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if line.endswith(','):
                # 检查是否在数组或对象中
                if not (line.endswith('},') or line.endswith('],')):
                    line = line[:-1]
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _analyze_search_intent_rule_based(self, search_query: str) -> Dict[str, Any]:
        """基于规则分析搜索意图"""
        query_lower = search_query.lower()

        # 默认分析结果
        analysis = {
            "primary_intent": "general",
            "company_filters": {},
            "scenario_filters": {},
            "risk_filters": {},
            "additional_requirements": ""
        }

        # 检查暗示可能撤否的关键词（表示存在撤否可能）- 优先检查
        potential_withdrawal_keywords = ["撤否可能", "可能撤否", "撤否风险", "撤否概率", "撤否预测", "撤否趋势", 
                                       "撤否预警", "撤否评估", "撤否可能性", "会撤否", "会被否决", 
                                       "审核风险", "IPO风险", "上市风险", "能否通过", "通过率", "审核概率"]
        if any(keyword in query_lower for keyword in potential_withdrawal_keywords):
            analysis["primary_intent"] = "撤否风险评估"  # 区分已撤否和可能撤否
            analysis["scenario_filters"]["scenario"] = "撤否风险评估分析"
            analysis["additional_requirements"] = "评估撤否可能性"
        # 检查明确的撤否相关关键词（表示已撤否）
        elif any(keyword in query_lower for keyword in ["已撤否", "撤否了", "撤单了", "终止审核", "撤回申请", "ipo失败", "上市失败", "撤否企业", "已撤否企业", "被否", "终止"]):
            analysis["primary_intent"] = "撤否企业"
            analysis["scenario_filters"]["scenario"] = "撤否企业分析"
        # 检查辅导相关
        elif any(keyword in query_lower for keyword in ["辅导", "备案", "辅导期", "辅导企业"]):
            analysis["primary_intent"] = "辅导企业"
            analysis["scenario_filters"]["scenario"] = "长期辅导企业分析"
        # 检查关联相关
        elif any(keyword in query_lower for keyword in ["关联", "上下游", "供应商", "客户", "关系网"]):
            analysis["primary_intent"] = "关联企业"
            analysis["scenario_filters"]["scenario"] = "上下游企业分析"

        # 检查风险相关
        if any(keyword in query_lower for keyword in ["高风险", "风险企业", "存在问题", "有问题的"]):
            analysis["risk_filters"]["min_level"] = "高"
        elif "风险" in query_lower:
            analysis["risk_filters"]["min_level"] = "中"

        return analysis

    def _retrieve_relevant_documents(self, search_query: str, search_intent: str, limit: int) -> List[Dict]:
        """检索相关文档"""
        try:
            # 根据搜索意图构建查询和过滤条件
            enhanced_query = search_query
            filters = None
            
            if search_intent == "撤否企业":
                # 已撤否企业：查找已发生撤否事件的企业
                enhanced_query = f"{search_query} 撤否 撤单 撤回 终止审核 ipo失败 上市失败 撤否企业列表"
                filters = {"document_type": {"$in": ["撤否企业列表", "撤否公告", "终止审核公告", "撤回申请文件"]}}
            elif search_intent == "撤否风险评估":
                # 撤否风险评估：查找可能存在撤否风险的企业
                enhanced_query = f"{search_query} 风险因素 审核问询 监管问题 上市风险 审核风险 撤否预警 撤否可能 撤否概率 撤否预测 审核关注点 问询函 监管问询 问询回复 问题清单"
                filters = {"document_type": {"$in": ["审核问询", "监管文件", "风险分析报告", "问询回复", "监管问询", "财务报告", "法律意见书"]}}
            elif search_intent == "辅导企业":
                enhanced_query = f"{search_query} 辅导 备案 辅导期 上市辅导 辅导企业列表"
                filters = {"document_type": {"$in": ["辅导企业列表", "辅导报告", "备案文件", "进展报告"]}}
            elif search_intent == "关联企业":
                enhanced_query = f"{search_query} 关联 上下游 供应商 客户 关联交易 关联企业列表"
                filters = {"document_type": {"$in": ["关联企业列表", "供应链报告", "关联交易报告", "业务往来"]}}
            else:
                # 对于一般查询，也尝试匹配列表文档类型
                enhanced_query = f"{search_query} 企业列表 列表文档"
                filters = {"document_type": {"$in": ["撤否企业列表", "辅导企业列表", "关联企业列表", "风险企业列表", "财务报告", "报告文档"]}}

            print(f"搜索查询: {enhanced_query}")
            print(f"搜索意图: {search_intent}, 过滤器: {filters}")
            
            # 检索文档
            docs = self.vectorizer.search_similar_documents(
                query=enhanced_query,
                top_k=limit,
                filters=filters,
                company_name=None,
                scenario=None
            )
            
            print(f"按类型过滤检索到 {len(docs)} 个文档")
            
            # 如果按类型检索结果较少，尝试更广泛的检索
            if len(docs) < limit // 2:
                print(f"按类型检索结果较少，尝试广泛检索")
                fallback_docs = self.vectorizer.search_similar_documents(
                    query=enhanced_query,
                    top_k=limit,
                    filters=None,  # 不限制文档类型
                    company_name=None,
                    scenario=None
                )
                print(f"广泛检索到 {len(fallback_docs)} 个文档")
                
                # 合并结果并去重
                all_docs = docs + [doc for doc in fallback_docs if doc not in docs]
                docs = all_docs[:limit]
                print(f"合并后返回 {len(docs)} 个文档")

            return docs

        except Exception as e:
            print(f"检索相关文档失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_companies_with_llm(
            self,
            documents: List[Dict],
            search_query: str,
            search_intent: str,
            company_filters: Dict,
            scenario_filters: Dict,
            risk_filters: Dict,
            limit: int
    ) -> List[Dict]:
        """使用大模型从文档中提取企业信息"""
        if not documents:
            return []

        try:
            # 准备文档内容供大模型分析
            doc_texts = []
            for i, doc in enumerate(documents[:20]):  # 限制文档数量以避免token超限
                # 检查文档是否为字典类型
                if not isinstance(doc, dict):
                    print(f"跳过非字典类型的文档: {type(doc)}")
                    continue
                    
                content = doc.get("content", "")[:500]  # 限制内容长度
                metadata = doc.get("metadata", {})
                source = doc.get("source", "未知")

                doc_text = f"文档{i + 1}:\n"
                doc_text += f"来源: {source}\n"
                doc_text += f"内容摘要: {content}\n"
                doc_text += f"元数据: {json.dumps(metadata, ensure_ascii=False)}\n"
                doc_texts.append(doc_text)

            documents_text = "\n---\n".join(doc_texts)

            # 构建大模型提示词
            if search_intent == "撤否风险评估":
                # 风险评估场景：重点提取可能存在撤否风险的企业
                prompt = f"""
    作为企业风险评估专家，请从以下文档中识别可能存在撤否风险的企业。

    搜索查询：{search_query}
    搜索意图：{search_intent}
    企业过滤条件：{json.dumps(company_filters, ensure_ascii=False)}
    场景过滤条件：{json.dumps(scenario_filters, ensure_ascii=False)}
    风险过滤条件：{json.dumps(risk_filters, ensure_ascii=False)}

    重要说明：
    - 搜索意图是"撤否风险评估"，目标是识别可能存在撤否风险的企业，而不是已经撤否的企业
    - 重点关注审核问询、监管问题、财务风险、合规风险等可能导致撤否的因素
    - 企业可能尚未撤否，但存在撤否风险

    需要提取的企业信息：
    1. 企业名称（中文全称，如：华为技术有限公司）
    2. 企业简称或常用名（如：华为）
    3. 企业代码或股票代码（如有）
    4. 相关风险信息（风险级别：高/中/低，风险类型）
    5. 所属场景（撤否风险评估/辅导/关联/一般）
    6. 从文档中提取的关键证据（简要描述）
    7. 置信度评分（0-100，基于证据充分性）

    文档内容：
    {documents_text}

    请严格遵循以下要求：
    1. 只提取确实在文档中提到的企业，不要编造
    2. 每个企业必须有具体的文档证据支持
    3. 优先提取与撤否风险相关的企业
    4. 准确判断企业的风险级别和场景分类
    5. 明确区分该企业是存在撤否风险还是已经撤否

    请以JSON数组格式返回结果，每个企业对象包含以下字段：
    - company_name: 企业全称
    - company_short_name: 企业简称
    - company_code: 企业代码（如有）
    - risk_assessment: 风险评估对象（包含level, types, evidence）
    - scenario_classification: 场景分类
    - confidence_score: 置信度评分
    - key_evidence: 关键证据列表
    - source_documents: 来源文档列表

    返回最多{limit}个最相关的企业。
    """
            else:
                # 其他场景的提示词
                prompt = f"""
    作为企业信息提取专家，请从以下文档中提取符合搜索要求的企业信息。

    搜索查询：{search_query}
    搜索意图：{search_intent}
    企业过滤条件：{json.dumps(company_filters, ensure_ascii=False)}
    场景过滤条件：{json.dumps(scenario_filters, ensure_ascii=False)}
    风险过滤条件：{json.dumps(risk_filters, ensure_ascii=False)}

    需要提取的企业信息：
    1. 企业名称（中文全称，如：华为技术有限公司）
    2. 企业简称或常用名（如：华为）
    3. 企业代码或股票代码（如有）
    4. 相关风险信息（风险级别：高/中/低，风险类型）
    5. 所属场景（撤否/辅导/关联/一般）
    6. 从文档中提取的关键证据（简要描述）
    7. 置信度评分（0-100，基于证据充分性）

    文档内容：
    {documents_text}

    请严格遵循以下要求：
    1. 只提取确实在文档中提到的企业，不要编造
    2. 每个企业必须有具体的文档证据支持
    3. 优先提取与搜索意图最相关的企业
    4. 准确判断企业的风险级别和场景分类

    请以JSON数组格式返回结果，每个企业对象包含以下字段：
    - company_name: 企业全称
    - company_short_name: 企业简称
    - company_code: 企业代码（如有）
    - risk_assessment: 风险评估对象（包含level, types, evidence）
    - scenario_classification: 场景分类
    - confidence_score: 置信度评分
    - key_evidence: 关键证据列表
    - source_documents: 来源文档列表

    返回最多{limit}个最相关的企业。
    """

            response = dashscope.Generation.call(
                model="qwen-max",
                prompt=prompt,
                temperature=0.1,
                top_p=0.9,
                result_format='message',
                max_tokens=2000
            )

            if response.status_code == 200:
                response_text = response.output.choices[0].message.content

                # 提取JSON数组
                import re
                json_match = re.search(r'\[[\s\S]*]', response_text)
                if json_match:
                    companies = json.loads(json_match.group())

                    # 验证和补充数据
                    validated_companies = []
                    for company in companies:
                        if self._validate_company_info(company):
                            # 补充文档来源信息
                            company["document_references"] = self._find_document_references(
                                company, documents
                            )
                            validated_companies.append(company)

                    return validated_companies

            return []

        except Exception as e:
            print(f"使用LLM提取企业信息失败: {e}")
            return []

    def _validate_company_info(self, company: Dict) -> bool:
        """验证企业信息有效性"""
        required_fields = ["company_name"]

        for field in required_fields:
            if field not in company or not company[field]:
                return False

        # 企业名称至少2个字符
        if len(company.get("company_name", "")) < 2:
            return False

        return True

    def _find_document_references(self, company: Dict, documents: List[Dict]) -> List[Dict]:
        """查找企业相关的具体文档引用"""
        references = []
        company_name = company.get("company_name", "")
        company_short_name = company.get("company_short_name", "")

        if not company_name:
            return references

        for doc in documents:
            # 检查文档是否为字典类型
            if not isinstance(doc, dict):
                print(f"跳过非字典类型的文档: {type(doc)}")
                continue
                
            content = doc.get("content", "").lower()
            metadata = doc.get("metadata", {})

            # 检查文档中是否提到该企业
            if (company_name.lower() in content or
                    (company_short_name and company_short_name.lower() in content)):
                # 获取文档来源信息，优先使用文件名
                source = doc.get("source", "")
                original_filename = metadata.get("original_filename", 
                                               metadata.get("file_name", 
                                                           metadata.get("source", "未知文件")))
                
                reference = {
                    "source": source,
                    "original_filename": original_filename,  # 添加原始文件名
                    "document_type": metadata.get("document_type", "未知"),
                    "relevance_score": doc.get("similarity", 0),
                    "content_snippet": doc.get("content", "")[:300],
                    "metadata": metadata  # 添加完整元数据
                }
                references.append(reference)

        return references

    def _extract_companies_traditional(self, documents: List[Dict]) -> List[Dict]:
        """传统方法提取企业信息（作为LLM的补充）"""
        companies = []
        seen_names = set()

        for doc in documents:
            # 检查文档是否为字典类型，如果不是则跳过
            if not isinstance(doc, dict):
                print(f"跳过非字典类型的文档: {type(doc)}")
                continue
                
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = doc.get("source", "")

            # 从metadata中提取企业名称
            company_name = metadata.get("company_name", "")
            if not company_name:
                # 尝试从文件名提取
                company_name = self._extract_company_name_from_filename(source)

            if company_name and company_name not in seen_names:
                seen_names.add(company_name)

                # 获取文档来源信息，优先使用文件名
                original_filename = metadata.get("original_filename", 
                                               metadata.get("file_name", 
                                                           metadata.get("source", "未知文件")))
                
                company_info = {
                    "company_name": company_name,
                    "company_short_name": company_name.replace("有限公司", "").replace("股份有限公司", ""),
                    "company_code": self._extract_company_code(company_name, content),
                    "risk_assessment": {
                        "level": "未知",
                        "types": [],
                        "evidence": "传统方法提取"
                    },
                    "scenario_classification": "待分析",
                    "confidence_score": 50,
                    "key_evidence": [f"从文档中提取: {source}"],
                    "source_documents": [{
                        "source": source,
                        "original_filename": original_filename,  # 添加原始文件名
                        "document_type": metadata.get("document_type", "未知"),
                        "relevance_score": doc.get("similarity", 0),
                        "metadata": metadata  # 添加完整元数据
                    }]
                }
                companies.append(company_info)

        return companies

    def _filter_and_sort_companies(
            self,
            companies: List[Dict],
            search_intent: str,
            company_filters: Dict,
            scenario_filters: Dict,
            risk_filters: Dict,
            limit: int
    ) -> List[Dict]:
        """过滤和排序企业"""
        if not companies:
            return []

        filtered_companies = []

        for company in companies:
            # 应用过滤器
            if not self._passes_filters(company, search_intent, company_filters, scenario_filters, risk_filters):
                continue

            # 计算匹配分数
            match_score = self._calculate_match_score(company, search_intent, company_filters, scenario_filters,
                                                      risk_filters)
            company["match_score"] = match_score

            filtered_companies.append(company)

        # 按匹配分数排序
        filtered_companies.sort(key=lambda x: x.get("match_score", 0), reverse=True)

        return filtered_companies[:limit]

    def _passes_filters(
            self,
            company: Dict,
            search_intent: str,
            company_filters: Dict,
            scenario_filters: Dict,
            risk_filters: Dict
    ) -> bool:
        """检查企业是否通过所有过滤器"""
        # 检查场景过滤器
        scenario = company.get("scenario_classification", "")
        if scenario_filters.get("scenario"):
            expected_scenario = scenario_filters["scenario"]
            if expected_scenario == "撤否企业分析" and search_intent != "撤否风险评估":
                # 对于一般撤否企业分析，要求场景中包含撤否
                if "撤否" not in scenario and "撤否风险评估" not in scenario:
                    return False
            elif expected_scenario == "撤否企业分析" and search_intent == "撤否风险评估":
                # 对于撤否风险评估，接受撤否风险评估或撤否相关的场景
                if "撤否" not in scenario:
                    return False
            elif expected_scenario == "长期辅导企业分析" and "辅导" not in scenario:
                return False
            elif expected_scenario == "上下游企业分析" and "关联" not in scenario:
                return False

        # 检查风险过滤器
        risk_level = company.get("risk_assessment", {}).get("level", "未知")
        min_risk = risk_filters.get("min_level")

        if min_risk == "高" and risk_level != "高":
            return False
        elif min_risk == "中" and risk_level not in ["高", "中"]:
            return False

        return True

    def _calculate_match_score(
            self,
            company: Dict,
            search_intent: str,
            company_filters: Dict,
            scenario_filters: Dict,
            risk_filters: Dict
    ) -> float:
        """计算企业匹配分数"""
        score = 50  # 基础分数

        # 基于置信度
        confidence = company.get("confidence_score", 0)
        score += confidence * 0.3

        # 基于风险匹配
        risk_level = company.get("risk_assessment", {}).get("level", "未知")
        if risk_filters.get("min_level") == "高" and risk_level == "高":
            score += 30
        elif risk_filters.get("min_level") == "中" and risk_level in ["高", "中"]:
            score += 20

        # 基于场景匹配
        scenario = company.get("scenario_classification", "")
        if search_intent == "撤否企业" and "撤否" in scenario and "风险评估" not in scenario:
            score += 25
        elif search_intent == "撤否风险评估" and ("撤否" in scenario or "风险评估" in scenario):
            score += 30  # 给撤否风险评估更高的分数
        elif search_intent == "辅导企业" and "辅导" in scenario:
            score += 25
        elif search_intent == "关联企业" and "关联" in scenario:
            score += 25

        # 基于文档数量
        doc_refs = company.get("document_references", [])
        score += min(len(doc_refs) * 5, 20)

        return min(score, 100)

    def _enrich_companies_with_llm(
            self,
            companies: List[Dict],
            search_query: str,
            relevant_docs: List[Dict]
    ) -> List[Dict]:
        """使用大模型丰富企业信息"""
        if not companies:
            return companies

        try:
            # 为每个企业生成详细分析
            for company in companies:
                if company.get("confidence_score", 0) < 30:
                    continue  # 置信度过低的不进行分析

                enriched_info = self._analyze_single_company(company, search_query, relevant_docs)
                if enriched_info:
                    company.update(enriched_info)

            return companies

        except Exception as e:
            print(f"丰富企业信息失败: {e}")
            return companies

    def _analyze_single_company(
            self,
            company: Dict,
            search_query: str,
            relevant_docs: List[Dict]
    ) -> Optional[Dict]:
        """分析单个企业"""
        company_name = company.get("company_name", "")
        if not company_name:
            return None

        # 查找与企业相关的文档
        company_docs = []
        for doc in relevant_docs:
            # 确保doc是字典类型才调用get方法
            if isinstance(doc, dict):
                content = doc.get("content", "").lower()
                if company_name.lower() in content:
                    company_docs.append(doc)

        if not company_docs:
            return None

        # 准备文档内容
        doc_texts = []
        for i, doc in enumerate(company_docs[:5]):  # 限制文档数量
            content = doc.get("content", "")[:300]
            doc_texts.append(f"文档{i + 1}: {content}")

        documents_text = "\n".join(doc_texts)

        # 构建分析提示词
        prompt = f"""
    基于以下文档信息，对企业进行分析：

    企业名称：{company_name}
    搜索查询：{search_query}

    相关文档内容：
    {documents_text}

    请分析以下内容：
    1. 企业面临的主要风险类型和具体问题
    2. 与搜索查询的相关性分析
    3. 建议的后续分析方向
    4. 信息可靠性评估

    请以JSON格式返回分析结果，包含以下字段：
    - risk_details: 风险详情（类型、具体问题、影响程度）
    - relevance_analysis: 与搜索查询的相关性分析
    - suggested_actions: 建议的后续分析方向
    - reliability_assessment: 信息可靠性评估（高/中/低）
    """

        try:
            response = dashscope.Generation.call(
                model="qwen-turbo",
                prompt=prompt,
                temperature=0.1,
                top_p=0.9,
                result_format='message',
                max_tokens=800
            )

            if response.status_code == 200:
                response_text = response.output.choices[0].message.content
                import re
                json_match = re.search(r'\{[\s\S]*}', response_text)
                if json_match:
                    return json.loads(json_match.group())

            return None

        except Exception as e:
            print(f"分析单个企业失败: {e}")
            return None

    def _build_intelligent_search_result(
            self,
            search_query: str,
            companies: List[Dict],
            intent_analysis: Dict,
            search_intent: str,
            relevant_docs_count: int,
            processing_time: float
    ) -> Dict[str, Any]:
        """构建智能搜索结果"""
        # 计算统计信息
        risk_distribution = {"高": 0, "中": 0, "低": 0, "未知": 0}
        scenario_distribution = {"撤否": 0, "撤否风险评估": 0, "辅导": 0, "关联": 0, "其他": 0}

        for company in companies:
            risk_level = company.get("risk_assessment", {}).get("level", "未知")
            if risk_level in risk_distribution:
                risk_distribution[risk_level] += 1

            scenario = company.get("scenario_classification", "")
            if "撤否风险评估" in scenario:
                scenario_distribution["撤否风险评估"] += 1
            elif "撤否" in scenario:
                scenario_distribution["撤否"] += 1
            elif "辅导" in scenario:
                scenario_distribution["辅导"] += 1
            elif "关联" in scenario:
                scenario_distribution["关联"] += 1
            else:
                scenario_distribution["其他"] += 1

        return {
            "search_query": search_query,
            "search_intent": search_intent,
            "intent_analysis": intent_analysis,
            "total_found": len(companies),
            "companies": companies,
            "statistics": {
                "relevant_documents_count": relevant_docs_count,
                "risk_distribution": risk_distribution,
                "scenario_distribution": scenario_distribution,
                "average_confidence": sum(c.get("confidence_score", 0) for c in companies) / max(len(companies), 1)
            },
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            "search_method": "intelligent_llm_extraction"
        }

    def _build_empty_search_result(self, search_query: str) -> Dict[str, Any]:
        """构建空搜索结果"""
        return {
            "search_query": search_query,
            "total_found": 0,
            "companies": [],
            "statistics": {
                "relevant_documents_count": 0,
                "risk_distribution": {"高": 0, "中": 0, "低": 0, "未知": 0},
                "scenario_distribution": {"撤否": 0, "辅导": 0, "关联": 0, "其他": 0}
            },
            "processing_time": 0,
            "timestamp": datetime.now().isoformat(),
            "message": "未找到相关企业信息"
        }

    def _build_error_search_result(self, search_query: str, error: str) -> Dict[str, Any]:
        """构建错误搜索结果"""
        return {
            "search_query": search_query,
            "total_found": 0,
            "companies": [],
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "message": "搜索过程中发生错误"
        }

    def search_companies(
            self,
            search_query: str,
            scenario: Optional[str] = None,
            risk_level: Optional[str] = None,
            limit: int = 20
    ) -> Dict[str, Any]:
        """
        检索企业信息
        """
        try:
            start_time = datetime.now()

            print(f"执行企业检索: {search_query}")
            print(f"场景过滤: {scenario or '无'}")
            print(f"风险级别: {risk_level or '无'}")

            # 构建检索查询
            search_terms = []

            # 解析搜索关键词
            if "撤否" in search_query or "撤销" in search_query:
                search_terms.append("撤否企业")
            if "风险" in search_query or "问题" in search_query:
                search_terms.append("风险")
            if "辅导" in search_query:
                search_terms.append("辅导企业")
            if "关联" in search_query or "关系" in search_query:
                search_terms.append("关联企业")

            # 如果没有明确的搜索词，使用原始查询
            if not search_terms:
                search_terms.append(search_query)

            # 构建检索条件
            search_conditions = {
                "search_terms": search_terms,
                "scenario": scenario,
                "risk_level": risk_level
            }

            # 执行检索
            results = []

            # 1. 尝试通过向量搜索查找相关企业
            try:
                # 使用向量搜索查找相关文档
                search_results = self.vectorizer.search_similar_documents(
                    query=search_query,
                    top_k=limit * 3,  # 获取更多结果用于筛选
                    filters=self._build_filters(scenario, None),
                    company_name=None,
                    scenario=scenario
                )

                # 从搜索结果中提取企业信息
                companies_info = self._extract_companies_from_documents(search_results)

                # 按相关性排序
                companies_info.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

                # 过滤和去重
                seen_companies = set()
                for company in companies_info:
                    company_name = company.get("company_name")
                    if company_name and company_name not in seen_companies:
                        seen_companies.add(company_name)
                        results.append(company)

                        if len(results) >= limit:
                            break

            except Exception as e:
                print(f"向量检索失败: {e}")

            # 2. 如果向量检索结果不足，尝试其他方式
            if len(results) < limit:
                try:
                    # 尝试直接从向量库中搜索企业
                    backup_results = self.vectorizer.search_by_company_name(
                        company_name=search_query,
                        query=search_query,
                        top_k=limit * 2,
                        similarity_threshold=0.1
                    )

                    # 提取企业信息
                    backup_companies = self._extract_companies_from_documents(backup_results)

                    # 合并结果（去重）
                    for company in backup_companies:
                        company_name = company.get("company_name")
                        if company_name and company_name not in seen_companies:
                            seen_companies.add(company_name)
                            results.append(company)

                            if len(results) >= limit:
                                break

                except Exception as e:
                    print(f"企业名称搜索失败: {e}")

            # 3. 如果仍然结果不足，尝试联网搜索
            if len(results) < 5 and self.web_search_enabled and self.web_searcher:
                try:
                    print("尝试联网搜索企业信息...")
                    web_results = self.web_searcher.search(
                        query=f"{search_query} 相关企业 上市",
                        company_name=None,
                        scenario=scenario,
                        model="qwen-turbo"
                    )

                    # 从网络结果中提取企业信息
                    web_companies = self._extract_companies_from_web_results(web_results)

                    for company in web_companies:
                        company_name = company.get("company_name")
                        if company_name and company_name not in seen_companies:
                            seen_companies.add(company_name)
                            results.append(company)

                except Exception as e:
                    print(f"联网搜索失败: {e}")

            # 4. 对结果进行风险评估和分类
            enriched_results = []
            for result in results[:limit]:
                enriched_result = self._enrich_company_info(result, search_query)
                enriched_results.append(enriched_result)

            # 排序：按风险等级和相关性
            enriched_results.sort(
                key=lambda x: (
                    {"高": 0, "中": 1, "低": 2, "未知": 3}.get(x.get("risk_assessment", {}).get("risk_level", "未知"),
                                                               3),
                    -x.get("relevance_score", 0)
                )
            )

            # 5. 构建最终结果
            processing_time = (datetime.now() - start_time).total_seconds()

            result_data = {
                "search_query": search_query,
                "total_found": len(enriched_results),
                "companies": enriched_results,
                "search_conditions": search_conditions,
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "search_stats": {
                    "local_sources": len([r for r in enriched_results if r.get("source_type") == "local"]),
                    "web_sources": len([r for r in enriched_results if r.get("source_type") == "web"]),
                    "scenario": scenario,
                    "risk_distribution": self._calculate_risk_distribution(enriched_results)
                }
            }

            print(f"检索完成，找到 {len(enriched_results)} 个相关企业")
            return result_data

        except Exception as e:
            print(f"企业检索失败: {e}")
            import traceback
            traceback.print_exc()

            return {
                "search_query": search_query,
                "total_found": 0,
                "companies": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_companies_from_documents(self, documents: List[Dict]) -> List[Dict]:
        """从文档中提取企业信息"""
        companies = []

        for doc in documents:
            try:
                # 检查文档是否为字典类型，如果不是则跳过
                if not isinstance(doc, dict):
                    print(f"跳过非字典类型的文档: {type(doc)}")
                    continue
                    
                metadata = doc.get("metadata", {})
                company_name = metadata.get("company_name", "")
                content = doc.get("content", "")
                similarity = doc.get("similarity", 0)
                source = doc.get("source", "")

                if company_name:
                    # 提取基本信息
                    company_info = {"company_name": company_name,
                                    "company_code": self._extract_company_code(company_name, content),
                                    "description": self._extract_company_description(content),
                                    "document_type": metadata.get("document_type", "未知"), "source_document": source,
                                    "relevance_score": similarity, "source_type": "local",
                                    "found_in_content": content[:500] if len(content) > 500 else content,
                                    "risk_info": self._extract_risk_info(content, company_name)}

                    # 提取风险相关信息

                    companies.append(company_info)

            except Exception as e:
                print(f"提取企业信息失败: {e}")
                continue

        return companies

    def _extract_companies_from_web_results(self, web_results: List[Dict]) -> List[Dict]:
        """从网络结果中提取企业信息"""
        companies = []

        for result in web_results:
            try:
                # 确保result是字典类型才调用get方法
                if not isinstance(result, dict):
                    continue
                    
                content = result.get("content", "")
                metadata = result.get("metadata", {})

                # 从网络内容中提取企业名称
                company_names = self._extract_company_names_from_text(content)

                for company_name in company_names[:3]:  # 最多取前3个
                    company_info = {
                        "company_name": company_name,
                        "description": metadata.get("title", "")[:200] + "...",
                        "document_type": "网络信息",
                        "source_document": metadata.get("source", "网络来源"),
                        "relevance_score": 0.5,  # 默认相关性
                        "source_type": "web",
                        "found_in_content": content[:300] if len(content) > 300 else content,
                        "risk_info": {
                            "source": "网络检索",
                            "risk_level": "待评估",
                            "description": f"来自网络搜索结果: {metadata.get('title', '')}"
                        }
                    }

                    companies.append(company_info)

            except Exception as e:
                print(f"从网络结果提取企业信息失败: {e}")
                continue

        return companies

    def _extract_company_name_from_filename(self, filename: str) -> str:
        """从文件名中提取企业名称"""
        if not filename:
            return ""
        
        import re
        
        # 获取文件名（不含扩展名）
        file_basename = os.path.basename(filename).split('.')[0]
        
        # 查找可能的企业名称模式
        # 匹配中文名称（包含"公司","股份","集团"等字样）
        patterns = [
            r'([一-龯]{2,15}(?:公司|股份|集团|有限|科技|电子|制造|投资|置业|贸易|发展|控股))',  # 中文公司名
            r'([A-Za-z\s]+(?:Company|Corp|Inc|Ltd|Limited|Group))',  # 英文公司名
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, file_basename)
            if matches:
                return matches[0].strip()
        
        # 如果没有找到标准格式，返回整个文件名（去除特殊字符）
        clean_name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', file_basename)
        return clean_name if len(clean_name) >= 2 else ""

    def _extract_company_names_from_text(self, text: str) -> List[str]:
        """从文本中提取企业名称"""
        import re

        # 常见企业名称模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,10}?(股份|科技|电子|集团|有限公司|公司|证券))',
            r'([A-Za-z0-9&\s]{3,20}?(Inc|Ltd|Corp|Group|Company))',
        ]

        company_names = set()

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    company_name = match[0]
                else:
                    company_name = match

                # 过滤掉太短或太长的情况
                if 2 <= len(company_name) <= 50:
                    company_names.add(company_name)

        return list(company_names)[:10]  # 最多返回10个

    def _extract_company_code(self, company_name: str, content: str) -> str:
        """提取企业代码"""
        import re

        # 尝试从内容中提取6位数字代码
        code_pattern = r'(\d{6})'
        matches = re.findall(code_pattern, content)

        if matches:
            # 返回第一个找到的代码
            return matches[0]

        # 如果没有数字代码，返回公司名称的哈希值前6位
        import hashlib
        return hashlib.md5(company_name.encode()).hexdigest()[:6].upper()

    def _extract_company_description(self, content: str) -> str:
        """提取企业描述"""
        # 取前200个字符作为描述
        if len(content) > 200:
            return content[:200] + "..."
        return content

    def _extract_risk_info(self, content: str, company_name: str) -> Dict:
        """从内容中提取风险信息"""
        risk_keywords = {
            "高": ["撤否", "终止", "失败", "重大", "严重", "违规", "处罚", "诉讼", "亏损", "st"],
            "中": ["问题", "风险", "关注", "问询", "整改", "调整", "波动", "下降"],
            "低": ["正常", "稳定", "良好", "增长", "改善", "通过", "合规"]
        }

        content_lower = content.lower()

        risk_level = "未知"
        risk_count = 0

        # 统计风险关键词
        for level, keywords in risk_keywords.items():
            count = sum(1 for keyword in keywords if keyword in content_lower)
            if count > risk_count:
                risk_count = count
                risk_level = level

        # 提取风险描述
        risk_description = ""
        sentences = content.split('。')
        for sentence in sentences:
            if any(keyword in sentence for keyword in risk_keywords["高"] + risk_keywords["中"]):
                if len(risk_description) < 200:  # 限制长度
                    risk_description += sentence + "。"

        return {
            "risk_level": risk_level,
            "risk_score": min(risk_count * 10, 100),
            "description": risk_description[:300] if risk_description else "未发现明显风险信息",
            "risk_factors": [k for k in risk_keywords["高"] + risk_keywords["中"] if k in content_lower][:5]
        }

    def _enrich_company_info(self, company_info: Dict, search_query: str) -> Dict:
        """丰富企业信息"""
        try:
            # 添加场景信息
            if "撤否" in search_query:
                company_info["scenario"] = "撤否企业分析"
                company_info["scenario_icon"] = "⚠️"
            elif "辅导" in search_query:
                company_info["scenario"] = "长期辅导企业分析"
                company_info["scenario_icon"] = "📅"
            elif "关联" in search_query or "关系" in search_query:
                company_info["scenario"] = "上下游企业分析"
                company_info["scenario_icon"] = "🔗"

            # 添加风险评估
            risk_info = company_info.get("risk_info", {})
            company_info["risk_assessment"] = {
                "risk_level": risk_info.get("risk_level", "未知"),
                "risk_score": risk_info.get("risk_score", 0),
                "description": risk_info.get("description", ""),
                "suggested_action": self._get_suggested_action(risk_info.get("risk_level", "未知"))
            }

            # 添加时间戳
            company_info["last_updated"] = datetime.now().isoformat()

            return company_info

        except Exception as e:
            print(f"丰富企业信息失败: {e}")
            return company_info

    def _get_suggested_action(self, risk_level: str) -> str:
        """根据风险等级获取建议操作"""
        actions = {
            "高": "建议深入分析，关注重大风险点",
            "中": "建议定期监控，注意风险变化",
            "低": "可保持关注，风险相对可控",
            "未知": "需要更多信息进行评估"
        }
        return actions.get(risk_level, "需要进一步分析")

    def _calculate_risk_distribution(self, companies: List[Dict]) -> Dict:
        """计算风险分布"""
        distribution = {"高": 0, "中": 0, "低": 0, "未知": 0}

        for company in companies:
            risk_level = company.get("risk_assessment", {}).get("risk_level", "未知")
            if risk_level in distribution:
                distribution[risk_level] += 1

        total = sum(distribution.values())
        if total > 0:
            for level in distribution:
                distribution[f"{level}_percentage"] = round(distribution[level] / total * 100, 1)

        return distribution
    def process_query(
            self,
            query: str,
            scenario: Optional[str] = None,
            company_code: Optional[str] = None,
            use_web_data: str = "auto",
            retrieval_count: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            web_search_model: Optional[str] = None,
            scenario_rule: Optional[ScenarioRule] = None
    ) -> Dict[str, Any]:
        """
        处理RAG查询（集成通义千问联网搜索）
        """
        try:
            start_time = datetime.now()

            # 使用参数或默认值
            top_k = retrieval_count or self.default_retrieval_count
            threshold = similarity_threshold or self.similarity_threshold

            print(f"开始处理查询: {query}")
            print(f"企业: {company_code or '无'}")
            print(f"场景: {scenario or '无'}")
            print(f"联网搜索模式: {use_web_data}")

            # 获取场景规则
            # 优先使用传入的场景规则，如果未传入则根据场景名称获取
            if not scenario_rule and scenario:
                scenario_rule = ScenarioConfig.get_scenario_by_name(scenario)

            # 1. 本地检索 - 修复这里，确保调用正确的方法
            local_docs = []
            try:
                if company_code:
                    # 使用企业名称搜索
                    local_docs = self.vectorizer.search_by_company_name(
                        company_name=company_code,  # 注意：这里传的是company_code，但方法期望company_name
                        query=query,
                        top_k=top_k * 2,
                        similarity_threshold=0.3
                    )
                else:
                    # 常规搜索
                    filters = self._build_filters(scenario, None)
                    local_docs = self.vectorizer.search_similar_documents(
                        query=query,
                        top_k=top_k,
                        filters=filters,
                        company_name=None,  # 添加这个参数
                        scenario=scenario  # 添加这个参数
                    )

                print(f"本地检索到 {len(local_docs)} 个文档")

            except Exception as e:
                print(f"本地检索失败: {e}")
                local_docs = []

            # 2. 联网搜索决策与执行
            web_docs = []
            web_search_analysis = {
                "performed": False,
                "reason": "未启用",
                "confidence": 0.0,
                "query": "",
                "model": web_search_model or self.model,
                "results_count": 0
            }

            if self.web_search_enabled and self.web_completer:
                # 分析搜索需求
                search_analysis = self.web_completer.analyze_search_need(
                    query=query,
                    local_docs=local_docs,
                    scenario=scenario,
                    company_name=company_code,
                    user_preference=use_web_data
                )

                web_search_analysis.update({
                    "performed": search_analysis["should_search"],
                    "reason": search_analysis["reasons"][0] if search_analysis["reasons"] else "未触发",
                    "confidence": search_analysis["confidence"],
                    "query": search_analysis["search_query"],
                    "model": search_analysis["model"],
                    "search_type": search_analysis["search_type"]
                })

                # 执行搜索
                if search_analysis["should_search"]:
                    print(f"执行联网搜索，类型: {search_analysis['search_type']}")

                    web_docs = self.web_searcher.search(
                        query=search_analysis["search_query"],
                        company_name=company_code,
                        scenario=scenario,
                        model=search_analysis["model"]
                    )

                    web_search_analysis["results_count"] = len(web_docs)
                    print(f"联网搜索获得 {len(web_docs)} 个结果")
                else:
                    print(f"不执行联网搜索，原因: {search_analysis['reasons']}")
            else:
                print("联网搜索功能未启用或不可用")

            # 3. 过滤本地文档
            filtered_local_docs = []
            for doc in local_docs:
                # 确保doc是字典类型才调用get方法
                if isinstance(doc, dict) and doc.get("similarity", 0) >= threshold:
                    filtered_local_docs.append(doc)
            print(f"本地文档过滤后剩余 {len(filtered_local_docs)} 个")

            # 4. 构建增强上下文
            context = self._build_scenario_context(
                local_docs=filtered_local_docs,
                web_docs=web_docs,
                query=query,
                company_code=company_code,
                scenario_rule=scenario_rule,
                web_search_info=web_search_analysis
            )

            # 5. 构建场景化提示词
            prompt = self._build_scenario_prompt(
                query=query,
                context=context,
                scenario_rule=scenario_rule,
                company_code=company_code,
                has_web_data=len(web_docs) > 0,
                web_search_info=web_search_analysis
            )

            # 6. 调用大模型生成响应
            response = self._call_llm(prompt, self.model)

            # 7. 解析响应
            parsed_response = self._parse_response(response, scenario_rule)

            # 8. 增强响应信息
            enhanced_response = self._enhance_response(
                parsed_response,
                filtered_local_docs,
                web_docs,
                web_search_analysis,
                scenario_rule
            )

            # 9. 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()

            # 10. 构建最终结果
            result = self._build_comprehensive_result(
                query=query,
                response=enhanced_response,
                local_docs=filtered_local_docs,
                web_docs=web_docs,
                web_search_analysis=web_search_analysis,
                processing_time=processing_time,
                scenario_name=scenario_rule.display_name if scenario_rule else "自定义分析",
                company_code=company_code,
                threshold=threshold,
                web_mode=use_web_data
            )

            print(f"查询处理完成，总耗时: {processing_time:.2f}秒")

            return result

        except Exception as e:
            print(f"处理查询失败: {e}")
            import traceback
            traceback.print_exc()

            return self._build_error_result(query, scenario, company_code, str(e))

    def _retrieve_local_documents(self, query: str, company_code: Optional[str],
                                  scenario: Optional[str], top_k: int) -> List[Dict]:
        """检索本地文档"""
        if company_code:
            # 使用企业名称搜索
            return self.vectorizer.search_by_company_name(
                company_name=company_code,
                query=query,
                top_k=top_k * 2,
                similarity_threshold=0.3
            )
        else:
            # 常规搜索
            filters = self._build_filters(scenario, None)
            return self.vectorizer.search_similar_documents(
                query=query,
                top_k=top_k,
                filters=filters
            )

    def _build_filters(self, scenario: Optional[str], company_code: Optional[str]) -> Optional[Dict]:
        """构建过滤条件"""
        if not scenario:
            return None

        scenario_to_type = {
            "撤否企业分析": ["撤否企业列表", "财务报告", "审核问询", "监管文件", "风险分析报告"],
            "长期辅导企业分析": ["辅导企业列表", "财务报告", "辅导报告", "备案文件"],
            "上下游企业分析": ["关联企业列表", "供应链报告", "关联交易报告", "业务往来"],
            "撤否企业": ["撤否企业列表", "财务报告", "审核问询", "监管文件", "风险分析报告"],
            "辅导企业": ["辅导企业列表", "辅导报告", "备案文件", "进展报告"],
            "关联企业": ["关联企业列表", "供应链报告", "关联交易报告", "业务往来"]
        }

        if scenario in scenario_to_type:
            types = scenario_to_type[scenario]
            if len(types) == 1:
                return {"document_type": {"$eq": types[0]}}
            else:
                return {"document_type": {"$in": types}}

        return None

    def _build_scenario_context(self, local_docs: List[Dict], web_docs: List[Dict],
                                query: str, company_code: Optional[str],
                                scenario_rule: Optional[ScenarioRule],
                                web_search_info: Dict) -> str:
        """构建场景化的上下文"""
        context_parts = []

        # 1. 查询信息概览
        context_parts.append("=== 分析任务概览 ===")
        context_parts.append(f"📋 原始查询: {query}")
        if company_code:
            context_parts.append(f"🏢 目标企业: {company_code}")
        if scenario_rule:
            context_parts.append(f"🎯 分析场景: {scenario_rule.display_name}")
            context_parts.append(f"📊 分析框架: {scenario_rule.framework}")
        context_parts.append("")

        # 2. 场景分析要求
        if scenario_rule:
            context_parts.append("=== 场景分析要求 ===")
            context_parts.append(f"📝 场景描述: {scenario_rule.description}")
            context_parts.append("🔍 重点关注领域:")
            for focus_area in scenario_rule.focus_areas[:5]:  # 显示前5个
                context_parts.append(f"  • {focus_area}")
            context_parts.append("")

        # 3. 本地文档信息
        if local_docs:
            context_parts.append("=== 本地文档库信息 ===")
            context_parts.append(f"共找到 {len(local_docs)} 个相关文档")

            for i, doc in enumerate(local_docs[:3], 1):  # 限制前3个
                # 确保doc是字典类型才调用get方法
                if not isinstance(doc, dict):
                    continue
                    
                content = doc.get("content", "")
                source = doc.get("source", "未知来源")
                similarity = doc.get("similarity", 0)
                metadata = doc.get("metadata", {})
                company = metadata.get("company", "未知企业")
                doc_type = metadata.get("document_type", "未知类型")

                context_parts.append(
                    f"\n【本地文档{i}】"
                    f"\n📄 来源: {source}"
                    f"\n🏭 企业: {company}"
                    f"\n🏷️ 类型: {doc_type}"
                    f"\n📊 相关度: {similarity:.3f}"
                    f"\n📝 内容: {content[:250]}..."
                )
        else:
            context_parts.append("=== 本地文档库信息 ===")
            context_parts.append("❌ 本地库中未找到相关文档")
        context_parts.append("")

        # 4. 网络搜索结果
        if web_docs:
            context_parts.append("=== 网络最新信息 ===")
            context_parts.append(f"🌐 联网搜索获得 {len(web_docs)} 条信息")

            for i, doc in enumerate(web_docs[:2], 1):  # 限制前2个
                # 确保doc是字典类型才调用get方法
                if not isinstance(doc, dict):
                    continue
                    
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                title = metadata.get("title", "网络信息")
                source = metadata.get("source", "网络来源")
                publish_date = metadata.get("publish_date", "未知日期")

                context_parts.append(
                    f"\n【网络信息{i}】"
                    f"\n📰 标题: {title}"
                    f"\n🏢 来源: {source} ({publish_date})"
                    f"\n📝 内容: {content[:200]}..."
                )
        else:
            if web_search_info.get("performed", False):
                context_parts.append("=== 网络最新信息 ===")
                context_parts.append("⚠️ 联网搜索未获得有效结果")
            else:
                context_parts.append("=== 网络最新信息 ===")
                context_parts.append("ℹ️ 未执行联网搜索")
        context_parts.append("")

        # 5. 综合分析指导
        context_parts.append("=== 分析指导 ===")

        if scenario_rule:
            context_parts.append("💡 场景特定分析提示:")
            for req in scenario_rule.output_requirements[:3]:  # 显示前3个要求
                context_parts.append(f"  • {req}")
            context_parts.append("")

        total_docs = len(local_docs) + len(web_docs)
        if total_docs == 0:
            context_parts.append("🚨 警告: 未找到任何相关信息")
            context_parts.append("请基于通用知识进行分析，并明确说明信息来源有限")
        else:
            context_parts.append(f"✅ 可用信息: 本地{len(local_docs)}个 + 网络{len(web_docs)}个")
            context_parts.append("请结合所有可用信息进行分析，并明确区分信息来源")

        return "\n".join(context_parts)

    def _build_scenario_prompt(self, query: str, context: str,
                               scenario_rule: Optional[ScenarioRule],
                               company_code: Optional[str], has_web_data: bool,
                               web_search_info: Dict) -> str:
        """构建场景化提示词"""

        # 基础系统角色
        if scenario_rule:
            system_role = f"""你是专业的{scenario_rule.display_name}专家，精通{scenario_rule.framework}。
你必须严格按照场景要求进行分析，确保分析的专业性和深度。"""
        else:
            system_role = "你是一个专业、严谨的企业分析专家，擅长综合分析各种信息源。"

        # 场景特定指导
        scenario_guidance = self._get_scenario_guidance(scenario_rule)

        # 信息来源说明
        source_instructions = ""
        if has_web_data:
            source_instructions = f"""
    🌐 网络搜索信息说明：
    - 搜索类型: {web_search_info.get('search_type', '未知')}
    - 搜索置信度: {web_search_info.get('confidence', 0):.2f}
    - 请特别关注网络信息的时效性和权威性"""

        # 输出模板
        output_template = self._get_scenario_output_template(scenario_rule)

        # 构建完整提示词
        prompt = f"""{system_role}

## 📋 分析任务
原始查询：{query}
{f'🏢 目标企业：{company_code}' if company_code else ''}
{f'🎯 分析场景：{scenario_rule.display_name if scenario_rule else "自定义分析"}'}

## 🎯 场景分析要求
{scenario_guidance}

## 📚 可用信息汇总
{context}

{source_instructions}

## 📄 输出格式要求
{output_template}

## ⚠️ 重要提示
1. 必须明确区分本地文档和网络信息的分析依据
2. 对不确定性保持诚实，不夸大或编造信息
3. 所有结论必须有信息支撑
4. 保持专业、客观、谨慎的分析态度
5. 严格按照场景要求的分析框架进行分析
"""

        return prompt

    def _get_scenario_guidance(self, scenario_rule: Optional[ScenarioRule]) -> str:
        """获取场景特定指导"""
        if not scenario_rule:
            return "请基于提供的所有信息进行全面、深入的分析。"

        guidance_map = {
            "撤否企业分析": f"""
【{scenario_rule.framework}】
请按以下维度进行分析：

1️⃣ 企业层面：
   - 财务数据真实性核查（收入确认、成本核算、毛利率异常等）
   - 内部控制有效性评估（资金管理、关联交易决策等）
   - 持续经营能力分析（业绩趋势、客户稳定性等）
   - 信息披露质量检查（招股书一致性、风险提示等）

2️⃣ 中介机构层面：
   - 保荐机构执业质量（尽职调查充分性）
   - 审计机构工作质量（审计程序适当性）
   - 律师核查充分性（法律事项完整性）

3️⃣ 监管审核层面：
   - 现场检查发现问题（主要违规事项）
   - 审核问询重点演变（监管关注点变化）
   - 撤否原因深度剖析（直接触发事件）

【重点关注】{', '.join(scenario_rule.focus_areas[:4])}
""",

            "长期辅导企业分析": f"""
【{scenario_rule.framework}】
请按以下阶段进行分析：

1️⃣ 辅导进度诊断：
   - 辅导历程时间线（备案时间、各阶段情况）
   - 中介机构变更及原因（保荐机构、审计机构等）
   - 主要工作内容质量评估（辅导报告、整改情况）

2️⃣ 障碍深度分析：
   - 财务规范性问题（会计政策、收入确认等）
   - 法律合规障碍（诉讼、处罚、知识产权等）
   - 业务独立性缺陷（关联交易、同业竞争等）
   - 行业定位问题（板块匹配度、政策支持度）

3️⃣ 上市可行性评估：
   - 近期上市可能性预测
   - 必要整改措施建议
   - 替代方案分析（新三板、并购重组等）

【重点关注】{', '.join(scenario_rule.focus_areas[:4])}
""",

            "上下游企业分析": f"""
【{scenario_rule.framework}】
请按以下层次进行分析：

1️⃣ 股权关联层：
   - 实际控制人穿透核查
   - 交叉持股和一致行动关系
   - 历史股权变更合规性

2️⃣ 业务关联层：
   - 关联交易公允性（价格、条款、结算方式）
   - 客户供应商依赖度分析（集中度、稳定性）
   - 同业竞争识别和影响

3️⃣ 人员关联层：
   - 关键人员兼职情况
   - 共同投资和利益关系
   - 历史任职关联性

4️⃣ 资金关联层：
   - 资金往来和担保情况
   - 资产租赁和共享安排
   - 其他潜在利益输送

【重点关注】{', '.join(scenario_rule.focus_areas[:4])}
"""
        }

        return guidance_map.get(scenario_rule.display_name,
                                "请基于提供的所有信息进行全面、深入的分析。")

    # 在 qwen_rag_processor.py 中找到 _get_scenario_output_template 方法，修改如下：

    def _get_scenario_output_template(self, scenario_rule: Optional[ScenarioRule]) -> str:
        """获取场景特定的输出模板"""

        if not scenario_rule:
            # 默认使用撤否企业分析模板
            scenario_rule = ScenarioConfig.get_all_scenarios()[ScenarioType.WITHDRAWAL]

        # 基础模板
        base_template = """请以JSON格式返回分析结果，必须包含以下字段：{{
        "summary": "总体结论摘要（200字以内，注明主要信息来源）",
        "detailed_analysis": {{
            "local_based": ["基于本地文档的分析要点1", "基于本地文档的分析要点2"],
            "web_based": ["基于网络信息的分析要点1", "基于网络信息的分析要点2"],
            "integrated": ["综合分析要点1", "综合分析要点2"]
        }},
        "key_findings": ["关键发现1", "关键发现2", "关键发现3"],
        "risk_assessment": {{
            "identified_risks": ["风险点1（注明来源）", "风险点2（注明来源）"],
            "risk_level": "高/中/低",
            "rationale": "风险评估依据"
        }},
        "recommendations": ["具体建议1", "具体建议2", "具体建议3"]"""

        # 场景特定字段
        scenario_fields = {
            "撤否企业分析": """,
        "withdrawal_analysis": {{
            "main_reasons": ["主要原因1", "主要原因2"],
            "timeline": [{{"date": "YYYY-MM-DD", "event": "事件描述", "type": "类型", "impact": "影响程度"}}],
            "inquiry_focus": ["问询重点1", "问询重点2"],
            "reapply_prediction": "预计重新申报时间",
            "success_probability": "重新上市成功率"
        }}""",

            "长期辅导企业分析": """,
        "tutoring_analysis": {{
            "start_date": "辅导开始时间",
            "duration_months": 0,
            "current_stage": "当前阶段",
            "ipo_obstacles": [{{"type": "障碍类型", "severity": "严重程度", "description": "具体描述"}}],
            "feasibility_assessment": {{
                "short_term_possibility": "近期上市可能性",
                "key_prerequisites": ["前提条件1", "前提条件2"]
            }}
        }}""",

            "上下游企业分析": """,
        "relationship_analysis": {{
            "entity_count": 0,
            "relation_count": 0,
            "relations": [{{"entity_a": "企业A", "entity_b": "企业B", "type": "关系类型", "risk_level": "风险等级"}}],
            "independence_issues": ["独立性问题1", "独立性问题2"],
            "risk_transmission_analysis": {{"paths": [{{"from": "源头", "to": "目标", "mechanism": "传导机制"}}]}}
        }}"""
        }

        # 添加场景特定字段
        enhancement = scenario_fields.get(scenario_rule.display_name, "")

        # 闭合JSON
        closing = """
    }"""

        return base_template + enhancement + closing

    def _call_llm(self, prompt: str, model: str) -> str:
        """调用通义千问API"""
        try:
            response = dashscope.Generation.call(
                model=model,
                prompt=prompt,
                temperature=0.2,
                top_p=0.9,
                result_format='message',
                max_tokens=5000,
                seed=12345
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                print(f"通义千问API调用失败: {response.code} - {response.message}")
                if model != "qwen-turbo":
                    print("尝试使用qwen-turbo模型...")
                    return self._call_llm(prompt, "qwen-turbo")
                else:
                    return f"API调用失败: {response.message}"

        except Exception as e:
            print(f"调用LLM失败: {e}")
            return f"模型调用错误: {str(e)}"

    def _parse_response(self, response_text: str, scenario_rule: Optional[ScenarioRule]) -> Dict[str, Any]:
        """解析模型响应"""
        try:
            import re

            # 查找JSON部分
            json_match = re.search(r'\{[\s\S]*}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # 确保包含场景特定字段
                if scenario_rule and scenario_rule.display_name == "撤否企业分析":
                    if "withdrawal_analysis" not in parsed:
                        parsed["withdrawal_analysis"] = {}
                elif scenario_rule and scenario_rule.display_name == "长期辅导企业分析":
                    if "tutoring_analysis" not in parsed:
                        parsed["tutoring_analysis"] = {}
                elif scenario_rule and scenario_rule.display_name == "上下游企业分析":
                    if "relationship_analysis" not in parsed:
                        parsed["relationship_analysis"] = {}

                return parsed

            # 如果不是标准JSON，返回结构化文本
            print("响应不是标准JSON格式")
            return self._parse_structured_response(response_text, scenario_rule)

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return self._create_fallback_response(response_text, scenario_rule)
        except Exception as e:
            print(f"响应解析失败: {e}")
            return self._create_error_response(str(e))

    def _parse_structured_response(self, text: str, scenario_rule: Optional[ScenarioRule]) -> Dict[str, Any]:
        """解析结构化文本响应"""
        base_response = {
            "summary": text[:300] + "..." if len(text) > 300 else text,
            "detailed_analysis": {
                "local_based": ["基于文本解析的分析"],
                "web_based": [],
                "integrated": ["综合信息分析"]
            },
            "key_findings": ["响应格式为非标准JSON"],
            "risk_assessment": {
                "identified_risks": ["数据格式风险"],
                "risk_level": "低",
                "rationale": "模型响应格式异常"
            },
            "recommendations": ["检查API响应格式"]
        }

        # 添加场景特定字段
        if scenario_rule and scenario_rule.display_name == "撤否企业分析":
            base_response["withdrawal_analysis"] = {"main_reasons": ["格式解析问题"]}
        elif scenario_rule and scenario_rule.display_name == "长期辅导企业分析":
            base_response["tutoring_analysis"] = {"current_stage": "分析阶段"}
        elif scenario_rule and scenario_rule.display_name == "上下游企业分析":
            base_response["relationship_analysis"] = {"relations": []}

        return base_response

    def _create_fallback_response(self, text: str, scenario_rule: Optional[ScenarioRule]) -> Dict[str, Any]:
        """创建降级响应"""
        base_response = {
            "summary": f"分析结果（原始响应）: {text[:200]}...",
            "detailed_analysis": {
                "local_based": ["本地信息分析"],
                "web_based": ["网络信息分析"],
                "integrated": ["综合分析"]
            },
            "key_findings": ["获取到分析结果"],
            "risk_assessment": {
                "identified_risks": ["响应格式异常"],
                "risk_level": "低",
                "rationale": "系统处理正常"
            },
            "recommendations": ["继续监控企业动态"]
        }

        # 添加场景特定字段
        if scenario_rule:
            base_response["information_quality"] = {
                "source_reliability": "中",
                "data_completeness": "一般",
                "timeliness": "最新",
                "limitations": ["响应格式需优化"]
            }

        return base_response

    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "summary": f"分析过程中遇到错误: {error}",
            "detailed_analysis": {
                "local_based": ["系统处理异常"],
                "web_based": [],
                "integrated": []
            },
            "key_findings": ["系统暂时不可用"],
            "risk_assessment": {
                "identified_risks": ["系统错误"],
                "risk_level": "高",
                "rationale": "技术故障"
            },
            "recommendations": ["联系技术支持", "稍后重试"]
        }

    def _enhance_response(self, response: Dict[str, Any],
                          local_docs: List[Dict],
                          web_docs: List[Dict],
                          web_search_info: Dict,
                          scenario_rule: Optional[ScenarioRule]) -> Dict[str, Any]:
        """增强响应信息"""
        # 添加信息来源统计
        response["source_statistics"] = {
            "local_documents": len(local_docs),
            "web_results": len(web_docs),
            "total_sources": len(local_docs) + len(web_docs),
            "web_search_performed": web_search_info.get("performed", False),
            "web_search_confidence": web_search_info.get("confidence", 0.0)
        }

        # 添加场景信息
        if scenario_rule:
            response["scenario_info"] = {
                "name": scenario_rule.display_name,
                "framework": scenario_rule.framework,
                "focus_areas": scenario_rule.focus_areas[:5]  # 取前5个
            }

        # 添加时间戳
        response["analysis_timestamp"] = datetime.now().isoformat()

        return response

    def _build_comprehensive_result(self, query: str, response: Dict,
                                    local_docs: List[Dict], web_docs: List[Dict],
                                    web_search_analysis: Dict, processing_time: float,
                                    scenario_name: str, company_code: Optional[str],
                                    threshold: float, web_mode: str) -> Dict[str, Any]:
        """构建全面的结果"""
        return {
            "query": query,
            "response": response,
            "retrieval_stats": {
                "local_documents": len(local_docs),
                "web_results": len(web_docs),
                "total_sources": len(local_docs) + len(web_docs),
                "similarity_threshold": threshold
            },
            "source_documents": local_docs + web_docs,
            "processing_time": round(processing_time, 2),
            "scenario_name": scenario_name,
            "company_code": company_code,
            "timestamp": datetime.now().isoformat(),
            "web_mode": web_mode,
            "web_search_info": web_search_analysis
        }

    def _build_error_result(self, query: str, scenario: Optional[str],
                            company_code: Optional[str], error: str) -> Dict[str, Any]:
        """构建错误结果"""
        return {
            "query": query,
            "response": self._create_error_response(error),
            "retrieval_stats": {
                "local_documents": 0,
                "web_results": 0,
                "total_sources": 0,
                "error": error
            },
            "source_documents": [],
            "processing_time": 0,
            "scenario_name": scenario or "自定义分析",
            "company_code": company_code,
            "timestamp": datetime.now().isoformat(),
            "web_mode": "none",
            "error": error
        }

    def clear_all_caches(self):
        """清空所有缓存"""
        if self.web_searcher:
            self.web_searcher.clear_cache()
        print("所有缓存已清空")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "rag_processor": "运行中",
            "web_search_enabled": self.web_search_enabled,
            "model": self.model,
            "scenario_support": True,
            "version": "3.0"
        }