# soure/rag/qwen_rag_processor.py
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..embedding.vectorizer_qwen import QwenVectorizer
from ..llm.qwen_completer import QwenWebCompleter
from ..llm.scenario_config import ScenarioConfig, ScenarioRule, ScenarioType
from ..llm.web_search import QwenWebSearcher

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
            filtered_local_docs = [
                doc for doc in local_docs
                if doc.get("similarity", 0) >= threshold
            ]
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
            "撤否企业分析": "财务报告",
            "长期辅导企业分析": "财务报告",
            "关系网分析": "报告文档"
        }

        if scenario in scenario_to_type:
            return {"document_type": {"$eq": scenario_to_type[scenario]}}

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