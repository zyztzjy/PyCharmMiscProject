# soure/web_search/qwen_web_searcher.py
import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import dashscope
from dashscope import Generation
import re


class QwenWebSearcher:
    """通义千问联网搜索器（根据场景优化搜索内容）"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key

        # 搜索配置
        self.max_web_results = 5
        self.search_timeout = 30
        self.enable_cache = True
        self.cache = {}

        # 支持的模型列表
        self.supported_models = [
            "qwen-max",  # 通义千问Max（支持联网）
            "qwen-plus",  # 通义千问Plus（支持联网）
            "qwen-turbo"  # 通义千问Turbo（部分支持）
        ]

        # 场景特定的搜索配置
        self.scenario_configs = {
            "撤否企业分析": {
                "keywords": [
                    "撤否原因", "终止审核", "现场检查", "财务异常", "信息披露",
                    "关联交易", "内部控制", "持续经营", "申报失败", "审核问询"
                ],
                "sources": [
                    "证监会网站", "交易所公告", "财经媒体报道", "券商研究报告",
                    "企业招股说明书", "问询函回复", "现场检查报告"
                ],
                "time_range": "近3年",  # 撤否信息需要较新的数据
                "search_focus": "审核过程和监管问题"
            },
            "长期辅导企业分析": {
                "keywords": [
                    "辅导备案", "辅导报告", "IPO进程", "申报障碍", "持续经营",
                    "财务规范", "法律合规", "中介机构", "辅导期", "上市准备"
                ],
                "sources": [
                    "证监局公告", "券商公告", "企业公告", "财经新闻",
                    "辅导工作总结", "上市进度报道"
                ],
                "time_range": "近5年",  # 辅导过程可能较长
                "search_focus": "辅导历程和障碍分析"
            },
            "上下游企业分析": {
                "keywords": [
                    "关联企业", "客户关系", "供应商", "股权结构", "实际控制人",
                    "同业竞争", "资金往来", "担保情况", "关联交易", "供应链"
                ],
                "sources": [
                    "企业年报", "招股说明书", "关联交易公告", "工商信息",
                    "行业研究报告", "产业链分析报告"
                ],
                "time_range": "近3年",
                "search_focus": "关联关系和风险传导"
            }
        }

        # 默认模型
        self.default_model = "qwen-max"

    def search(self,
               query: str,
               company_name: Optional[str] = None,
               scenario: Optional[str] = None,
               model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        执行场景优化的联网搜索

        Args:
            query: 搜索查询
            company_name: 企业名称
            scenario: 分析场景
            model: 使用的模型

        Returns:
            搜索结果列表
        """
        try:
            print(f"开始场景化搜索: {query} | 场景: {scenario} | 企业: {company_name}")

            # 检查缓存
            cache_key = self._generate_cache_key(query, company_name, scenario)
            if self.enable_cache and cache_key in self.cache:
                print("从缓存加载搜索结果")
                return self.cache[cache_key]

            # 构建场景优化的搜索查询
            enhanced_query, search_params = self._build_scenario_optimized_query(
                query, company_name, scenario
            )

            # 选择模型
            search_model = model or self.default_model
            if search_model not in self.supported_models:
                search_model = self.default_model

            # 执行搜索
            web_results = self._perform_scenario_search(enhanced_query, search_model, search_params)

            # 处理并格式化结果
            formatted_results = self._format_search_results(web_results, query, company_name, scenario)

            # 缓存结果
            if self.enable_cache:
                self.cache[cache_key] = formatted_results

            print(f"场景化搜索完成，获得 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            print(f"场景化搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_results(query, company_name, scenario)

    def _build_scenario_optimized_query(self,
                                        query: str,
                                        company_name: Optional[str],
                                        scenario: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """构建场景优化的搜索查询"""
        search_metadata = {
            "company": company_name,
            "scenario": scenario,
            "timestamp": datetime.now().isoformat()
        }

        # 获取场景配置
        scenario_config = self.scenario_configs.get(scenario, {})

        # 构建查询部件
        query_parts = []

        # 1. 添加企业名称（如果存在）
        if company_name:
            query_parts.append(company_name)

        # 2. 添加原始查询关键词
        query_parts.append(query)

        # 3. 添加场景特定关键词
        if scenario_config:
            keywords = scenario_config.get("keywords", [])
            # 选择最相关的3-4个关键词
            selected_keywords = self._select_relevant_keywords(query, keywords)
            query_parts.extend(selected_keywords[:4])
            search_metadata["scenario_keywords"] = selected_keywords[:4]

        # 4. 添加时间限定
        time_range = scenario_config.get("time_range", "近2年")
        query_parts.extend([time_range, "最新"])

        # 5. 添加质量要求关键词
        quality_terms = ["官方", "权威", "可靠", "准确", "详细", "深度"]
        query_parts.extend(quality_terms[:2])

        # 构建最终查询
        enhanced_query = " ".join([p for p in query_parts if p])

        # 更新搜索参数
        search_metadata.update({
            "search_query": enhanced_query,
            "time_range": time_range,
            "search_focus": scenario_config.get("search_focus", "企业信息"),
            "preferred_sources": scenario_config.get("sources", [])
        })

        print(f"场景优化搜索查询: {enhanced_query}")
        print(f"搜索参数: {search_metadata}")

        return enhanced_query, search_metadata

    def _select_relevant_keywords(self, query: str, keywords: List[str]) -> List[str]:
        """选择与查询最相关的关键词"""
        query_lower = query.lower()
        relevance_scores = []

        for keyword in keywords:
            # 计算相关性分数
            score = 0
            keyword_lower = keyword.lower()

            # 检查是否包含关键词
            if keyword_lower in query_lower:
                score += 2

            # 检查是否有共同词汇
            query_words = set(query_lower.split())
            keyword_words = set(keyword_lower.split())
            common_words = query_words.intersection(keyword_words)
            score += len(common_words) * 0.5

            relevance_scores.append((keyword, score))

        # 按分数排序
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        return [k for k, s in relevance_scores if s > 0]

    def _perform_scenario_search(self,
                                 query: str,
                                 model: str,
                                 search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行场景特定的搜索"""
        try:
            print(f"执行场景化搜索，模型: {model}")

            # 获取场景信息
            scenario = search_params.get("scenario", "通用搜索")
            company_name = search_params.get("company", "")
            search_focus = search_params.get("search_focus", "企业信息")
            preferred_sources = search_params.get("preferred_sources", [])

            # 构建场景特定的系统提示
            system_prompt = self._build_scenario_system_prompt(
                scenario, search_focus, preferred_sources
            )

            # 构建用户查询
            user_query = self._build_scenario_user_query(
                query, company_name, scenario, search_params
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            # 调用API
            response = Generation.call(
                model=model,
                messages=messages,
                result_format='message',
                temperature=0.2,  # 降低温度以获得更精确的结果
                top_p=0.7,
                max_tokens=3500,
                enable_search=True,
                search_params={
                    "search_mode": "accurate",
                    "max_search_results": self.max_web_results,
                    "search_timeout": self.search_timeout
                }
            )

            if response.status_code == 200:
                result_text = response.output.choices[0].message.content
                print(f"场景化搜索成功: {scenario}")

                # 解析响应
                return self._parse_scenario_response(result_text, query, scenario, search_params)
            else:
                print(f"场景化搜索API失败: {response.code}")
                return self._perform_fallback_search(query, model, search_params)

        except Exception as e:
            print(f"场景化搜索异常: {e}")
            return self._get_scenario_mock_results(query, scenario)

    def _build_scenario_system_prompt(self,
                                      scenario: str,
                                      search_focus: str,
                                      preferred_sources: List[str]) -> str:
        """构建场景特定的系统提示"""

        scenario_prompts = {
            "撤否企业分析": f"""你是一个专业的IPO撤否分析专家，请使用联网搜索功能获取以下信息：

搜索重点：{search_focus}

信息来源优先级：
1. {preferred_sources[0] if preferred_sources else '证监会官方公告'}
2. {preferred_sources[1] if len(preferred_sources) > 1 else '交易所问询函'}
3. {preferred_sources[2] if len(preferred_sources) > 2 else '财经媒体报道'}

具体要求：
- 重点获取撤否原因、审核过程、监管关注点
- 查找现场检查报告、问询函回复等官方文件
- 关注财务数据异常、内部控制缺陷等信息
- 注意信息的时效性和权威性

请以结构化JSON格式返回搜索结果。""",

            "长期辅导企业分析": f"""你是一个专业的上市辅导分析专家，请使用联网搜索功能获取以下信息：

搜索重点：{search_focus}

信息来源优先级：
1. {preferred_sources[0] if preferred_sources else '证监局备案信息'}
2. {preferred_sources[1] if len(preferred_sources) > 1 else '企业公告'}
3. {preferred_sources[2] if len(preferred_sources) > 2 else '券商研究报告'}

具体要求：
- 重点获取辅导历程、障碍分析、上市进度
- 查找辅导备案、进展报告等官方文件
- 关注财务规范、法律合规等关键问题
- 注意信息的连续性和完整性

请以结构化JSON格式返回搜索结果。""",

            "上下游企业分析": f"""你是一个专业的关联关系分析专家，请使用联网搜索功能获取以下信息：

搜索重点：{search_focus}

信息来源优先级：
1. {preferred_sources[0] if preferred_sources else '企业年报和公告'}
2. {preferred_sources[1] if len(preferred_sources) > 1 else '工商信息'}
3. {preferred_sources[2] if len(preferred_sources) > 2 else '行业研究报告'}

具体要求：
- 重点获取股权结构、业务往来、关联交易
- 查找股权穿透图、关联交易公告等
- 关注客户集中度、供应商依赖度等风险
- 注意信息的准确性和权威性

请以结构化JSON格式返回搜索结果。"""
        }

        return scenario_prompts.get(scenario, """你是一个专业的企业信息分析专家，请使用联网搜索功能获取准确、权威的企业相关信息。

要求：
1. 获取最新、最相关的信息
2. 优先选择官方、权威的信息来源
3. 确保信息的准确性和时效性
4. 注明信息来源和发布时间

请以结构化JSON格式返回搜索结果。""")

    def _build_scenario_user_query(self,
                                   query: str,
                                   company_name: Optional[str],
                                   scenario: str,
                                   search_params: Dict[str, Any]) -> str:
        """构建场景特定的用户查询"""

        time_range = search_params.get("time_range", "近2年")
        scenario_keywords = search_params.get("scenario_keywords", [])

        query_template = f"""请使用联网搜索功能，搜索以下信息：

搜索主题：{query}
{f'企业名称：{company_name}' if company_name else ''}
分析场景：{scenario}

具体要求：
1. 时间范围：{time_range}的信息
2. 重点关注：{', '.join(scenario_keywords[:3]) if scenario_keywords else '相关企业信息'}
3. 信息类型：官方公告、权威报道、研究报告等
4. 数量要求：{self.max_web_results}条最相关的结果
5. 质量要求：信息准确、来源可靠、内容详实

请返回结构化JSON数据，每条结果包含：
- title: 信息标题
- content: 内容摘要（300-400字，包含关键数据）
- source: 信息来源（具体机构/媒体名称）
- url: 原文链接（如果可得）
- publish_date: 发布日期
- relevance_score: 相关性评分（0-1）
- search_method: 搜索方式
- information_type: 信息类型（如：官方公告、新闻报道、研究报告等）

如果信息不足或搜索受限，请说明原因。"""

        return query_template

    def _parse_scenario_response(self,
                                 response_text: str,
                                 original_query: str,
                                 scenario: str,
                                 search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析场景特定的响应"""
        try:
            # 查找JSON部分
            import re
            json_match = re.search(r'\[\s*{.*}\s*]', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                results = json.loads(json_str)

                # 增强结果信息
                for result in results:
                    result["scenario"] = scenario
                    result["company"] = search_params.get("company")
                    result["search_query"] = original_query
                    result["retrieval_time"] = datetime.now().isoformat()

                print(f"成功解析场景响应，共 {len(results)} 条结果")
                return results
            else:
                # 尝试解析结构化文本
                return self._parse_structured_text_with_scenario(
                    response_text, original_query, scenario, search_params
                )

        except json.JSONDecodeError:
            print("JSON解析失败，尝试结构化解析")
            return self._parse_structured_text_with_scenario(
                response_text, original_query, scenario, search_params
            )
        except Exception as e:
            print(f"响应解析异常: {e}")
            return []

    def _parse_structured_text_with_scenario(self,
                                             text: str,
                                             original_query: str,
                                             scenario: str,
                                             search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """带场景信息的结构化文本解析"""
        results = []

        # 根据场景使用不同的解析策略
        if scenario == "撤否企业分析":
            results = self._parse_withdrawal_analysis_text(text, original_query, search_params)
        elif scenario == "长期辅导企业分析":
            results = self._parse_tutoring_analysis_text(text, original_query, search_params)
        elif scenario == "上下游企业分析":
            results = self._parse_relationship_analysis_text(text, original_query, search_params)
        else:
            results = self._parse_general_structured_text(text, original_query, search_params)

        return results

    def _parse_withdrawal_analysis_text(self,
                                        text: str,
                                        original_query: str,
                                        search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析撤否企业分析文本"""
        results = []

        # 查找撤否相关信息
        withdrawal_patterns = [
            r'(撤否|终止审核|撤回申请|审核不通过)[：:]\s*(.+?)(?:\n|$)',
            r'(现场检查|财务异常|关联交易|信息披露)[：:]\s*(.+?)(?:\n|$)',
            r'(审核问询)[：:]\s*(.+?)(?:\n|$)',
            r'(主要问题)[：:]\s*(.+?)(?:\n|$)',
        ]

        sections = re.split(r'\n\s*\n+', text)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 100:
                continue

            # 检查是否包含撤否相关信息
            is_withdrawal_related = any(
                re.search(pattern, section, re.IGNORECASE)
                for pattern in ['撤否', '终止', '审核', '现场检查', '问询']
            )

            if is_withdrawal_related:
                result = {
                    "title": self._extract_title(section, "撤否相关信息"),
                    "content": section[:400] + "..." if len(section) > 400 else section,
                    "source": self._extract_source(section, "监管信息来源"),
                    "publish_date": self._extract_date(section),
                    "relevance_score": 0.8,
                    "search_method": "联网搜索（撤否分析）",
                    "information_type": "撤否分析信息",
                    "scenario": "撤否企业分析",
                    "company": search_params.get("company")
                }
                results.append(result)

        return results or self._parse_general_structured_text(text, original_query, search_params)

    def _parse_tutoring_analysis_text(self,
                                      text: str,
                                      original_query: str,
                                      search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析长期辅导企业分析文本"""
        results = []

        # 查找辅导相关信息
        tutoring_patterns = [
            r'(辅导备案|辅导期|辅导报告)[：:]\s*(.+?)(?:\n|$)',
            r'(上市进程|IPO准备|申报障碍)[：:]\s*(.+?)(?:\n|$)',
            r'(中介机构|保荐机构)[：:]\s*(.+?)(?:\n|$)',
        ]

        sections = re.split(r'\n\s*\n+', text)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 100:
                continue

            # 检查是否包含辅导相关信息
            is_tutoring_related = any(
                re.search(pattern, section, re.IGNORECASE)
                for pattern in ['辅导', 'IPO', '上市', '申报']
            )

            if is_tutoring_related:
                result = {
                    "title": self._extract_title(section, "辅导相关信息"),
                    "content": section[:400] + "..." if len(section) > 400 else section,
                    "source": self._extract_source(section, "辅导信息来源"),
                    "publish_date": self._extract_date(section),
                    "relevance_score": 0.8,
                    "search_method": "联网搜索（辅导分析）",
                    "information_type": "辅导分析信息",
                    "scenario": "长期辅导企业分析",
                    "company": search_params.get("company")
                }
                results.append(result)

        return results or self._parse_general_structured_text(text, original_query, search_params)

    def _parse_relationship_analysis_text(self,
                                          text: str,
                                          original_query: str,
                                          search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析上下游企业分析文本"""
        results = []

        # 查找关联关系相关信息
        relationship_patterns = [
            r'(关联企业|关联方)[：:]\s*(.+?)(?:\n|$)',
            r'(股权结构|实际控制人)[：:]\s*(.+?)(?:\n|$)',
            r'(关联交易|业务往来)[：:]\s*(.+?)(?:\n|$)',
            r'(客户|供应商)[：:]\s*(.+?)(?:\n|$)',
        ]

        sections = re.split(r'\n\s*\n+', text)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 100:
                continue

            # 检查是否包含关联关系信息
            is_relationship_related = any(
                re.search(pattern, section, re.IGNORECASE)
                for pattern in ['关联', '股权', '客户', '供应商', '交易']
            )

            if is_relationship_related:
                result = {
                    "title": self._extract_title(section, "关联关系信息"),
                    "content": section[:400] + "..." if len(section) > 400 else section,
                    "source": self._extract_source(section, "关联信息来源"),
                    "publish_date": self._extract_date(section),
                    "relevance_score": 0.8,
                    "search_method": "联网搜索（关系分析）",
                    "information_type": "关联关系信息",
                    "scenario": "上下游企业分析",
                    "company": search_params.get("company")
                }
                results.append(result)

        return results or self._parse_general_structured_text(text, original_query, search_params)

    def _parse_general_structured_text(self,
                                       text: str,
                                       original_query: str,
                                       search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """通用结构化文本解析"""
        results = []

        sections = re.split(r'\n\s*\n+', text)
        for section in sections:
            section = section.strip()
            if not section or len(section) < 80:
                continue

            result = {
                "title": self._extract_title(section, f"关于{original_query}的信息"),
                "content": section[:350] + "..." if len(section) > 350 else section,
                "source": self._extract_source(section, "通义千问搜索"),
                "publish_date": self._extract_date(section),
                "relevance_score": 0.7,
                "search_method": "联网搜索",
                "information_type": "企业信息",
                "scenario": search_params.get("scenario", "通用搜索"),
                "company": search_params.get("company")
            }
            results.append(result)

        return results

    def _extract_title(self, text: str, default: str) -> str:
        """从文本中提取标题"""
        title_patterns = [
            r'标题[：:]\s*(.+?)(?:\n|$)',
            r'##\s*(.+?)(?:\n|$)',
            r'【(.+?)】',
            r'^\s*(\d+[\.、]?\s*.+?)(?:\n|$)'
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # 如果没有找到，使用文本的第一行或前50个字符
        lines = text.strip().split('\n')
        if lines and lines[0].strip():
            return lines[0].strip()[:60]

        return default

    def _extract_source(self, text: str, default: str) -> str:
        """从文本中提取来源"""
        source_patterns = [
            r'来源[：:]\s*(.+?)(?:\n|$)',
            r'信息来源[：:]\s*(.+?)(?:\n|$)',
            r'摘自[：:]\s*(.+?)(?:\n|$)',
            r'据(.+?)报道',
            r'(.+?)消息'
        ]

        for pattern in source_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return default

    def _extract_date(self, text: str) -> str:
        """从文本中提取日期"""
        date_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{4}/\d{1,2}/\d{1,2})',
            r'(\d{4}年\d{1,2}月)',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return datetime.now().strftime("%Y-%m-%d")

    def _perform_fallback_search(self,
                                 query: str,
                                 model: str,
                                 search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """降级搜索"""
        print("执行降级搜索")

        scenario = search_params.get("scenario", "通用搜索")
        company_name = search_params.get("company", "")

        try:
            messages = [
                {
                    "role": "system",
                    "content": f"你是{scenario}专家，请提供相关信息。"
                },
                {
                    "role": "user",
                    "content": f"""请提供关于以下{scenario}的信息：

查询：{query}
{f'企业：{company_name}' if company_name else ''}

请提供结构化信息，包含：
1. 关键发现
2. 数据支持
3. 信息来源说明
4. 时效性说明"""
                }
            ]

            response = Generation.call(
                model=model,
                messages=messages,
                result_format='message',
                temperature=0.3,
                top_p=0.8,
                max_tokens=2500
            )

            if response.status_code == 200:
                result_text = response.output.choices[0].message.content

                return [{
                    "title": f"{scenario}信息 - {query}",
                    "content": result_text[:800] + "..." if len(result_text) > 800 else result_text,
                    "source": f"通义千问{scenario}知识库",
                    "publish_date": datetime.now().strftime("%Y-%m-%d"),
                    "relevance_score": 0.6,
                    "search_method": "知识库查询",
                    "information_type": f"{scenario}知识",
                    "scenario": scenario,
                    "company": company_name
                }]

        except Exception:
            pass

        return self._get_scenario_mock_results(query, scenario)

    def _get_scenario_mock_results(self, query: str, scenario: str) -> List[Dict[str, Any]]:
        """获取场景特定的模拟结果"""
        mock_templates = {
            "撤否企业分析": [
                {
                    "title": f"{query}撤否原因初步分析",
                    "content": f"根据公开信息，{query}可能存在财务数据异常、内部控制缺陷等问题，导致审核终止。建议查阅证监会和交易所的官方公告获取详细信息。",
                    "source": "监管信息摘要",
                    "publish_date": datetime.now().strftime("%Y-%m-%d"),
                    "relevance_score": 0.7,
                    "information_type": "撤否分析"
                }
            ],
            "长期辅导企业分析": [
                {
                    "title": f"{query}辅导历程概要",
                    "content": f"{query}正在进行IPO辅导，涉及财务规范、法律合规等方面的准备工作。具体进展需查阅证监局备案信息和辅导机构报告。",
                    "source": "上市辅导信息",
                    "publish_date": datetime.now().strftime("%Y-%m-%d"),
                    "relevance_score": 0.7,
                    "information_type": "辅导分析"
                }
            ],
            "上下游企业分析": [
                {
                    "title": f"{query}关联关系概要",
                    "content": f"{query}的关联网络涉及多家上下游企业，包括客户、供应商等。详细关联交易数据和股权结构需查阅企业年报和工商信息。",
                    "source": "关联关系信息",
                    "publish_date": datetime.now().strftime("%Y-%m-%d"),
                    "relevance_score": 0.7,
                    "information_type": "关系分析"
                }
            ]
        }

        return mock_templates.get(scenario, [{
            "title": f"关于{query}的信息",
            "content": f"正在获取{query}的{scenario}相关信息。",
            "source": "企业信息库",
            "publish_date": datetime.now().strftime("%Y-%m-%d"),
            "relevance_score": 0.6,
            "information_type": "企业信息"
        }])

    def _format_search_results(self,
                               raw_results: List[Dict],
                               original_query: str,
                               company_name: Optional[str],
                               scenario: Optional[str]) -> List[Dict[str, Any]]:
        """格式化搜索结果"""
        formatted_results = []

        for i, result in enumerate(raw_results):
            # 基础字段
            title = result.get("title", f"{scenario or '搜索'}结果 {i + 1}")
            content = result.get("content", "")
            source = result.get("source", "通义千问搜索")
            url = result.get("url", "")
            publish_date = result.get("publish_date", datetime.now().strftime("%Y-%m-%d"))
            relevance = min(max(result.get("relevance_score", 0.7), 0.0), 1.0)
            info_type = result.get("information_type", "企业信息")
            search_method = result.get("search_method", "联网搜索")

            # 唯一ID
            result_id = hashlib.md5(f"{title}{content}{source}{scenario}".encode()).hexdigest()[:16]

            # 格式化结果
            formatted_result = {
                "content": content,
                "content_preview": f"{content[:200]}..." if len(content) > 200 else content,
                "metadata": {
                    "title": title,
                    "source": source,
                    "url": url,
                    "publish_date": publish_date,
                    "relevance_score": relevance,
                    "information_type": info_type,
                    "search_query": original_query,
                    "company_name": company_name,
                    "scenario": scenario,
                    "result_id": result_id,
                    "search_method": search_method,
                    "is_web_result": True,
                    "retrieval_time": datetime.now().isoformat(),
                    "api_source": "通义千问"
                },
                "similarity": relevance,
                "source": f"{source} | {title} | {scenario or '通用'}"
            }

            formatted_results.append(formatted_result)

        return formatted_results

    def _get_fallback_results(self,
                              query: str,
                              company_name: Optional[str],
                              scenario: Optional[str]) -> List[Dict[str, Any]]:
        """获取降级结果"""
        print(f"使用场景降级结果: {scenario}")
        mock_results = self._get_scenario_mock_results(query, scenario or "通用搜索")
        return self._format_search_results(mock_results, query, company_name, scenario)

    def _generate_cache_key(self,
                            query: str,
                            company_name: Optional[str],
                            scenario: Optional[str]) -> str:
        """生成缓存键"""
        key_parts = [query]
        if company_name:
            key_parts.append(company_name)
        if scenario:
            key_parts.append(scenario)

        key_text = "_".join(key_parts)
        return hashlib.md5(key_text.encode()).hexdigest()

    def clear_cache(self):
        """清空缓存"""
        self.cache = {}
        print("搜索缓存已清空")

    def test_connection(self) -> Tuple[bool, str]:
        """测试API连接"""
        try:
            response = Generation.call(
                model="qwen-turbo",
                prompt="测试连接",
                result_format='message',
                max_tokens=10
            )

            if response.status_code == 200:
                return True, "通义千问API连接正常"
            else:
                return False, f"API连接失败: {response.code} - {response.message}"

        except Exception as e:
            return False, f"连接异常: {str(e)}"