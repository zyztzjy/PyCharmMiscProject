# soure/scenarios/llm_extractor.py
"""
使用大模型进行企业和场景识别
"""
import json
import re
from typing import Dict, List, Optional, Any

from soure.llm.scenario_config import ScenarioConfig, ScenarioType
import dashscope

class LLMExtractor:
    """使用大模型的智能识别器"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key

        # 强制映射所有场景到专业分析场景，不保留自定义
        self.scenario_normalization = {
            # 撤否相关场景 -> ScenarioType.WITHDRAWAL
            "撤否企业分析": ScenarioType.WITHDRAWAL,
            "撤否分析": ScenarioType.WITHDRAWAL,
            "撤单分析": ScenarioType.WITHDRAWAL,
            "撤回分析": ScenarioType.WITHDRAWAL,
            "终止审核分析": ScenarioType.WITHDRAWAL,
            "ipo撤单分析": ScenarioType.WITHDRAWAL,
            "ipo撤回分析": ScenarioType.WITHDRAWAL,
            "上市失败分析": ScenarioType.WITHDRAWAL,
            "审核终止分析": ScenarioType.WITHDRAWAL,
            "ipo终止分析": ScenarioType.WITHDRAWAL,
            "ipo分析": ScenarioType.WITHDRAWAL,  # IPO分析映射到撤否
            "上市分析": ScenarioType.WITHDRAWAL,  # 上市分析映射到撤否
            "撤否风险评估分析": "撤否风险评估分析",  # 特殊处理撤否风险评估

            # 辅导相关场景 -> ScenarioType.TUTORING
            "长期辅导企业分析": ScenarioType.TUTORING,
            "辅导分析": ScenarioType.TUTORING,
            "ipo辅导分析": ScenarioType.TUTORING,
            "上市辅导分析": ScenarioType.TUTORING,

            # 上下游相关场景 -> ScenarioType.RELATIONSHIP
            "上下游企业分析": ScenarioType.RELATIONSHIP,
            "关联企业分析": ScenarioType.RELATIONSHIP,
            "关系网分析": ScenarioType.RELATIONSHIP,
            "股权结构分析": ScenarioType.RELATIONSHIP,
            "关联方分析": ScenarioType.RELATIONSHIP,
            "供应链分析": ScenarioType.RELATIONSHIP,
            "产业链分析": ScenarioType.RELATIONSHIP,

            # 财务分析 -> 保留为财务分析（虽然不是ScenarioType，但会显示特定框架）
            "财务分析": "财务分析",
            "财报分析": "财务分析",
            "财务数据": "财务分析",
            "财务报表": "财务分析",

            # 行业分析 -> 保留为行业分析
            "行业分析": "行业分析",
            "市场分析": "行业分析",
            "竞争分析": "行业分析",
            "趋势分析": "行业分析",

            # 默认映射 - 所有其他情况都映射到撤否企业分析
            "默认": ScenarioType.WITHDRAWAL,
            "通用": ScenarioType.WITHDRAWAL,
            "综合": ScenarioType.WITHDRAWAL,
            "基础": ScenarioType.WITHDRAWAL,
            "一般": ScenarioType.WITHDRAWAL
        }

    def _validate_and_standardize(self, result: Dict, original_query: str) -> Dict[str, any]:
        """验证和标准化结果"""
        validated = {
            "company_code": None,
            "company_name": None,
            "scenario": ScenarioType.WITHDRAWAL,  # 默认设为撤否企业分析
            "scenario_name": "撤否企业分析",  # 默认设为撤否企业分析
            "confidence": {
                "company": 0.0,
                "scenario": 0.0
            },
            "reasoning": result.get("reasoning", ""),
            "source": "llm_extraction",
            "original_scenario": result.get("scenario", "撤否企业分析")  # 保存原始场景名称
        }

        # 处理企业名称
        company_name = result.get("company_name", "")
        company_code = result.get("company_code", "")

        if company_name and company_name not in ["", "无", "未知"]:
            cleaned_name = self._clean_company_name(company_name, original_query)
            if cleaned_name:
                validated["company_name"] = cleaned_name
                validated["confidence"]["company"] = min(result.get("confidence", 0.8), 0.95)

        if company_code and company_code not in ["", "无", "未知"]:
            if self._is_valid_stock_code(company_code):
                validated["company_code"] = company_code
                if not validated["company_name"]:
                    validated["confidence"]["company"] = min(result.get("confidence", 0.8), 0.95)

        # 处理场景 - 强制映射到专业场景
        scenario_name = result.get("scenario", "撤否企业分析")
        validated["original_scenario"] = scenario_name

        # 1. 首先尝试精确匹配
        normalized_scenario = self._normalize_scenario_name(scenario_name, original_query)

        if isinstance(normalized_scenario, ScenarioType):
            # 映射到标准的场景类型
            scenario_rule = ScenarioConfig.get_scenario_rule(normalized_scenario)
            validated["scenario"] = normalized_scenario
            validated["scenario_name"] = scenario_rule.display_name
        elif normalized_scenario in ["财务分析", "行业分析", "撤否风险评估分析"]:
            # 虽然不是ScenarioType，但保留为特定分析类型
            validated["scenario_name"] = normalized_scenario
        else:
            # 默认使用撤否企业分析
            scenario_rule = ScenarioConfig.get_scenario_rule(ScenarioType.WITHDRAWAL)
            validated["scenario"] = ScenarioType.WITHDRAWAL
            validated["scenario_name"] = scenario_rule.display_name

        validated["confidence"]["scenario"] = min(result.get("confidence", 0.8), 0.95)

        return validated

    def _normalize_scenario_name(self, scenario_name: str, original_query: str):
        """标准化场景名称"""
        # 清理场景名称
        cleaned_name = scenario_name.strip().lower()
        query_lower = original_query.lower()

        # 1. 尝试精确匹配
        for key, value in self.scenario_normalization.items():
            if key.lower() in cleaned_name or cleaned_name in key.lower():
                return value

        # 2. 基于查询内容的关键词匹配
        # 检查明确的撤否相关关键词（表示已撤否）
        explicit_withdrawal_keywords = ["撤否", "撤单", "撤回", "终止审核", "撤回申请", "ipo失败", "上市失败", "被否", "终止"]
        if any(keyword in query_lower for keyword in explicit_withdrawal_keywords):
            return ScenarioType.WITHDRAWAL

        # 检查暗示可能撤否的关键词（表示存在撤否可能）
        elif any(keyword in query_lower for keyword in ["撤否可能", "可能撤否", "撤否风险", "撤否概率", "撤否预测", "撤否趋势", 
                                       "撤否预警", "撤否评估", "撤否可能性", "会撤否", "会被否决", 
                                       "审核风险", "IPO风险", "上市风险", "能否通过", "通过率", "审核概率"]):
            return "撤否风险评估分析"

        # 检查辅导相关关键词
        elif any(keyword in query_lower for keyword in ["辅导", "长期辅导", "辅导期", "辅导备案", "辅导过程", "上市辅导"]):
            return ScenarioType.TUTORING

        # 检查上下游相关关键词
        elif any(keyword in query_lower for keyword in ["上下游", "关联", "关系网", "股权", "控制", "供应链", "产业链", "客户", "供应商",
                                 "合作伙伴"]):
            return ScenarioType.RELATIONSHIP

        # 检查财务相关关键词
        elif any(keyword in query_lower for keyword in ["财务", "报表", "利润", "收入", "成本", "资产", "负债", "现金流", "毛利率", "净利率"]):
            return "财务分析"

        # 检查行业相关关键词
        elif any(keyword in query_lower for keyword in ["行业", "市场", "竞争", "趋势", "发展", "前景", "政策", "监管", "市场规模", "竞争格局"]):
            return "行业分析"

        # 3. 检查IPO相关关键词
        elif any(keyword in query_lower for keyword in ["ipo", "上市", "创业板", "科创板", "主板", "招股书", "申报", "审核", "证监会"]):
            return ScenarioType.WITHDRAWAL  # IPO相关默认映射到撤否分析

        # 4. 默认返回撤否企业分析
        return ScenarioType.WITHDRAWAL

    def _build_extraction_prompt(self, query: str) -> str:
        """构建识别提示词 - 强制要求选择专业场景"""
        return f"""你是一个专业的企业信息分析专家。请从用户查询中提取以下信息：

用户查询："{query}"

请提取：
1. 企业名称/代码（如果有）：明确的企业名称或股票代码
2. 分析场景类型：必须从以下专业分析场景中选择最匹配的一个：
   - 撤否企业分析（分析已被撤否、撤单、终止审核、撤回IPO申请的企业）
   - 撤否风险评估分析（评估企业IPO撤否可能性的风险分析）
   - 长期辅导企业分析（分析长期处于IPO辅导期的企业）
   - 上下游企业分析（分析企业的关联方、供应商、客户关系）
   - 财务分析（分析企业财务数据、报表、盈利能力）
   - 行业分析（分析行业趋势、竞争格局、市场前景）

重要规则：
1. 禁止选择"自定义分析"、"一般分析"、"综合分折"等非专业场景
2. 如果查询提到明确的撤否结果（如"已撤否"、"撤否了"、"撤单了"、"终止审核"、"撤回申请"、"ipo失败"、"上市失败"、"撤否企业"、"已撤否企业"），请选择"撤否企业分析"
3. 如果查询询问撤否可能性（如"撤否可能"、"可能撤否"、"撤否风险"、"撤否概率"、"能否通过审核"、"通过率"、"撤否预测"、"撤否倾向"），选择"撤否风险评估分析"，并在推理中说明这是风险评估而非已发生的撤否
4. 如果查询提到辅导、上市前准备，请选择"长期辅导企业分析"
5. 如果查询提到关联、上下游、供应链，请选择"上下游企业分析"
6. 如果查询提到财务数据、报表、利润，请选择"财务分析"
7. 如果查询提到行业、市场、竞争，请选择"行业分析"
8. 企业名称只提取核心名称，不要包含"分析"、"查询"等词语

请以JSON格式返回结果，格式如下：
{{
  "company_name": "提取的企业名称（如：欣强电子、300745等）",
  "company_code": "股票代码（如果有）",
  "scenario": "场景名称（必须从上述五个专业场景中选择）",
  "confidence": 0.95,
  "reasoning": "提取的理由说明，特别是为什么选择这个专业场景，以及是否是评估撤否可能性还是分析已撤否情况"
}}

现在请分析查询并返回JSON结果："""


    def extract_company_and_scenario(self, query: str) -> Dict[str, any]:
        """使用大模型提取企业信息和场景"""

        # 构建提示词
        prompt = self._build_extraction_prompt(query)

        try:
            # 调用通义千问API
            response = dashscope.Generation.call(
                model="qwen-turbo",  # 使用轻量模型，响应更快
                prompt=prompt,
                temperature=0.1,  # 低温度以获得更确定的结果
                top_p=0.9,
                result_format='message',
                max_tokens=500
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                return self._parse_llm_response(content, query)
            else:
                print(f"大模型识别失败: {response.code} - {response.message}")
                return self._get_fallback_result(query)

        except Exception as e:
            print(f"大模型调用失败: {e}")
            return self._get_fallback_result(query)



    def _parse_llm_response(self, response_text: str, original_query: str) -> Dict[str, any]:
        """解析大模型响应"""
        try:
            # 尝试解析JSON
            json_match = None
            import re
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                json_str = match.group()
                result = json.loads(json_str)

                # 验证和标准化结果
                validated_result = self._validate_and_standardize(result, original_query)
                return validated_result
            else:
                print(f"未找到JSON格式的响应: {response_text[:100]}")
                return self._get_fallback_result(original_query)

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return self._get_fallback_result(original_query)
        except Exception as e:
            print(f"解析响应失败: {e}")
            return self._get_fallback_result(original_query)



    def _clean_company_name(self, company_name: str, original_query: str) -> str:
        """清理企业名称"""
        # 移除常见前缀
        prefixes = ["分析", "查询", "了解", "查找", "搜索", "请问", "帮我", "我想"]
        cleaned = company_name

        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # 移除常见后缀
        suffixes = ["公司", "的", "企业", "集团"]
        for suffix in suffixes:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[:-len(suffix)].strip()

        # 如果清理后太短或包含分析词语，尝试从原始查询中提取
        if len(cleaned) < 2 or any(word in cleaned for word in ["分析", "关系", "查询"]):
            # 尝试从原始查询提取
            import re
            # 查找可能的企业名称模式
            patterns = [
                r'([\u4e00-\u9fa5]{2,6}?(?:股份|科技|电子|集团|有限|公司))',
                r'([\u4e00-\u9fa5]{2,8})的',  # "XX的"
                r'关于([\u4e00-\u9fa5]{2,8})'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, original_query)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            candidate = match[0]
                        else:
                            candidate = match

                        # 检查候选名称是否合理
                        if len(candidate) >= 2 and not any(
                                word in candidate for word in ["分析", "查询", "关系", "什么", "如何"]):
                            return candidate

        return cleaned if len(cleaned) >= 2 else None

    def _is_valid_stock_code(self, code: str) -> bool:
        """验证股票代码格式"""
        # A股代码：6位数字，以6、0、3、9开头
        if len(code) == 6 and code.isdigit():
            first_digit = code[0]
            if first_digit in ['6', '0', '3', '9']:
                return True

        # 新三板代码：8位数字
        if len(code) == 8 and code.isdigit():
            return True

        # 港股代码：字母+数字
        if re.match(r'^[A-Z]{2}\d{4}$', code):
            return True

        # 美股代码：1-5位字母
        if re.match(r'^[A-Z]{1,5}$', code):
            return True

        return False

    def _get_fallback_result(self, query: str) -> Dict[str, any]:
        """获取降级结果（当大模型失败时）"""
        # 使用简单的正则匹配作为降级方案
        import re

        result = {
            "company_code": None,
            "company_name": None,
            "scenario": ScenarioType.CUSTOM,
            "scenario_name": "自定义分析",
            "confidence": {
                "company": 0.0,
                "scenario": 0.0
            },
            "reasoning": "使用降级识别方案",
            "source": "fallback"
        }

        # 简单提取企业名称
        company_patterns = [
            r'([\u4e00-\u9fa5]{2,6}?(?:股份|科技|电子|集团|有限|公司))',
            r'(\d{6})',  # 股票代码
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, query)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        candidate = match[0]
                    else:
                        candidate = match

                    if pattern == r'(\d{6})':
                        result["company_code"] = candidate
                        result["confidence"]["company"] = 0.7
                    else:
                        result["company_name"] = candidate
                        result["confidence"]["company"] = 0.6
                    break

        # 简单场景识别
        scenario_keywords = {
            "撤否风险评估": ("撤否风险评估分析", "撤否风险评估分析"),  # 特殊处理撤否风险评估
            "撤否": ("撤否企业分析", ScenarioType.WITHDRAWAL),
            "辅导": ("长期辅导企业分析", ScenarioType.TUTORING),
            "关联": ("上下游企业分析", ScenarioType.RELATIONSHIP),
            "上下游": ("上下游企业分析", ScenarioType.RELATIONSHIP),
        }

        for keyword, (scenario_name, scenario_type) in scenario_keywords.items():
            if keyword in query:
                if isinstance(scenario_type, ScenarioType):
                    scenario_rule = ScenarioConfig.get_scenario_rule(scenario_type)
                    result["scenario"] = scenario_type
                    result["scenario_name"] = scenario_rule.display_name
                elif scenario_type == "撤否风险评估分析":
                    result["scenario_name"] = scenario_name
                else:
                    result["scenario_name"] = scenario_name
                result["confidence"]["scenario"] = 0.7
                break

        # 额外检查：如果查询包含"可能撤否"或"撤否可能"等关键词，明确为撤否风险评估
        if any(keyword in query for keyword in ["撤否可能", "可能撤否", "撤否风险", "撤否概率", "撤否预测"]):
            result["scenario_name"] = "撤否风险评估分析"
            result["confidence"]["scenario"] = 0.9

        return result