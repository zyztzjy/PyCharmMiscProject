# soure/scenarios/scenario_config.py
"""
场景配置模块 - 统一管理所有分析场景的配置
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """场景类型枚举"""
    WITHDRAWAL = "撤否企业分析"
    TUTORING = "长期辅导企业分析"
    RELATIONSHIP = "上下游企业分析"


@dataclass
class ScenarioRule:
    """场景规则数据类"""
    name: str
    display_name: str
    description: str
    framework: str
    focus_areas: List[str]
    risk_metrics: List[Dict[str, Any]]
    analysis_template: Dict[str, Any]
    output_requirements: List[str]
    icon: str = "📊"
    color: str = "blue"


class ScenarioConfig:
    """场景配置管理器"""

    # 场景关键词映射
    SCENARIO_KEYWORDS = {
        ScenarioType.WITHDRAWAL: ["撤否", "撤销", "终止审核", "审核终止", "撤回", "ipo失败", "上市失败", "撤否原因"],
        ScenarioType.TUTORING: ["长期辅导", "辅导备案", "辅导期", "辅导超过", "辅导时间", "辅导过程", "辅导企业"],
        ScenarioType.RELATIONSHIP: [
            "关系网", "关联企业", "股权结构", "控股", "持股", "关联方", "关联交易",
            "上下游", "供应商", "客户", "供应链", "产业链", "业务往来", "关联关系"
        ],
    }

    # 企业识别模式
    COMPANY_PATTERNS = [
        r'(\d{6})',  # 股票代码（6位数字）
        r'([\u4e00-\u9fa5]{2,10}?(股份|科技|电子|集团|有限公司|公司|证券))',  # 中文企业名称
    ]

    @classmethod
    def get_all_scenarios(cls) -> Dict[ScenarioType, ScenarioRule]:
        """获取所有场景配置"""
        return {
            ScenarioType.WITHDRAWAL: ScenarioRule(
                name="withdrawal",
                display_name="撤否企业分析",
                description="分析被撤否企业的原因、问题和整改建议",
                framework="三维度分析框架（企业层面-中介机构层面-监管审核层面）",
                focus_areas=[
                    "现场检查经历及问题",
                    "审核问询重点及回复质量",
                    "财务数据真实性及异常",
                    "内部控制有效性缺陷",
                    "持续盈利能力疑虑",
                    "信息披露合规性问题",
                    "行业政策与定位匹配度",
                    "关联交易与独立性"
                ],
                risk_metrics=[
                    {"name": "现场检查风险指数", "weight": 0.3},
                    {"name": "财务异常指标数", "weight": 0.25},
                    {"name": "问询回复质量评分", "weight": 0.2},
                    {"name": "内控缺陷严重程度", "weight": 0.15},
                    {"name": "行业监管风险", "weight": 0.1}
                ],
                analysis_template={
                    "sections": [
                        {
                            "title": "撤否原因深度剖析",
                            "subsections": [
                                "主要撤否原因归类分析",
                                "关键问题发生时间线与影响",
                                "同类企业对比参考"
                            ]
                        },
                        {
                            "title": "审核过程还原",
                            "subsections": [
                                "审核轮次与问询重点演变",
                                "企业回复与整改措施评估",
                                "监管关注点变化趋势"
                            ]
                        },
                        {
                            "title": "风险评估与预警",
                            "subsections": [
                                "撤否风险等级综合评估",
                                "问题可整改性分析",
                                "重新申报时间预测"
                            ]
                        }
                    ]
                },
                output_requirements=[
                    "必须明确标注信息来源（本地文档/网络信息）",
                    "每个分析结论需附带证据支持",
                    "风险提示需量化评估",
                    "提供具体整改建议",
                    "包含重新上市可行性分析"
                ],
                icon="⚠️",
                color="red"
            ),

            ScenarioType.TUTORING: ScenarioRule(
                name="tutoring",
                display_name="长期辅导企业分析",
                description="分析长期辅导企业的上市障碍和可行性",
                framework="三阶段评估模型（辅导进度-障碍诊断-上市可行性）",
                focus_areas=[
                    "辅导备案时间与进度",
                    "辅导机构变更及原因",
                    "财务数据波动与趋势",
                    "法律合规问题整改",
                    "行业竞争地位变化",
                    "募投项目合理性",
                    "实际控制人稳定性",
                    "信息披露一致性"
                ],
                risk_metrics=[
                    {"name": "辅导停滞风险指数", "weight": 0.35},
                    {"name": "财务规范度评分", "weight": 0.25},
                    {"name": "法律障碍严重程度", "weight": 0.2},
                    {"name": "行业前景匹配度", "weight": 0.15},
                    {"name": "团队稳定性风险", "weight": 0.05}
                ],
                analysis_template={
                    "sections": [
                        {
                            "title": "辅导历程诊断",
                            "subsections": [
                                "辅导阶段划分与关键节点",
                                "主要障碍问题时间线",
                                "中介机构工作质量评估"
                            ]
                        },
                        {
                            "title": "上市障碍分析",
                            "subsections": [
                                "财务规范性问题清单",
                                "法律合规风险点",
                                "业务独立性缺陷",
                                "行业定位匹配度"
                            ]
                        },
                        {
                            "title": "可行性评估",
                            "subsections": [
                                "近期上市可能性预测",
                                "必要整改措施建议",
                                "替代方案分析（并购/新三板等）"
                            ]
                        }
                    ]
                },
                output_requirements=[
                    "按时间线整理辅导历程",
                    "量化评估各项障碍严重程度",
                    "提供分阶段的整改路线图",
                    "预测不同情景下的时间表"
                ],
                icon="📅",
                color="orange"
            ),

            ScenarioType.RELATIONSHIP: ScenarioRule(
                name="relationship",
                display_name="上下游企业分析",
                description="分析上下游关联企业的关系网络和风险传导",
                framework="四层次关联分析（股权-业务-人员-资金）",
                focus_areas=[
                    "股权结构穿透与实际控制人",
                    "关联方交易规模与公允性",
                    "客户供应商集中度风险",
                    "同业竞争与利益冲突",
                    "资金往来与担保情况",
                    "人员兼职与共同投资",
                    "技术合作与知识产权",
                    "历史重组与业务剥离"
                ],
                risk_metrics=[
                    {"name": "关联交易依赖度", "weight": 0.3},
                    {"name": "客户集中风险指数", "weight": 0.25},
                    {"name": "同业竞争严重程度", "weight": 0.2},
                    {"name": "资金占用风险", "weight": 0.15},
                    {"name": "人员独立性风险", "weight": 0.1}
                ],
                analysis_template={
                    "sections": [
                        {
                            "title": "关联网络图谱分析",
                            "subsections": [
                                "股权控制关系可视化分析",
                                "业务往来依赖度评估",
                                "关键人员重叠情况"
                            ]
                        },
                        {
                            "title": "风险传导机制",
                            "subsections": [
                                "财务风险传导路径",
                                "经营风险关联影响",
                                "合规风险连带效应"
                            ]
                        },
                        {
                            "title": "独立性整改评估",
                            "subsections": [
                                "关联交易规范方案",
                                "业务资产重组建议",
                                "人员机构分离措施"
                            ]
                        }
                    ]
                },
                output_requirements=[
                    "提供关联关系结构图描述",
                    "量化分析各项关联指标",
                    "评估风险传导的可能性与影响",
                    "提供具体的独立性整改方案"
                ],
                icon="🔗",
                color="purple"
            )
        }

    @classmethod
    def extract_scenario_and_company(cls, query: str) -> Dict[str, Any]:
        """从查询中智能提取场景和企业信息"""
        import re

        result = {
            "scenario": None,
            "scenario_name": "未识别到场景",
            "company_code": None,
            "company_name": None
        }

        # 1. 查找企业
        for pattern in cls.COMPANY_PATTERNS:
            matches = re.findall(pattern, query)
            if matches:
                if pattern == r'(\d{6})':  # 股票代码
                    result["company_code"] = matches[0]
                else:  # 中文企业名称
                    for match in matches:
                        if isinstance(match, tuple):
                            result["company_name"] = match[0]
                        else:
                            result["company_name"] = match
                        break
                if result["company_name"]:
                    result["company_code"] = result["company_name"]
                    break

        # 2. 查找场景
        query_lower = query.lower()
        found_scenario = False

        for scenario_type, keywords in cls.SCENARIO_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scenario_rule = cls.get_all_scenarios().get(scenario_type)
                    result["scenario"] = scenario_type
                    result["scenario_name"] = scenario_rule.display_name
                    found_scenario = True
                    break
            if found_scenario:
                break

        # 3. 如果未识别到场景，则尝试基于企业信息推断
        if not found_scenario and result["company_name"]:
            # 可以根据企业类型或历史记录推断场景
            if any(x in result["company_name"] for x in ["证券", "银行", "保险", "基金"]):
                # 金融机构通常需要撤否分析或关系分析
                result["scenario"] = ScenarioType.WITHDRAWAL
                result["scenario_name"] = cls.get_all_scenarios()[ScenarioType.WITHDRAWAL].display_name

        return result

    @classmethod
    def get_scenario_rule(cls, scenario_type: ScenarioType) -> Optional[ScenarioRule]:
        """获取特定场景的规则"""
        if scenario_type:
            return cls.get_all_scenarios().get(scenario_type)
        return None

    @classmethod
    def get_scenario_by_name(cls, scenario_name: str) -> Optional[ScenarioRule]:
        """通过场景名称获取规则"""
        for scenario_type, rule in cls.get_all_scenarios().items():
            if rule.display_name == scenario_name:
                return rule
        return None

    @classmethod
    def get_default_scenario(cls) -> ScenarioRule:
        """获取默认场景（撤否企业分析）"""
        return cls.get_all_scenarios()[ScenarioType.WITHDRAWAL]