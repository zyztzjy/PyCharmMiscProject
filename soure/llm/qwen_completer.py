# soure/web_search/qwen_web_completer.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re


class QwenWebCompleter:
    """通义千问智能联网补全决策器"""

    def __init__(self, web_searcher, config: Optional[Dict] = None):
        self.web_searcher = web_searcher
        self.config = config or {}

        # 配置参数
        self.enable_auto_search = self.config.get('enable_auto_search', True)
        self.min_local_docs = self.config.get('min_local_docs', 2)  # 降低阈值
        self.min_avg_similarity = self.config.get('min_avg_similarity', 0.5)  # 降低阈值
        self.max_web_results = self.config.get('max_web_results', 3)

        # 通义千问特定配置
        self.preferred_model = self.config.get('preferred_model', 'qwen-max')
        self.enable_search_cache = self.config.get('enable_search_cache', True)

        # 搜索触发规则
        self.search_triggers = {
            # 高优先级：必须联网搜索
            "mandatory": {
                "keywords": ["最新", "新闻", "近期", "动态", "2024", "2023", "今年", "本月", "本周"],
                "scenarios": ["舆情分析", "行业分析"],
                "confidence": 0.9
            },
            # 中优先级：建议联网搜索
            "recommended": {
                "keywords": ["趋势", "发展", "前景", "预测", "变化", "更新", "政策", "监管"],
                "scenarios": ["撤否企业分析", "供应链分析", "新三板企业分析"],
                "confidence": 0.7
            },
            # 低优先级：可选联网搜索
            "optional": {
                "keywords": ["概况", "简介", "基本", "一般", "概述", "历史"],
                "scenarios": ["财务分析", "长期辅导企业分析", "关系网分析"],
                "confidence": 0.5
            }
        }

        # 信息缺口检测
        self.info_gap_indicators = [
            # (模式, 置信度增加)
            (r"缺乏.*(数据|信息)", 0.3),
            (r"信息.*不足", 0.3),
            (r"没有.*找到", 0.2),
            (r"无法.*获取", 0.2),
            (r"需要.*最新", 0.4),
            (r"想要.*了解", 0.1),
            (r"查看.*详情", 0.2)
        ]

    def analyze_search_need(self,
                            query: str,
                            local_docs: List[Dict],
                            scenario: Optional[str] = None,
                            company_name: Optional[str] = None,
                            user_preference: Optional[str] = None) -> Dict[str, Any]:
        """
        全面分析搜索需求

        Returns:
            包含搜索决策的字典
        """
        analysis = {
            "should_search": False,
            "search_query": "",
            "confidence": 0.0,
            "reasons": [],
            "model": self.preferred_model,
            "search_type": "none",  # none, auto, user_requested, mandatory
            "expected_results": 0
        }

        # 1. 检查用户偏好
        if user_preference == "always":
            analysis.update({
                "should_search": True,
                "search_type": "user_requested",
                "confidence": 1.0,
                "reasons": ["用户明确要求联网搜索"]
            })

        elif user_preference == "never":
            analysis.update({
                "should_search": False,
                "search_type": "user_disabled",
                "confidence": 0.0,
                "reasons": ["用户禁用联网搜索"]
            })
            return analysis

        # 2. 检查是否必须搜索（高优先级触发）
        if self._is_mandatory_search(query, scenario):
            analysis.update({
                "should_search": True,
                "search_type": "mandatory",
                "confidence": self.search_triggers["mandatory"]["confidence"],
                "reasons": ["查询包含时间敏感关键词或属于必须联网的场景"]
            })

        # 3. 评估本地信息质量
        local_info_quality = self._assess_local_info_quality(local_docs, query)

        if not local_info_quality["is_sufficient"]:
            analysis["should_search"] = True
            analysis["search_type"] = "auto"
            analysis["confidence"] = max(analysis["confidence"], local_info_quality["confidence"])
            analysis["reasons"].extend(local_info_quality["deficiencies"])

        # 4. 检查查询特征
        query_analysis = self._analyze_query_features(query)

        if query_analysis["requires_web"]:
            analysis["should_search"] = True
            analysis["search_type"] = "auto" if analysis["search_type"] == "none" else analysis["search_type"]
            analysis["confidence"] = max(analysis["confidence"], query_analysis["confidence"])
            analysis["reasons"].append(query_analysis["reason"])

        # 5. 如果决定搜索，构建搜索查询
        if analysis["should_search"]:
            analysis["search_query"] = self._build_optimized_search_query(
                query, company_name, scenario, local_info_quality
            )
            analysis["expected_results"] = self._estimate_expected_results(
                query, company_name, scenario
            )

            # 根据置信度调整预期结果
            if analysis["confidence"] > 0.8:
                analysis["expected_results"] = min(analysis["expected_results"] + 2, self.max_web_results)

        # 6. 最终置信度调整
        analysis["confidence"] = self._adjust_final_confidence(
            analysis["confidence"],
            local_info_quality,
            query_analysis
        )

        return analysis

    def _is_mandatory_search(self, query: str, scenario: Optional[str]) -> bool:
        """检查是否必须进行联网搜索"""
        query_lower = query.lower()

        # 检查关键词
        mandatory_keywords = self.search_triggers["mandatory"]["keywords"]
        for keyword in mandatory_keywords:
            if keyword in query_lower:
                return True

        # 检查场景
        if scenario in self.search_triggers["mandatory"]["scenarios"]:
            return True

        return False

    def _assess_local_info_quality(self, local_docs: List[Dict], query: str) -> Dict[str, Any]:
        """评估本地信息质量"""
        assessment = {
            "is_sufficient": True,
            "confidence": 0.0,
            "deficiencies": [],
            "doc_count": len(local_docs),
            "avg_similarity": 0.0,
            "time_relevance": 0.0
        }

        if not local_docs:
            assessment.update({
                "is_sufficient": False,
                "confidence": 0.8,
                "deficiencies": ["本地库中未找到相关文档"]
            })
            return assessment

        # 计算平均相似度
        similarities = [doc.get("similarity", 0) for doc in local_docs]
        avg_similarity = sum(similarities) / len(similarities)
        assessment["avg_similarity"] = avg_similarity

        # 规则1: 文档数量
        if len(local_docs) < self.min_local_docs:
            assessment["is_sufficient"] = False
            assessment["confidence"] += 0.3
            assessment["deficiencies"].append(f"本地文档数量不足（{len(local_docs)} < {self.min_local_docs}）")

        # 规则2: 相似度
        if avg_similarity < self.min_avg_similarity:
            assessment["is_sufficient"] = False
            assessment["confidence"] += 0.3
            assessment["deficiencies"].append(f"平均相似度过低（{avg_similarity:.2f} < {self.min_avg_similarity}）")

        # 规则3: 时间相关性
        time_relevance = self._assess_time_relevance(local_docs, query)
        assessment["time_relevance"] = time_relevance

        if time_relevance < 0.3 and self._contains_time_keywords(query):
            assessment["is_sufficient"] = False
            assessment["confidence"] += 0.4
            assessment["deficiencies"].append("本地文档时间相关性不足")

        # 规则4: 内容覆盖度
        coverage = self._assess_content_coverage(local_docs, query)
        if coverage < 0.5:
            assessment["is_sufficient"] = False
            assessment["confidence"] += 0.2
            assessment["deficiencies"].append("本地文档内容覆盖不足")

        # 调整置信度
        if not assessment["is_sufficient"]:
            assessment["confidence"] = min(assessment["confidence"], 0.9)

        return assessment

    def _assess_time_relevance(self, docs: List[Dict], query: str) -> float:
        """评估时间相关性"""
        if not docs:
            return 0.0

        # 检查查询是否包含时间关键词
        if not self._contains_time_keywords(query):
            return 0.5  # 中性评分

        # 检查文档的时间信息
        recent_docs = 0
        current_year = datetime.now().year

        for doc in docs:
            metadata = doc.get("metadata", {})

            # 检查上传时间
            upload_time = metadata.get("upload_time", "")
            if upload_time:
                try:
                    upload_year = int(upload_time[:4]) if len(upload_time) >= 4 else 0
                    if upload_year >= current_year - 1:  # 近1-2年
                        recent_docs += 1
                except:
                    pass

            # 检查发布时间
            publish_date = metadata.get("publish_date", "")
            if publish_date:
                try:
                    publish_year = int(publish_date[:4]) if len(publish_date) >= 4 else 0
                    if publish_year >= current_year - 1:
                        recent_docs += 1
                except:
                    pass

        return recent_docs / len(docs)

    def _contains_time_keywords(self, query: str) -> bool:
        """检查是否包含时间关键词"""
        time_keywords = ["最新", "近期", "今年", "本月", "当前", "最近", "2024", "2023", "2022"]
        query_lower = query.lower()

        for keyword in time_keywords:
            if keyword in query_lower:
                return True

        return False

    def _assess_content_coverage(self, docs: List[Dict], query: str) -> float:
        """评估内容覆盖度"""
        if not docs:
            return 0.0

        # 提取查询关键词
        keywords = self._extract_query_keywords(query)

        if not keywords:
            return 0.5  # 中性评分

        # 计算关键词覆盖度
        covered_keywords = 0

        for keyword in keywords:
            keyword_found = False

            for doc in docs:
                content = doc.get("content", "").lower()
                source = doc.get("source", "").lower()

                if keyword in content or keyword in source:
                    keyword_found = True
                    break

            if keyword_found:
                covered_keywords += 1

        return covered_keywords / len(keywords) if keywords else 0.0

    def _extract_query_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        # 移除常见疑问词
        stop_words = {"的", "了", "和", "是", "在", "有", "我", "他", "她", "它", "这", "那",
                      "什么", "怎么", "如何", "为什么", "哪些", "哪个", "分析", "查询", "请问"}

        # 提取中文字词
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', query)

        # 过滤停用词
        keywords = [w for w in words if w not in stop_words]

        return keywords

    def _analyze_query_features(self, query: str) -> Dict[str, Any]:
        """分析查询特征"""
        analysis = {
            "requires_web": False,
            "confidence": 0.0,
            "reason": "",
            "features": []
        }

        query_lower = query.lower()

        # 检查各优先级关键词
        for priority, config in self.search_triggers.items():
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    analysis["requires_web"] = True
                    analysis["confidence"] = config["confidence"]
                    analysis["reason"] = f"查询包含{priority}关键词: {keyword}"
                    analysis["features"].append(f"{priority}_keyword_{keyword}")
                    return analysis  # 返回最高优先级匹配

        # 检查信息缺口模式
        for pattern, confidence_boost in self.info_gap_indicators:
            if re.search(pattern, query):
                analysis["requires_web"] = True
                analysis["confidence"] = 0.5 + confidence_boost
                analysis["reason"] = "查询表明存在信息缺口"
                analysis["features"].append("info_gap_indicator")
                break

        return analysis

    def _build_optimized_search_query(self,
                                      original_query: str,
                                      company_name: Optional[str],
                                      scenario: Optional[str],
                                      local_quality: Dict[str, Any]) -> str:
        """构建优化的搜索查询"""
        query_parts = []

        # 1. 添加企业名称（如果存在）
        if company_name:
            query_parts.append(company_name)

        # 2. 根据本地信息质量调整查询
        if local_quality["deficiencies"]:
            # 针对本地信息不足的领域进行加强
            for deficiency in local_quality["deficiencies"]:
                if "时间" in deficiency:
                    query_parts.extend(["最新", "2024年", "近期"])
                elif "数量" in deficiency or "相似度" in deficiency:
                    query_parts.append("详细")
                elif "覆盖" in deficiency:
                    # 提取原始查询的核心部分
                    core_query = self._extract_core_query(original_query)
                    if core_query:
                        query_parts.append(core_query)

        # 3. 添加场景特定关键词
        scenario_keywords = {
            "撤否企业分析": "撤否原因 审核问题 证监会",
            "长期辅导企业分析": "辅导备案 IPO",
            "上下游企业分析": "关联企业 投资关系"
        }

        if scenario in scenario_keywords:
            query_parts.append(scenario_keywords[scenario])

        # 4. 添加原始查询的核心部分
        core_query = self._extract_core_query(original_query)
        if core_query and core_query not in query_parts:
            query_parts.append(core_query)


        # 构建最终查询
        optimized_query = " ".join([p for p in query_parts if p])

        print(f"构建依据: 企业={company_name}, 场景={scenario}, 本地质量={local_quality['confidence']:.2f}")

        return optimized_query

    def _extract_core_query(self, query: str) -> str:
        """提取查询的核心部分"""
        # 移除常见疑问词和修饰词
        remove_patterns = [
            r'分析一下', r'请问', r'什么', r'如何', r'怎样', r'为什么',
            r'哪些', r'哪个', r'查询', r'搜索', r'了解', r'查看',
            r'的', r'了', r'和', r'是', r'在', r'有'
        ]

        core_query = query
        for pattern in remove_patterns:
            core_query = re.sub(pattern, '', core_query)

        # 清理多余空格
        core_query = re.sub(r'\s+', ' ', core_query).strip()

        return core_query if len(core_query) >= 2 else query

    def _estimate_expected_results(self,
                                   query: str,
                                   company_name: Optional[str],
                                   scenario: Optional[str]) -> int:
        """预估预期结果数量"""
        base_count = 3

        # 有企业名称通常能获得更多结果
        if company_name and len(company_name) >= 2:
            base_count += 1

        # 特定场景可能结果更多
        high_result_scenarios = ["舆情分析", "行业分析", "财务分析"]
        if scenario in high_result_scenarios:
            base_count += 1

        # 查询长度影响
        if len(query) >= 10:
            base_count += 1

        return min(base_count, self.max_web_results)

    def _adjust_final_confidence(self,
                                 current_confidence: float,
                                 local_quality: Dict[str, Any],
                                 query_analysis: Dict[str, Any]) -> float:
        """调整最终置信度"""
        confidence = current_confidence

        # 基于本地信息质量调整
        if not local_quality["is_sufficient"]:
            # 本地信息越差，搜索置信度越高
            deficiency_boost = 1.0 - min(local_quality["avg_similarity"], 0.7)
            confidence += deficiency_boost * 0.2

        # 基于查询特征调整
        if query_analysis["requires_web"]:
            confidence = max(confidence, query_analysis["confidence"])

        # 确保在合理范围内
        confidence = min(max(confidence, 0.0), 1.0)

        return round(confidence, 2)