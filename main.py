# soure/main.py
import tempfile
from typing import Dict, List, Optional
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import sys
import os
import yaml
import hashlib
import time

from matplotlib import pyplot as plt
from soure.document.document_processor import PDFProcessor
from soure.llm.intel_extractor import LLMExtractor
from soure.llm.scenario_config import ScenarioConfig, ScenarioType, ScenarioRule

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from soure.embedding.vectorizer_qwen import QwenVectorizer
from soure.rag.qwen_rag_processor import QwenRAGProcessor


st.set_page_config(
    page_title="企业智能分析助手",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0


def generate_unique_key(prefix: str, data: Optional[Dict] = None) -> str:
    """生成唯一的key"""
    st.session_state.message_counter += 1
    counter = st.session_state.message_counter

    if data:
        timestamp = data.get("timestamp", str(time.time()))
        query = data.get("query", "")
        key_str = f"{prefix}_{timestamp}_{query}_{counter}"
    else:
        key_str = f"{prefix}_{time.time()}_{counter}"

    return f"{prefix}_{hashlib.md5(key_str.encode()).hexdigest()[:8]}"


@st.cache_resource
def init_system():
    """初始化系统组件"""
    try:
        with open("config/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        api_key = "sk-6892cc65b78941e7a6981cae25997c0b"
        if not api_key:
            st.error("请设置DASHSCOPE_API_KEY环境变量或Streamlit secrets")
            st.stop()

        vectorizer = QwenVectorizer(config)
        rag_processor = QwenRAGProcessor(
            vectorizer=vectorizer,
            api_key=api_key,
            model=config['llm'].get('model', 'qwen-max')
        )

        # 初始化大模型识别器
        llm_extractor = LLMExtractor(api_key)

        return {
            "config": config,
            "vectorizer": vectorizer,
            "rag_processor": rag_processor,
            "llm_extractor": llm_extractor,  # 替换为新的识别器
            "api_key": api_key
        }
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.stop()


def add_message(role: str, content: str, analysis_result: Optional[Dict] = None):
    """添加消息到聊天历史"""
    if analysis_result:
        analysis_result["unique_id"] = generate_unique_key("analysis", {
            "timestamp": analysis_result.get("timestamp", str(datetime.now().timestamp())),
            "query": analysis_result.get("query", "")
        })

    message_id = generate_unique_key("msg", {"content": content, "role": role})

    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "analysis_result": analysis_result,
        "unique_id": message_id
    })

    if analysis_result:
        st.session_state.analysis_history.append(analysis_result)


def display_message(message: Dict):
    """显示单条消息"""
    role = message["role"]
    content = message["content"]
    analysis_result = message.get("analysis_result")
    message_id = message.get("unique_id", "")

    with st.chat_message(role):
        st.markdown(content)

        if analysis_result and role == "assistant":
            display_analysis_details(analysis_result, message_id)


def display_scenario_specific_analysis(response: Dict, scenario_rule: ScenarioRule, company_code: str):
    """根据场景显示特定的分析内容"""

    # 显示场景标题和框架
    st.subheader(f"{scenario_rule.icon} {scenario_rule.display_name}")
    st.caption(f"分析框架: {scenario_rule.framework}")

    # 创建场景特定组件
    create_scenario_specific_components(scenario_rule, response, company_code)


    # 显示详细分析内容
    st.subheader("详细分析内容")

    # 本地文档分析
    detailed_analysis = response.get("detailed_analysis", {})
    if detailed_analysis and isinstance(detailed_analysis, dict):
        local_based = detailed_analysis.get("local_based", [])
        if local_based and isinstance(local_based, list):
            with st.expander("基于本地文档的分析", expanded=False):
                for i, item in enumerate(local_based):
                    if isinstance(item, str):
                        st.markdown(f"**{i + 1}.** {item}")

        # 网络信息分析
        web_based = detailed_analysis.get("web_based", [])
        if web_based and isinstance(web_based, list):
            with st.expander("基于网络信息的分析", expanded=False):
                for i, item in enumerate(web_based):
                    if isinstance(item, str):
                        st.markdown(f"**{i + 1}.** {item}")

        # 综合分析
        integrated = detailed_analysis.get("integrated", [])
        if integrated and isinstance(integrated, list):
            with st.expander("综合分析结论", expanded=False):
                for i, item in enumerate(integrated):
                    if isinstance(item, str):
                        st.markdown(f"**{i + 1}.** {item}")
    else:
        analysis_list = response.get("analysis", [])
        if analysis_list and isinstance(analysis_list, list):
            with st.expander("分析要点", expanded=False):
                for i, item in enumerate(analysis_list):
                    if isinstance(item, str):
                        st.markdown(f"**{i + 1}.** {item}")
        else:
            st.info("暂无详细分析内容")


def display_analysis_details(result: Dict, message_id: str = ""):
    """显示分析结果的详细信息"""
    if not isinstance(result, dict):
        st.warning("分析结果格式异常")
        return

    # 生成唯一key
    unique_key = result.get("unique_id", message_id)
    if not unique_key:
        unique_key = generate_unique_key("analysis", result)

    response = result.get("response", {})
    retrieval_stats = result.get("retrieval_stats", {})
    scenario_name = result.get("scenario_name", "自定义分析")
    company_code = result.get("company_code", "未识别到企业")

    # 获取场景规则
    scenario_rule = ScenarioConfig.get_scenario_by_name(scenario_name)
    if not scenario_rule:
        # 如果没有识别到场景，使用撤否企业分析作为默认
        scenario_rule = ScenarioConfig.get_default_scenario()
        scenario_name = scenario_rule.display_name

    # 创建可折叠的详细信息区域
    with st.expander("查看分析详情", expanded=False):
        if response and isinstance(response, dict):
            display_scenario_specific_analysis(response, scenario_rule, company_code)

            st.divider()

            # 企业风险提示
            st.subheader("企业风险提示")
            risk_assessment = response.get("risk_assessment", {})

            if risk_assessment and isinstance(risk_assessment, dict):
                identified_risks = risk_assessment.get("identified_risks", [])
                risk_level = risk_assessment.get("risk_level", "未知")
                rationale = risk_assessment.get("rationale", "")

                col1, col2 = st.columns([1, 3])
                with col1:
                    risk_color = {
                        "高": "🔴",
                        "中": "🟡",
                        "低": "🟢",
                        "未知": "⚪"
                    }
                    risk_icon = risk_color.get(risk_level, "⚪")
                    st.metric("风险级别", f"{risk_icon} {risk_level}")

                with col2:
                    if rationale:
                        st.caption(f"风险评估依据: {rationale}")

                if identified_risks and isinstance(identified_risks, list):
                    with st.expander(f"具体风险点 ({len(identified_risks)}个)", expanded=True):
                        for i, risk in enumerate(identified_risks):
                            if isinstance(risk, str):
                                st.warning(f"**• 风险{i + 1}:** {risk}")
                else:
                    st.info("暂无具体风险信息")
            else:
                risks_list = response.get("risks", [])
                if risks_list and isinstance(risks_list, list):
                    for i, risk in enumerate(risks_list):
                        if isinstance(risk, str):
                            st.warning(f"**• 风险{i + 1}:** {risk}")
                else:
                    st.info("暂无风险提示")

        st.divider()

        # 参考文档
        source_docs = result.get("source_documents", [])
        if source_docs and isinstance(source_docs, list):
            with st.expander(f"参考文档 ({len(source_docs)}个)", expanded=False):
                for i, doc in enumerate(source_docs):
                    doc_title = "未知文档"
                    doc_content = "无内容预览"
                    doc_source = "未知来源"

                    if isinstance(doc, dict):
                        doc_title = doc.get('title', doc.get('source', '未知文档'))
                        doc_content = doc.get('content_preview', doc.get('content', '无预览内容'))
                        doc_source = doc.get('source', '未知来源')
                    elif isinstance(doc, str):
                        doc_title = f"文档 {i + 1}"
                        doc_content = doc[:200] + "..." if len(doc) > 200 else doc
                        doc_source = "文本内容"

                    with st.expander(f"{doc_title}", expanded=False):
                        st.caption(f"**来源:** {doc_source}")
                        st.write(doc_content)
        else:
            st.info("暂无参考文档")

        # 操作按钮
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            export_key = f"export_{unique_key}"
            if st.button("导出报告", key=export_key, width='stretch'):
                report = {
                    "生成时间": result.get("timestamp", datetime.now().isoformat()),
                    "分析场景": scenario_rule.display_name,
                    "目标企业": result.get("company_code"),
                    "查询内容": result.get("query"),
                    "分析结果": result.get("response", {}),
                    "检索统计": result.get("retrieval_stats", {}),
                    "参考文档": result.get("source_documents", [])
                }
                st.download_button(
                    label="下载JSON报告",
                    data=json.dumps(report, ensure_ascii=False, indent=2),
                    file_name=f"企业分析报告_{company_code or '未知企业'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key=f"download_{unique_key}"
                )

def clear_chat_history():
    """清空聊天历史"""
    st.session_state.messages = []
    st.session_state.analysis_history = []
    st.success("聊天记录已清空")


def create_scenario_specific_components(scenario_rule: ScenarioRule, response: Dict, company_code: str):
    """创建场景特定的展示组件"""

    scenario_name = scenario_rule.display_name

    # 根据场景名称选择组件
    if scenario_name == "撤否企业分析":
        create_withdrawal_analysis_components(response, scenario_rule, company_code)
    elif scenario_name == "长期辅导企业分析":
        create_tutoring_analysis_components(response, scenario_rule, company_code)
    elif scenario_name == "上下游企业分析":
        create_relationship_analysis_components(response, scenario_rule, company_code)


def create_withdrawal_analysis_components(response: Dict, scenario_rule: ScenarioRule, company_code: str):
    """创建撤否企业分析专用组件"""

    # 1. 撤否风险仪表盘
    st.subheader("撤否风险综合评估")

    risk_assessment = response.get("risk_assessment", {})
    withdrawal_analysis = response.get("withdrawal_analysis", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_level = risk_assessment.get("risk_level", "未知")
        risk_config = {
            "高": {"color": "🔴", "desc": "存在重大审核障碍"},
            "中": {"color": "🟡", "desc": "部分问题需重点整改"},
            "低": {"color": "🟢", "desc": "问题相对可控"},
            "未知": {"color": "⚪", "desc": "信息不足无法评估"}
        }
        config = risk_config.get(risk_level, risk_config["未知"])
        st.metric("撤否风险等级", f"{config['color']} {risk_level}")
        st.caption(config['desc'])

    with col2:
        risks = risk_assessment.get("identified_risks", [])
        if isinstance(risks, list):
            st.metric("主要问题数量", len(risks))
        else:
            st.metric("主要问题数量", 0)


    # 2. 撤否原因时间线
    st.subheader("撤否关键事件时间线")

    timeline_data = withdrawal_analysis.get("timeline", [])
    if timeline_data and isinstance(timeline_data, list):
        for event in timeline_data:
            if isinstance(event, dict):
                with st.expander(f"{event.get('date', '未知日期')} - {event.get('event', '未知事件')}"):
                    st.write(f"**事件类型**: {event.get('type', '未知')}")
                    st.write(f"**影响程度**: {event.get('impact', '未知')}")
                    if event.get('description'):
                        st.write(f"**详细描述**: {event['description']}")
            elif isinstance(event, str):
                st.info(f"• {event}")
    else:
        st.info("暂无详细时间线信息")

    # 3. 审核问询重点分析
    st.subheader("审核问询重点分析")

    inquiry_analysis = withdrawal_analysis.get("inquiry_analysis", {})
    if inquiry_analysis and isinstance(inquiry_analysis, dict):
        rounds = inquiry_analysis.get("inquiry_rounds", [])

        if rounds and isinstance(rounds, list):
            for i, round_data in enumerate(rounds):
                if isinstance(round_data, dict):
                    with st.expander(f"第{round_data.get('round_number', i + 1)}轮问询 ({round_data.get('date', '')})",
                                     expanded=i == 0):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**问题数量**: {round_data.get('question_count', 0)}")

                            focus_areas = round_data.get('focus_areas', [])
                            if isinstance(focus_areas, list):
                                st.write(f"**重点领域**: {', '.join(focus_areas)}")
                            else:
                                st.write(f"**重点领域**: {focus_areas}")

                        with col_b:
                            st.write(f"**回复质量**: {round_data.get('reply_quality', '未知')}")
                            st.write(f"**整改情况**: {round_data.get('rectification', '未知')}")

                        key_questions = round_data.get('key_questions', [])
                        if key_questions and isinstance(key_questions, list):
                            st.write("**关键问题**:")
                            for q in key_questions:
                                if isinstance(q, str):
                                    st.write(f"• {q}")
                elif isinstance(round_data, str):
                    st.info(f"• 问询轮次 {i + 1}: {round_data}")
    else:
        st.info("暂无审核问询分析数据")

    # 4. 整改建议与重新上市路径
    st.subheader("整改建议与重新上市路径")

    recommendations = response.get("recommendations", [])
    if recommendations and isinstance(recommendations, list):
        tab1, tab2, tab3 = st.tabs(["立即整改项", "中期改进项", "长期优化项"])

        with tab1:
            urgent_items = [r for r in recommendations if isinstance(r, dict) and r.get('priority') == 'urgent']
            if urgent_items:
                for item in urgent_items:
                    st.warning(f"**{item.get('title', '')}**")
                    st.write(item.get('description', ''))
                    st.write(f"*预计耗时: {item.get('duration', '未知')}*")
            else:
                st.info("无立即整改项")

        with tab2:
            medium_items = [r for r in recommendations if isinstance(r, dict) and r.get('priority') == 'medium']
            if medium_items:
                for item in medium_items:
                    st.info(f"**{item.get('title', '')}**")
                    st.write(item.get('description', ''))
                    st.write(f"*预计耗时: {item.get('duration', '未知')}*")
            else:
                st.info("无中期改进项")

        with tab3:
            long_items = [r for r in recommendations if isinstance(r, dict) and r.get('priority') == 'long']
            if long_items:
                for item in long_items:
                    st.success(f"**{item.get('title', '')}**")
                    st.write(item.get('description', ''))
                    st.write(f"*预计耗时: {item.get('duration', '未知')}*")
            else:
                st.info("无长期优化项")
    else:
        st.info("暂无整改建议")
        # 添加调试信息
        st.write("**调试信息**: ")
        st.write(f"- recommendations字段类型: {type(recommendations)}")
        st.write(f"- recommendations内容: {recommendations}")
        # 显示整个response结构的一部分用于调试
        st.write(f"- response中包含的键: {list(response.keys())}")


# 修改 create_tutoring_analysis_components 函数中的相关部分：

def create_tutoring_analysis_components(response: Dict, scenario_rule: ScenarioRule, company_code: str):
    """创建长期辅导企业分析专用组件"""

    # 1. 辅导历程概览
    st.subheader("辅导历程概览")

    tutoring_analysis = response.get("tutoring_analysis", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = tutoring_analysis.get('start_date', '未知')
        st.metric("辅导开始时间", start_date)

    with col2:
        duration = tutoring_analysis.get('duration_months', 0)
        st.metric("辅导时长(月)", duration)

    with col3:
        stage = tutoring_analysis.get('current_stage', '未知')
        st.metric("当前阶段", stage)

    # 2. 辅导阶段时间线
    st.subheader("辅导阶段分析")

    stages = tutoring_analysis.get('stages', [])
    if stages and isinstance(stages, list):
        for stage in stages:
            if isinstance(stage, dict):
                status_icon = "✅" if stage.get('completed') else "⏳"
                with st.expander(f"{status_icon} {stage.get('name', '未知阶段')} ({stage.get('date_range', '')})"):
                    st.write(f"**主要内容**: {stage.get('content', '')}")
                    st.write(f"**完成情况**: {'已完成' if stage.get('completed') else '进行中/未开始'}")

                    issues = stage.get('issues', [])
                    if issues and isinstance(issues, list):
                        st.write("**存在问题**:")
                        for issue in issues:
                            if isinstance(issue, str):
                                st.warning(f"• {issue}")
            elif isinstance(stage, str):
                st.info(f"• {stage}")
    else:
        st.info("暂无辅导阶段信息")

    # 3. 上市障碍分析
    st.subheader("主要上市障碍分析")

    obstacles = tutoring_analysis.get("ipo_obstacles", [])
    if obstacles and isinstance(obstacles, list):
        obstacle_data = []
        for obs in obstacles:
            if isinstance(obs, dict):
                obstacle_data.append({
                    "障碍类型": obs.get('type', ''),
                    "严重程度": obs.get('severity', ''),
                    "影响环节": obs.get('impact_stage', ''),
                    "整改难度": obs.get('rectification_difficulty', '')
                })

        if obstacle_data:
            obstacle_df = pd.DataFrame(obstacle_data)
            st.dataframe(obstacle_df, width='stretch')

            # 障碍严重程度分布
            severity_counts = obstacle_df['严重程度'].value_counts()
            if not severity_counts.empty:
                fig, ax = plt.subplots()
                severity_counts.plot(kind='bar', ax=ax, color=['#ff6b6b', '#ffa726', '#66bb6a'])
                ax.set_ylabel('障碍数量')
                ax.set_title('上市障碍严重程度分布')
                st.pyplot(fig)
        else:
            st.info("暂无有效的障碍数据")
    else:
        st.info("暂无上市障碍分析")

# soure/main.py
# 修改 create_relationship_analysis_components 函数中的相关部分：

def create_relationship_analysis_components(response: Dict, scenario_rule: ScenarioRule, company_code: str):
    """创建上下游企业分析专用组件"""

    # 1. 关联网络概览
    st.subheader("🔗 关联网络概览")

    relationship_analysis = response.get("relationship_analysis", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        entity_count = relationship_analysis.get('entity_count', 0)
        st.metric("关联实体数量", entity_count)

    with col2:
        relation_count = relationship_analysis.get('relation_count', 0)
        st.metric("关联关系数量", relation_count)

    with col3:
        core_entities = relationship_analysis.get('core_entities', 0)
        st.metric("核心关联实体", core_entities)

    # 2. 关联关系矩阵
    relations = relationship_analysis.get('relations', [])
    if relations:
        relation_data = []
        for rel in relations:
            # 安全处理：确保rel是字典
            if isinstance(rel, dict):
                relation_data.append({
                    "关联方A": rel.get('entity_a', ''),
                    "关联方B": rel.get('entity_b', ''),
                    "关系类型": rel.get('relation_type', ''),
                    "交易金额": rel.get('transaction_amount', 'N/A'),
                    "比例(%)": rel.get('percentage', 'N/A'),
                    "公允性": rel.get('fairness', '未知')
                })
            elif isinstance(rel, str):
                # 如果是字符串，简单显示
                relation_data.append({
                    "关联关系": rel
                })

        if relation_data:
            relation_df = pd.DataFrame(relation_data)
            st.dataframe(relation_df, width='stretch')
        else:
            st.info("暂无有效的关联关系数据")

    # 3. 风险传导分析
    st.subheader("关联风险传导分析")

    risk_transmission = relationship_analysis.get("risk_transmission_analysis", {})
    if risk_transmission and isinstance(risk_transmission, dict):
        transmission_paths = risk_transmission.get('paths', [])

        if transmission_paths and isinstance(transmission_paths, list):
            for path in transmission_paths:
                if isinstance(path, dict):
                    with st.expander(f"风险传导路径: {path.get('from', '')} → {path.get('to', '')}"):
                        st.write(f"**传导机制**: {path.get('mechanism', '')}")
                        st.write(f"**影响程度**: {path.get('impact_level', '')}")
                        st.write(f"**发生概率**: {path.get('probability', '')}")

                        if path.get('mitigation_measures'):
                            st.write("**防范措施**:")
                            for measure in path.get('mitigation_measures', []):
                                st.info(f"• {measure}")
                elif isinstance(path, str):
                    st.info(f"• {path}")
        else:
            st.info("暂无风险传导路径信息")
    else:
        st.info("暂无风险传导分析数据")

    # 4. 独立性整改建议 - 修复这里的错误
    st.subheader("独立性整改建议")

    independence_issues = relationship_analysis.get("independence_issues", [])
    if independence_issues and isinstance(independence_issues, list):
        for i, issue in enumerate(independence_issues):
            if isinstance(issue, dict):
                # 使用安全的get方法，提供默认值
                issue_type = issue.get('type', f'独立性问题{i + 1}')
                severity = issue.get('severity', '未知')

                with st.expander(f"{issue_type} - 严重程度: {severity}"):
                    st.write(f"**问题描述**: {issue.get('description', '')}")
                    st.write(f"**影响分析**: {issue.get('impact_analysis', '')}")

                    suggestions = issue.get('rectification_suggestions', [])
                    if suggestions and isinstance(suggestions, list):
                        st.write("**整改建议**:")
                        for suggestion in suggestions:
                            if isinstance(suggestion, str):
                                st.success(f"• {suggestion}")
                    elif isinstance(suggestions, str):
                        st.success(f"• {suggestions}")
            elif isinstance(issue, str):
                # 如果是字符串，直接显示
                with st.expander(f"独立性问题 {i + 1}"):
                    st.write(f"**问题**: {issue}")
    else:
        st.info("暂无独立性整改建议")


def create_default_analysis_components(response: Dict, scenario_rule: ScenarioRule, company_code: str):
    """创建默认分析组件"""

    st.subheader("综合分析概览")

    # 显示关键指标
    key_metrics = response.get("key_metrics", {})
    if key_metrics:
        cols = st.columns(min(4, len(key_metrics)))
        for idx, (metric_name, metric_value) in enumerate(key_metrics.items()):
            with cols[idx % len(cols)]:
                st.metric(metric_name, metric_value)


def main():
    # ========== 侧边栏 ==========
    with st.sidebar:
        st.title("⚙️ 文档管理")

        # 文档管理
        st.subheader("上传企业文档")

        uploaded_files = st.file_uploader(
            "选择PDF文档",
            type=['pdf'],
            accept_multiple_files=True,
            help="上传企业相关PDF文档，支持多文件上传"
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            with st.expander(f"已上传 ({len(uploaded_files)}个文件)", expanded=False):
                for idx, file in enumerate(uploaded_files):
                    file_size_mb = file.size / (1024 * 1024)
                    st.write(f"{idx + 1}. **{file.name}** ({file_size_mb:.2f} MB)")

        st.divider()
        # 操作按钮
        st.subheader("🛠️ 操作")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空对话", width='stretch', help="清除所有聊天记录"):
                clear_chat_history()

    # ========== 主聊天界面 ==========
    # 初始化系统
    if "system" not in st.session_state:
        with st.spinner("正在初始化系统..."):
            st.session_state.system = init_system()

    # 显示聊天历史
    chat_container = st.container()

    with chat_container:
        # 显示所有消息
        for idx, message in enumerate(st.session_state.messages):
            display_message(message)

        # 如果还没有消息，显示欢迎信息
        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #666;'>
                <h3>👋 欢迎使用企业智能分析助手</h3>
                <p>💡 示例：</p>
                <p>• "分析欣强电子(300745)的撤否原因"</p>
                <p>• "评估某科技公司长期辅导的上市障碍"</p>
                <p>• "分析某集团上下游关联关系"</p>
            </div>
            """, unsafe_allow_html=True)

    # ========== 输入区域 ==========
    input_container = st.container()

    with input_container:
        st.divider()

        # 创建输入表单
        with st.form(key="chat_input_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                prompt = st.text_area(
                    "输入您的问题",
                    height=80,
                    placeholder="例如：分析欣强电子(300745)的撤否原因、评估某公司的上市可行性、了解行业最新趋势等...",
                    key="chat_input",
                    label_visibility="collapsed",
                    value=st.session_state.get("auto_query", "")
                )

            with col2:
                submit_button = st.form_submit_button(
                    "发送",
                    type="primary",
                    width='stretch'
                )

            if "auto_query" in st.session_state:
                del st.session_state.auto_query

    # ========== 处理用户输入 ==========
    if submit_button and prompt:
        add_message("user", prompt)
        st.rerun()

    elif submit_button and not prompt:
        st.warning("请输入您的问题")

    # ========== 处理AI响应 ==========
    if (st.session_state.messages and
            st.session_state.messages[-1]["role"] == "user" and
            not hasattr(st.session_state, "processing_message")):

        user_message = st.session_state.messages[-1]["content"]
        st.session_state.processing_message = True

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("正在分析..."):
                    try:
                        # 获取系统实例
                        system = st.session_state.system
                        rag_processor = system["rag_processor"]
                        vectorizer = system["vectorizer"]
                        llm_extractor = system["llm_extractor"]  # 获取大模型识别器

                        # 使用大模型智能提取场景和企业信息

                        extracted_info = llm_extractor.extract_company_and_scenario(user_message)

                        company_code = extracted_info["company_code"]
                        company_name = extracted_info["company_name"]
                        scenario_type = extracted_info["scenario"]
                        scenario_name = extracted_info["scenario_name"]

                        # 获取场景规则
                        scenario_rule = ScenarioConfig.get_scenario_rule(scenario_type)

                        # 确保场景名称与场景规则匹配
                        if scenario_rule:
                            scenario_name = scenario_rule.display_name

                        # 显示提取的信息
                        info_text = []
                        if company_name:
                            confidence = extracted_info["confidence"]["company"]
                            info_text.append(f"识别到企业: {company_name} (置信度: {confidence:.0%})")
                            if company_code:
                                info_text[-1] += f" [代码: {company_code}]"
                        elif company_code:
                            confidence = extracted_info["confidence"]["company"]
                            info_text.append(f"识别到企业代码: {company_code} (置信度: {confidence:.0%})")

                        if scenario_name != "自定义分析":
                            confidence = extracted_info["confidence"]["scenario"]
                            info_text.append(f"识别到场景: {scenario_name} (置信度: {confidence:.0%})")

                        # 使用企业名称进行搜索（优先使用名称，其次使用代码）
                        search_company = company_name or company_code

                        # 处理上传的PDF文档
                        if st.session_state.uploaded_files:
                            pdf_processor = PDFProcessor(system["config"])
                            processed_count = 0

                            for uploaded_file in st.session_state.uploaded_files:
                                temp_dir = tempfile.mkdtemp()
                                temp_path = os.path.join(temp_dir, uploaded_file.name)

                                try:
                                    with open(temp_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

                                    chunks = pdf_processor.extract_text_from_pdf(temp_path)
                                    if chunks and isinstance(chunks, list):
                                        success_count = vectorizer.store_documents(chunks)
                                        processed_count += success_count if success_count else 0

                                except Exception as e:
                                    st.error(f"处理文件 {uploaded_file.name} 失败: {e}")
                                finally:
                                    try:
                                        os.remove(temp_path)
                                        os.rmdir(temp_dir)
                                    except:
                                        pass

                            if processed_count > 0:
                                st.info(f"已处理 {processed_count} 个文档片段")

                        # 执行分析
                        result = rag_processor.process_query(
                            query=user_message,
                            scenario=scenario_rule.display_name if scenario_rule else (scenario_name if scenario_name != "自定义分析" else None),
                            company_code=search_company,
                            use_web_data="auto",
                            scenario_rule=scenario_rule
                        )

                        # 确保result是字典
                        if not isinstance(result, dict):
                            result = {
                                "response": {
                                    "summary": str(result) if result else "分析结果为空",
                                    "detailed_analysis": {
                                        "local_based": ["本地文档分析"],
                                        "web_based": ["网络信息分析"],
                                        "integrated": ["综合分析"]
                                    },
                                    "key_findings": ["分析完成"],
                                    "risk_assessment": {
                                        "identified_risks": [],
                                        "risk_level": "未知",
                                        "rationale": "分析完成"
                                    },
                                    "recommendations": []
                                },
                                "retrieval_stats": {},
                                "source_documents": [],
                                "query": user_message,
                                "scenario_name": scenario_name,
                                "company_code": company_code,
                                "timestamp": datetime.now().isoformat()
                            }

                        # 获取响应内容
                        response = result.get("response", {})
                        if not isinstance(response, dict):
                            response = {"summary": str(response)}

                        answer = response.get("summary") or "\n".join(response.get("analysis", []))

                        if not answer:
                            answer = "抱歉，我没有找到相关信息。请尝试更具体的问题或上传相关文档。"

                        # 添加AI消息到历史
                        add_message("assistant", answer, result)

                    except Exception as e:
                        error_msg = f"分析过程中出现错误: {str(e)}"
                        add_message("assistant", error_msg)
                        st.error(f"详细错误: {e}")

                    finally:
                        if "processing_message" in st.session_state:
                            del st.session_state.processing_message
                        st.rerun()


if __name__ == "__main__":
    main()