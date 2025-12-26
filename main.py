# src/main.py (包含 display_analysis_result 函数)
from typing import Dict
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

from src.data_ingestion.data_collector import collect_data

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embedding.vectorizer_qwen import QwenVectorizer
from src.rag.qwen_rag_processor import QwenRAGProcessor

import yaml

# 页面配置
st.set_page_config(
    page_title="企业智能分析系统",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化函数
@st.cache_resource
def init_system():
    """初始化系统"""
    # 读取配置
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 从环境变量或secrets获取API Key
    api_key = st.secrets.get("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
    if not api_key:
        st.error("请设置DASHSCOPE_API_KEY环境变量或Streamlit secrets")
        st.stop()

    # 初始化向量化器
    vectorizer = QwenVectorizer(config)

    # 初始化RAG处理器
    rag_processor = QwenRAGProcessor(
        vectorizer=vectorizer,
        api_key=api_key,
        model=config['llm'].get('model', 'qwen-max')
    )
    return {
        "config": config,
        "vectorizer": vectorizer,
        "rag_processor": rag_processor,
        "api_key": api_key
    }

def display_analysis_result(result: Dict):
    """展示分析结果"""
    response = result.get("response", {})
    source_docs = result.get("source_documents", [])
    retrieval_stats = result.get("retrieval_stats", {})
    timestamp = result.get("timestamp", "未知时间")
    scenario = result.get("scenario", "未知场景")
    company_code = result.get("company_code", "未知企业")
    query = result.get("query", "无查询")

    # 概要
    summary = response.get("summary", "")
    if summary:
        st.subheader("📋 概要")
        st.info(summary)

    # 分析详情
    analysis_list = response.get("analysis", [])
    if analysis_list:
        st.subheader("🔍 详细分析")
        for i, item in enumerate(analysis_list):
            st.write(f"**要点 {i+1}**: {item}")

    # 风险点
    risks_list = response.get("risks", [])
    if risks_list:
        st.subheader("⚠️ 识别风险")
        for risk in risks_list:
            st.warning(f"• {risk}")

    # 建议
    recommendations_list = response.get("recommendations", [])
    if recommendations_list:
        st.subheader("💡 分析建议")
        for rec in recommendations_list:
            st.success(f"• {rec}")

    # 置信度 (如果有的话)
    confidence = response.get("confidence")
    if confidence is not None:
        st.subheader("📊 分析置信度")
        st.metric(label="置信度", value=f"{confidence:.2f}")

    # 检索统计信息
    if retrieval_stats:
        st.subheader("📈 检索统计")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总检索文档数", retrieval_stats.get("total_docs_retrieved", 0))
        with col2:
            st.metric("本地文档数", retrieval_stats.get("local_docs_count", 0))
        with col3:
            st.metric("网络文档数", retrieval_stats.get("web_docs_count", 0))
        with col4:
            st.metric("处理耗时(秒)", retrieval_stats.get("processing_time_seconds", 0))

    # 参考文档
    if source_docs:
        st.subheader("📚 参考信息来源")
        with st.expander("点击查看详细来源"):
            for i, doc in enumerate(source_docs):
                source = doc.get("source", "未知来源")
                content_preview = doc.get("content_preview", "无预览内容")
                metadata = doc.get("metadata", {})
                st.write(f"**来源 {i+1}: {source}**")
                st.write(f"预览: {content_preview}")
                if metadata:
                    st.write(f"元数据: {metadata}")
                st.write("---")

    # 分析元信息
    st.subheader("ℹ️ 分析元信息")
    meta_info = pd.DataFrame({
        "项目": ["分析场景", "目标企业", "查询内容", "分析时间"],
        "内容": [scenario, company_code, query, timestamp]
    })
    st.dataframe(meta_info, use_container_width=True, hide_index=True)

# ... (main 函数保持不变) ...

def main():
    st.title("🏢 企业智能分析系统 (集成实时数据)")
    st.markdown("基于通义千问大模型的非结构化数据分析平台")
    st.markdown("---")

    # 初始化系统
    with st.spinner("正在初始化系统..."):
        try:
            system = init_system()
            st.success("系统初始化完成！")
        except Exception as e:
            st.error(f"系统初始化失败: {e}")
            st.stop()

    rag_processor = system["rag_processor"]
    vectorizer = system["vectorizer"]

    # 侧边栏
    with st.sidebar:
        st.header("分析场景")
        scenario = st.selectbox(
            "选择分析场景：",
            [
                "撤否企业分析",
                "长期辅导企业分析",
                "新三板企业分析",
                "供应链分析",
                "关系网分析",
                "财务分析",
                "舆情分析",
                "行业分析",
                "自定义分析"
            ]
        )
        st.markdown("---")
        st.header("目标企业")
        company_code = st.text_input("企业代码/名称", placeholder="如：600000 或 浦发银行", help="请输入股票代码或企业全称")
        st.markdown("---")
        st.header("分析设置")
        # 高级选项
        with st.expander("高级选项"):
            retrieval_count = st.slider("检索文档数量", 5, 30, 15)
            similarity_threshold = st.slider("相似度阈值", 0.5, 0.9, 0.7, 0.05)
            model_choice = st.selectbox("选择模型", ["qwen-turbo", "qwen-plus", "qwen-max"])
            # 添加是否使用网络数据的选项
            use_web_data = st.checkbox("启用实时网络数据", value=True, help="勾选后将抓取并分析最新的网络信息")
        st.markdown("---")
        # 行动按钮
        col1, col2 = st.columns(2)
        with col1:
            analyze_clicked = st.button("开始分析", type="primary", width='stretch')
        with col2:
            if st.button("重置", width='stretch'):
                st.session_state.clear()
                st.rerun()

        st.markdown("---")
        # 系统状态
        st.header("系统状态")
        try:
            stats = vectorizer.get_collection_stats()
            st.metric("本地文档总数", stats.get("total_documents", 0))
            st.metric("状态", stats.get("status", "未知"))
        except:
            st.metric("本地文档总数", "N/A")

        # --- 新增：数据更新按钮 ---
        if st.button("🔄 更新网络数据", type="secondary"):
            with st.spinner("正在从网络抓取最新数据..."):
                try:
                    new_docs = collect_data()
                    if new_docs:
                        vectorizer.store_documents(new_docs) # 存储到向量数据库
                        st.success(f"成功抓取并更新了 {len(new_docs)} 条网络数据！")
                        # 可选：清除旧的分析结果，因为数据已更新
                        if "analysis_result" in st.session_state:
                            del st.session_state["analysis_result"]
                        if "quick_result" in st.session_state:
                            del st.session_state["quick_result"]
                    else:
                        st.info("未抓取到新的网络数据。")
                except Exception as e:
                    st.error(f"更新网络数据失败: {e}")

        # 清空缓存按钮
        if st.button("清空缓存", type="secondary"):
            st.cache_resource.clear()
            st.success("缓存已清除")
            st.rerun()

    # 主界面 (保持不变)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("分析输入")
        # 场景说明 (保持不变)
        scenario_descriptions = {
            "撤否企业分析": "分析撤否原因、现场检查、审核问题、财务异常",
            "长期辅导企业分析": "分析辅导备案超过1年未申报企业的原因和风险",
            "新三板企业分析": "分析新三板企业转板可能性和障碍",
            "供应链分析": "分析上下游关系和行业竞争格局",
            "关系网分析": "分析企业投资、客户、招投标等关系网络",
            "财务分析": "深度分析企业财务状况和风险",
            "舆情分析": "分析企业舆情情绪和媒体关注度",
            "行业分析": "分析行业趋势、政策和竞争格局"
        }
        if scenario in scenario_descriptions:
            st.info(f"**{scenario}**：{scenario_descriptions[scenario]}")

        # 查询输入 (保持不变)
        if scenario == "自定义分析":
            query = st.text_area("请输入分析需求：", height=150, placeholder="例如：分析XX公司的上市可能性、风险评估、投资价值等...", help="请详细描述您的分析需求")
        else:
            query = st.text_area("补充分析需求（可选）：", height=100, placeholder="可以补充具体的关注点，如特定风险、时间范围等...")

        # 构建完整查询 (保持不变)
        if scenario != "自定义分析":
            if company_code:
                base_query = f"分析{company_code}的"
            else:
                base_query = "分析"
            if query:
                full_query = f"{base_query}{scenario}，具体要求：{query}"
            else:
                full_query = f"{base_query}{scenario}"
        else:
            full_query = query

    with col2:
        st.subheader("⚡ 快速分析")
        # 快速分析选项 (保持不变)
        quick_options = {
            "财务健康度": "财务健康度分析",
            "合规风险": "合规风险评估",
            "舆情监控": "近期舆情分析",
            "行业地位": "行业竞争地位分析",
            "供应链风险": "供应链风险评估",
            "核心团队": "核心团队背景分析"
        }
        for option_text, option_desc in quick_options.items():
            if st.button(option_text, width='stretch'):
                if company_code:
                    quick_query = f"分析{company_code}的{option_desc}"
                else:
                    quick_query = option_desc
                with st.spinner(f"正在{option_desc}..."):
                    try:
                        # 在快速分析中也使用 use_web_data 选项
                        result = rag_processor.process_query(quick_query, scenario, company_code, use_web_data=use_web_data)
                        st.session_state["quick_result"] = result
                        st.success("快速分析完成！")
                    except Exception as e:
                        st.error(f"快速分析失败: {e}")

    # 执行分析 (更新调用，传入 use_web_data)
    if analyze_clicked and full_query:
        with st.spinner("🤖 AI正在深度分析中，请稍候..."):
            try:
                # 执行RAG查询，传入 use_web_data 选项
                result = rag_processor.process_query(
                    query=full_query,
                    scenario=scenario if scenario != "自定义分析" else None,
                    company_code=company_code if company_code else None,
                    use_web_data=use_web_data # 传入选项
                )
                # 保存结果
                st.session_state["analysis_result"] = result
                st.success("分析完成！")
            except Exception as e:
                st.error(f"分析失败: {e}")

    # 显示主分析结果 (使用 display_analysis_result)
    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        st.markdown("---")
        st.header("AI分析报告")
        display_analysis_result(result) # 调用函数显示结果

        # 导出功能 (保持不变)
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("导出JSON报告", width='stretch'):
                report = {
                    "生成时间": datetime.now().isoformat(),
                    "分析场景": result.get("scenario"),
                    "目标企业": result.get("company_code"),
                    "查询内容": result.get("query"),
                    "分析结果": result.get("response"),
                    "检索统计": result.get("retrieval_stats"),
                    "参考文档": result.get("source_documents", [])
                }
                st.download_button(
                    label="点击下载JSON",
                    data=json.dumps(report, ensure_ascii=False, indent=2),
                    file_name=f"企业分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        with col2:
            if st.button("重新分析", width='stretch'):
                st.session_state.pop("analysis_result", None)
                st.rerun()
        with col3:
            if st.button("生成PPT摘要", width='stretch'):
                st.info("PPT生成功能开发中...")

    # 显示快速分析结果 (保持不变)
    if "quick_result" in st.session_state:
        st.markdown("---")
        st.header("⚡ 快速分析结果")
        quick_result = st.session_state["quick_result"]
        response = quick_result.get("response", {})
        summary = response.get("summary", "")
        if summary:
            st.write(summary)
        elif response.get("analysis"):
            analysis = response["analysis"]
            if isinstance(analysis, list):
                for item in analysis[:3]:
                    st.write(f"• {item}")
            else:
                st.write(analysis[:300] + "..." if len(analysis) > 300 else analysis)

        if st.button("清除快速结果"):
            st.session_state.pop("quick_result", None)
            st.rerun()

if __name__ == "__main__":
    main()