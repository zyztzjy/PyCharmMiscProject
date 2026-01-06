# test_qwen_web_search.py
import sys
import os

from soure.llm.web_search import QwenWebSearcher
from soure.rag.qwen_rag_processor import QwenRAGProcessor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from soure.embedding.vectorizer_qwen import QwenVectorizer
import yaml


def test_qwen_web_search():
    """测试通义千问联网搜索"""

    print("=== 测试通义千问联网搜索功能 ===\n")

    # 读取配置
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 需要API密钥
    api_key = "sk-6892cc65b78941e7a6981cae25997c0b"
    # api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置DASHSCOPE_API_KEY环境变量")
        print("请在环境变量中设置您的通义千问API密钥")
        return

    print("1. 测试QwenWebSearcher基本功能...")
    try:
        searcher = QwenWebSearcher(api_key)

        # 测试连接
        success, message = searcher.test_connection()
        print(f"   连接测试: {message}")

        if success:
            # 测试搜索
            test_queries = [
                ("欣强电子最新财务情况", "欣强电子", "财务分析"),
                ("撤否企业常见问题", None, "撤否企业分析"),
                ("半导体行业2024年趋势", None, "行业分析")
            ]

            for query, company, scenario in test_queries:
                print(f"\n   测试搜索: '{query}'")
                print(f"   企业: {company or '无'}, 场景: {scenario}")

                results = searcher.search(query, company, scenario)

                if results:
                    print(f"   获得 {len(results)} 个结果")
                    for i, result in enumerate(results[:2]):
                        metadata = result.get("metadata", {})
                        print(f"     结果{i + 1}: {metadata.get('title', '无标题')}")
                        print(f"       来源: {metadata.get('source', '未知')}")
                        print(f"       搜索方式: {metadata.get('search_method', '未知')}")
                else:
                    print("   未获得结果")
        else:
            print("   API连接失败，无法继续测试")

    except Exception as e:
        print(f"   搜索器测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n2. 测试UnifiedRAGProcessor集成...")
    try:
        # 初始化向量化器
        vectorizer = QwenVectorizer(config)

        # 初始化统一处理器
        processor = QwenRAGProcessor(
            vectorizer=vectorizer,
            api_key=api_key,
            model="qwen-max",
            config=config
        )

        # 测试系统状态
        status = processor.get_system_status()
        print(f"   系统状态: {status}")

        # 测试简单查询
        print("\n   测试简单查询处理...")
        test_query = "分析欣强电子的基本情况"

        result = processor.process_query(
            query=test_query,
            company_code="欣强电子",
            scenario="财务分析",
            use_web_data="auto",
            retrieval_count=5
        )

        if "error" not in result:
            print(f"   查询处理成功")
            print(f"   处理时间: {result['processing_metrics']['total_time_seconds']:.2f}秒")
            print(f"   本地文档: {result['source_documents']['local']}")
            print(f"   网络结果: {result['source_documents']['web']}")
            print(f"   联网搜索: {result['web_search_details']['performed']}")

            if result['response'].get('summary'):
                print(f"   分析摘要: {result['response']['summary'][:100]}...")
        else:
            print(f"   查询处理失败: {result.get('error', '未知错误')}")

    except Exception as e:
        print(f"   处理器测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_web_search_modes():
    """测试不同的联网搜索模式"""

    print("\n=== 测试不同联网搜索模式 ===")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return

    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    vectorizer = QwenVectorizer(config)
    processor = QwenRAGProcessor(vectorizer, api_key, "qwen-max", config)

    # 测试不同模式
    test_modes = ["auto", "always", "never"]

    for mode in test_modes:
        print(f"\n   测试模式: {mode}")

        result = processor.process_query(
            query="测试企业最新动态",
            company_code="测试企业",
            use_web_data=mode,
            retrieval_count=3
        )

        web_performed = result['web_search_details']['performed']
        print(f"   是否执行联网搜索: {web_performed}")
        print(f"   原因: {result['web_search_details']['reason']}")
        print(f"   置信度: {result['web_search_details']['confidence']:.2f}")


if __name__ == "__main__":
    test_qwen_web_search()
    test_web_search_modes()