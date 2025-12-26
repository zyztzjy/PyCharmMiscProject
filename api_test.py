# test_api.py
import os
import sys
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.embedding.vectorizer_qwen import QwenVectorizer
from src.llm.qwen_client import QwenClient


def test_api_configuration():
    """测试 API 配置"""
    print("=== 测试 API 配置 ===")

    # 读取配置
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    api_key = config.get('llm', {}).get('api_key')
    print(f"API Key 配置: {'已配置' if api_key else '未配置'}")

    if not api_key:
        print("❌ API Key 未配置，请在 config.yaml 中设置")
        return False

    # 检查是否为环境变量格式
    if isinstance(api_key, str) and '${' in api_key:
        env_var = api_key.replace('${', '').replace('}', '')
        actual_key = os.getenv(env_var)
        print(f"环境变量 {env_var}: {'已设置' if actual_key else '未设置'}")
        if not actual_key:
            print("❌ 环境变量未设置")
            return False

    print("✅ API 配置检查通过")
    return True


def test_qwen_client():
    """测试 Qwen 客户端"""
    print("\n=== 测试 Qwen 客户端 ===")

    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        # 获取 API 密钥
        api_key = config.get('llm', {}).get('api_key')
        if isinstance(api_key, str) and '${' in api_key:
            env_var = api_key.replace('${', '').replace('}', '')
            api_key = os.getenv(env_var)

        if not api_key:
            print("❌ 无法获取 API 密钥")
            return False

        client = QwenClient(api_key=api_key)

        # 测试简单调用
        test_messages = [
            {"role": "system", "content": "你是一个测试助手"},
            {"role": "user", "content": "Hello, 测试 API 连接"}
        ]

        response = client.chat_completion(test_messages, max_tokens=100)

        if response and len(response) > 0:
            print("✅ Qwen 客户端连接成功")
            print(f"响应示例: {response[:100]}...")
            return True
        else:
            print("❌ Qwen 客户端连接失败")
            return False

    except Exception as e:
        print(f"❌ Qwen 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vectorizer():
    """测试向量化器"""
    print("\n=== 测试向量化器 ===")

    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        vectorizer = QwenVectorizer(config)

        # 测试文本向量化
        test_texts = [
            "这是测试文本1",
            "这是测试文本2，用于验证向量化功能"
        ]

        print("正在测试文本向量化...")
        embeddings = vectorizer.create_embeddings(test_texts)

        print(f"✅ 向量化成功，维度: {embeddings.shape}")
        print(f"嵌入向量形状: {embeddings.shape}")

        return True

    except Exception as e:
        print(f"❌ 向量化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_store_documents():
    """测试文档存储功能"""
    print("\n=== 测试文档存储功能 ===")

    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        vectorizer = QwenVectorizer(config)

        # 创建测试文档
        test_docs = [
            {
                "content": "这是一份测试文档，用于验证向量化和存储功能",
                "metadata": {
                    "company_name": "测试公司",
                    "doc_type": "test",
                    "source": "test"
                },
                "doc_id": "test_doc_1"
            },
            {
                "content": "第二份测试文档，包含更多内容以验证存储功能",
                "metadata": {
                    "company_name": "测试公司2",
                    "doc_type": "test",
                    "source": "test"
                },
                "doc_id": "test_doc_2"
            }
        ]

        print("正在测试文档存储...")
        vectorizer.store_documents(test_docs)

        print("✅ 文档存储测试成功")

        # 验证存储结果
        stats = vectorizer.get_collection_stats()
        print(f"向量库中现有文档数: {stats.get('total_documents', 0)}")

        return True

    except Exception as e:
        print(f"❌ 文档存储测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_functionality():
    """测试搜索功能"""
    print("\n=== 测试搜索功能 ===")

    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        vectorizer = QwenVectorizer(config)

        # 测试语义搜索
        print("正在测试语义搜索...")
        results = vectorizer.search_similar("测试", top_k=2)

        print(f"✅ 搜索成功，返回 {len(results)} 个结果")
        if results:
            print(f"最相似文档相似度: {results[0].get('similarity', 0):.3f}")

        return True

    except Exception as e:
        print(f"❌ 搜索功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始 API 测试...")

    tests = [
        ("API 配置", test_api_configuration),
        ("Qwen 客户端", test_qwen_client),
        ("向量化器", test_vectorizer),
        ("文档存储", test_store_documents),
        ("搜索功能", test_search_functionality)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        success = test_func()
        results.append((test_name, success))

    print(f"\n{'=' * 50}")
    print("📊 测试结果汇总:")

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")

    passed_count = sum(1 for _, success in results if success)
    total_count = len(results)

    print(f"\n总览: {passed_count}/{total_count} 项测试通过")

    if passed_count == total_count:
        print("🎉 所有测试均通过！")
    else:
        print("⚠️  部分测试失败，请检查配置和环境")


if __name__ == "__main__":
    run_all_tests()
