# init_data_qwen.py
import sys
import os

from soure.data_ingestion.data_collector import RobustDataCollector

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from soure.data_ingestion.collector import DataCollector, collect_government_lists, clean_and_process_data
from soure.embedding.vectorizer_qwen import QwenVectorizer
import yaml


def main():
    print("开始初始化企业分析系统...")

    with open("/Users/zjy/PyCharmMiscProject/config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    collector = RobustDataCollector()  # 使用增强版采集器
    vectorizer = QwenVectorizer(config)

    # 采集各类数据
    all_docs = []

    try:
        # 1. 采集深交所IPO数据
        print("正在采集深交所IPO项目动态数据...")
        szse_ipo_docs = collector.collect_szse_ipo_data()
        all_docs.extend(szse_ipo_docs)
        print(f"采集到 {len(szse_ipo_docs)} 条深交所IPO数据")

        # 2. 从API获取数据（推荐）
        print("正在从API采集数据...")
        if config.get('data_sources', {}).get('akshare', {}).get('enabled', True):
            print("正在采集A股公告数据...")
            announcement_docs = collector.collect_from_akshare()
            all_docs.extend(announcement_docs)
            print(f"采集到 {len(announcement_docs)} 条公告数据")

        # 3. 采集监管数据
        print("正在采集监管数据...")
        regulatory_docs = collector.collect_regulatory_data(days=30)
        all_docs.extend(regulatory_docs)
        print(f"采集到 {len(regulatory_docs)} 条监管数据")

        # 4. 采集政府榜单
        print("正在采集政府榜单数据...")
        list_docs = collect_government_lists()
        all_docs.extend(list_docs)
        print(f"采集到 {len(list_docs)} 条政府榜单数据")

        # 5. 数据清洗
        print("正在清洗数据...")
        cleaned_docs = clean_and_process_data(all_docs)
        print(f"清洗后剩余 {len(cleaned_docs)} 条文档")

        # 6. 批量存储 - 添加API可用性检查
        if cleaned_docs:
            print(f"正在存储 {len(cleaned_docs)} 条文档到向量数据库...")
            try:
                vectorizer.store_documents(cleaned_docs)
                print(f"成功入库 {len(cleaned_docs)} 条文档")
            except Exception as e:
                print(f"向量存储失败: {e}")
                print("尝试使用备用方案...")
                # 这里可以实现备用存储方案
        else:
            print("没有获取到数据")

    except Exception as e:
        print(f"初始化过程出错: {e}")
        import traceback
        traceback.print_exc()

    print("系统初始化完成！")



if __name__ == "__main__":
    main()
