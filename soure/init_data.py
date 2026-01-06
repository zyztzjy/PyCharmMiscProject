# init_data_qwen.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from soure.data_ingestion.collector import DataCollector, collect_government_lists, clean_and_process_data
from soure.embedding.vectorizer_qwen import QwenVectorizer
import yaml


def main():
    print("开始初始化企业分析系统...")

    with open("/Users/zjy/PyCharmMiscProject/config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)


    vectorizer = QwenVectorizer(config)

    # 采集各类数据
    all_docs = []

    try:
        # 1. 采集撤否企业数据
        print("正在采集撤否企业数据...")
        withdrawal_docs = DataCollector.collect_withdrawal_analysis_data()
        all_docs.extend(withdrawal_docs)
        print(f"采集到 {len(withdrawal_docs)} 条撤否企业数据")

        # 2. 采集长期辅导企业数据
        print("正在采集长期辅导企业数据...")
        long_guidance_docs = DataCollector.collect_long_guidance_companies()
        all_docs.extend(long_guidance_docs)
        print(f"采集到 {len(long_guidance_docs)} 条长期辅导企业数据")

        # 3. 采集新三板长期挂牌企业数据
        print("正在采集新三板长期挂牌企业数据...")
        long_nse_docs = DataCollector.collect_long_nse_companies()
        all_docs.extend(long_nse_docs)
        print(f"采集到 {len(long_nse_docs)} 条新三板长期挂牌企业数据")

        # 4. 采集主要榜单数据
        print("正在采集主要榜单数据...")
        major_lists_docs = DataCollector.collect_major_lists_data()
        all_docs.extend(major_lists_docs)
        print(f"采集到 {len(major_lists_docs)} 条榜单数据")

        # 5. 从API获取其他数据
        print("正在从API采集数据...")
        if config.get('data_sources', {}).get('akshare', {}).get('enabled', True):
            print("正在采集A股公告数据...")
            announcement_docs = DataCollector.collect_from_akshare()
            all_docs.extend(announcement_docs)
            print(f"采集到 {len(announcement_docs)} 条公告数据")

        # 6. 采集监管数据
        print("正在采集监管数据...")
        regulatory_docs = DataCollector.collect_regulatory_data(days=30)
        all_docs.extend(regulatory_docs)
        print(f"采集到 {len(regulatory_docs)} 条监管数据")

        # 7. 数据清洗
        print("正在清洗数据...")
        # 在这里清洗数据（如果需要）

        # 8. 批量存储
        if all_docs:
            print(f"正在存储 {len(all_docs)} 条文档到向量数据库...")
            try:
                vectorizer.store_documents(all_docs)
                print(f"成功入库 {len(all_docs)} 条文档")
            except Exception as e:
                print(f"向量存储失败: {e}")
                print("尝试使用备用方案...")
        else:
            print("没有获取到数据")

    except Exception as e:
        print(f"初始化过程出错: {e}")
        import traceback
        traceback.print_exc()

    print("系统初始化完成！")




if __name__ == "__main__":
    main()
