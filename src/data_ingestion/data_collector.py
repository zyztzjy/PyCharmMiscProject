# src/data/data_collector.py (更新版 - 集成 web_scraper)
import logging
import time
from typing import List, Dict

from src.data.web_scraper import scrape_all_sources
from src.processing.text_prc import process_raw_text

logger = logging.getLogger(__name__)

def collect_data_from_web() -> List[Dict]:
    """从网络抓取最新数据并进行处理"""
    logger.info("开始从网络收集数据...")
    scraped_data = scrape_all_sources()

    processed_docs = []
    for source, items in scraped_data.items():
        for item in items:
            # 假设 item 是一个字典，包含 'company_name', 'content', 'status' 等字段
            # 需要根据 web_scraper.py 返回的 item 结构进行调整
            company_name = item.get("company_name", "未知公司")
            content = f"企业名称: {company_name}. 状态: {item.get('status', '')}. 详情: {str(item)}" # 构建内容
            doc = {
                "content": process_raw_text(content), # 假设 process_raw_text 用于清洗
                "metadata": {
                    "company_name": company_name,
                    "source": item.get("source", "Unknown"),
                    "status": item.get("status", ""),
                    "filing_date": item.get("filing_date", ""), # 对于辅导备案
                    "doc_type": "web_scraped", # 标记来源类型
                    "scraped_at": time.time(), # 添加抓取时间戳
                },
                "doc_id": f"web_{hash(content[:50]) % 1000000}" # 生成文档ID
            }
            processed_docs.append(doc)

    logger.info(f"网络数据收集完成，处理了 {len(processed_docs)} 条记录。")
    return processed_docs

def collect_data() -> List[Dict]:
    """主收集函数，可以合并网络数据和可能的其他数据源"""
    docs = []
    # 收集网络数据
    web_docs = collect_data_from_web()
    docs.extend(web_docs)

    # 可以在此处添加其他数据源的收集逻辑
    # e.g., docs.extend(collect_from_database())
    # e.g., docs.extend(collect_from_filesystem())

    return docs

# --- 示例：如何使用 ---
# if __name__ == "__main__":
#     collected_docs = collect_data()
#     print(f"总共收集到 {len(collected_docs)} 份文档。")
#     for doc in collected_docs[:5]: # 打印前5份
#         print(doc)