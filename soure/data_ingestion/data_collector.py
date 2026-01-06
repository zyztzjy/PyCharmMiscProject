# soure/data_ingestion/data_collector.py (更新版)
import logging
import time
from typing import List, Dict

from soure.data.web_scraper import scrape_all_sources
from soure.processing.text_prc import process_raw_text

logger = logging.getLogger(__name__)


# soure/data_ingestion/data_collector.py
def collect_withdrawal_data() -> List[Dict]:
    """专门收集撤否企业数据 - 使用Selenium"""
    logger.info("开始使用Selenium收集撤否企业数据...")

    try:
        from soure.data.web_scraper import scrape_all_withdrawals_selenium
        scraped_data = scrape_all_withdrawals_selenium()

        processed_docs = []
        for source, items in scraped_data.items():
            for item in items:
                # 构建内容用于向量化
                content_parts = []
                if item.get("company_name"):
                    content_parts.append(f"企业名称: {item['company_name']}")
                if item.get("status"):
                    content_parts.append(f"状态: {item['status']}")
                if item.get("reason"):
                    content_parts.append(f"原因: {item['reason']}")
                if item.get("registration_place"):
                    content_parts.append(f"注册地: {item['registration_place']}")
                if item.get("industry"):
                    content_parts.append(f"行业: {item['industry']}")
                if item.get("sponsor"):
                    content_parts.append(f"保荐机构: {item['sponsor']}")
                if item.get("update_date"):
                    content_parts.append(f"更新日期: {item['update_date']}")
                if item.get("accept_date"):
                    content_parts.append(f"受理日期: {item['accept_date']}")

                content = ". ".join(content_parts) + "."

                doc = {
                    "content": content,
                    "metadata": {
                        "company_name": item.get("company_name", ""),
                        "status": item.get("status", ""),
                        "reason": item.get("reason", ""),
                        "registration_place": item.get("registration_place", ""),
                        "industry": item.get("industry", ""),
                        "sponsor": item.get("sponsor", ""),
                        "update_date": item.get("update_date", ""),
                        "accept_date": item.get("accept_date", ""),
                        "source": item.get("source", "Unknown"),
                        "doc_type": "withdrawal_company",
                        "scraped_at": time.time(),
                    },
                    "doc_id": f"withdrawal_{hash(content[:50]) % 1000000}"
                }
                processed_docs.append(doc)

        logger.info(f"使用Selenium的撤否企业数据收集完成，处理了 {len(processed_docs)} 条记录。")
        return processed_docs

    except Exception as e:
        logger.error(f"使用Selenium收集撤否企业数据失败: {e}")
        import traceback
        traceback.print_exc()
        return []


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
            content = f"企业名称: {company_name}. 状态: {item.get('status', '')}. 详情: {str(item)}"  # 构建内容
            doc = {
                "content": process_raw_text(content),  # 假设 process_raw_text 用于清洗
                "metadata": {
                    "company_name": company_name,
                    "source": item.get("source", "Unknown"),
                    "status": item.get("status", ""),
                    "filing_date": item.get("filing_date", ""),  # 对于辅导备案
                    "doc_type": "web_scraped",  # 标记来源类型
                    "scraped_at": time.time(),  # 添加抓取时间戳
                },
                "doc_id": f"web_{hash(content[:50]) % 1000000}"  # 生成文档ID
            }
            processed_docs.append(doc)

    logger.info(f"网络数据收集完成，处理了 {len(processed_docs)} 条记录。")
    return processed_docs


# soure/data_ingestion/data_collector.py
def collect_data() -> List[Dict]:
    """主收集函数，可以合并网络数据和可能的其他数据源"""
    docs = []

    try:
        # 收集撤否企业数据（使用Selenium）
        withdrawal_docs = collect_withdrawal_data()
        docs.extend(withdrawal_docs)
        logger.info(f"收集到 {len(withdrawal_docs)} 条撤否企业数据")

        # 收集其他网络数据
        web_docs = collect_data_from_web()
        docs.extend(web_docs)
        logger.info(f"收集到 {len(web_docs)} 条其他网络数据")

    except Exception as e:
        logger.error(f"数据收集过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    return docs
