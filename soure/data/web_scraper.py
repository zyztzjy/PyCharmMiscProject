# soure/data/web_scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)

def scrape_sse_withdrawals() -> List[Dict]:
    """抓取上交所撤否企业信息 (示例 URL 需要根据实际网站结构更新)"""
    url = "http://www.sse.com.cn/assortment/stock/list/namechange/"
    # 注意：上交所的撤否信息可能不在这个 URL，需要根据实际网站查找
    # 例如，可能在 IPO 审核专栏或信息披露 -> 首次公开发行 -> 审核状态
    # 这里使用一个假设的 URL 和结构作为示例
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 示例：查找包含“终止”、“撤回”关键词的列表项或表格行
        # 需要根据实际网页结构调整选择器
        # withdrawal_elements = soup.find_all('a', href=True, text=re.compile(r'终止|撤回'))
        # 示例结构假设 (请根据实际网站替换)
        # table = soup.find('table', {'class': '...'})
        # rows = table.find_all('tr')[1:] # 跳过表头
        # withdrawals = []
        # for row in rows:
        #     cells = row.find_all('td')
        #     if len(cells) >= 2:
        #         company_name = cells[0].get_text(strip=True)
        #         reason = cells[1].get_text(strip=True)
        #         withdrawals.append({"company_name": company_name, "reason": reason, "source": "SSE"})
        # return withdrawals

        # 由于无法直接访问真实 URL 获取结构，这里返回空列表
        logger.warning("上交所撤否信息抓取 URL 需要根据实际网站更新。")
        return []
    except Exception as e:
        logger.error(f"抓取上交所撤否信息失败: {e}")
        return []

def scrape_szse_withdrawals() -> List[Dict]:
    """抓取深交所撤否企业信息 (示例 URL 需要根据实际网站更新)"""
    url = "http://www.szse.cn/disclosure/refinancing/withdraw/index.html"
    # 注意：深交所的撤否信息 URL 可能已变更
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 示例：查找包含“终止”、“撤回”关键词的列表项或表格行
        # 需要根据实际网页结构调整选择器
        # withdrawal_elements = soup.find_all('a', href=True, text=re.compile(r'终止|撤回'))
        # 示例结构假设 (请根据实际网站替换)
        # table = soup.find('table', {'class': '...'})
        # rows = table.find_all('tr')[1:] # 跳过表头
        # withdrawals = []
        # for row in rows:
        #     cells = row.find_all('td')
        #     if len(cells) >= 2:
        #         company_name = cells[0].get_text(strip=True)
        #         reason = cells[1].get_text(strip=True)
        #         withdrawals.append({"company_name": company_name, "reason": reason, "source": "SZSE"})
        # return withdrawals

        # 由于无法直接访问真实 URL 获取结构，这里返回空列表
        logger.warning("深交所撤否信息抓取 URL 需要根据实际网站更新。")
        return []
    except Exception as e:
        logger.error(f"抓取深交所撤否信息失败: {e}")
        return []

def scrape_bse_withdrawals() -> List[Dict]:
    """抓取北交所撤否企业信息 (示例 URL 需要根据实际网站更新)"""
    url = "https://www.bse.cn/disclosure/withdrawal_list.html" # 假设 URL
    # 注意：北交所的撤否信息 URL 可能已变更
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 示例：查找包含“终止”、“撤回”关键词的列表项或表格行
        # 需要根据实际网页结构调整选择器
        # withdrawal_elements = soup.find_all('a', href=True, text=re.compile(r'终止|撤回'))
        # 示例结构假设 (请根据实际网站替换)
        # table = soup.find('table', {'class': '...'})
        # rows = table.find_all('tr')[1:] # 跳过表头
        # withdrawals = []
        # for row in rows:
        #     cells = row.find_all('td')
        #     if len(cells) >= 2:
        #         company_name = cells[0].get_text(strip=True)
        #         reason = cells[1].get_text(strip=True)
        #         withdrawals.append({"company_name": company_name, "reason": reason, "source": "BSE"})
        # return withdrawals

        # 由于无法直接访问真实 URL 获取结构，这里返回空列表
        logger.warning("北交所撤否信息抓取 URL 需要根据实际网站更新。")
        return []
    except Exception as e:
        logger.error(f"抓取北交所撤否信息失败: {e}")
        return []

def scrape_csrc_guidance() -> List[Dict]:
    """抓取证监会辅导备案信息"""
    # 这个 URL 来自您提供的文件内容
    url = "http://eid.csrc.gov.cn/csrcfd/index_f.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找包含辅导信息的表格
        table = soup.find('table', {'id': 'exposureTable'}) # 假设 ID，需根据实际页面确认
        if not table:
            logger.warning("在证监会网站未找到辅导信息表格。")
            return []

        rows = table.find_all('tr')[1:] # 跳过表头
        guidance_info = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4: # 确保有足够的列
                company_name_elem = cells[0].find('a')
                company_name = company_name_elem.get_text(strip=True) if company_name_elem else cells[0].get_text(strip=True)
                filing_date = cells[2].get_text(strip=True) # 备案时间
                status = cells[3].get_text(strip=True) # 辅导状态
                # 只抓取辅导备案状态的企业
                if status == '辅导备案':
                     guidance_info.append({
                         "company_name": company_name,
                         "filing_date": filing_date,
                         "status": status,
                         "source": "CSRC_Guidance"
                     })
        return guidance_info
    except Exception as e:
        logger.error(f"抓取证监会辅导备案信息失败: {e}")
        return []

def scrape_sse_star_neeq() -> List[Dict]:
    """抓取上交所星企航新三板拟上市信息"""
    # 这个 URL 来自您提供的文件内容
    url = "https://star.sseinfo.com/#/capitalRode/middleCompany?type=expense"
    # 注意：这个 URL 是一个 SPA (单页应用) 的路由，直接 requests.get 可能无法获取动态内容
    # 需要使用 Selenium 或 Playwright 等工具模拟浏览器行为
    logger.warning("上交所星企航网站为单页应用，需使用 Selenium/Playwright 等工具抓取动态内容。")
    # 这里无法直接抓取，返回空列表
    # 示例 (需要安装 selenium: pip install selenium)
    # from selenium import webdriver
    # from selenium.webdriver.common.by import By
    # from selenium.webdriver.chrome.options import Options
    # options = Options()
    # options.add_argument('--headless') # 无头模式
    # driver = webdriver.Chrome(options=options)
    # driver.get(url)
    # # 等待页面加载，查找元素...
    # # neeq_companies = ...
    # driver.quit()
    # return neeq_companies
    return []

def scrape_all_sources() -> Dict[str, List[Dict]]:
    """抓取所有指定来源的数据"""
    logger.info("开始从多个网站抓取数据...")
    data = {}
    data['SSE_Withdrawals'] = scrape_sse_withdrawals()
    data['SZSE_Withdrawals'] = scrape_szse_withdrawals()
    data['BSE_Withdrawals'] = scrape_bse_withdrawals()
    data['CSRC_Guidance'] = scrape_csrc_guidance()
    data['SSE_STAR_NEEQ'] = scrape_sse_star_neeq()

    total_items = sum(len(v) for v in data.values())
    logger.info(f"数据抓取完成，共获取 {total_items} 条记录。")
    return data

# --- 示例：如何使用 ---
# if __name__ == "__main__":
#     scraped_data = scrape_all_sources()
#     for source, items in scraped_data.items():
#         print(f"\n--- {source} ---")
#         for item in items[:5]: # 打印前5条
#             print(item)