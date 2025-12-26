import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


# 使用Selenium来处理动态页面
def get_dynamic_page(url):
    options = Options()
    options.headless = True  # 无头模式
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # 等待页面加载完成
    time.sleep(5)

    page_source = driver.page_source
    driver.quit()
    return page_source


# 解析深交所的IPO页面
def parse_szse_ipo(url):
    html = get_dynamic_page(url)
    soup = BeautifulSoup(html, 'html.parser')

    # 找到所有“终止阶段”相关的元素，根据页面结构调整
    ipo_list = soup.find_all('tr', {'class': 'ipo-row'})  # 假设‘ipo-row’是IPO企业的行
    terminated_companies = []

    for row in ipo_list:
        status = row.find('td', {'class': 'status'}).get_text(strip=True)
        if "终止" in status:
            company_name = row.find('td', {'class': 'company-name'}).get_text(strip=True)
            terminated_companies.append(company_name)

    return terminated_companies


# 解析上交所的IPO页面
def parse_sse_ipo(url):
    html = get_dynamic_page(url)
    soup = BeautifulSoup(html, 'html.parser')

    # 假设每个IPO项目有一个特定的class来标识
    ipo_list = soup.find_all('div', {'class': 'ipo-item'})
    terminated_companies = []

    for item in ipo_list:
        status = item.find('span', {'class': 'status'}).get_text(strip=True)
        if "终止" in status:
            company_name = item.find('a', {'class': 'company-name'}).get_text(strip=True)
            terminated_companies.append(company_name)

    return terminated_companies


# 解析北交所的IPO页面
def parse_bse_ipo(url):
    html = get_dynamic_page(url)
    soup = BeautifulSoup(html, 'html.parser')

    # 假设每个IPO项目信息的class为‘ipo-entry’
    ipo_list = soup.find_all('div', {'class': 'ipo-entry'})
    terminated_companies = []

    for entry in ipo_list:
        status = entry.find('span', {'class': 'status'}).get_text(strip=True)
        if "终止" in status:
            company_name = entry.find('a', {'class': 'company-name'}).get_text(strip=True)
            terminated_companies.append(company_name)

    return terminated_companies


# 示例URL
szse_url = "https://listing.szse.cn/projectdynamic/ipo/index.html"
sse_url = "http://www.sse.com.cn/listing/renewal/ipo/"
bse_url = "https://www.bse.cn/audit/project_news.html"

# 爬取数据
szse_terminated = parse_szse_ipo(szse_url)
sse_terminated = parse_sse_ipo(sse_url)
bse_terminated = parse_bse_ipo(bse_url)

# 输出结果
print("深交所终止阶段企业:", szse_terminated)
print("上交所终止阶段企业:", sse_terminated)
print("北交所终止阶段企业:", bse_terminated)
