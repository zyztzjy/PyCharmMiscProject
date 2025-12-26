# src/data_ingestion/collector.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
import logging
from pathlib import Path
import yaml
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class DataCollector:
    """数据采集器 - 支持多种数据源"""

    def __init__(self, config_path: str = "/Users/zjy/PyCharmMiscProject/config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def collect_guidance_reports(self) -> List[Dict]:
        """采集辅导备案报告"""
        reports = []

        # 爬取证监会辅导备案信息
        csrc_url = "http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zqdcl_1.shtml"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            response = requests.get(csrc_url, headers=headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # 解析辅导备案信息
            items = soup.find_all('li', class_='zx_list')
            for item in items:
                title_elem = item.find('a')
                date_elem = item.find('span', class_='date')

                if title_elem:
                    title = title_elem.get_text(strip=True)
                    date = date_elem.get_text(strip=True) if date_elem else "2024-01-01"

                    reports.append({
                        "title": title,
                        "publish_date": date,
                        "company_name": self._extract_company_name(title),
                        "doc_type": "guidance_report",
                        "source": "CSRC",
                        "url": title_elem.get('href', '')
                    })

        except Exception as e:
            logger.error(f"采集辅导报告失败: {e}")

        return reports

    def collect_ntb_announcements(self) -> List[Dict]:
        """采集新三板公告"""
        announcements = []

        # 新三板公告页面
        ntb_url = "http://www.neeq.com.cn/disclosure/notice.html"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            response = requests.get(ntb_url, headers=headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # 解析公告信息
            notice_items = soup.find_all('tr')[1:21]  # 获取前20条公告
            for item in notice_items:
                cols = item.find_all('td')
                if len(cols) >= 3:
                    code = cols[0].get_text(strip=True)
                    company = cols[1].get_text(strip=True)
                    title = cols[2].get_text(strip=True)
                    date = cols[3].get_text(strip=True) if len(cols) > 3 else "2024-01-01"

                    announcements.append({
                        "stock_code": code,
                        "company_name": company,
                        "title": title,
                        "publish_date": date,
                        "doc_type": "ntb_announcement",
                        "source": "NEEQ"
                    })

        except Exception as e:
            logger.error(f"采集新三板公告失败: {e}")

        return announcements

    def collect_financial_data_batch(self, company_codes: List[str]) -> List[Dict]:
        """批量采集财务数据（使用公开API）"""
        financial_data = []

        # 使用公开财务数据API（如Tushare等）
        # 这里以模拟调用为例
        for code in company_codes[:10]:  # 限制数量
            try:
                # 实际应调用真实API
                data = self._fetch_financial_from_api(code)
                if data:
                    financial_data.append(data)
                time.sleep(0.5)  # 控制请求频率
            except Exception as e:
                logger.error(f"获取{code}财务数据失败: {e}")

        return financial_data

    def _fetch_financial_from_api(self, code: str) -> Dict:
        """从API获取财务数据"""
        # 这里应该调用真实的财务数据API
        # 示例：使用Tushare、AkShare等
        return {
            "company_code": code,
            "company_name": f"公司{code}",
            "revenue": "1000万",
            "net_profit": "200万",
            "total_assets": "5000万",
            "total_liabilities": "2000万",
            "report_period": "2023年",
            "doc_type": "financial_report",
            "source": "financial_api"
        }

    def collect_regulatory_data(self, days: int = 30) -> List[Dict]:
        """采集监管数据（公告、问询函等）"""
        data = []

        # 上交所公告
        sse_data = self._collect_sse_announcements(days)
        data.extend(sse_data)

        # 深交所公告
        szse_data = self._collect_szse_announcements(days)
        data.extend(szse_data)

        # 新三板公告
        nse_data = self._collect_nse_announcements(days)
        data.extend(nse_data)

        logger.info(f"采集到 {len(data)} 条监管数据")
        return data

    def _collect_sse_announcements(self, days: int) -> List[Dict]:
        """采集上交所公告"""
        # 实际实现需要处理反爬和API调用
        # 这里提供示例代码结构
        base_url = self.config['data_sources']['regulatory']['sse']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        announcements = []

        # 模拟数据
        announcements.append({
            "source": "SSE",
            "stock_code": "600000",
            "company_name": "浦发银行",
            "title": "关于收到上海证券交易所问询函的公告",
            "content": "上海证券交易所对公司年报提出问询...",
            "publish_date": "2024-01-15",
            "doc_type": "inquiry_letter",
            "url": "http://example.com/doc1.pdf"
        })

        return announcements

    def _collect_szse_announcements(self, days: int) -> List[Dict]:
        """采集深交所公告"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.szse.cn/',
        }

        try:
            # 深交所信息披露页面
            szse_url = "https://www.szse.cn/disclosure/listing/index.html"
            response = requests.get(szse_url, headers=headers, timeout=10)
            response.encoding = 'utf-8'

            if response.status_code != 200:
                logger.warning(f"访问深交所公告页面失败: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            # 解析公告列表
            announcement_items = soup.find_all('tr')[1:21]  # 获取前20条，跳过表头
            announcements = []

            for item in announcement_items:
                tds = item.find_all('td')
                if len(tds) >= 3:
                    code = tds[0].get_text(strip=True)
                    company = tds[1].get_text(strip=True)
                    title = tds[2].get_text(strip=True)

                    announcements.append({
                        "stock_code": code,
                        "company_name": company,
                        "title": title,
                        "publish_date": "2024-01-01",  # 实际应从页面获取
                        "doc_type": "company_announcement",
                        "source": "SZSE",
                        "content": f"{code} {company} {title}"
                    })

            return announcements

        except Exception as e:
            logger.error(f"采集深交所公告失败: {e}")
            return []

    def _collect_nse_announcements(self, days: int) -> List[Dict]:
        """采集北交所公告"""
        # 模拟实现
        announcements = []
        announcements.append({
            "source": "BSE",
            "stock_code": "830799",
            "company_name": "艾融软件",
            "title": "2023年年度报告",
            "content": "公司2023年实现营业收入...",
            "publish_date": "2024-04-20",
            "doc_type": "annual_report",
            "url": "http://www.bse.cn/disclosure/file_000001.html"
        })
        return announcements

    def collect_financial_data(self, company_codes: List[str]) -> Dict:
        """采集企业财务数据"""
        financial_data = {}

        for code in company_codes:
            # 实际应调用财务数据API
            financial_data[code] = {
                "company_code": code,
                "revenue": "1000万元",
                "net_profit": "200万元",
                "assets": "5000万元",
                "liabilities": "2000万元",
                "report_period": "2023Q3"
            }

        return financial_data

    def collect_news_sentiment(self, keywords: List[str], days: int = 7) -> List[Dict]:
        """采集新闻舆情数据"""
        news_items = []

        for keyword in keywords:
            # 调用新闻API或爬虫
            news_items.extend(self._search_news(keyword, days))

        return news_items

    def collect_industry_data(self, industry_codes: List[str]) -> Dict:
        """采集行业数据"""
        industry_data = {}

        for code in industry_codes:
            # 获取行业分析报告、榜单等
            industry_data[code] = {
                "industry_name": self._get_industry_name(code),
                "ranking": self._get_industry_ranking(code),
                "growth_rate": "15.5%",
                "listed_companies": ["600000", "000001"],
                "policy_support": ["专精特新支持政策"]
            }

        return industry_data

    def collect_supply_chain_data(self, company_code: str) -> Dict:
        """采集供应链数据"""
        # 实际实现需要从企查查等数据库获取
        supply_chain = {
            "company_code": company_code,
            "upstream_suppliers": [
                {"name": "供应商A", "relationship": "主要原材料供应商"},
                {"name": "供应商B", "relationship": "零部件供应商"}
            ],
            "downstream_customers": [
                {"name": "客户A", "relationship": "第一大客户", "revenue_share": "30%"},
                {"name": "客户B", "relationship": "战略客户"}
            ],
            "competitors": [
                {"name": "竞争对手A", "market_share": "25%"},
                {"name": "竞争对手B", "market_share": "20%"}
            ]
        }

        return supply_chain

    def collect_from_akshare(self) -> List[Dict]:
        """使用AkShare获取数据"""
        try:
            import akshare as ak

            # 获取IPO数据
            ipo_data = ak.stock_ipo_info()

            # 获取公告数据
            announcement_data = []
            for _, row in ipo_data.iterrows():
                announcement_data.append({
                    "company_name": row.get('sec_name', ''),
                    "stock_code": row.get('sec_code', ''),
                    "title": row.get('title', ''),
                    "publish_date": row.get('publish_date', ''),
                    "content": row.get('summary', ''),
                    "doc_type": "ipo_announcement",
                    "source": "akshare"
                })

            return announcement_data
        except ImportError:
            logger.error("AkShare未安装")
            return []
        except Exception as e:
            logger.error(f"从AkShare采集数据失败: {e}")
            return []

    def collect_from_juchao(self) -> List[Dict]:
        """从巨潮资讯网获取公告数据"""
        try:
            import akshare as ak

            # 获取最新公告
            announcement = ak.stock_announcement_cninfo(date="20240101")

            processed_data = []
            for _, row in announcement.iterrows():
                processed_data.append({
                    "title": row.get('announcement_title', ''),
                    "company_name": row.get('sec_name', ''),
                    "stock_code": row.get('sec_code', ''),
                    "publish_date": row.get('announcement_date', ''),
                    "content": row.get('announcement_content', '')[:1000],  # 限制长度
                    "doc_type": "company_announcement",
                    "source": "juchao"
                })

            return processed_data
        except ImportError:
            logger.error("AkShare未安装")
            return []
        except Exception as e:
            logger.error(f"从巨潮资讯采集数据失败: {e}")
            return []

    def collect_from_exchange(self) -> List[Dict]:
        """从交易所获取数据"""
        try:
            import akshare as ak

            # 获取上交所主板退市整理期公司
            sse_delist = ak.stock_info_sh_delist()

            processed_data = []
            for _, row in sse_delist.iterrows():
                processed_data.append({
                    "company_name": row.get('COMPANY_ABBR', ''),
                    "stock_code": row.get('SECURITY_CODE', ''),
                    "delist_reason": row.get('DELIST_REASON', ''),
                    "delist_date": row.get('DELIST_DATE', ''),
                    "doc_type": "delist_company",
                    "source": "sse"
                })

            return processed_data
        except ImportError:
            logger.error("AkShare未安装")
            return []
        except Exception as e:
            logger.error(f"从交易所采集数据失败: {e}")
            return []

    def collect_szse_ipo_data(self) -> List[Dict]:
        """采集深交所IPO项目动态数据"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.szse.cn/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # 深交所IPO项目动态页面
        szse_ipo_url = "https://listing.szse.cn/projectdynamic/ipo/index.html"

        try:
            response = requests.get(szse_ipo_url, headers=headers, timeout=15)
            response.encoding = 'utf-8'

            if response.status_code != 200:
                logger.error(f"访问深交所IPO页面失败: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            # 查找IPO项目列表 - 根据实际页面结构调整选择器
            # 这里是示例选择器，需要根据实际页面结构进行调整
            project_container = soup.find('div', {'id': 'project-list'}) or soup.find('table', class_='tbl-list')

            if not project_container:
                # 尝试其他可能的容器
                project_container = soup.find('div', class_='project-dynamic') or soup.find('ul', class_='project-list')

            ipo_data = []

            if project_container:
                # 根据页面结构解析项目数据
                if project_container.name == 'table':
                    # 如果是表格结构
                    rows = project_container.find_all('tr')[1:]  # 跳过表头
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            company_name = cells[0].get_text(strip=True)
                            current_status = cells[1].get_text(strip=True)
                            update_date = cells[2].get_text(strip=True)

                            # 获取详情链接
                            detail_link = cells[3].find('a')
                            detail_url = ""
                            if detail_link:
                                href = detail_link.get('href')
                                if href:
                                    if href.startswith('http'):
                                        detail_url = href
                                    else:
                                        detail_url = urljoin(szse_ipo_url, href)

                            ipo_data.append({
                                "company_name": company_name,
                                "current_status": current_status,
                                "update_date": update_date,
                                "detail_url": detail_url,
                                "doc_type": "ipo_process",
                                "source": "SZSE",
                                "content": f"【深交所IPO项目动态】公司名称：{company_name}，当前状态：{current_status}，更新日期：{update_date}，详情链接：{detail_url}"
                            })
                else:
                    # 如果是列表结构
                    items = project_container.find_all(['li', 'div'], class_=['project-item', 'item'])
                    for item in items:
                        # 根据实际页面结构调整解析逻辑
                        company_elem = item.find(class_='company-name') or item.find('h3')
                        status_elem = item.find(class_='status') or item.find('span')
                        date_elem = item.find(class_='date') or item.find('time')

                        company_name = company_elem.get_text(strip=True) if company_elem else ""
                        current_status = status_elem.get_text(strip=True) if status_elem else ""
                        update_date = date_elem.get_text(strip=True) if date_elem else ""

                        # 获取详情链接
                        detail_link = item.find('a')
                        detail_url = ""
                        if detail_link:
                            href = detail_link.get('href')
                            if href:
                                if href.startswith('http'):
                                    detail_url = href
                                else:
                                    detail_url = urljoin(szse_ipo_url, href)

                        if company_name:  # 确保公司名称存在
                            ipo_data.append({
                                "company_name": company_name,
                                "current_status": current_status,
                                "update_date": update_date,
                                "detail_url": detail_url,
                                "doc_type": "ipo_process",
                                "source": "SZSE",
                                "content": f"【深交所IPO项目动态】公司名称：{company_name}，当前状态：{current_status}，更新日期：{update_date}，详情链接：{detail_url}"
                            })
            else:
                logger.warning("未找到IPO项目列表容器")

                # 尝试另一种解析方式
                # 查找所有包含项目信息的元素
                potential_items = soup.find_all(['tr', 'li', 'div'], recursive=True)
                for item in potential_items[:50]:  # 限制数量
                    text = item.get_text(strip=True)
                    if '发行人' in text or '保荐机构' in text or '审核状态' in text:
                        # 尝试从文本中提取信息
                        import re
                        company_match = re.search(r'发行人[:：]?\s*([^\s\n\r]+)', text)
                        status_match = re.search(r'审核状态[:：]?\s*([^\s\n\r]+)', text)
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)

                        company_name = company_match.group(1) if company_match else ""
                        current_status = status_match.group(1) if status_match else ""
                        update_date = date_match.group(1) if date_match else ""

                        if company_name:
                            ipo_data.append({
                                "company_name": company_name,
                                "current_status": current_status,
                                "update_date": update_date,
                                "detail_url": szse_ipo_url,
                                "doc_type": "ipo_process",
                                "source": "SZSE",
                                "content": f"【深交所IPO项目动态】公司名称：{company_name}，当前状态：{current_status}，更新日期：{update_date}，详情链接：{szse_ipo_url}"
                            })

            logger.info(f"成功采集深交所IPO数据 {len(ipo_data)} 条")
            return ipo_data

        except Exception as e:
            logger.error(f"采集深交所IPO数据失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _search_news(self, keyword: str, days: int) -> List[Dict]:
        """内部方法：搜索新闻"""
        # 模拟实现
        return []

    def _get_industry_name(self, code: str) -> str:
        """内部方法：获取行业名称"""
        # 模拟实现
        return f"行业{code}"

    def _get_industry_ranking(self, code: str) -> str:
        """内部方法：获取行业排名"""
        # 模拟实现
        return "前10名"

    def _extract_company_name(self, text: str) -> str:
        """内部方法：从文本中提取公司名称"""
        import re
        # 尝试匹配公司名称模式
        patterns = [
            r'关于(?:.*)终止对(.*)首次公开发行股票',
            r'(.*)撤回上市申请',
            r'(.*)终止审核',
            r'(.*)撤回',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return "未知公司"


def collect_government_lists():
    """
    爬取真实政府榜单数据
    """
    all_listings = []

    # 1. 爬取工信部专精特新名单
    try:
        xinxin_data = _scrape_xinxin_list()
        all_listings.extend(xinxin_data)
    except Exception as e:
        logger.error(f"爬取专精特新名单失败: {e}")

    # 2. 爬取各省上市后备企业名单
    try:
        houbei_data = _scrape_houbei_list()
        all_listings.extend(houbei_data)
    except Exception as e:
        logger.error(f"爬取上市后备名单失败: {e}")

    # 3. 爬取IPO审核撤否企业信息
    try:
        withdraw_data = _scrape_withdraw_list()
        all_listings.extend(withdraw_data)
    except Exception as e:
        logger.error(f"爬取撤否企业名单失败: {e}")

    return all_listings


def _scrape_xinxin_list():
    """改进的专精特新企业名单爬取"""
    import requests
    from bs4 import BeautifulSoup
    import time

    listings = []

    # 使用真实的工信部网站数据
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    # 实际的专精特新公布页面
    try:
        # 使用真实的专精特新发布页面
        url = "https://www.miit.gov.cn/jgsj/zbes/gzdt/index.html"
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找包含专精特新内容的链接
        links = soup.find_all('a', href=lambda x: x and '专精特新' in x)

        for link in links[:5]:  # 限制数量
            href = link.get('href')
            if href.startswith('/'):
                full_url = 'https://www.miit.gov.cn' + href
            else:
                full_url = href

            # 访问具体页面获取企业信息
            detail_response = requests.get(full_url, headers=headers)
            detail_response.encoding = 'utf-8'
            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

            # 根据实际页面结构提取企业信息
            # 这里需要根据实际页面结构调整
            company_elements = detail_soup.find_all(text=lambda text: text and '公司' in text)

            for elem in company_elements[:10]:  # 限制数量
                if len(str(elem)) < 100:  # 过滤过短的文本
                    company_name = str(elem).strip()
                    if '公司' in company_name:
                        listings.append({
                            "company_name": company_name,
                            "stock_code": "",  # 通常专精特新名单不包含股票代码
                            "list_name": "工信部专精特新企业",
                            "publish_date": datetime.now().strftime('%Y-%m-%d'),
                            "source_url": full_url
                        })

            time.sleep(2)  # 控制请求频率

    except Exception as e:
        logger.error(f"爬取专精特新列表失败: {e}")

    return listings


def _scrape_houbei_list():
    """爬取各省上市后备企业名单 - 修复版本"""
    import requests
    from bs4 import BeautifulSoup
    import time

    listings = []

    # 更新的政府网站URL
    province_urls = {
        "广东": "http://gdii.gd.gov.cn/zwgk_n/zdgk_n/qtbt_1/index.html",  # 更新的广东工信厅URL
        "江苏": "http://gxt.jiangsu.gov.cn/col/col70796/index.html",  # 更新的江苏工信厅URL
        "浙江": "http://jxt.zj.gov.cn/col/col152898/index.html",  # 更新的浙江经信厅URL
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    for province, url in province_urls.items():
        try:
            # 使用RobustDataCollector的重试机制
            from src.data_ingestion.data_collector import RobustDataCollector
            collector = RobustDataCollector()
            response = collector.collect_with_retry(url, max_retries=3)

            if not response:
                logger.error(f"无法访问{province}网站: {url}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # 根据实际页面结构调整选择器
            # 尝试多种可能的元素选择器
            selectors = [
                '.company-item', '.company-name', 'li', 'div', 'p'
            ]

            elements = []
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    break

            for elem in elements:
                company_text = elem.get_text(strip=True)
                if company_text and len(company_text) > 2 and '公司' in company_text:
                    listings.append({
                        "company_name": company_text,
                        "stock_code": "",
                        "list_name": f"{province}上市后备企业",
                        "publish_date": datetime.now().strftime('%Y-%m-%d'),
                        "source_url": url
                    })

            time.sleep(2)

        except Exception as e:
            logger.error(f"爬取{province}上市后备名单失败: {e}")
            import traceback
            traceback.print_exc()

    return listings


def _scrape_withdraw_list():
    """爬取IPO撤否企业信息"""
    import requests
    from bs4 import BeautifulSoup
    import time

    listings = []

    # 交易所撤否企业信息页面
    exchange_urls = {
        "上交所": "http://www.sse.com.cn/disclosure/credibility/supervision/inquiries/",
        "深交所": "http://www.szse.cn/disclosure/supervision/inquiry/index.html",
        "北交所": "http://www.bse.cn/disclosure/inquiries/"
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'http://www.baidu.com'
    }

    for exchange, url in exchange_urls.items():
        try:
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # 查找撤否公告
            announcements = soup.find_all('a', href=lambda x: x and 'withdraw' in x.lower() or '撤回' in x)
            for ann in announcements[:10]:  # 限制数量
                title = ann.get_text(strip=True)
                link = ann.get('href')

                if link and not link.startswith('http'):
                    link = url + link

                listings.append({
                    "company_name": _extract_company_name(title),
                    "stock_code": _extract_stock_code(title),
                    "list_name": f"{exchange}IPO撤否企业",
                    "publish_date": "2024-01-01",
                    "source_url": link,
                    "title": title
                })

            time.sleep(3)

        except Exception as e:
            logger.error(f"爬取{exchange}撤否信息失败: {e}")

    return listings


def _extract_company_name(text):
    """从文本中提取公司名称"""
    import re
    # 尝试匹配公司名称模式
    patterns = [
        r'关于(?:.*)终止对(.*)首次公开发行股票',
        r'(.*)撤回上市申请',
        r'(.*)终止审核',
        r'(.*)撤回',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return "未知公司"


def _extract_stock_code(text):
    """从文本中提取股票代码"""
    import re
    # 匹配股票代码模式
    match = re.search(r'(?:股票代码|证券代码|股票简称)?\s*([0-9]{6})', text)
    if match:
        return match.group(1)
    return ""


def clean_and_process_data(raw_data: List[Dict]) -> List[Dict]:
    """清洗和处理原始数据"""
    processed = []

    for item in raw_data:
        # 数据清洗
        cleaned_item = {
            "content": _clean_text(item.get("title", "") + " " + item.get("content", "")),
            "metadata": {
                "company_name": item.get("company_name", ""),
                "stock_code": item.get("stock_code", ""),
                "doc_type": item.get("doc_type", "general"),
                "source": item.get("source", "unknown"),
                "publish_date": item.get("publish_date", ""),
                "source_url": item.get("source_url", ""),
                "list_name": item.get("list_name", ""),
                "title": item.get("title", "")
            },
            "doc_id": f"doc_{hash(item.get('content', '')[:50]) % 1000000}"
        }
        processed.append(cleaned_item)

    return processed


def _clean_text(text: str) -> str:
    """清理文本数据"""
    import re
    # 移除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()（）【】\[\]]', ' ', text)
    return text.strip()
