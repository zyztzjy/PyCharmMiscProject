# src/processing/document_processor.py
import re
import pdfplumber
import docx
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import Dict, List, Any
import hashlib
from datetime import datetime


class DocumentProcessor:
    """文档处理器 - 支持多种格式"""

    @staticmethod
    def process_pdf(file_path: str) -> Dict:
        """处理PDF文档"""
        text_content = ""
        metadata = {}

        try:
            with pdfplumber.open(file_path) as pdf:
                # 提取文本
                for page in pdf.pages:
                    text_content += page.extract_text() + "\n"

                # 提取元数据
                metadata = {
                    "pages": len(pdf.pages),
                    "format": "PDF",
                    "extracted_at": datetime.now().isoformat()
                }

        except Exception as e:
            print(f"处理PDF失败: {e}")

        return {
            "content": text_content,
            "metadata": metadata,
            "file_hash": DocumentProcessor._calculate_hash(file_path)
        }

    @staticmethod
    def process_html(html_content: str) -> Dict:
        """处理HTML文档"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        # 提取正文
        text = soup.get_text()

        # 清理文本
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # 提取标题和关键信息
        title = soup.title.string if soup.title else ""

        return {
            "content": text,
            "metadata": {
                "title": title,
                "format": "HTML",
                "extracted_at": datetime.now().isoformat()
            }
        }

    @staticmethod
    def process_financial_table(table_data: List[Dict]) -> str:
        """处理财务表格数据"""
        # 将表格数据转换为结构化文本
        structured_text = "财务数据摘要：\n"

        for item in table_data:
            structured_text += f"- {item.get('指标', '')}: {item.get('数值', '')} {item.get('单位', '')}\n"

        return structured_text

    @staticmethod
    def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """文档分块"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 尝试在句子边界处截断
            if end < len(text):
                # 找最近的句号
                period_pos = text.rfind('。', start, end)
                if period_pos != -1 and period_pos > start + chunk_size // 2:
                    end = period_pos + 1

            chunk = text[start:end]

            chunks.append({
                "content": chunk,
                "start": start,
                "end": end,
                "chunk_id": f"chunk_{len(chunks)}"
            })

            start = end - overlap

        return chunks

    @staticmethod
    def _calculate_hash(file_path: str) -> str:
        """计算文件哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()