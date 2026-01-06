# soure/document_processor/pdf_processor.py
import os
import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Optional, Any
import hashlib
import tempfile
from datetime import datetime
import re


class PDFProcessor:
    """PDF文档处理器"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_chunk_size = self.config.get('pdf', {}).get('max_chunk_size', 1000)
        self.overlap_size = self.config.get('pdf', {}).get('overlap_size', 200)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF提取文本并分块"""
        try:
            if not os.path.exists(pdf_path):
                print(f"PDF文件不存在: {pdf_path}")
                return []

            doc = fitz.open(pdf_path)
            full_text = ""

            # 从文件路径中提取企业名称
            file_name = os.path.basename(pdf_path)
            file_dir = os.path.dirname(pdf_path)

            # 尝试从目录名提取企业名称（如"撤否企业/欣强电子"）
            company_name = self._extract_company_name_from_path(file_dir, file_name)

            metadata = {
                "source": file_name,  # 文件名
                "file_path": pdf_path,  # 完整路径
                "file_name": file_name,  # 文件名
                "file_dir": file_dir,  # 目录路径
                "company_name": company_name,  # 新增：企业名称
                "file_size": os.path.getsize(pdf_path),
                "page_count": len(doc),
                "extraction_time": datetime.now().isoformat(),
                "document_type": self._detect_document_type(pdf_path)
            }

            # 提取所有页面文本
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text += f"第{page_num + 1}页:\n{text}\n\n"

            doc.close()

            if not full_text.strip():
                print(f"PDF文件没有提取到文本: {pdf_path}")
                return []

            # 对文本进行分块
            chunks = self._chunk_text(full_text, metadata)

            print(f"从 {pdf_path} 提取了 {len(chunks)} 个文本块，企业名称: {company_name}")
            return chunks

        except Exception as e:
            print(f"提取PDF文本失败: {e}")
            return []

    def _extract_company_name_from_path(self, file_dir: str, file_name: str) -> str:
        """从文件路径中提取企业名称"""
        try:
            # 方法1：从文件名中提取（移除扩展名和常见词汇）
            base_name = os.path.splitext(file_name)[0]

            # 移除常见的文件描述词汇
            common_terms = [
                "股份有限公司", "有限公司", "公司",
                "招股说明书", "招股书", "招股",
                "审计报告", "审计", "报告",
                "法律意见书", "法律意见", "意见书",
                "发行保荐书", "发行保荐", "保荐书",
                "上市保荐书", "上市保荐",
                "财务报表", "财务报告", "财务",
                "年度报告", "年报", "报告"
            ]

            # 尝试从文件名中提取
            for term in common_terms:
                if term in base_name:
                    # 保留企业名称部分
                    company_part = base_name.split(term)[0].strip()
                    if company_part:  # 如果分割后有内容
                        return company_part

            # 方法2：从目录名中提取
            dir_name = os.path.basename(file_dir)
            if dir_name and dir_name != ".":
                # 移除目录中的分类词汇
                category_terms = ["撤否企业", "长期辅导企业", "新三板企业",
                                  "供应链分析", "财务报告", "舆情报告", "行业分析"]
                for term in category_terms:
                    dir_name = dir_name.replace(term, "").strip()
                if dir_name:
                    return dir_name

            # 方法3：返回清理后的文件名
            return base_name

        except Exception as e:
            print(f"提取企业名称失败: {e}")
            return file_name  # 失败时返回原始文件名

    def _detect_document_type(self, pdf_path: str) -> str:
        """检测文档类型"""
        filename = os.path.basename(pdf_path).lower()

        if any(keyword in filename for keyword in ["财务", "年报", "审计", "报表", "finance"]):
            return "财务报告"
        elif any(keyword in filename for keyword in ["研报", "分析", "研究", "industry"]):
            return "研究报告"
        elif any(keyword in filename for keyword in ["招股", "ipo", "prospectus"]):
            return "招股说明书"
        elif any(keyword in filename for keyword in ["新闻", "舆情", "报道", "news"]):
            return "新闻舆情"
        elif any(keyword in filename for keyword in ["专利", "技术", "patent"]):
            return "技术专利"
        elif any(keyword in filename for keyword in ["报告", "document"]):
            return "报告文档"
        else:
            return "其他文档"

    def _chunk_text(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """将长文本分块"""
        chunks = []

        # 清理文本
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            # 如果没有明显段落，尝试按句子分割
            sentences = re.split(r'(?<=[。！？\.!?])\s+', text)
            paragraphs = [s.strip() for s in sentences if s.strip()]

        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # 如果当前段落加上新段落超过限制，保存当前块并开始新块
            if current_length + para_length > self.max_chunk_size and current_chunk:
                # 保存当前块
                chunk_dict = self._create_chunk_dict(current_chunk, metadata)
                chunks.append(chunk_dict)

                # 开始新块（带重叠）
                if self.overlap_size > 0 and len(current_chunk) > self.overlap_size:
                    # 取最后overlap_size个字符作为重叠
                    overlap_text = current_chunk[-self.overlap_size:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
                current_length = len(current_chunk)
            else:
                # 添加到当前块
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_length = len(current_chunk)

            # 如果单个段落超过最大块大小，需要强制分割
            if para_length > self.max_chunk_size:
                # 先保存当前块
                if current_chunk and current_chunk != para:
                    chunk_dict = self._create_chunk_dict(current_chunk, metadata)
                    chunks.append(chunk_dict)

                # 分割超大段落
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    chunk_dict = self._create_chunk_dict(sub_chunk, metadata)
                    chunks.append(chunk_dict)

                current_chunk = ""
                current_length = 0

        # 保存最后一个块
        if current_chunk:
            chunk_dict = self._create_chunk_dict(current_chunk, metadata)
            chunks.append(chunk_dict)

        return chunks

    def _create_chunk_dict(self, content: str, metadata: dict) -> Dict[str, Any]:
        """创建分块字典"""
        chunk_id = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]

        return {
            "content": content,
            "metadata": {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_size": len(content),
                "is_partial": True,
                "created_time": datetime.now().isoformat()
            }
        }

    def _split_large_paragraph(self, text: str) -> List[str]:
        """分割大段落文本"""
        # 按句子分割
        sentences = re.split(r'(?<=[。！？\.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_pdf_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """处理目录中的所有PDF文件"""
        all_chunks = []

        if not os.path.exists(directory_path):
            print(f"目录不存在: {directory_path}")
            return all_chunks

        # 查找所有PDF文件
        pdf_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        print(f"找到 {len(pdf_files)} 个PDF文件")

        # 处理每个PDF文件
        for pdf_file in pdf_files:
            print(f"处理文件: {pdf_file}")

            # 提取文本
            text_chunks = self.extract_text_from_pdf(pdf_file)
            if text_chunks:
                all_chunks.extend(text_chunks)

        print(f"总共提取了 {len(all_chunks)} 个文本块")
        return all_chunks