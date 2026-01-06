# scripts/init_pdf_documents.py
import os
import sys

from soure.document.document_processor import PDFProcessor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from soure.embedding.vectorizer_qwen import QwenVectorizer
import yaml
import argparse


def init_pdf_documents(config_path: str = "config/config.yaml"):
    """初始化PDF文档到向量数据库"""

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化向量化器
    vectorizer = QwenVectorizer(config)

    # 初始化PDF处理器
    pdf_processor = PDFProcessor(config)

    # 获取PDF目录配置
    pdf_directories = config.get('pdf', {}).get('pdf_directories', [])

    total_chunks = 0

    # 处理每个目录
    for directory in pdf_directories:
        if os.path.exists(directory):
            print(f"处理目录: {directory}")

            # 处理目录中的PDF文件
            chunks = pdf_processor.process_pdf_directory(directory)

            if chunks:
                # 存储到向量数据库
                success_count = vectorizer.store_documents(chunks)
                total_chunks += success_count
                print(f"  成功存储 {success_count} 个文档块")
            else:
                print(f"  目录中没有找到PDF文件或处理失败")
        else:
            print(f"  目录不存在: {directory}")

    print(f"\n初始化完成！总共存储了 {total_chunks} 个文档块")

    # 显示统计信息
    stats = vectorizer.get_collection_stats()
    print(f"\n集合统计:")
    print(f"  总文档数: {stats.get('total_documents', 0)}")
    print(f"  文档类型分布:")
    for doc_type, count in stats.get('document_types', {}).items():
        print(f"    {doc_type}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初始化PDF文档到向量数据库")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    init_pdf_documents(args.config)