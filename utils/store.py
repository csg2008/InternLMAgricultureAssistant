# -*- encoding: utf-8 -*-
"""
向量数据库
"""

import os
from typing import List

from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 获取文件路径函数
def get_dir_files(dir_path: str, suffix: List[str]) -> List[str]:
    """
    获取指定目录中的所有文本与markdown文件列表

    args:
        dir_path 目标文件夹路径
        suffix   文件后缀名列表

    return:
        file_list 文件路径列表
    """

    file_list = []
    for filepath, _, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            _, file_ext = os.path.splitext(filename)
            if file_ext in suffix:
                file_list.append(os.path.join(filepath, filename))
    return file_list

def load_documents(dir_path: str) -> List[Document]:
    """
    加载指定目录中的文档

    args:
        dir_path 目标文件夹路径

    return:
        docs     文档列表
    """

    # 首先调用上文定义的函数得到目标文件路径列表
    files= get_dir_files(dir_path, ['.txt', '.md'])
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for file in tqdm(files):
        file_type = file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(file)
        elif file_type == 'txt':
            loader = TextLoader(file, autodetect_encoding = True)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue

        docs.extend(loader.load())
    return docs

def save_to_chroma(persist_directory: str, embedding, chunks: List[Document]) -> None:
    """
    将向量数据库持久化到磁盘上

    args:
        persist_directory  持久化目录
        embedding          词向量模型
        chunks             文档列表
    """

    db = Chroma.from_documents(
        documents = chunks,
        embedding = embedding,
        persist_directory = persist_directory
    )

    # 将加载的向量数据库持久化到磁盘上
    db.persist()

def generate_vector_store(doc_path: str, db_path: str, embedding_model: str) -> None:
    """
    生成向量数据库

    args:
        doc_path         文档路径
        db_path          向量数据库路径
        embedding_model  词向量模型名
    """
    # 加载目标文件
    all_docs = load_documents(doc_path)

    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(all_docs)

    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings( model_name = embedding_model)

    # 构建向量数据库
    save_to_chroma(db_path, embeddings, split_docs)

def load_vector_store(db_path: str, embedding_model: str) -> Chroma:
    """
    加载向量数据库

    args:
        db_path          向量数据库路径
        embedding_model  词向量模型名

    return:
        db               向量数据库
    """

    embeddings = HuggingFaceEmbeddings(model_name = embedding_model)

    # 加载数据库
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    return db
