"""extract feature and search with user query."""

import os
from typing import List

from BCEmbedding.tools.langchain import BCERerank
from langchain.docstore.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from loguru import logger


class Retriever:
    """Tokenize and extract features from the project's documents, for use in
    the reject pipeline and response pipeline."""

    def __init__(self, embeddings, reranker, retriever_path: str = '') -> None:
        """Init with model device type and config."""
        self.retriever = None

        if not os.path.exists(embeddings) or not os.path.exists(reranker):
            raise RuntimeError('embeddings and reranker path not found')

        # load text2vec and rerank model
        logger.info('loading test2vec and rerank models')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={
                'batch_size': 1,
                'normalize_embeddings': True
            })
        self.embeddings.client = self.embeddings.client.half()
        reranker_args = {
            'model': reranker,
            'top_n': 7,
            'device': 'cuda',
            'use_fp16': True
        }
        self.reranker = BCERerank(**reranker_args)

        if os.path.exists(retriever_path):
            self.retriever = Chroma(
                persist_directory=retriever_path,
                embedding_function=self.embeddings
            ).as_retriever(
                search_type='similarity',
                search_kwargs={
                    # 'score_threshold': 0.15,
                    'k': 30
                })
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.retriever
            )

    def query(self,
              question: str,
              sep: str = '\n',
              context_max_length: int = 16000):
        """Processes a query and returns the best match from the vector store
        database. If the question is rejected, returns None.

        Args:
            question (str): The question asked by the user.

        Returns:
            str: The best matching chunk, or None.
            str: The best matching text, or None
        """
        if question is None or len(question) < 1:
            return None, None, []

        if len(question) > 512:
            logger.warning('input too long, truncate to 512')
            question = question[0:512]

        chunks = []
        context = ''
        references = []

        docs = self.compression_retriever.invoke(question)
        for doc in docs:
            chunk = doc.page_content
            chunks.append(chunk)


            source = doc.metadata['source']
            logger.info(f'target {source} file length {len(doc.page_content)}')
            if len(doc.page_content) + len(context) > context_max_length:
                if source in references:
                    continue
                references.append(source)
                # add and break
                add_len = context_max_length - len(context)
                if add_len <= 0:
                    break

                # chunk not in file_text
                context += chunk
                context += sep
                context += doc.page_content[0:add_len - len(chunk) - 1]

                break

            if source not in references:
                context += doc.page_content
                context += sep
                references.append(source)

        context = context[0:context_max_length]
        logger.debug(f'query:{question} top1 file:{references[0]}')
        return context, chunks, references

    def build_query_with_context(self, question: str, template: str) -> str:
        """
        构建问答模板

        args:
            db           向量数据库
            question     原始问题
            template     问答模板

        return:
            query        带上下文的查询
        """

        context = []
        docs, _, references = self.query(question, sep = "\n-------\n")
        if docs and len(docs) > 0:
            return template.replace('{context}', context).replace('{question}', question).replace('{reference}', '\n'.join(references))

        return question

    def build(self, doc_path: str, db_path: str):
        '''
        构建向量数据库

        args:
            doc_path    文档路径
            db_path     向量数据库路径
        '''

        docs = []
        files = self.get_dir_files(doc_path, ['.txt'])

        for file in files:
            docs.extend(self.get_documents(file))

        self.save_to_chroma(db_path, self.embeddings, docs)

    # 获取文件路径函数
    def get_dir_files(self, dir_path: str, suffix: List[str]) -> List[str]:
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

    def get_documents(self, file: str):
        '''
        获取文档

        args:
            file    文档路径

        return:
            documents    文档列表
        '''

        documents = []
        text = ''
        with open(file, encoding='utf8') as f:
            text = f.read()

        if len(text) <= 1:
            return []

        chunks = text.split('\n')
        for idx, chunk in enumerate(chunks):
            if len(chunk) < 15:
                continue

            doc = Document(page_content=chunk,
                            metadata={
                                'source': os.path.basename(file),
                                'chunk': idx,
                            })
            documents.append(doc)
        return documents

    def save_to_chroma(self, persist_directory: str, embedding, chunks: List[Document]) -> None:
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
