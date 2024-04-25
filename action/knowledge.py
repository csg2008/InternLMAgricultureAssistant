from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class KnowledgeQuery(BaseAction):
    """知识库信息查询"""

    def __init__(self,
                 db_path: str,
                 embedding_model: str,
                 template: str = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)

        self.Retriever = self.build_vector_retriever(db_path, embedding_model)

        if template is None:
            self.template = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
提供的上下文：
···
{context}
···
用户的问题: {question}
你给的回答:"""
        else:
            self.template = template

    @tool_api(explode_return=True)
    def run(self, query: str) -> ActionReturn:
        """根据查询从知识库搜索文章信息.

        Args:
            query (:class:`str`): the content of search query

        Returns:
            :class:`str`: article information
        """
        context = []
        question = query
        tool_return = ActionReturn(type=self.name)
        docs = self.Retriever.get_relevant_documents(query)
        if docs and len(docs) > 0:
            for doc in docs:
                context.append(doc.page_content)


            question = self.template.replace('{context}', "\n-------\n".join(context)).replace('{question}', query)

        tool_return.result = [dict(type='text', content=question)]

        return tool_return

    def build_vector_retriever(self, db_path: str, embedding_model: str) -> VectorStoreRetriever:
        """
        构建向量数据库查询器

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

        return db.as_retriever()
