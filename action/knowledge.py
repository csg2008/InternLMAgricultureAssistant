from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn

from utils.retriever import Retriever

class KnowledgeQuery(BaseAction):
    """知识库信息查询"""

    def __init__(self,
                 db_path: str,
                 embedding_model: str,
                 reranker_model: str,
                 template: str = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)

        self.Retriever = Retriever(embedding_model, reranker_model, db_path)

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

        question = self.Retriever.build_query_with_context(query, self.template)
        tool_return = ActionReturn(type=self.name)

        tool_return.result = [dict(type='text', content=question)]

        return tool_return
