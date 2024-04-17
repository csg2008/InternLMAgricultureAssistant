from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def test_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="E:\\Python\\workspace\\paraphrase-multilingual-MiniLM-L12-v2")

    # 向量数据库持久化路径
    persist_directory = 'E:\\Python\\workspace\\InternLMAgricultureAssistant\\data\\vector_db'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )


    # 你可以修改这里的 prompt template 来试试不同的问答效果
    template = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
    提供的上下文：
    ···
    {context}
    ···
    用户的问题: {question}
    你给的回答:"""


    retriever=vectordb.as_retriever()


    question = "你是谁"

    context = []
    docs = retriever.get_relevant_documents(question)
    if docs and len(docs) > 0:
        for doc in docs:
            context.append(doc.page_content)

        return template.replace('{context}', "\n-------\n".join(context)).replace('{question}', question)

    return question

