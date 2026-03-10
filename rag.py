import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document


class SimpleRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.vectorstore = None

    def ingest_documents(self, texts: List[str]):
        docs = [Document(page_content=t) for t in texts]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        split_docs = splitter.split_documents(docs)

        self.vectorstore = FAISS.from_documents(
            split_docs,
            self.embeddings
        )

    def retrieve(self, query: str, k: int = 3):
        return self.vectorstore.similarity_search(query, k=k)

    def answer(self, query: str):
        docs = self.retrieve(query)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

        response = self.llm.predict(prompt)

        return response
