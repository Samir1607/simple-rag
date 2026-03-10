from rag import SimpleRAG

documents = [
    "Python is a programming language created by Guido van Rossum.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Retrieval Augmented Generation combines search with LLMs."
]

rag = SimpleRAG()

rag.ingest_documents(documents)

while True:
    question = input("Ask: ")

    if question.lower() == "exit":
        break

    answer = rag.answer(question)

    print("Answer:", answer)
