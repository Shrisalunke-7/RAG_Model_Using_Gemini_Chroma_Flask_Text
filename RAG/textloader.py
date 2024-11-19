from langchain_community.document_loaders import TextLoader
loader = TextLoader("C:/Users/raj/Documents/Naresh It/RAG/Datafiles/mbox-short.txt")

documents = loader.load()

for doc in documents:
    print(f"Document Text:{doc.page_content}")