from langchain.document_loaders import PyPDFLoader
file_path='C:/Users/raj/Documents/Naresh It/RAG/Datafiles/budget_speech.pdf'
loader=PyPDFLoader(file_path)
documents = loader.load()
print(documents)
for doc in documents:
    print(f"Page {doc.metadata['page']} Text: {doc.page_content}")
    