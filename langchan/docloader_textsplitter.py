from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


# load the pdf document
loader = PyPDFLoader("Eye_tracking_H4.pdf")
pages = loader.load()
print(f"Total Pages: {len(pages)}")

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator="")

docs = splitter.split_documents(pages)
print(f"Total Chunks: {len(docs)}")
print(docs[0].page_content)

# Recursive Text Splitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

docs_recursive = recursive_splitter.split_documents(pages)
print(f"Total Chunks: {len(docs_recursive)}")
print(docs_recursive[0].page_content)