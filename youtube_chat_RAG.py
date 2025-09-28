from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()


# Step 1a - Indexing (Document Ingestion)
print("Loading document...")
loader = PyPDFLoader("langchan/Eye_tracking_H4.pdf")
pages = loader.load()

print("Document loaded successfully.")
print(f"Total Pages: {len(pages)}")

# Step 1b - Indexing (Text Splitting)
print("Splitting document into chunks...")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

chunks = recursive_splitter.split_documents(pages)
print("Document split into chunks successfully.")
print(f"Total Chunks: {len(chunks)}")

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
print("Generating embeddings and storing in vector store...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Embeddings generated and stored in vector store successfully.")

# Step 2 - Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
print("Retriever created successfully.")

# Step 3 - Augmentation
print("Setting up LLM...")
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",
                            task="text-generation")

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

while True:
    question = input("You: ")
    if question.lower() == 'exit':
        print("Exiting the program.")
        break


    # question = "What is the main focus of the document?"
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})

    response = model.invoke(final_prompt)
    print("AI:", response.content)
