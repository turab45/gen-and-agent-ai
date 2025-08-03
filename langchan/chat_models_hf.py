from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables from .env file (API key)
load_dotenv()

# Chat Models
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-3B-Instruct",
                          task="text-generation",)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of France? Tell Faiz Ahmed Faiz Poetry Style.")
print(result.content)  # Access the content of the response
