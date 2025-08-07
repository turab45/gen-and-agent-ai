from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

# Load environment variables from .env file (API key)
load_dotenv()

llm = HuggingFacePipeline.from_model_id(model_id="microsoft/Phi-tiny-MoE-instruct",
                                         task="text-generation",
                                         pipeline_kwargs={
                                                "temperature": 0.7,
                                                "max_new_tokens": 512,
                                         })
model = ChatHuggingFace(llm=llm)
# Chat Models
result = model.invoke("What is the capital of France?")
print(result.content)  # Access the content of the response