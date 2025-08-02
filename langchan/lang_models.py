# This script is all about the language models both the closed
# and open source ones.


import os
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (API key)
load_dotenv()

# LLMs
llm = OpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)   

result = llm.invoke("What is the capital of France?")

print(result)
