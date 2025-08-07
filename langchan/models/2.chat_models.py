from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file (API key)
load_dotenv()

# Chat Models
# temp value is used to control the randomness or the creativity of the model's responses
model = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

result = model.invoke("What is the capital of France?")

print(result.content)  # Access the content of the response