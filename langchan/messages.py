from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",
                            task="text-generation")
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print("AI:", response.content)

print("Messages history:")
print(messages)