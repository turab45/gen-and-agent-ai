from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",
                            task="text-generation")
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")


print("Messages history:")
print(chat_history)
