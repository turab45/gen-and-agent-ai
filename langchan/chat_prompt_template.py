from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",
                            task="text-generation")
model = ChatHuggingFace(llm=llm)


template = ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain the concept of {concept} in simple terms.'),
])


prompt = template.invoke({'domain': 'AI', 'concept': 'concept activation vectors'})

response = model.invoke(prompt)

print("AI:", response.content)