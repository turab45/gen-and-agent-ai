from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
 
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",
                            task="text-generation")

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = "You are a helpful medical doctor assistant. You will be given a patient's symptoms. " \
    "Provide a concise diagnosis and recommended treatment plan.\n\n" \
    "Patient Symptoms: {symptoms}\n\n",
    input_variables = ["symptoms"] 
)

parser = StrOutputParser()

chain = RunnableSequence(template, model, parser)

response = chain.invoke({"symptoms": "Fever, cough, shortness of breath"})

print("AI:", response)

