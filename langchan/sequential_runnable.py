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

# This is just a simple example of a sequential runnable chain.
# We expand it by adding another chain to summarize the response in a few lines.
print("##############################################################################")
print("Now summarizing the diagnosis and treatment plan:")
print("##############################################################################")


template2 = PromptTemplate(
    template = "Summarize the following medical diagnosis and treatment plan in a few lines:\n\n" \
    "{diagnosis_and_treatment}\n\n",
    input_variables = ["diagnosis_and_treatment"]
)

chain2 = RunnableSequence(template2, model, parser)

response2 = chain2.invoke({"diagnosis_and_treatment": response})


print("AI:", response2)
