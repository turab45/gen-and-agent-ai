from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

"The idea of using RunnablePassthrough is to get the input of the chain as it is."
"Like in this case when we want both the detailed and summary. So we can get the details"
"as it is and then use it in the summary chain. At the end we get both the detailed and summary"
 
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

template2 = PromptTemplate(
    template = "Summarize the following medical diagnosis and treatment plan in a few lines:\n\n" \
    "{symptoms}\n\n",
    input_variables = ["symptoms"]
)

parser = StrOutputParser()


detailed_chain = RunnableSequence(template, model, parser)


parallel_chain = RunnableParallel({
    "detailed": RunnablePassthrough(),
    "summary": RunnableSequence(template2, model, parser)
})

final_chain = RunnableSequence(detailed_chain, parallel_chain)

response = final_chain.invoke({"symptoms": "Fever, cough, shortness of breath"})

print("AI:", response)
