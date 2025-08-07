from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Load environment variables from .env file (API key)
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b",
                          task="text-generation")

st.header("Research Assistatnt Tool")

# drop down to select the paper
paper_input = st.selectbox(
    "Select a paper",
    ["Select...", "Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
     "GPT-3: Language Models are Few-Shot Learners"]
)

style_input = st.selectbox(
    "Select a style",
    ["Select...", "Beginner friendly", "Technical", "Code oriented", "Poetry Style", "Mathematical oriented"]
)

length_input = st.selectbox(
    "Select the length of the response",
    ["Select...", "Short (1-2 paragraphs)", "Medium", "Long"]
)


template = load_prompt("prompt_template.json")

prompt = template.invoke({'paper_input': paper_input, 'style_input': style_input, 'length_input': length_input})



if st.button("Send"):
    model = ChatHuggingFace(llm=llm)
    response = model.invoke(prompt)
    st.write(response.content)