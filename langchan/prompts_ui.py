from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

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


template = PromptTemplate(
   template = "Please summarize the research paper titled \"{paper_input}\" with the following specifications:" \
   "Explanation Style: {style_input}  \nExplanation Length: {length_input}" \
   "1. Mathematical Details:  \n   - Include relevant mathematical equations if present in the paper."
   "- Explain the mathematical concepts using simple, intuitive code snippets where applicable.  " \
   "2. Analogies:  \n   - Use relatable analogies to simplify complex ideas.  " \
   "If certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing." \
   "Ensure the summary is clear, accurate, and aligned with the provided style and length.",
    input_variables=["paper_input", "style_input", "length_input"]
)

prompt = template.invoke({'paper_input': paper_input, 'style_input': style_input, 'length_input': length_input})



if st.button("Send"):
    model = ChatHuggingFace(llm=llm)
    response = model.invoke(prompt)
    st.write(response.content)