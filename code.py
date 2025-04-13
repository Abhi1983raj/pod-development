# AI-Powered Virtual Development Pod (Simplified Demo with Gemini)
# Libraries: streamlit, google-generativeai, langchain (for agents)

import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDbC-YuqkXY08L8d8HZczzyc-n3gbBvS9U"  # Replace with your actual API key

# Initialize Gemini Pro model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

# Templates for agents
BA_PROMPT = PromptTemplate(
    input_variables=["req"],
    template="""
    You are a Business Analyst. Convert the following requirement into user stories:
    Requirement: {req}
    """,
)

DEV_PROMPT = PromptTemplate(
    input_variables=["stories"],
    template="""
    You are a Software Developer. Generate Python Streamlit code based on these user stories:
    {stories}
    """,
)

TESTER_PROMPT = PromptTemplate(
    input_variables=["code"],
    template="""
    You are a QA Tester. Write Pytest test cases for this code:
    {code}
    """,
)

PM_PROMPT = PromptTemplate(
    input_variables=["stories", "code", "tests"],
    template="""
    You are a Project Manager bot. Summarize and assess the quality and completeness of this project based on:
    - User stories: {stories}
    - Code: {code}
    - Tests: {tests}
    Give suggestions and flag any missing parts.
    """,
)

# Define Streamlit UI
st.title("AI-Powered Virtual Development Pod (Gemini)")
req = st.text_area("Enter a High-Level Requirement:", "Build a task manager app")

if st.button("Generate Project"):
    with st.spinner("Business Analyst is working..."):
        # BA Agent
        ba_chain = LLMChain(llm=llm, prompt=BA_PROMPT)
        user_stories = ba_chain.run(req)
        st.subheader("User Stories")
        st.code(user_stories)

    with st.spinner("Developer is coding..."):
        # Developer Agent
        dev_chain = LLMChain(llm=llm, prompt=DEV_PROMPT)
        generated_code = dev_chain.run(user_stories)
        st.subheader("Generated Code")
        st.code(generated_code, language="python")

    with st.spinner("QA Tester is testing..."):
        # Tester Agent
        tester_chain = LLMChain(llm=llm, prompt=TESTER_PROMPT)
        test_cases = tester_chain.run(generated_code)
        st.subheader("Generated Test Cases")
        st.code(test_cases, language="python")

    with st.spinner("Project Manager is reviewing..."):
        # Project Manager Agent
        pm_chain = LLMChain(llm=llm, prompt=PM_PROMPT)
        pm_feedback = pm_chain.run({"stories": user_stories, "code": generated_code, "tests": test_cases})
        st.subheader("PM Bot Feedback")
        st.write(pm_feedback)

    st.success("Project generated successfully!")
