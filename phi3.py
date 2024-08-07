from langchain.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st

# Initialize Ollama model
try:
    llm = Ollama(model="phi3", temperature=0)
    st.write("Ollama model initialized successfully!")
except Exception as e:
    st.write(f"Error initializing Ollama model: {e}")

# Function to retrieve EDA steps
@st.cache_data
def steps_eda():
    try:
        steps_eda_text = llm.run('What are the steps of EDA')
        return steps_eda_text
    except Exception as e:
        st.write(f"Error retrieving steps of EDA: {e}")
        return "Error retrieving steps of EDA"

# Continue with your existing code
# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> made with ❤️ by Ana</p>", unsafe_allow_html=True)