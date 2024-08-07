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
