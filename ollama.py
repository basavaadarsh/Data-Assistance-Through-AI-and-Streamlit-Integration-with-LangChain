# Import required libraries
import os
import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Ollama model
try:
    llm = Ollama(model="phi", temperature=0)
    st.write("Ollama model initialized successfully!")
except Exception as e:
    st.write(f"Error initializing Ollama model: {e}")

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

# Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)
        st.write("CSV file uploaded successfully!")

        # Create Pandas agent
        try:
            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
            st.write("Pandas agent created successfully!")
        except Exception as e:
            st.write(f"Error creating pandas agent: {e}")

        # Function to retrieve EDA steps
        @st.cache_data
        def steps_eda():
            try:
                steps_eda_text = llm.run('What are the steps of EDA')
                st.write("Steps of EDA retrieved successfully!")
                return steps_eda_text
            except Exception as e:
                st.write(f"Error retrieving steps of EDA: {e}")
                return "Error retrieving steps of EDA"

        # Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")

            try:
                columns_df = pandas_agent.run("What are the meanings of the columns?")
                st.write(columns_df)
            except Exception as e:
                st.write(f"Error retrieving column meanings: {e}")

            try:
                missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                st.write(missing_values)
            except Exception as e:
                st.write(f"Error retrieving missing values: {e}")

            try:
                duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                st.write(duplicates)
            except Exception as e:
                st.write(f"Error retrieving duplicates: {e}")

            st.write("**Data Summarisation**")
            st.write(df.describe())

            try:
                correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
                st.write(correlation_analysis)
            except Exception as e:
                st.write(f"Error retrieving correlation analysis: {e}")

            try:
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
            except Exception as e:
                st.write(f"Error retrieving outliers: {e}")

            try:
                new_features = pandas_agent.run("What new features would be interesting to create?")
                st.write(new_features)
            except Exception as e:
                st.write(f"Error retrieving new features: {e}")

            return

        @st.cache_data
        def function_question_variable():
            try:
                st.line_chart(df, y=[user_question_variable])
                summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
                st.write(summary_statistics)
                normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
                st.write(normality)
                outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
                st.write(outliers)
                trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
                st.write(trends)
                missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
                st.write(missing_values)
            except Exception as e:
                st.write(f"Error processing variable {user_question_variable}: {e}")
            return

        @st.cache_data
        def function_question_dataframe():
            try:
                dataframe_info = pandas_agent.run(user_question_dataframe)
                st.write(dataframe_info)
            except Exception as e:
                st.write(f"Error processing dataframe query: {e}")
            return

        # Main
        st.header('Exploratory Data Analysis')
        st.subheader('General Information about the Dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of Study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable and user_question_variable != "":
            function_question_variable()

            st.subheader('Further Study')

        if user_question_variable:
            user_question_dataframe = st.text_input("Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe and user_question_dataframe.lower() not in ("no", "n"):
                function_question_dataframe()
            if user_question_dataframe.lower() in ("no", "n"):
                st.write("")
