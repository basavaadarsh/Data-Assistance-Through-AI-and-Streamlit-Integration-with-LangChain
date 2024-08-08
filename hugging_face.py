# streamlit run hugging_face.py --server.enableXsrfProtection false
# use above command for running the code
# Import required libraries
import os
from huggingface_hub import login
from transformers import pipeline
import streamlit as st
import pandas as pd

# Define your Hugging Face API key here
apikey = ""

# Login with your Hugging Face API key
login(token=apikey)

# Initialize the Hugging Face pipeline for text generation
generator = pipeline('text-generation', model='gpt2')  # You can choose a different model if needed

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

        # Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = generator('What are the steps of EDA?', max_length=100)[0]['generated_text']
            return steps_eda

        # Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = generator('What are the meaning of the columns?', max_length=100)[0]['generated_text']
            st.write(columns_df)
            missing_values = generator("How many missing values does this dataframe have? Start the answer with 'There are'", max_length=100)[0]['generated_text']
            st.write(missing_values)
            duplicates = generator("Are there any duplicate values and if so where?", max_length=100)[0]['generated_text']
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = generator("Calculate correlations between numerical variables to identify potential relationships.", max_length=100)[0]['generated_text']
            st.write(correlation_analysis)
            outliers = generator("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.", max_length=100)[0]['generated_text']
            st.write(outliers)
            new_features = generator("What new features would be interesting to create?.", max_length=100)[0]['generated_text']
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y=[user_question_variable])
            summary_statistics = generator(f"Give me a summary of the statistics of {user_question_variable}", max_length=100)[0]['generated_text']
            st.write(summary_statistics)
            normality = generator(f"Check for normality or specific distribution shapes of {user_question_variable}", max_length=100)[0]['generated_text']
            st.write(normality)
            outliers = generator(f"Assess the presence of outliers of {user_question_variable}", max_length=100)[0]['generated_text']
            st.write(outliers)
            trends = generator(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}", max_length=100)[0]['generated_text']
            st.write(trends)
            missing_values = generator(f"Determine the extent of missing values of {user_question_variable}", max_length=100)[0]['generated_text']
            st.write(missing_values)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = generator(user_question_dataframe, max_length=100)[0]['generated_text']
            st.write(dataframe_info)
            return

        # Main

        st.header('Exploratory Data Analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable:
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input("Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe and user_question_dataframe not in ("", "no", "No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")
