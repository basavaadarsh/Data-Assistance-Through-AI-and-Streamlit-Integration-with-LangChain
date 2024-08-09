# Import required libraries
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain
from langchain.utilities import WikipediaAPIWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def prompt_templates():
    data_problem_template = PromptTemplate(
        input_variables=['business_problem'],
        template='Convert the following business problem into a data science problem: {business_problem}.'
    )
    model_selection_template = PromptTemplate(
        input_variables=['data_problem', 'wikipedia_research'],
        template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}.'
    )
    return data_problem_template, model_selection_template

def chains():
    generate_text = HuggingFaceRunnable(model_name="gpt2")  # Replace with your model name
    data_problem_chain = LLMChain(
        llm=generate_text,  # Ensure this is a valid Runnable object
        prompt=prompt_templates()[0],
        verbose=True,
        output_key='data_problem'
    )
    model_selection_chain = LLMChain(
        llm=generate_text,  # Ensure this is a valid Runnable object
        prompt=prompt_templates()[1],
        verbose=True,
        output_key='model_selection'
    )
    sequential_chain = SequentialChain(
        chains=[data_problem_chain, model_selection_chain],
        input_variables=['business_problem', 'wikipedia_research'],
        output_variables=['data_problem', 'model_selection'],
        verbose=True
    )
    return sequential_chain


# Define the model name and initialize the custom runnable class
model_name = "gpt2"  # Replace with the model you need
generate_text = HuggingFaceRunnable(model_name)

def generate_text_func(prompt):
    return generate_text(prompt)

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

    st.caption("<p style='text-align:center'> made with ❤️ by Ana</p>", unsafe_allow_html=True)

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
            steps_eda = generate_text_func('What are the steps of EDA')
            return steps_eda

        # Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = generate_text_func("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = generate_text_func("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = generate_text_func("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = generate_text_func("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = generate_text_func("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = generate_text_func("What new features would be interesting to create?")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y=[user_question_variable])
            summary_statistics = generate_text_func(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = generate_text_func(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = generate_text_func(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = generate_text_func(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = generate_text_func(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = generate_text_func(user_question_dataframe)
            st.write(dataframe_info)
            return

        @st.cache_data
        def wiki(prompt):
            wiki_research = WikipediaAPIWrapper().run(prompt)
            return wiki_research

        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}.'
            )
            model_selection_template = PromptTemplate(
                input_variables=['data_problem', 'wikipedia_research'],
                template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}.'
            )
            return data_problem_template, model_selection_template

        @st.cache_data
        def chains():
            data_problem_chain = LLMChain(llm=generate_text_func, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
            model_selection_chain = LLMChain(llm=generate_text_func, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
            sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
            return sequential_chain

        @st.cache_data
        def chains_output(prompt, wiki_research):
            my_chain = chains()
            my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
            my_data_problem = my_chain_output["data_problem"]
            my_model_selection = my_chain_output["model_selection"]
            return my_data_problem, my_model_selection

        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            algorithm_lines = my_model_selection_input.split('\n')
            algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
            algorithms.insert(0, "Select Algorithm")
            formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
            return formatted_list_output

        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, user_csv):
            # Replace this with a method for generating code or providing a solution
            solution = generate_text_func(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}.")
            return solution

        # Main
        st.header('Exploratory Data Analysis')
        st.subheader('General Information About the Dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of Study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable:
            function_question_variable()

            st.subheader('Further Study')

            user_question_dataframe = st.text_input("Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe not in ("", "no", "No"):
                function_question_dataframe()
            elif user_question_dataframe in ("no", "No"):
                st.write("")

            if user_question_dataframe:
                st.divider()
                st.header("Data Science Problem")
                st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")

                prompt = st.text_area('What is the business problem you would like to solve?')

                if prompt:
                    wiki_research = wiki(prompt)
                    my_data_problem, my_model_selection = chains_output(prompt, wiki_research)
                    
                    st.write(my_data_problem)
                    st.write(my_model_selection)

                    formatted_list = list_to_selectbox(my_model_selection)
                    selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                    if selected_algorithm != "Select Algorithm":
                        st.subheader("Solution")
                        solution = python_solution(my_data_problem, selected_algorithm, user_csv)
                        st.write(solution)
