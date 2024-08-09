# Import required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_experimental.agents import create_pandas_dataframe_agent
# updated libraries
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_community.agent_toolkits.powerbi.base import create_pbi_agent
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper

from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    # Generate text based on the prompt
    results = generator(prompt, max_length=150, num_return_sequences=1)
    return results[0]['generated_text']

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)


# Define your Hugging Face API key here
apikey = "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"

# Initialize the tokenizer and model
model_name = "gpt2"  # Replace with the model you need
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant, and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('''**Upload a CSV file to begin. I will analyze the data, provide insights, generate visualizations, 
    and suggest appropriate machine learning models to tackle your problem. Let’s dive into your data science journey!**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> made with ❤️ by Ana</p>",
               unsafe_allow_html=True)

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
            steps_eda = generate_text('What are the steps of EDA?')
            return steps_eda

        # Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())

            st.write("**Data Cleaning**")
            st.write("Missing values in each column:")
            null_values = df.isnull().sum()
            st.write(null_values[null_values > 0])

            st.write("Duplicate values in the dataset:")
            duplicates = df[df.duplicated()]
            if not duplicates.empty:
                st.write(duplicates)
            else:
                st.write("No duplicate values found.")

            st.write("**Data Summarisation**")
            st.write(df.describe())

            st.write("**Visualizations**")

            # Heatmap of missing values
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Heatmap of Missing Values')  # Title for the heatmap
            st.pyplot(fig)

            # Histograms of numeric features
            num_features = df.select_dtypes(include=['number']).columns
            if len(num_features) > 0:
                fig, ax = plt.subplots()
                df[num_features].hist(ax=ax, bins=30, figsize=(10, 7))
                # Title for the histograms
                ax.set_title('Histograms of Numeric Features')
                st.pyplot(fig)
            else:
                st.write("No numeric features available for histograms.")

            # Pairplot for visualizing relationships between numeric features
            if len(num_features) > 1:
                pairplot_fig = sns.pairplot(df[num_features])
                # Title for the pairplot
                pairplot_fig.fig.suptitle(
                    'Pairplot of Numeric Features', y=1.02)
                st.pyplot(pairplot_fig.fig)

            # Suggest ML models based on data types
            st.markdown(
                """
            <p style='font-size: 25px;'><strong>Suggested Machine Learning Models</strong></p>
            """,
                unsafe_allow_html=True
            )

            # Using st.write() for other messages
            if 'object' in df.dtypes.values:
                st.write(
                    "Since your dataset contains categorical data, consider using models like Decision Trees, Random Forests, or Gradient Boosting.")
            else:
                st.write(
                    "Since your dataset is numeric, consider using models like Linear Regression, Support Vector Machines, or Neural Networks.")

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y=[user_question_variable])
            summary_statistics = generate_text(
                f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = generate_text(
                f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = generate_text(
                f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = generate_text(
                f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = generate_text(
                f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = generate_text(user_question_dataframe)
            st.write(dataframe_info)
            return

        @st.cache_resource
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

            data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[
                                          0], verbose=True, output_key='data_problem')

            model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[
                                             1], verbose=True, output_key='model_selection')

            sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=[
                                               'business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)

            return sequential_chain

        @st.cache_data
        def chains_output(prompt, wiki_research):

            my_chain = chains()

            my_chain_output = my_chain(
                {'business_problem': prompt, 'wikipedia_research': wiki_research})

            my_data_problem = my_chain_output["data_problem"]

            my_model_selection = my_chain_output["model_selection"]

            return my_data_problem, my_model_selection

        @st.cache_data
        def list_to_selectbox(my_model_selection_input):

            algorithm_lines = my_model_selection_input.split('\n')

            algorithms = [algorithm.split(':')[-1].split('.')[-1].strip()
                          for algorithm in algorithm_lines if algorithm.strip()]

            algorithms.insert(0, "Select Algorithm")

            formatted_list_output = [
                f"{algorithm}" for algorithm in algorithms if algorithm]

            return formatted_list_output

        @st.cache_resource
        def python_agent():

            agent_executor = create_python_agent(

                llm=llm,

                tool=PythonREPLTool(),

                verbose=True,

                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,

                handle_parsing_errors=True,

            )

            return agent_executor

        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, user_csv):

            solution = python_agent().run(f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."

                                          )

            return solution

        # Main
        st.header('Exploratory Data Analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of Study')
        user_question_variable = st.text_input(
            'What variable are you interested in')
        if user_question_variable:
            function_question_variable()

        st.subheader('Further Study')
        user_question_dataframe = st.text_input(
            "Is there anything else you would like to know about your dataframe?")
        if user_question_dataframe and user_question_dataframe not in ("", "no", "No"):
            function_question_dataframe()
        elif user_question_dataframe in ("no", "No"):
            st.write("")

        if user_question_dataframe:

            st.divider()

            st.header("Data Science Problem")

            st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")

            prompt = st.text_area(
                'What is the business problem you would like to solve?')

            if prompt:

                wiki_research = wiki(prompt)

                my_data_problem = chains_output(prompt, wiki_research)[0]

                my_model_selection = chains_output(prompt, wiki_research)[1]

                st.write(my_data_problem)

                st.write(my_model_selection)

                formatted_list = list_to_selectbox(my_model_selection)

                selected_algorithm = st.selectbox(
                    "Select Machine Learning Algorithm", formatted_list)

                if selected_algorithm is not None and selected_algorithm != "Select Algorithm":

                    st.subheader("Solution")

                    solution = python_solution(
                        my_data_problem, selected_algorithm, user_csv)

                    st.write(solution)
