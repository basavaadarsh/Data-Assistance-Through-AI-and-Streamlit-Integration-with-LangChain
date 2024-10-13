import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
import base64
import pdfkit

# Define your Hugging Face API key here
apikey = "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"

# Initialize the tokenizer and model
model_name = "gpt2"  # Replace with the model you need
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize encryption key and cipher (For real usage, load this from a secure location)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Encryption and Decryption Functions
def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        return decrypted_data
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        raise

def plt_to_base64(fig):
    """Convert matplotlib figure to base64 for embedding in HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_pdf_report(content):
    """Generate a PDF report from HTML content"""
    pdf_path = "analysis_report.pdf"
    pdfkit.from_string(content, pdf_path)
    return pdf_path

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
    st.caption("<p style ='text-align:center'> made with ❤️ by Kare</p>", unsafe_allow_html=True)

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
        # Encrypt the file for demonstration
        file_data = user_csv.read()
        encrypted_file = encrypt_data(file_data.decode(errors='ignore'))

        # Notify user about encryption
        st.write("Your data has been successfully encrypted.")

        try:
            decrypted_file = decrypt_data(encrypted_file)
            df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
        except Exception as e:
            st.error(f"An error occurred during decryption: {e}")
            st.stop()

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
                ax.set_title('Histograms of Numeric Features')  # Title for the histograms
                st.pyplot(fig)
            else:
                st.write("No numeric features available for histograms.")

            # Pairplot for visualizing relationships between numeric features
            if len(num_features) > 1:
                pairplot_fig = sns.pairplot(df[num_features])
                pairplot_fig.fig.suptitle('Pairplot of Numeric Features', y=1.02)  # Title for the pairplot
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
                st.write("Since your dataset contains categorical data, consider using models like Decision Trees, Random Forests, or Gradient Boosting.")
            else:
                st.write("Since your dataset is numeric, consider using models like Linear Regression, Support Vector Machines, or Neural Networks.")

        @st.cache_data
        def function_question_variable(variable):
            # Summary statistics
            st.write(f"**Summary Statistics for {variable}**")
            summary_statistics = df[variable].describe()
            st.write(summary_statistics)

            # Normality check
            st.write(f"**Normality Check for {variable}**")
            fig, ax = plt.subplots()
            sns.histplot(df[variable], kde=True, ax=ax)
            ax.set_title(f'Normality Distribution for {variable}')
            st.pyplot(fig)

            # Outlier assessment
            st.write(f"**Assessing Outliers for {variable}**")
            Q1 = df[variable].quantile(0.25)
            Q3 = df[variable].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = (df[variable] < (Q1 - 1.5 * IQR)) | (df[variable] > (Q3 + 1.5 * IQR))
            outliers = df[outlier_condition]
            st.write(f"Number of outliers for {variable}: {outliers.shape[0]}")

            # Missing values check
            missing_values = df[variable].isnull().sum()
            st.write(f"**Missing Values in {variable}**: {missing_values}")

            # Trends analysis
            if pd.api.types.is_numeric_dtype(df[variable]):
                st.write(f"**Trends Analysis for {variable}**")
                fig, ax = plt.subplots()
                df[variable].plot(title=f'Trends of {variable}', ax=ax)
                st.pyplot(fig)

        # Main
        st.header('Exploratory Data Analysis')
        st.subheader('General information about the data')
        function_agent()

        # Variable Study
        st.header('Variable Study')
        user_question_variable = st.selectbox("Select a variable for study", df.columns)
        if user_question_variable:
            function_question_variable(user_question_variable)

        # Dataframe Study
        st.header('DataFrame Study')
        user_question_dataframe = st.text_area("Ask a question about the DataFrame")
        if user_question_dataframe:
            dataframe_info = generate_text(user_question_dataframe)
            st.write(dataframe_info)

        # Predictive Modeling
        st.header('Predictive Modeling')
        st.write("**Select Feature Columns and Target Column for Prediction**")
        feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist())
        target_col = st.selectbox("Select Target Column", df.columns.tolist())
        num_years = st.number_input("Enter number of years for prediction", min_value=1, value=1)
        if st.button("Train Model"):
            train_predictive_model(feature_cols, target_col, num_years)

        # Download Report
        st.header('Download Analysis Report')
        if st.button("Generate Report"):
            content = """
            <html>
            <head>
                <style>
                    body {font-family: Arial, sans-serif; margin: 20px;}
                    h1, h2 {color: #333;}
                    table {width: 100%; border-collapse: collapse;}
                    table, th, td {border: 1px solid #ddd;}
                    th, td {padding: 8px; text-align: left;}
                    th {background-color: #f2f2f2;}
                    .chart {margin-bottom: 20px;}
                </style>
            </head>
            <body>
                <h1>Data Analysis Report</h1>
                <h2>Exploratory Data Analysis</h2>
                <p>This report provides insights from the data analysis performed on your dataset.</p>
                <!-- You can include charts and tables here -->
            </body>
            </html>
            """
            pdf_path = generate_pdf_report(content)
            st.success(f"Report generated: [Download it here](/{pdf_path})")
