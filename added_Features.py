import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
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
        # Encrypt the file for demonstration (you can test decryption with a sample encrypted file)
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
        def function_question_variable():
            st.line_chart(df, y=[user_question_variable])
            summary_statistics = generate_text(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = generate_text(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = generate_text(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = generate_text(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = generate_text(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = generate_text(user_question_dataframe)
            st.write(dataframe_info)
            return

        @st.cache_data
        def train_predictive_model(feature_cols, target_col, num_years):
            # Check for NaN values in the target column
            if df[target_col].isnull().any():
                st.error("The target column contains NaN values. Please handle them before training.")
                return
            
            # Prepare data
            X = df[feature_cols].dropna()
            y = df[target_col].dropna()
            X = X.loc[y.index]  # Align X and y indices
            
            if X.empty or y.empty:
                st.error("Feature or target columns contain no valid data for training.")
                return

            # Split the data
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R^2 Score: {r2:.2f}")

            # Predict future values
            future_X = pd.DataFrame([[X.mean()] * len(feature_cols)], columns=feature_cols)  # Simple example
            future_y_pred = model.predict(future_X)

            st.write(f"Predictions for the next {num_years} years: {future_y_pred.tolist()}")

        # Main
        st.header('Exploratory Data Analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Data Cleaning')
        st.write("Null values in the dataset:")
        null_values = df.isnull().sum()
        st.write(null_values[null_values > 0])

        # Add a button to remove null values
        if st.button('Remove Null Values'):
            df_cleaned = df.dropna()
            st.write("Null values have been removed. Here are the first few rows of the cleaned dataset:")
            st.write(df_cleaned.head())
        else:
            df_cleaned = df

        st.subheader('Predictive Modeling')
        st.write("Select the feature columns for prediction:")
        feature_columns = st.multiselect('Feature Columns', df_cleaned.select_dtypes(include=['number']).columns)
        
        if feature_columns:
            st.write("Select the target variable for prediction:")
            target_column = st.selectbox('Target Column', df_cleaned.select_dtypes(include=['number']).columns)
            
            if st.button('Train Model'):
                num_years = st.number_input('Enter number of years to predict:', min_value=1, value=1)
                if feature_columns and target_column:
                    train_predictive_model(feature_columns, target_column, num_years)
                else:
                    st.error("Please select both feature columns and target variable.")

        # Add a button to download the report as PDF
        if st.button('Download Analysis Report'):
            html_content = """
            <html>
            <head><title>Analysis Report</title></head>
            <body>
            <h1>Analysis Report</h1>
            <h2>Data Overview</h2>
            <p>The first rows of your dataset:</p>
            {data_head}

            <h2>Data Cleaning</h2>
            <p>Missing values in each column:</p>
            {null_values_table}
            <p>Duplicate values in the dataset:</p>
            {duplicates_table}

            <h2>Data Summarisation</h2>
            {summary_statistics}

            <h2>Visualizations</h2>
            <img src="data:image/png;base64,{heatmap_image}" alt="Heatmap of Missing Values">
            {histograms_images}
            {pairplot_image}

            <h2>Suggested Machine Learning Models</h2>
            {ml_suggestions}

            <h2>Predictive Modeling</h2>
            {model_results}
            </body>
            </html>
            """.format(
                data_head=df.head().to_html(),
                null_values_table=null_values[null_values > 0].to_frame().to_html(),
                duplicates_table=df[df.duplicated()].to_html() if not df[df.duplicated()].empty else "No duplicate values found.",
                summary_statistics=df.describe().to_html(),
                heatmap_image=plt_to_base64(plt.figure()),  # Placeholder for the actual image
                histograms_images="",
                pairplot_image=plt_to_base64(plt.figure()),  # Placeholder for the actual image
                ml_suggestions="Your ML suggestions here",  # Placeholder for ML suggestions
                model_results="Model results will be included here"  # Placeholder for model results
            )

            pdf_path = generate_pdf_report(html_content)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="analysis_report.pdf",
                    mime="application/pdf"
                )

