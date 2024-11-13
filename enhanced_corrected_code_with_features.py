import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pickle
from fpdf import FPDF

# Initialize API key, tokenizer, and model
apikey = "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"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize encryption key and cipher
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Title
st.title('AI Assistant for Data Science 🤖')

# Sidebar Information
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('Upload a CSV file to begin. I will analyze the data, provide insights, generate visualizations, and suggest appropriate machine learning models to tackle your problem.')
    st.divider()
    st.caption("<p style ='text-align:center'> made with ❤️ by Team Avengers</p>", unsafe_allow_html=True)

# Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

# Main App Functionality
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        file_data = user_csv.read()
        encrypted_file = encrypt_data(file_data.decode(errors='ignore'))
        st.write("Your data has been successfully encrypted.")

        try:
            decrypted_file = decrypt_data(encrypted_file)
            df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
        except Exception as e:
            st.error(f"An error occurred during decryption: {e}")
            st.stop()

        st.header('Exploratory Data Analysis')
        st.subheader('General information about the data')
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())

        # Data Cleaning Section
        st.write("**Data Cleaning**")
        null_values = df.isnull().sum()
        st.write("Missing values in each column:")
        st.write(null_values[null_values > 0])

        # Sidebar for preprocessing options
        st.sidebar.subheader("Data Preprocessing Options")
        remove_missing_values = st.sidebar.checkbox("Remove Missing Values")
        if remove_missing_values:
            missing_value_method = st.sidebar.selectbox("Choose method for handling missing values", ["Mean", "Median", "Mode", "K-Nearest Neighbors"])
            if missing_value_method == "Mean":
                imputer = SimpleImputer(strategy='mean')
            elif missing_value_method == "Median":
                imputer = SimpleImputer(strategy='median')
            elif missing_value_method == "Mode":
                imputer = SimpleImputer(strategy='most_frequent')
            elif missing_value_method == "K-Nearest Neighbors":
                imputer = KNNImputer(n_neighbors=5)
            
            numeric_df = df.select_dtypes(include=[np.number])
            df[numeric_df.columns] = imputer.fit_transform(numeric_df)

            st.write("**Updated Missing Values**")
            updated_null_values = df.isnull().sum()
            st.write("Missing values in each column after imputation:")
            st.write(updated_null_values)

        # Outlier Removal
        remove_outliers = st.sidebar.checkbox("Outlier Removal")
        if remove_outliers:
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Feature Scaling
        scale_features = st.sidebar.checkbox("Feature Scaling")
        if scale_features:
            scaler = StandardScaler()
            numeric_df = df.select_dtypes(include=[np.number])
            df[numeric_df.columns] = scaler.fit_transform(numeric_df)

        # Visualization Functions
        def plot_missing_values_heatmap(df):
            fig = px.imshow(df.isnull(), title='Missing Values Heatmap')
            st.plotly_chart(fig)

        def plot_correlation_heatmap(df):
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
                st.plotly_chart(fig)
            else:
                st.write("No numeric features available for correlation.")

        def plot_histograms(df, selected_columns):
            for col in selected_columns:
                fig = px.histogram(df, x=col, nbins=30, title=f'Histogram of {col}')
                st.plotly_chart(fig)

        # Display visualizations
        st.write("### Missing Values Heatmap")
        plot_missing_values_heatmap(df)

        st.write("### Histograms of Selected Numerical Features")
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        selected_columns = st.multiselect("Select Variables for Histograms", numeric_columns)

        if selected_columns:
            plot_histograms(df, selected_columns)

        st.write("### Correlation Heatmap")
        plot_correlation_heatmap(df)

        # Variable Study Function
        def function_question_variable(selected_variable):
            st.write(f"**Analyzing Variable: {selected_variable}**")
            summary_statistics = df[selected_variable].describe()
            st.write("**Summary Statistics**")
            st.write(summary_statistics)

            fig, ax = plt.subplots()
            sns.histplot(df[selected_variable], kde=True, color="skyblue", ax=ax)
            ax.set_title(f'Distribution of {selected_variable}', fontsize=14, weight='bold')
            ax.set_xlabel(selected_variable, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            st.pyplot(fig)

            lower_bound = df[selected_variable].quantile(0.25) - 1.5 * (df[selected_variable].quantile(0.75) - df[selected_variable].quantile(0.25))
            upper_bound = df[selected_variable].quantile(0.75) + 1.5 * (df[selected_variable].quantile(0.75) - df[selected_variable].quantile(0.25))
            outlier_count = ((df[selected_variable] < lower_bound) | (df[selected_variable] > upper_bound)).sum()
            st.write(f"**Number of Outliers for {selected_variable}:** {outlier_count}")

            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_variable], ax=ax, color="lightcoral")
            ax.set_title(f'Boxplot of {selected_variable}', fontsize=14, weight='bold')
            st.pyplot(fig)

            missing_values = df[selected_variable].isnull().sum()
            st.write(f"**Missing Values in {selected_variable}:** {missing_values}")

        # Variable Study
        st.header('Variable Study')
        user_question_variable = st.selectbox("Select a variable for study", [''] + df.columns.tolist())
        if user_question_variable:
            function_question_variable(user_question_variable)

        # DataFrame Study
        st.header('DataFrame Study')
        user_question_dataframe = st.text_area("Ask a question about the DataFrame")
        if user_question_dataframe:
            dataframe_info = generate_text(user_question_dataframe)
            if "dataframe_responses" not in st.session_state:
                st.session_state["dataframe_responses"] = []
            if dataframe_info not in st.session_state["dataframe_responses"]:
                st.write(dataframe_info)
                st.session_state["dataframe_responses"].append(dataframe_info)

        # Predictive Modeling with Phase 4 enhancements
        st.header('Predictive Modeling')
        feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist())
        target_col = st.selectbox("Select Target Column", df.columns.tolist())
        num_years = st.number_input("Enter number of years for prediction", min_value=1, value=1)
        model_choice = st.selectbox("Select Machine Learning Model", ["Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regression"])
        perform_cv = st.checkbox("Perform Cross-Validation")

        # Hyperparameter grid for selected models
        param_grids = {
            "Linear Regression": {},
            "Random Forest": {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            "Support Vector Regression": {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf']
            }
        }

        if st.button("Train Model"):
            if target_col not in df.columns:
                st.error(f"Target column '{target_col}' does not exist in the dataset.")
            elif not set(feature_cols).issubset(df.columns):
                st.error("One or more feature columns do not exist in the dataset.")
            else:
                if remove_missing_values:
                    df = df.dropna(subset=[target_col] + feature_cols)

                X = df[feature_cols]
                y = df[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Support Vector Regression": SVR()
                }[model_choice]
                
                if param_grids[model_choice]:
                    grid_search = GridSearchCV(model, param_grids[model_choice], cv=5, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    st.write(f"Best Parameters: {grid_search.best_params_}")

                if perform_cv:
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    st.write(f"Cross-Validation RMSE: {(-cv_scores.mean()) ** 0.5:.2f}")

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                mse = mean_squared_error(y_test, predictions)
                rmse = mse ** 0.5
                st.write(f"**Model Accuracy (RMSE):** {rmse:.2f}")

                # Future prediction logic
                future_prediction = model.predict(pd.DataFrame(X_test.iloc[-num_years:]))
                trend = "high" if future_prediction.mean() > y.mean() else "low"
                st.write(f"The market is expected to be {trend} in the future after {num_years} years.")

                st.write("**Predictions:**")
                st.write(predictions)

                # Model Comparison Section
                st.header('Model Comparison')

                if st.button("Run Model Comparison"):
                    # Define models to compare
                    models_to_compare = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
                        "Support Vector Regression": SVR(C=1.0, kernel='rbf')
                    }

                    # Dictionary to store RMSE results for each model
                    comparison_results = {}

                    # Loop over each model, train it, and calculate RMSE
                    for model_name, model_instance in models_to_compare.items():
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Train model
                        model_instance.fit(X_train, y_train)

                        # Predict on test data
                        preds = model_instance.predict(X_test)

                        # Calculate RMSE
                        mse = mean_squared_error(y_test, preds)
                        rmse = mse ** 0.5

                        # Store the result
                        comparison_results[model_name] = rmse

                    # Create a DataFrame from the comparison results for better display
                    comparison_df = pd.DataFrame.from_dict(comparison_results, orient='index', columns=['RMSE']).sort_values(by="RMSE")
                    
                    # Display the comparison table in Streamlit
                    st.write("### Model Comparison Results")
                    st.write(comparison_df)

                    # Plot the comparison as a bar chart
                    st.write("### Model Comparison (RMSE)")
                    st.bar_chart(comparison_df)


                # Save Model
                if st.button("Save Model"):
                    try:
                        model_file_path = "trained_model.pkl"
                        with open(model_file_path, "wb") as f:
                            pickle.dump(model, f)
                        st.success("Model saved successfully as 'trained_model.pkl'.")
                    except Exception as e:
                        st.error(f"An error occurred while saving the model: {e}")

                # Load Model
                if st.button("Load Model"):
                    model_file_path = "trained_model.pkl"
                    try:
                        if os.path.exists(model_file_path):
                            with open(model_file_path, "rb") as f:
                                loaded_model = pickle.load(f)
                            st.success("Model loaded successfully.")
                        else:
                            st.error("Model file 'trained_model.pkl' not found. Please save a model first.")
                    except Exception as e:
                        st.error(f"An error occurred while loading the model: {e}")

                # Generate PDF Report
                if st.button("Generate PDF Report"):
                    try:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(200, 10, "Data Science Report", ln=True, align="C")
                        
                        pdf.set_font("Arial", "", 12)
                        pdf.cell(200, 10, f"Model: {model_choice}", ln=True)
                        pdf.cell(200, 10, f"RMSE: {rmse:.2f}", ln=True)

                        pdf_buffer = io.BytesIO()
                        pdf.output(pdf_buffer)
                        pdf_buffer.seek(0)

                        st.download_button(
                            label="Download Report",
                            data=pdf_buffer,
                            file_name="data_science_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"An error occurred while generating the report: {e}")
