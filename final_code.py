import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from tpot import TPOTRegressor  # TPOT for AutoML

# Define your Hugging Face API key here
apikey = "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"

# Initialize the tokenizer and model
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

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant, and I am here to help you with your data science projects.")

# Sidebar information
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('Upload a CSV file to start.')
    st.divider()
    st.caption("<p style ='text-align:center'> made with ❤️ by Team Avengers</p>", unsafe_allow_html=True)

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
        file_data = user_csv.read()
        encrypted_file = encrypt_data(file_data.decode(errors='ignore'))
        st.write("Your data has been successfully encrypted.")

        try:
            decrypted_file = decrypt_data(encrypted_file)
            df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
        except Exception as e:
            st.error(f"An error occurred during decryption: {e}")
            st.stop()

        # EDA section functions remain the same
        # (Omitted here to focus on predictive modeling modifications)

        # Predictive Modeling Enhancements: Model Comparison and AutoML
        st.header('Predictive Modeling with Model Comparison')
        st.write("**Select Feature Columns and Target Column for Prediction**")
        feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist())
        target_col = st.selectbox("Select Target Column", df.columns.tolist())

        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]

            if not X.empty and not y.empty:
                # Model Comparison
                if st.button("Run Model Comparison"):
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Support Vector Regression": SVR()
                    }
                    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
                    results = {}

                    for name, model in models.items():
                        cv_results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
                        rmse = (-cv_results.mean()) ** 0.5
                        results[name] = rmse

                    st.write("### Model Comparison (RMSE)")
                    for model_name, rmse in results.items():
                        st.write(f"{model_name}: RMSE = {rmse:.2f}")

                # AutoML with TPOT
                if st.button("Run AutoML (TPOT)"):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42, scoring='neg_mean_squared_error')
                        tpot.fit(X_train, y_train)
                        best_model = tpot.fitted_pipeline_

                        # Predict and evaluate TPOT best model
                        predictions = best_model.predict(X_test)
                        rmse = mean_squared_error(y_test, predictions) ** 0.5
                        r2 = r2_score(y_test, predictions)

                        st.write(f"AutoML Best Model (TPOT) RMSE: {rmse:.2f}")
                        st.write(f"AutoML Best Model (TPOT) R²: {r2:.2f}")
                        st.write("Best Model Pipeline Steps:", tpot.fitted_pipeline_)

                    except Exception as e:
                        st.error(f"Error running TPOT: {e}")
            else:
                st.warning("Please ensure feature columns and target column are correctly selected and contain data.")
        else:
            st.warning("Please select feature columns and a target column.")