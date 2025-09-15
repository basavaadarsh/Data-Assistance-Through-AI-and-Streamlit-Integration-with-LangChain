# Data Assistance Through AI and Streamlit (LangChain)

A Streamlit app using LangChain + LLMs to assist data exploration of CSVs in `Datasets/`. Includes basic EDA, charts, and simple ML helpers.

## Requirements
- Python 3.93.11, pip, Git

## Setup
1) git clone https://github.com/basavaadarsh/Data-Assistance-Through-AI-and-Streamlit-Integration-with-LangChain.git
2) cd Data-Assistance-Through-AI-and-Streamlit-Integration-with-LangChain
3) Create venv and activate:
   - Windows:  python -m venv .venv; .\.venv\Scripts\Activate.ps1
   - macOS/Linux:  python3 -m venv .venv; source .venv/bin/activate
4) pip install -r requirements.txt

## Environment (.env in repo root)
OPENAI_API_KEY=your_openai_api_key   (or)   HUGGINGFACEHUB_API_TOKEN=your_hf_token
Optional: LANGCHAIN_TRACING_V2=true, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, SERPAPI_API_KEY

## Run (use any script that exists locally)
streamlit run app_DS.py --server.enableXsrfProtection false
streamlit run final_code.py --server.enableXsrfProtection false
streamlit run enhanced_corrected_code_with_features.py --server.enableXsrfProtection false
streamlit run added_Features.py --server.enableXsrfProtection false

## Datasets
Place CSVs in `Datasets/` or upload via UI.

## Troubleshooting
- CPU Torch on Windows: pip install --upgrade pip; pip install torch --index-url https://download.pytorch.org/whl/cpu; pip install -r requirements.txt
- SSL errors: pip install --upgrade cryptography certifi
- OpenAI 401/429: verify API key and limits

## Deploy
- Streamlit Cloud: set secrets from .env; choose an entry script above
- Docker: use python:3.11-slim, install requirements, CMD streamlit run final_code.py
