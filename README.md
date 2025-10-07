# Data Assistance Through AI and Streamlit (with LangChain)

An interactive Streamlit application suite for fast EDA, feature handling, visualization, and lightweight ML modeling on your CSV datasets. It integrates Hugging Face text generation for guidance and insights, optional encryption/decryption for uploaded data, and multiple modeling options (Linear Regression, Random Forest, Gradient Boosting, SVR).

### Key Features
- EDA: head, summary stats, missing values, outlier detection, correlations, histograms, pair plots
- Preprocessing: imputation (mean/median/mode/KNN), optional scaling and outlier removal, one-hot encoding
- Modeling: train/test split, RMSE metric, cross-validation and simple hyperparameter tuning (in enhanced app)
- Guidance: calls a small text-generation model for explanatory text
- Optional export: PDF stub available in one variant

### Repository Structure
```
Data-Assistance-Through-AI-and-Streamlit-Integration-with-LangChain/
  added_Features.py                      # Streamlit app: preprocessing, EDA, basic models
  enhanced_corrected_code_with_features.py # Streamlit app: preprocessing, model comparison, CV/grid search
  final_code.py                          # Streamlit app: EDA + modeling baseline
  test.py                                # Streamlit app variant with extra preprocessing options
  Datasets/                              # Sample CSVs for testing
  requirements.txt
  README.md
```

Note: Multiple Streamlit entry points are provided; you can run any of them.

## Getting Started

### Prerequisites
- Python 3.9+ recommended
- Windows, macOS, or Linux
- Internet access on first run (to download Hugging Face model/tokenizer, e.g., `gpt2`)

### Clone the Repository
```bash
git clone https://github.com/basavaadarsh/Data-Assistance-Through-AI-and-Streamlit-Integration-with-LangChain.git
cd Data-Assistance-Through-AI-and-Streamlit-Integration-with-LangChain
```

### Create and Activate a Virtual Environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

If `torch` wheels fail on your platform, install a compatible wheel from PyTorch per your OS/CPU/GPU: see `https://pytorch.org/get-started/locally/`.

## Run the Apps

Run any one of the Streamlit apps below from the project root.

- Baseline app (EDA + simple modeling):
```bash
streamlit run final_code.py
```

- Enhanced app (preprocessing options, model comparison, CV/grid search):
```bash
streamlit run enhanced_corrected_code_with_features.py
```

- Added features variant (EDA + preprocessing + modeling):
```bash
streamlit run added_Features.py
```

- Extra variant:
```bash
streamlit run test.py
```

Optionally disable XSRF protection if you need to run behind certain proxies:
```bash
streamlit run final_code.py --server.enableXsrfProtection false
```

## Using the App
1. Upload a CSV file via the sidebar/main uploader.
2. Explore the data: preview, summary stats, missing values, heatmaps, histograms, pair plots.
3. (Optional) Apply preprocessing (imputation, outlier removal, scaling/encoding depending on app).
4. Choose features and a target, pick a model, and train.
5. Review RMSE and predictions; in the enhanced app, compare models and view CV results.

### Sample Data
The `Datasets/` folder contains example CSVs (e.g., stock datasets) you can use for testing. You can also upload your own CSVs.

## Configuration Notes
- Hugging Face usage: the apps currently initialize a small model (`gpt2`) for text generation the first time you run them. This download happens automatically and requires internet access.
- API keys: the code contains an `apikey` variable for demonstration. For production, prefer environment variables instead of hardcoding. Example (PowerShell):
```powershell
$env:HF_TOKEN="<your_hf_token>"
```
Then update the code to read from the environment if you intend to use private models or the Inference API.
- Encryption: uploaded files are encrypted/decrypted in-memory as a demo; do not treat this as production-grade key management.

## Troubleshooting
- Model/tokenizer download errors: ensure your machine is online and try again. Clear `~/.cache/huggingface/` if corrupted.
- Torch install issues: install a compatible wheel from the official PyTorch instructions for your OS/compute stack.
- Streamlit not found: verify your virtual environment is active and `pip install -r requirements.txt` completed successfully.
- Large CSVs: consider sampling or increasing system memory; some visuals (pair plots) can be heavy for wide datasets.

## License
Add a license file (e.g., MIT) if you plan to share or open-source.

## Acknowledgements
- Streamlit, scikit-learn, seaborn, plotly
- Hugging Face Transformers
- Inspiration from EDA and AutoML best practices

