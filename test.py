from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor, AutoModelForSeq2SeqLM

model_name = "Qwen/Qwen2-Audio-7B-Instruct"

# Load models with error handling to confirm successful setup
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
