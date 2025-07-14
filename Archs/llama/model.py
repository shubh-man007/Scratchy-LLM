from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")

login(token=HF_TOKEN)

model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN) 
