"""
Construct and return the model, tokenizer, and generation config.
"""
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def construct_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, GenerationConfig]:
    i = input("Warning. This will download model weights [roughly 15GB]. Proceed? (y/n)")
    if i.lower() != 'y':
        print("Model download aborted.")
        return None, None, None

    # The paper found Baichuan2-7B performed well. Other options include Llama-3, Mistral, etc.
    model_name = "baichuan-inc/Baichuan2-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload"
    )

    # Optional: You can define generation parameters
    generation_config = GenerationConfig(max_new_tokens=512)

    return model, tokenizer, generation_config
