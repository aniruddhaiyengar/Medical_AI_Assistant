from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch

def load_reason_model():
    reasoning_model_name = "FreedomIntelligence/HuatuoGPT-o1-8B"
    device_map = "auto"
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    reason_model = AutoModelForCausalLM.from_pretrained(
        reasoning_model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map=device_map,
        offload_folder="./offload",
        torch_dtype=torch.float16
    )

    tokenizer_reason = AutoTokenizer.from_pretrained(reasoning_model_name, use_fast=True, trust_remote_code=True)
    reason_model.generation_config = GenerationConfig.from_pretrained(reasoning_model_name)
    reason_model.generation_config.max_length = 256

    print("Loaded diagnosis reasoning model and tokenizer.")
    return reason_model, tokenizer_reason