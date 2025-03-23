from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_sft_model():
    sft_model_name = "AniruddhAiyengar/Llama-3.1-8B-SOAP-notes-finetuned"
    device_map = {"": 0}
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained(sft_model_name, trust_remote_code=True)
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loaded SOAP note generation model and tokenizer.")
    return sft_model, tokenizer