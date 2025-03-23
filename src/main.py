from build_gradio_ui import build_ui
from models.asr_model import load_asr_model
from models.soap_model import load_sft_model
from models.reasoning_model import load_reason_model
from huggingface_hub import login

def main():
    
    # Huggingface login
    login(token="HF_TOKEN")

    # Load models
    asr_model = load_asr_model()
    sft_model, sft_tokenizer = load_sft_model()
    reason_model, reason_tokenizer = load_reason_model()

    # Build and launch Gradio UI
    demo = build_ui(share=True)
    demo.launch()

if __name__ == "__main__":
    main()