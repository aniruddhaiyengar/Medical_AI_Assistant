import torch
import whisper

def load_asr_model():
    model_size = "medium"
    asr_model = whisper.load_model(model_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    asr_model.to(device)
    print(f"Loaded Whisper ASR model ({model_size}) on {device}.")
    return asr_model