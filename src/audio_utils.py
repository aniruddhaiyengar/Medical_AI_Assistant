import torch
from pyannote.audio import Pipeline

def load_diarization_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token="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"
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline.to(device)
    return pipeline

def transcribe_audio(audio_input, asr_model, pipeline):
    if isinstance(audio_input, str):
        audio_path = audio_input

    result = asr_model.transcribe(audio_path)
    diarization = pipeline(audio_path)
    transcript = combined_segments(result, diarization)
    transcript = clean_transcript(transcript)

    if not isinstance(audio_input, str):
        os.remove(audio_path)

    return transcript

def combined_segments(result, diarization):
    combined_transcript = []
    segments = result["segments"]

    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]

        speaker = None
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= segment_start <= turn.end:
                speaker = spk
                break

        if speaker:
            combined_transcript.append({
                "speaker": speaker,
                "text": segment_text
            })

    transcript_string = ""
    for entry in combined_transcript:
        transcript_string += f"{entry['speaker']}: {entry['text']}\\n"

    return transcript_string

def clean_transcript(transcript):
    lines = [line.strip() for line in transcript.split("\\n") if line.strip()]
    trivial = {"bye", "bye."}
    last_index = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lower() in trivial:
            last_index = i
        else:
            break
    cleaned_transcript = "\\n".join(lines[:last_index])
    return cleaned_transcript