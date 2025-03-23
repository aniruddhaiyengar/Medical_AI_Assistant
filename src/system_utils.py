from soap_generation import generate_soap_note
from audio_utils import *
from diagnosis import generate_diagnosis

def generate_followup_answer(context):
    """
    Generates a follow-up answer based on the provided context,
    using the already loaded reasoning model (reason_model, tokenizer_reason).

    context (str): The minimal context string, typically containing:
      - SOAP note
      - Diagnosis
      - Doctor's new question

    Returns:
      A string representing the assistant's answer to the follow-up question.
    """
    system_prompt = (
        "You are a professional medical assistant. Based on the SOAP note, diagnosis, and the doctor's new question, "
        "please provide a clear, medically accurate answer. Keep it concise and relevant without additional reasoning steps or self-reflection. "
         "Do **not** explain your thought process."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]

    conversation_text = "system: " + system_prompt + "\nuser: " + context + "\nassistant:"

    inputs = reason_tokenizer(conversation_text, return_tensors="pt")
    device = reason_model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = reason_model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=1,
        do_sample=True,
        temperature=0.7
    )

    answer_raw = reason_tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = answer_raw.replace(conversation_text, "").strip()

    return answer

def chat_with_assistant(user_query, conv_state):
    """
    Handles a new question from the doctor in multi-turn conversation.

    conv_state: list of (role, text).

    Returns:
      conv_state, status
    """
    # 1) Add the new user query as "Doctor"
    conv_state.append(("Doctor", user_query))

    # 2) Optionally retrieve existing SOAP note & Diagnosis from conv_state
    #    We'll parse the last Assistant message that contained them
    #    Or store them in a global variable after initial generation
    soap_note, diagnosis = parse_soap_and_diagnosis(conv_state)

    # 3) Build minimal context: just SOAP + Diagnosis + user_query
    context = f"SOAP NOTE:\n{soap_note}\n\nDiagnosis:\n{diagnosis}\n\nNew Question:\n{user_query}"

    # 4) Call reasoning model
    #    (If you want RAG again, do retrieve_documents(soap_note, k=3) etc.)
    answer = generate_followup_answer(context)

    # 5) Append assistant reply
    conv_state.append(("Assistant", answer))

    status = "Follow-up question answered."
    return conv_state, status


def parse_soap_and_diagnosis(conv_state):
    """
    Parse the last SOAP note & diagnosis from conv_state, or from a global variable.
    For example, look for the last Assistant message that starts with "SOAP NOTE:".
    """
    soap = ""
    diag = ""
    for role, msg in reversed(conv_state):
        if role == "Assistant" and msg.startswith("SOAP NOTE:"):
            # parse
            # Example:
            # "SOAP NOTE:\n{soap_note}\n\nDiagnosis:\n{diagnosis}"
            lines = msg.split("\n")
            # find lines after "SOAP NOTE:" and lines after "Diagnosis:"
            # or do a more robust parse
            # for simplicity:
            soap_idx = 0
            diag_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("SOAP NOTE:"):
                    soap_idx = i
                if line.startswith("Diagnosis:"):
                    diag_idx = i
            # gather
            soap_lines = lines[soap_idx+1:diag_idx]
            diag_lines = lines[diag_idx+1:]
            soap = "\n".join(soap_lines).strip()
            diag = "\n".join(diag_lines).strip()
            break
    return soap, diag

def archive_current(patient_id, conv_state):
    """
    Archives the conversation (conv_state) into a TXT file in "files" folder,
    naming the file with the patient_id.

    conv_state: list of (role, text), e.g. [("Doctor", "..."), ("Assistant", "...")]
    """
    if not patient_id:
        return "Archive failed: patient ID is required."

    import os
    archive_dir = "files"
    os.makedirs(archive_dir, exist_ok=True)
    file_path = os.path.join(archive_dir, f"{patient_id}.txt")

    # Convert the conv_state list into a string
    lines = []
    for role, msg in conv_state:
        lines.append(f"{role}: {msg}")
    conversation_str = "\n".join(lines)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(conversation_str)
        return f"Archive successful: saved to {file_path}"
    except Exception as e:
        return f"Archive failed: {str(e)}"

def reset_all():
    """
    Resets all conversation states.
    Returns empty strings for chat conversation, SOAP note, retrieved documents, diagnosis,
    and conversation state (to clear all UI outputs).

    Returns:
      A tuple of five empty strings: (chat_history, soap_note, retrieved_docs, diagnosis, conv_state)
    """
    return "", "", "", "", ""

def update_dialogue_generator(new_manual_input, audio_file, conv_state):
    """
    A generator function that processes the audio/manual input step-by-step,
    yielding partial updates (conv_state, status).

    conv_state: list of (role, text), e.g. [("Doctor", "..."), ("Assistant", "...")]
    """
    steps = []

    # Step 1: Transcribe audio if provided
    if audio_file:
        steps.append("Step 1: Transcribing audio...")
        yield (conv_state, steps[-1])
        pipeline = load_diarization_pipeline()
        transcript_text = transcribe_audio(audio_file, asr_model, pipeline)
        steps.append("Step 1: Audio transcription done.")
        yield (conv_state, steps[-1])
    else:
        transcript_text = ""

    # Step 2: Append manual input if provided
    if new_manual_input.strip():
        steps.append("Step 2: Appending manual input...")
        yield (conv_state, steps[-1])

        # conv_state.append(("Doctor", new_manual_input.strip()))
        steps.append("Step 2: Manual input appended.")
        yield (conv_state, steps[-1])

    # Combine text for SOAP note generation
    conversation_text = (transcript_text + "\n" + new_manual_input.strip()).strip()

    # Step 3: Generate SOAP note
    steps.append("Step 3: Generating SOAP note...")
    yield (conv_state, steps[-1])
    soap_note = generate_soap_note(conversation_text)
    steps.append("Step 3: SOAP note generation done.")
    yield (conv_state, steps[-1])

    # Step 4: RAG retrieval
    steps.append("Step 4: Retrieving relevant documents...")
    yield (conv_state, steps[-1])
    docs = retrieve_documents(soap_note, k=1)
    rag_context = combine_context("", docs)
    steps.append("Step 4: Documents retrieved.")
    yield (conv_state, steps[-1])

    # Step 5: Generate diagnosis
    steps.append("Step 5: Generating diagnosis...")
    yield (conv_state, steps[-1])
    print("soap_note",soap_note)
    diagnosis = generate_diagnosis(soap_note, rag_context, "")  # 不传入 conversation_text
    steps.append("Step 5: Diagnosis done.")
    yield (conv_state, steps[-1])

    # Step 6: Append assistant message
    assistant_msg = f"SOAP NOTE:\n{soap_note}\n\nDiagnosis:\n{diagnosis}"
    conv_state.append(("Assistant", assistant_msg))
    steps.append("All steps complete.")
    yield (conv_state, steps[-1])

#   - we only want to append the final result to the conversation
#     and show the status in status_box
def handle_report_result(conv_history, soap, docs, diag, status):
    """
    Combine SOAP and Diagnosis into one assistant message,
    append to conv_history, and return updated conv_history plus status.
    """
    assistant_msg = f"SOAP NOTE:\n{soap}\n\nDiagnosis:\n{diag}"
    # conv_history is a list of (user, assistant)
    # Let's append a single assistant message:
    conv_history.append(("Assistant", assistant_msg))
    return conv_history, status

def handle_chat_output(updated_history, response):
    """
    Called after chat_with_assistant. We get updated_history (string) and
    response (string). We convert them into chat_list for the Chatbot.
    """
    # parse updated_history if it is a string "Doctor: ...\nAssistant: ..."
    # or maintain a list of (user, assistant) in memory.

    lines = updated_history.strip().split("\n")
    chat_list = []
    user_buf, assistant_buf = None, None
    for line in lines:
        if line.startswith("Doctor:"):
            if user_buf and assistant_buf:
                chat_list.append((user_buf, assistant_buf))
                user_buf, assistant_buf = None, None
            user_buf = line.replace("Doctor:", "").strip()
        elif line.startswith("Assistant:"):
            assistant_buf = line.replace("Assistant:", "").strip()
            # Each time we see Assistant, we finalize a pair
            if user_buf and assistant_buf:
                chat_list.append((user_buf, assistant_buf))
                user_buf, assistant_buf = None, None

    # If there's leftover
    if user_buf and not assistant_buf:
        # user typed something but no assistant reply
        chat_list.append((user_buf, ""))
    return chat_list