from text_utils import * 

def generate_soap_note(dialogue_history):
    """
    Generates a SOAP note from the complete dialogue history using the fine-tuned SOAP note generation model.

    Parameters:
      dialogue_history (str): The complete dialogue text (accumulated audio transcript and manual inputs).

    Returns:
      A string representing the generated SOAP note.
    """
    soap_system_prompt = (
        "You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting. Create a Medical SOAP note summary from the dialogue, following these guidelines:\n S (Subjective): Summarize the patient's reported symptoms, including chief complaint and relevant history. Rely on the patient's statements as the primary source and ensure standardized terminology.\n O (Objective): Highlight critical findings such as vital signs, lab results, and imaging, emphasizing important details like the side of the body affected and specific dosages. Include normal ranges where relevant.\n A (Assessment): Offer a concise assessment combining subjective and objective data. State the primary diagnosis and any differential diagnoses, noting potential complications and the prognostic outlook.\n P (Plan): Outline the management plan, covering medication, diet, consultations, and education. Ensure to mention necessary referrals to other specialties and address compliance challenges.\n Considerations: Compile the report based solely on the transcript provided. Maintain confidentiality and document sensitively. Use concise medical jargon and abbreviations for effective doctor communication.\n Please format the summary in a clean, simple list format without using markdown or bullet points. Use 'S:', 'O:', 'A:', 'P:' directly followed by the text. Avoid any styling or special characters. "
    )

    messages = [
        {"role": "system", "content": soap_system_prompt},
        {"role": "user", "content": dialogue_history}
    ]

    # Format the conversation using the chat template function
    formatted_input = apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Tokenize the formatted input and move it to the model's device
    inputs = sft_tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)
    device = sft_model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the SOAP note using the SOAP note model
    output = sft_model.generate(**inputs, max_new_tokens=400, num_beams=3, early_stopping=True)

    # Decode and post-process the output
    soap_note = sft_tokenizer.decode(output[0], skip_special_tokens=True)

    soap_note = trim_incomplete_sentence(soap_note)
    soap_note = extract_after_pattern(soap_note)
    soap_note = trim_incomplete_sentence(soap_note)

    return soap_note