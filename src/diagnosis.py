from text_utils import *

def generate_diagnosis(soap_note, rag_context, chat_history):
    """
    Generates a diagnosis suggestion using the reasoning model based on the SOAP note,
    retrieved context (from RAG), and the conversation history.

    Parameters:
      soap_note (str): The generated SOAP note.
      rag_context (str): The context retrieved via the RAG system.
      chat_history (str): The accumulated conversation history (can be empty).

    Returns:
      A string representing the generated diagnosis suggestion.
    """
    # Define the system prompt for the reasoning model
    diagnosis_system_prompt = (
        "You are a professional medical assistant. Using the provided SOAP note and related context, "
        "please provide a clear and concise diagnosis along with any recommendations. "
        "Ensure your response follows the format: "
        "Diagnosis: <diagnosis details> "
        "Recommendations: <recommendations if any>."
    )

    # Combine the inputs into a single conversation input.
    # If chat_history exists, include it to provide more context.
    if chat_history:
        combined_input = (
            f"Conversation History:\n{chat_history}\n\n"
            f"SOAP Note:\n{soap_note}\n\n"
            f"Context:\n{rag_context}"
        )
    else:
        combined_input = f"SOAP Note:\n{soap_note}\n\nContext:\n{rag_context}"

    # Construct conversation messages
    messages = [
        {"role": "system", "content": diagnosis_system_prompt},
        {"role": "user", "content": combined_input}
    ]

    # Format the conversation using the reasoning model's tokenizer's chat template.
    formatted_input = reason_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize the input and move tensors to the same device as the model
    inputs = reason_tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)
    device = reason_model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the output using the reasoning model
    outputs = reason_model.generate(
        **inputs,
        max_new_tokens=800,   # larger token limit
        num_beams=1,
    )

    diagnosis = reason_tokenizer.decode(outputs[0], skip_special_tokens=True)
    diagnosis = extract_reason_response(diagnosis)


    return diagnosis