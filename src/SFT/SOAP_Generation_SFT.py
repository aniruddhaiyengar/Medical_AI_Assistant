import os
import torch
from datasets import load_dataset
from transformers import(
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  HfArgumentParser,
  TrainingArguments,
  pipeline,
  logging,
)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import platform
import locale
import matplotlib.pyplot as plt
from huggingface_hub import login

def print_system_specs():
  # Check if CUDA is available
  is_cuda_available = torch.cuda.is_available()
  print("CUDA Available:", is_cuda_available)
  # Get the number of available CUDA devices
  num_cuda_devices = torch.cuda.device_count()
  print("Number of CUDA devices:", num_cuda_devices)
  if is_cuda_available:
    for i in range(num_cuda_devices):
      # Get CUDA device properties
      device = torch.device('cuda', i)
      print(f"---CUDA Device {i}---")
      print("Name:", torch.cuda.get_device_name(i))
      print("Compute Capability:", torch.cuda.get_device_capability(i))
      print("Total Memory:", torch.cuda.get_device_properties(i).total_memory, "bytes")
      # Get CPU information
      print("---CPU Information ---")
      print("Processor:", platform.processor())
      print("System:", platform.system(), platform.release())
      print("Python Version:", platform.python_version())

print_system_specs()

def format_chat_example(dialogue, soap):
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": dialogue
        },
        {
            "role": "assistant",
            "content": soap
        }
    ]

def load_model_from_huggingface(model_name, hf_token):
    """Loads a model and tokenizer from Hugging Face."""
    login(token=hf_token)  # Authenticate
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # trust_remote_code=True is necessary for loading models with custom code. Asks the
    # to trust and execute the code from the model repository.

    base_model.config.use_cache = False  # Disable model's cache to save memory
    base_model.config.pretraining_tp = 1 # Disabling tensor parallelism since running on only a single GPU

    # Loading the Tokenizer associated with the base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Set the padding token to be the same as the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' # Padding added to the right of shorter sequences

    tokenizer.special_tokens_map # Observing the special tokens list of tokenizer
    return model, tokenizer

def push_model_to_huggingface(model, tokenizer, new_model_name, hf_token):
    """Pushes a model and tokenizer to a Hugging Face repository."""
    login(token=hf_token)  # Authenticate
    model.push_to_hub(new_model_name, check_pr=True, create_pr=1) # Pusing Model to HuggingFace
    tokenizer.push_to_hub(new_model_name, check_pr=True, create_pr=1) # Pushing Tokenizer to HuggingFace

# Model from Hugging Face hub
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# New instruction dataset
soap_dataset = "omi-health/medical-dialogue-to-soap-summary"

# Fine-tuned model name
new_model_name = "AniruddhAiyengar/Llama-3.1-8B-SOAP-notes-finetuned"

# Load the entire model on the GPU 0
device_map= {"": 0}

hf_token = "HF_TOKEN"

# Load the dataset from the specified name and select the "train" split
dataset = load_dataset(soap_dataset, split="train").remove_columns(["prompt", "messages", "messages_nosystem"])

system_prompt = "You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting. Create a Medical SOAP note summary from the dialogue, following these guidelines:\n S (Subjective): Summarize the patient's reported symptoms, including chief complaint and relevant history. Rely on the patient's statements as the primary source and ensure standardized terminology.\n O (Objective): Highlight critical findings such as vital signs, lab results, and imaging, emphasizing important details like the side of the body affected and specific dosages. Include normal ranges where relevant.\n A (Assessment): Offer a concise assessment combining subjective and objective data. State the primary diagnosis and any differential diagnoses, noting potential complications and the prognostic outlook.\n P (Plan): Outline the management plan, covering medication, diet, consultations, and education. Ensure to mention necessary referrals to other specialties and address compliance challenges.\n Considerations: Compile the report based solely on the transcript provided. Maintain confidentiality and document sensitively. Use concise medical jargon and abbreviations for effective doctor communication.\n Please format the summary in a clean, simple list format without using markdown or bullet points. Use 'S:', 'O:', 'A:', 'P:' directly followed by the text. Avoid any styling or special characters."

# Apply formatting to the dataset
dataset = dataset.map(lambda x: {"messages": format_chat_example(x["dialogue"], x["soap"])})

# Load Model and Tokenizer
base_model, tokenizer = load_model_from_huggingface(model_name, hf_token)

# Load LoRA Configuration

# QLoRA parameters
# LoRAattention dimension
lora_r= 64
# Alpha parameter for LoRAscaling
lora_alpha= 16
# Dropout probability for LoRAlayers
lora_dropout= 0.1

# Initializing LoRA configuration for Causal Language Model
peft_config = LoraConfig(
  lora_alpha=lora_alpha,
  lora_dropout=lora_dropout,
  r=lora_r,
  bias="none",
  task_type="CAUSAL_LM"
)
# No bias term reduces the number of trainable parameters

# TrainingArgumentsparameters
# Output directory where the model predictions and checkpoints will be stored
output_dir= "./results"

# Number of training epochs
num_train_epochs= 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False  # Enables mixed precision training to speed up training and improve memory usage
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size= 8

# Batch size per GPU for evaluation
per_device_eval_batch_size= 8

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps= 2

# Enable gradient checkpointing
# Trades computation time for memory by only storing a subset of activations during
# forward pass and recomputing them during backward pass on demand
gradient_checkpointing= True

# Maximum gradient normal (gradient clipping to prevent exploding gradients)
max_grad_norm= 0.3

# Initial learning rate (AdamWoptimizer)
learning_rate= 2e-4
# Weight decay to apply to all layers except bias/LayerNormweights
weight_decay= 0.001
# Optimizer to use
optim= "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type= "cosine" # Cosine Annealing
# Number of training steps (overrides num_train_epochs)
max_steps= -1    # No impact
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio= 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length= True
# Save checkpoint every X updates steps
save_steps= 10
# Log every X updates steps
logging_steps= 10

# Cell
# Training Parameters used to Initialize TrainingArguments object
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type
)

# Initialize SFTTrainer

# SFT parameters

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Create a trainer instance using SFTTrainer

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)

# For stable training, convert all Layer Normalization layers to float32 datatype
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
# Train the model for finetuning
trainer.train()

# Save trained QLoRA Adapter locally
adapter_path = "/content/qlora_adapter"
trainer.model.save_pretrained(adapter_path)

# Cell
# Load the QLoRA adapter on top of the 16-bit precision base model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Cell
# Merge the adapter to get standalone fine-tuned model
model = model.merge_and_unload()

# Assuming trainer.state.log_history contains your training logs
loss_history = trainer.state.log_history

# Extract loss values and corresponding steps
train_loss_values = [entry['loss'] for entry in loss_history if 'loss' in entry]
train_steps = [entry['step'] for entry in loss_history if 'loss' in entry]

# Create the plot
plt.plot(train_steps, train_loss_values)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()


locale.getpreferredencoding = lambda: "UTF-8"

push_model_to_huggingface(model, tokenizer, new_model_name, hf_token)
