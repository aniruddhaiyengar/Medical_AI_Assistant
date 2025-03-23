# eecse_6895_midterm_project

The goal of this project to create a Medical AI Assistant that generates SOAP Notes along with expert feedback and recommendations based on Docotor-Patient clinical audio recordings. A RAG Framework is implemented to get the latest medical guidelines. The user can ask follow-up questions and save the audio trasncript as well as the conversation history.  

## Data Source

Supervised Finetuning (SFT) is performed uisng the OMI Health medical dialogue to SOAP summary dataset https://huggingface.co/datasets/omi-health/medical-dialogue-to-soap-summary . This dataset consists of 10,000 synthetic dialogues between a patient and clinician, created using the GPT-4 dataset from NoteChat, based on PubMed Central (PMC) case-reports. Accompanying these dialogues are SOAP summaries generated through GPT-4. The dataset is split into 9250 training, 500 validation, and 250 test entries, each containing a dialogue column, a SOAP column, a prompt column, and a ChatML-style conversation format column.

The RAG Vector Store is created by web-scraping guidelines from the National Institute for Health and Care website https://www.nice.org.uk/guidance/ and combining it with the Orange Book PDF releasd by the FDA. 

### Build

```
pip install -r requirements.txt
```

## Create RAG Vector Store

Run`src/rag_embedding.py` : Scrape guidelines from NICE, load Orange Book PDF,  combine the 2 documents and use PubMedBERT to create and save the FAISS Vector Store.

## SFT

Run `src/SFT/SOAP_Generation_SFT.py`: Use LORA Finetuning to finetune the Llama 3.1 8B model for SOAP Note Generation.

## Run Gradio App for Medical AI Assistant

Run `src/main.py` to run the interactive Gradio app.


