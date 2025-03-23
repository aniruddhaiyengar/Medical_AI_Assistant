from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

zip_path = "/rag/doctor_ai_faiss_index.zip"
VECTORSTORE_DIR = "/rag/doctor_ai_faiss_index"
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

def get_vectorstore():
    # Load RAG Embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(VECTORSTORE_DIR)
    
    # Load Vector Store
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def retrieve_documents(soap_note, k=3):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(soap_note)
    return retrieved_docs

def combine_context(soap_note, docs):
    context = "\\n".join([doc.page_content for doc in docs])
    combined = f"{soap_note}\\n\\nContext:\\n{context}"
    return combined