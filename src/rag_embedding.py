import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer
import torch
import re
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

def scrape_and_load_nice_guidance(start_url="https://www.nice.org.uk/guidance/published?ps=9999", max_depth=3):
    """
    Crawls NICE guidance pages, extracts valid links, loads them using WebBaseLoader,
    and returns a single combined document.

    Args:
        start_url (str): The starting URL for crawling.
        max_depth (int): Maximum recursion depth for crawling sublinks.

    Returns:
        str: Combined text content from all crawled NICE guidance pages.
    """
    def get_sublinks(url, depth, visited):
        """ Recursively extracts sublinks from NICE guidance pages. """
        if depth == 0 or url in visited:
            return []

        visited.add(url)
        print(f"Crawling: {url}")

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            sublinks = []
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                if href.startswith("/"):
                    href = "https://www.nice.org.uk" + href

                # First loop: Only include links starting with https://www.nice.org.uk/guidance/
                if depth == max_depth and href.startswith("https://www.nice.org.uk/guidance/"):
                    sublinks.append(href)
                # Second loop: Only include links matching the format for recommendations
                elif depth < max_depth and (
                    href.endswith("/chapter/1-Recommendations") or href.endswith("/chapter/Recommendations")
                ):
                    sublinks.append(href)

            # Recursively get sublinks from each discovered link
            for sublink in sublinks:
                sublinks.extend(get_sublinks(sublink, depth - 1, visited))

            return sublinks

        except requests.exceptions.RequestException as e:
            print(f"Error crawling {url}: {e}")
            return []

    # Step 1: Extract links
    visited_urls = set()
    all_links = [start_url] + get_sublinks(start_url, max_depth, visited_urls)

    # Step 2: Filter links to only include recommendation pages
    recommendation_links = [
        link for link in all_links if link.endswith("/chapter/1-Recommendations") or link.endswith("/chapter/Recommendations")
    ]

    # Step 3: Load pages using WebBaseLoader
    print("\nLoading pages into LangChain...\n")
    loader = WebBaseLoader(recommendation_links)
    docs = loader.load()

    # Step 4: Combine all page content into one string document
    combined_document = "\n\n".join(doc.page_content for doc in docs)

    return combined_document  # Return the final combined text

def extract_chunks(text):
    # Define the regex pattern
    pattern = r"Recommendations(.*?)Next page"
    # Find all matches
    chunks = re.findall(pattern, text, re.DOTALL)
    return chunks

def save_vector_store(chunks, embedding_model="pritamdeka/S-PubMedBert-MS-MARCO"):
    """Generate text embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Save the FAISS index to disk
    vector_store.save_local("rag/doctor_ai_faiss_index")

# Scrape Web Guidelines for Diagnosis and Treatment from nice.gov.uk 
doc_string = scrape_and_load_nice_guidance()
doc_chunks = extract_chunks(doc_string)
doc_chunks = [Document(page_content=chunk) for chunk in doc_chunks]

loader = PyPDFLoader("/rag/Orange_Book_45th_Annual.pdf")
orange_book = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

# Adding documnet chunks from Orange Book PDF
orange_book_chunks = text_splitter.split_documents(orange_book)
doc_chunks.extend(orange_book_chunks)

# Generating and saving vector store embeddings
save_vector_store(doc_chunks)