from dotenv import load_dotenv

import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.schema import HumanMessage
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.retrievers import ParentDocumentRetriever

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

from langchain_community.embeddings import OpenAIEmbeddings

import chromadb.api

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
model = AutoModelForCausalLM.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")

chromadb.api.client.SharedSystemClient.clear_system_cache() #no idea why but this fixed the Chroma issue

#Upload PDF files
st.header("PDF Reader")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")
        
def get_llm_response(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")  # Convert text to token IDs
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Handle padding appropriately
        temperature=0.7,  # Control randomness in output
        top_p=0.9  # Nucleus sampling for diversity
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode to text
    return response


def detect_intent(user_query):
    prompt = f"What is the user asking? Options: 'summarize', 'explain', 'specific info', 'general info'\nUser query: {user_query}"
    return get_llm_response(prompt)  # Assume the LLM can classify intent based on the query

def reformulate_query(intent, user_query):
    prompt = f"Reformulate the following query for '{intent}': {user_query}"
    return get_llm_response(prompt)
    
#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    documents = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            # Create a Document object for each page
            doc = Document(page_content=text, metadata={"page_number": i + 1})
            documents.append(doc)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    )
    
    store = InMemoryStore()
    
    full_doc_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    full_doc_retriever.add_documents(documents, ids=None)


    # get user question
    user_query = st.text_input("Type Your question here")

    # do similarity search
    if user_query:
        intent = detect_intent(user_query)
    
        reformulated_query = reformulate_query(intent, user_query)
    
        search_results = vectorstore.similarity_search(reformulated_query)
        
        final_prompt = f"User query: {user_query}\nBased on the following information, please {intent}:\n{search_results}"
        
        answer = get_llm_response(final_prompt)

        st.write(answer)
