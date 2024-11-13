from dotenv import load_dotenv

import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.schema import HumanMessage
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.chat_models import ChatOpenAI

import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache() #no idea why but this fixed the Chroma issue

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Upload PDF files
st.header("PDF Reader")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

    my_llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        temperature = 0.6,
        max_tokens = 1000,
        model_name = "gpt-4o-mini"
    )
        
def get_llm_response(llm, query):
    message = [HumanMessage(content=query)]
    response = llm(messages = message)
    return response.content

def detect_intent(user_query):
    prompt = f"What is the user asking? Options: 'summarize', 'explain', 'specific info', 'general info'\nUser query: {user_query}"
    return get_llm_response(my_llm, prompt)  # Assume the LLM can classify intent based on the query

def reformulate_query(intent, user_query):
    prompt = f"Reformulate the following query for '{intent}': {user_query}"
    return get_llm_response(my_llm, prompt)
    
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
        
        answer = get_llm_response(my_llm, final_prompt)

        st.write(answer)
