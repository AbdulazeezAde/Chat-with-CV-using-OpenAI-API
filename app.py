from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import streamlit as st

load_dotenv()
api_key = os.getenv("openai_api_key")

client = OpenAI(api_key=api_key)

def get_text_input():
    user_input = st.text_input("Enter your question or message:")
    return user_input

def create_vector_store(pdf_files):
    loaders = [PyPDFLoader(file) for file in pdf_files]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15)
    text_chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

def get_conversation_chain(vector_store):
     llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, openai_api_key=api_key)
     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
     conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
     return conversation_chain


def get_answer(messages):
    user_question = messages[-1]["content"]
    response = conversation_chain({"question": user_question, "chat_history": messages})
    memory = conversation_chain.memory
    if isinstance(memory, ConversationBufferMemory):
        memory.clear()
    return response["answer"]

pdf_files = [
    "ABDULAZEEZ ADEDAYO CV.pdf"
   ]
vector_store = create_vector_store(pdf_files)
conversation_chain = get_conversation_chain(vector_store)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How may I assist you today?"}
        ]
    if "audio_initialized" not in st.session_state:
        st.session_state.audio_initialized = False



initialize_session_state()
def main():
    # Chat display
    st.subheader("Chat with your CV")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    
        # Text input
    user_text = get_text_input()

        # Create button
    cols = st.columns(1)

        # Text-to-Text (Send button)
    if cols[0].button("Send"):
        if user_text:
            messages = [{"role": "user", "content": user_text}]
            response = get_answer(messages)
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append({"role": "assistant", "content": response})

 #main_container.float("bottom: 0rem;")

if __name__ == '__main__':
    main()
