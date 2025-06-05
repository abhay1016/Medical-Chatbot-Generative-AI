import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Pinecone index
index_name = "medical-chatbot-index"  # Replace with your actual index name
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

# Load Llama2 model (adjust model name as per your repo)
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example, adjust to your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Streamlit app
st.title("Medical Chatbot")
st.write("Ask your medical questions, and the AI will provide informed responses based on a medical knowledge base.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user query
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = qa_chain({"query": prompt})
            answer = response["result"]
            st.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})