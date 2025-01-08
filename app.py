import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Hugging Face Model Repository Details
REPO_ID = "krishkpatil/legal_llm"
FILE_NAME = "unsloth.Q4_K_M.gguf"

# Streamlit App
st.title("Legal Advisor Chatbot")
st.write("This chatbot specializes in Indian law. Ask your legal questions!")

# Download the GGUF model from Hugging Face
@st.cache_resource
def download_model():
    st.info("Downloading model from Hugging Face. Please wait...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME)
    return model_path

MODEL_PATH = download_model()

# Load the GGUF model
@st.cache_resource
def load_model(model_path):
    st.info("Loading model into memory...")
    llama = Llama(model_path=model_path, n_threads=8, n_ctx=2048)
    return llama

llama = load_model(MODEL_PATH)

# User input
user_input = st.text_area("Enter your query:")

if st.button("Submit"):
    if user_input.strip():
        # Generate response
        messages = [{"role": "user", "content": user_input}]
        with st.spinner("Generating response..."):
            response = llama.chat(
                messages=messages,
                max_tokens=128,
                temperature=0.7,
                top_p=0.9
            )
        # Display response
        st.success("Response:")
        st.write(response["choices"][0]["content"])
    else:
        st.error("Please enter a query.")
