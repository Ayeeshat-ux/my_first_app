import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Environment & API Setup
load_dotenv()

# Safely fetch keys from Streamlit Secrets or environment variables
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
base_url = "https://api.groq.com/openai/v1"
chat_model_name = "openai/gpt-oss-20b"

# Fallback embedding API key to main OPENAI_API_KEY if EMBEDDING_API_KEY isn't explicitly defined
embedding_api_key = os.getenv("EMBEDDING_API_KEY") or st.secrets.get("EMBEDDING_API_KEY") or api_key
embedding_base_url = "https://qwen-embed.publicaai.com/v1"
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"

# Build robust dynamic absolute path for store_path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
store_path = os.path.join(BASE_DIR, "health_store", "health_index.json")

st.set_page_config(page_title="Health Assistant RAG", page_icon="🩺", layout="centered")
st.title("🩺 Health Specialist Assistant")

# 2. Resource Caching for Models & Vector Database
@st.cache_resource
def load_rag_components():
    # Embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )

    # Persistent Vector Store
    if not os.path.exists(store_path):
        st.error(f"Vector store not found at `{store_path}`. Ensure your index is built and pushed to GitHub.")
        st.stop()

    healthdb = SKLearnVectorStore(
        embedding=embeddings,
        persist_path=store_path,
        serializer="json",
    )
    health_retriever = healthdb.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 2, "fetch_k": 10}
    )

    # Chat Model
    chatmodel = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=chat_model_name,
        temperature=0,
        streaming=True,
    )

    # Prompt Template & Chain Definition
    prompt = ChatPromptTemplate.from_template(
        """You are a health specialist providing insights on health-related topics.
You will be provided with the context: {context} to answer the user's question.
The context includes health information and guidelines.
Provide a comprehensive response.

question: {question}"""
    )

    chain = prompt | chatmodel | StrOutputParser()
    return health_retriever, chain

retriever, chain = load_rag_components()

# 3. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    st.info(f"**Model:** {chat_model_name}")
    st.info(f"**Embeddings:** {embedding_model_name}")
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# 4. Display Existing Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input and Streaming Response
if user_question := st.chat_input("Ask a health question..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate streaming assistant response
    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        # Retrieve relevant context from persistent store
        retrieved_docs = retriever.invoke(user_question)
        chain_input = {"context": retrieved_docs, "question": user_question}

        # Stream LLM outputs token by token
        for chunk in chain.stream(chain_input):
            full_response += chunk
            response_box.markdown(full_response + "▌")

        response_box.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})