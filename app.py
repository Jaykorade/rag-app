import streamlit as st
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------- ENV ----------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# ---------------- PASSWORD ----------------
def check_password():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        pwd = st.text_input("Enter Password", type="password")

        if pwd == APP_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        elif pwd:
            st.error("Wrong password")

        st.stop()

check_password()

# ---------------- UI ----------------
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📚 RAG Document Q&A")

uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])

# ---------------- CACHE ----------------
@st.cache_resource
def create_vectorstore(file_path, file_type):
    if file_type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    return db

# ---------------- PROCESS ----------------
if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    db = create_vectorstore(file_path, uploaded_file.type)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """Answer ONLY from the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask your question")

    if query:
        with st.spinner("Thinking..."):
            result = chain.invoke(query)

        st.subheader("Answer")
        st.write(result)