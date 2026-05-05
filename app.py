import streamlit as st
import os
import uuid
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Advanced RAG", layout="wide")
st.title("📚 PDF Assistant (With Memory)")

# ---------------- LOAD SECRETS ----------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

APP_PASSWORD = os.getenv("APP_PASSWORD")

# fallback for Streamlit Cloud
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if not APP_PASSWORD:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", "admin")

# ---------------- LOGIN ----------------
def login():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        pwd = st.text_input("Enter Password", type="password")

        if pwd == APP_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        elif pwd:
            st.error("Incorrect password")

        st.stop()

login()

# ---------------- STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------- LOAD DOCUMENTS ----------------
def load_docs(files):
    docs = []

    for file in files:
        file_id = str(uuid.uuid4())
        path = f"/tmp/{file_id}_{file.name}"

        with open(path, "wb") as f:
            f.write(file.read())

        if file.type == "application/pdf":
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        docs.extend(loader.load())

    return docs

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(chunks, embeddings)

    return db

# ---------------- PROCESS FILES ----------------
if uploaded_files and st.session_state.db is None:
    docs = load_docs(uploaded_files)
    st.session_state.db = build_vectorstore(docs)
    st.success("Documents indexed successfully!")

# ---------------- RETRIEVER ----------------
def get_retriever(db):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

# ---------------- FORMAT CONTEXT ----------------
def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

# ---------------- CHAT HISTORY ----------------
def format_chat_history(messages):
    history = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template(
"""
You are a strict document assistant.

RULES:
- Use ONLY the provided context
- Use chat history to understand follow-up questions
- If answer is not found, say "Not found in document"
- Do NOT guess

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = prompt | llm | StrOutputParser()

# ---------------- CHAT UI ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask something from your documents")

# ---------------- QUERY FLOW ----------------
if user_query and st.session_state.db:

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    retriever = get_retriever(st.session_state.db)

    # Retrieve docs
    docs = retriever.invoke(user_query)
    context = format_docs(docs)

    # ✅ MEMORY ADDED HERE
    history = format_chat_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "history": history,
                "context": context,
                "question": user_query
            })

        st.write(response)

        # ---------------- SOURCES ----------------
        with st.expander("📄 Sources"):
            for i, d in enumerate(docs):
                st.markdown(f"**Chunk {i+1} | Page {d.metadata.get('page', 'N/A')}**")
                st.write(d.page_content)

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )