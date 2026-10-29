# Fix for old SQLite on Streamlit Cloud (does nothing on your Mac)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# On Streamlit Cloud the key comes from Secrets; locally it comes from .env
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

BASE_DIR = Path(__file__).parent
DB_DIR = str(BASE_DIR / "chroma_db")
PDF_PATH = BASE_DIR / "document_loader" / "NCERT-Class-10-History.pdf"

st.set_page_config(page_title="NCERT History Chatbot", page_icon="📚")
st.title("📚 NCERT Class 10 History Chatbot")
st.caption("Ask questions from the book. Answers come from the document.")


@st.cache_resource(show_spinner="Loading the book (first start takes a few minutes)...")
def load_rag():
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.isdir(DB_DIR) and os.listdir(DB_DIR):
        vector_store = Chroma(
            persist_directory=DB_DIR, embedding_function=embeddings_model
        )
    else:
        docs = PyPDFLoader(str(PDF_PATH)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=DB_DIR,
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant for a Class 10 history book.
Answer using the provided context.
If the context only partly answers the question, answer with what it does say
and mention that the book covers it only briefly.
Say "I could not find the answer in the document." only if the context has nothing relevant.
Write clear, simple answers suitable for a student."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    return retriever, prompt | llm


retriever, chain = load_rag()

with st.sidebar:
    st.header("About")
    st.write("RAG app: Chroma + HuggingFace embeddings + Groq.")
    st.write("Tip: ask full questions, like *What did Napoleon do in Europe?*")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.write(msg["sources"])

question = st.chat_input("Ask a question about the book...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the book..."):
                docs = retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)
                result = chain.invoke({"context": context, "question": question})

            answer = result.content
            st.markdown(answer)

            pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
            sources = "Pages: " + ", ".join(map(str, pages))
            with st.expander("Sources"):
                st.write(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")