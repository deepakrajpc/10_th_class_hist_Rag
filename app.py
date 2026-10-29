import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="NCERT History Chatbot", page_icon="📚")
st.title("📚 NCERT Class 10 History Chatbot")
st.caption("Ask questions from the book. Answers come from the document.")


# Load the heavy things once, not on every message
@st.cache_resource
def load_rag():
    # Must be the same model used in create_database.py
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings_model,
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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("RAG app: Chroma + HuggingFace embeddings + Groq.")
    st.write("Tip: ask full questions, like *What did Napoleon do in Europe?*")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.write(msg["sources"])

# ---------- New question ----------
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

            # PyPDF pages start at 0, so add 1
            pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
            sources = "Pages: " + ", ".join(map(str, pages))
            with st.expander("Sources"):
                st.write(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")