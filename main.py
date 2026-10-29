from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# FIXED: must be the same model used in create_database.py
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db", embedding_function=embeddings_model
)

retrievers = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,                # FIXED: ":" instead of "="
        "fetch_k": 10,
        "lambda_mult": 0.5,    # FIXED: name is lambda_mult
    },
)

llm = ChatGroq(model="openai/gpt-oss-20b")  # reads GROQ_API_KEY from .env

prompts = ChatPromptTemplate.from_messages([
    ("system", """
    you are a helpful ai assistent .
    Use only the provided context to answer the questions. if ansers not presernt in the context . say: "i could not find the answer in the document
    """),
    ("human", """ context : {context} Question:{Question}""")
])

print("Rag system is created")

# ---- completed part ----
chain = prompts | llm

while True:
    Question = input("\nAsk a question (or type 'exit'): ")
    if Question.strip().lower() == "exit":
        break

    docs = retrievers.invoke(Question)
    context = "\n\n".join(doc.page_content for doc in docs)

    result = chain.invoke({"context": context, "Question": Question})
    print("\nAnswer:", result.content)
    print("Pages used:", sorted({doc.metadata.get("page") for doc in docs}))