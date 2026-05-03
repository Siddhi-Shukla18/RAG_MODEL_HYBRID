

#  Hybrid RAG Pipeline


import os
from dotenv import load_dotenv

# Document loading
from langchain_community.document_loaders import PyPDFLoader

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# LLM (Groq)
from langchain_groq import ChatGroq



#  Load Environment Variables

load_dotenv()



#  Load Documents
def load_documents(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at {file_path}")
    
    loader = PyPDFLoader(file_path)
    return loader.load()



#  Split Documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)



#  Embeddings

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )



#  Vector Store (FAISS)

def create_vector_store(chunks, embeddings):
    if os.path.exists("faiss_index"):
        print("📂 Loading existing FAISS index...")
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("⚡ Creating new FAISS index...")
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local("faiss_index")
        return vs



#  Hybrid Retriever

def create_hybrid_retriever(vector_store, chunks):
    
    # Semantic (Dense)
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Keyword (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # Hybrid (Ensemble)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6]
    )

    return retriever



#  Load LLM

def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024
    )


#  RAG Answer Function

def rag_answer(query, retriever, llm):

    docs = retriever.invoke(query)

    # Remove duplicates
    unique_docs = []
    seen = set()

    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    # Filter + limit
    docs = [doc for doc in unique_docs if len(doc.page_content.strip()) > 200]
    docs = docs[:3]

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant.

Answer the question using the provided context.
Do not add unnecessary external knowledge.
Explain clearly and simply.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content



#  Main Execution

if __name__ == "__main__":

    FILE_PATH = "data/knowledge_base.pdf"  # change if needed

    print(" Loading documents...")
    docs = load_documents(FILE_PATH)

    print(" Splitting...")
    chunks = split_documents(docs)

    print(" Creating embeddings...")
    embeddings = create_embeddings()

    print(" Creating vector store...")
    vector_store = create_vector_store(chunks, embeddings)

    print(" Creating hybrid retriever...")
    retriever = create_hybrid_retriever(vector_store, chunks)

    print(" Loading LLM...")
    llm = load_llm()

    # Test
    query = "What is Artificial Intelligence?"

    print("\n Query:", query)
    answer = rag_answer(query, retriever, llm)

    print("\n Answer:\n", answer)
