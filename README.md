- 🔍 Hybrid Retrieval (FAISS + BM25) for improved accuracy

## 🔄 Upgrade: From Basic RAG to Hybrid RAG

### 🧠 Basic RAG

The initial system used **semantic retrieval** with FAISS:

* Text chunks were converted into embeddings
* Similarity search was used to retrieve relevant documents
* The retrieved context was passed to the LLM for answer generation

**Limitation:**

* Struggles with exact keyword matching
* May miss relevant results for short or ambiguous queries

---

### 🚀 Hybrid RAG (Improved System)

To improve retrieval quality, the system was upgraded to **Hybrid RAG**, combining:

#### 🔹 1. Semantic Retrieval (FAISS)

* Captures meaning and context
* Handles synonyms and natural language queries

#### 🔹 2. Keyword Retrieval (BM25)

* Matches exact words and phrases
* Works well for precise or short queries

---

### 🔀 Fusion Strategy

Both retrievers are combined using an **Ensemble Retriever**:

* FAISS (dense retrieval)
* BM25 (sparse retrieval)

Weighted combination:

* Semantic: 0.6
* Keyword: 0.4

This improves both:

* Recall (finding relevant chunks)
* Precision (ranking correct chunks higher)

---

### 📈 Result

* More accurate answers
* Better handling of diverse queries
* Improved robustness in real-world scenarios

---

### 🧠 Key Insight

Hybrid RAG leverages the strengths of both:

* **Understanding (semantic search)**
* **Exact matching (keyword search)**

making it more effective than Basic RAG.
