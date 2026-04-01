"""
Observability example for a simple RAG pipeline using Langfuse (v4.0+).

FLOW TỔNG QUÁT - Quy trình RAG từng bước (FOLLOW THIS):
================================================================================
  STEP 1: Load environment (.env)
           ↓
           Get: LANGFUSE_PUBLIC_KEY, SECRET_KEY, proxy settings
           
  STEP 2: Normalize proxy environment
           ↓
           Fix Windows proxy format (semicolon → comma)
           Set NO_PROXY for localhost bypass
           
  STEP 3: Initialize Langfuse client
           ↓
           Connect to Langfuse cloud (https://cloud.langfuse.com)
           
  STEP 4: Prepare RAG data
           ↓
           Load 4 documents about AI/ML
           Create: TinyEmbedding (model), TinyVectorStore (storage)
           
  STEP 5: Run RAG pipeline with observability tracking
           ├─ SPAN "index_documents"
           │   ├─ Embed 4 documents
           │   └─ Store vectors in database
           │
           ├─ SPAN "retrieve"
           │   ├─ Embed user query
           │   └─ Search for top-2 similar documents (cosine similarity)
           │
           └─ GENERATION "generate_answer"
               ├─ Take retrieved documents as context
               └─ Generate answer (mock LLM)
           
  STEP 6: Flush/send all traces to Langfuse cloud
           ↓
           All spans + generation uploaded as a trace
           
  STEP 7: Print results to console
           ↓
           Query, Retrieved docs, Generated answer

FILE REQUIREMENTS:
  - .env (LANGFUSE_PUBLIC_KEY, SECRET_KEY, proxy)
  - python-dotenv (optional, fallback parser exists)

INSTALL & RUN:
    pip install langfuse numpy python-dotenv
    python observability_langfuse_example.py
================================================================================
"""

# ============= IMPORTS & SETUP =============
import os
import importlib
from typing import List, Tuple

import numpy as np

# Load python-dotenv nếu cài, không thì fallback
load_dotenv = None
try:
    dotenv_module = importlib.import_module("dotenv")
    load_dotenv = getattr(dotenv_module, "load_dotenv", None)
except Exception:
    load_dotenv = None

# Load Langfuse SDK
try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None


# ============= HELPER CLASSES: EMBEDDING & VECTOR STORE =============

class TinyEmbedding:
    """
    EMBEDDING MODEL: Convert text → vector representation.
    
    Method: Word Frequency (đơn giản dùng để demo)
    
    Flow:
      Input: "RAG combines retrieval" 
        1. Tokenize: ["rag", "combines", "retrieval"]
        2. Map to ID: {rag→0, combines→1, retrieval→2}
        3. Count freq: vector[0]+=1, vector[1]+=1, vector[2]+=1
                      → [1, 1, 1, 0, 0, ..., 0]  (128 dims)
        4. Normalize: divide by L2 norm (length)
      Output: [0.577, 0.577, 0.577, 0, ..., 0]  (unit vector)
    
    Dùng để: vector search (tính cosine similarity)
    """
    def __init__(self, vocab_size: int = 128):
        """vocab_size: số từ trong vocabulary (= vector dimension)."""
        self.vocab_size = vocab_size
        self.word_to_id = {}  # Map: từ → index (0, 1, 2, ...)
        self.next_id = 0      # Counter cho từ mới

    def embed(self, text: str) -> np.ndarray:
        """Chuyển text thành embedding vector."""
        vec = np.zeros(self.vocab_size)
        
        # Tokenize: tách từ bằng space, convert to lowercase
        for token in text.lower().split():
            # Gán id cho từ mới (nếu chưa vượt vocab limit)
            if token not in self.word_to_id and self.next_id < self.vocab_size:
                self.word_to_id[token] = self.next_id
                self.next_id += 1
            
            # Tăng tần số
            token_id = self.word_to_id.get(token, -1)
            if token_id >= 0:
                vec[token_id] += 1

        # Normalize bằng L2 norm (chia cho độ dài)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec


class TinyVectorStore:
    """
    VECTOR STORE: Lưu trữ documents + embeddings, hỗ trợ semantic search.
    
    Flow:
      1. ADD: Document + embedding → store
      2. SEARCH: Query string → embedding → cosine similarity
                 → top-k most similar docs
    
    Dùng để: Retrieve relevant documents based on query
    """
    def __init__(self):
        self.docs: List[str] = []              # Documents gốc
        self.vectors: List[np.ndarray] = []    # Embeddings tương ứng

    def add(self, doc: str, vec: np.ndarray) -> None:
        """Thêm document + embedding vào store."""
        self.docs.append(doc)
        self.vectors.append(vec)

    def search(self, query_vec: np.ndarray, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Tìm top-k documents tương tự nhất dùng cosine similarity.
        
        Formula: cos_sim(v1, v2) = (v1 · v2) / (||v1|| * ||v2||)
                 where · = dot product, || || = L2 norm
        
        Return: List of (document, similarity_score) sorted by score DESC
        """
        scored = []
        
        # Tính similarity với tất cả docs
        for doc, vec in zip(self.docs, self.vectors):
            # Cosine similarity
            score = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-12))
            scored.append((doc, score))

        # Sort theo score giảm dần (cao nhất lên trước)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_k]  # Return top-k


# ============= UTILITY FUNCTIONS =============

def fake_llm_answer(query: str, context_docs: List[str]) -> str:
    """
    Tạo câu trả lời từ query + retrieved context (mock LLM).
    Trong app thực, đây sẽ là call tới OpenAI/Llama/etc.
    """
    context = " ".join(context_docs)
    return f"Answer for '{query}': based on context -> {context}"


def load_dotenv_fallback(file_path: str = ".env") -> None:
    """
    Fallback parser để load .env nếu python-dotenv chưa cài.
    
    Đọc file .env dòng-by-dòng, parse key=value, ghi vào os.environ.
    Format hỗ trợ: KEY="value" hoặc KEY=value hoặc KEY='value'
    """
    if not os.path.exists(file_path):
        return

    with open(file_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            
            # Bỏ qua: comment (#...), dòng trống
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            # Parse key=value
            key, value = stripped.split("=", 1)
            key = key.strip()
            # Bỏ dấu ngoặc kép/ngoặc đơn nếu có
            value = value.strip().strip('"').strip("'")
            
            # Ghi vào env (chỉ nếu chưa set)
            if key and key not in os.environ:
                os.environ[key] = value


def normalize_proxy_env() -> None:
    """
    Chuẩn hóa proxy environment variables để httpx tương thích.
    
    Vấn đề: Windows proxy settings dùng dấu `;` để phân cách
            (VD: "localhost;127.0.0.1;::1")
            Nhưng httpx dùng `,`
            
    Giải pháp: Chuyển `;` → `,` trong NO_PROXY
    """
    # Fix NO_PROXY format: convert semicolon to comma
    for no_proxy_key in ("NO_PROXY", "no_proxy"):
        raw_value = os.getenv(no_proxy_key)
        if raw_value and ";" in raw_value:
            os.environ[no_proxy_key] = raw_value.replace(";", ",")
    
    # Đảm bảo localhost bypass được set
    if "NO_PROXY" not in os.environ and "no_proxy" not in os.environ:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"


def run_demo() -> None:
    """
    Main function: Chạy RAG observability demo complete end-to-end.
    
    STEP-BY-STEP EXECUTION (xem FLOW TỔNG QUÁT ở trên):
    """
    
    # ─────── STEP 1: Load environment variables ───────
    print("\n" + "=" * 70)
    print("[STEP 1] Loading environment variables from .env")
    print("=" * 70)
    
    if load_dotenv is not None:
        load_dotenv()  # Dùng python-dotenv
        print("✓ Using python-dotenv")
    else:
        load_dotenv_fallback()  # Fallback parser
        print("✓ Using fallback .env parser")

    # ─────── STEP 2: Normalize proxy environment ───────
    normalize_proxy_env()
    print("✓ Proxy environment normalized")

    # Check Langfuse installed
    if Langfuse is None:
        print("\n✗ ERROR: langfuse package not found")
        print("  Install with: pip install langfuse")
        return

    # ─────── STEP 3: Validate credentials ───────
    print("\n" + "=" * 70)
    print("[STEP 2] Validating Langfuse credentials")
    print("=" * 70)
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"

    if not public_key or not secret_key:
        print("✗ ERROR: Missing credentials")
        print(f"  LANGFUSE_PUBLIC_KEY: {public_key is not None}")
        print(f"  LANGFUSE_SECRET_KEY: {secret_key is not None}")
        print("  Please set them in .env file")
        return

    print(f"✓ Credentials found")
    print(f"  Host: {host}")

    # ─────── STEP 3: Initialize Langfuse client ───────
    print("\n" + "=" * 70)
    print("[STEP 3] Initializing Langfuse client")
    print("=" * 70)
    
    try:
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        print(f"✓ Connected to Langfuse at {host}")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize Langfuse")
        print(f"  {e}")
        print(f"  Check: host={host}")
        print(f"  Check: proxy HTTP_PROXY={os.getenv('HTTP_PROXY')}, HTTPS_PROXY={os.getenv('HTTPS_PROXY')}")
        return

    # ─────── STEP 4: Prepare RAG data ───────
    print("\n" + "=" * 70)
    print("[STEP 4] Preparing RAG data")
    print("=" * 70)
    
    # Knowledge base (4 documents)
    docs = [
        "RAG combines retrieval and generation.",
        "Vector search returns the most similar chunks.",
        "Observability helps debug prompt, latency, and quality.",
        "Langfuse tracks traces, spans, generations, and scores.",
    ]

    query = "How to observe a RAG pipeline?"  # User query
    
    # Initialize embedding model & vector store
    embedding = TinyEmbedding()  # Will embed text to vectors
    store = TinyVectorStore()    # Will store documents + vectors
    
    print(f"✓ Loaded {len(docs)} documents")
    print(f"✓ Query: '{query}'")

    try:
        # ─────── STEP 5: Run RAG pipeline with observability ───────
        print("\n" + "=" * 70)
        print("[STEP 5] Running RAG pipeline with observability")
        print("=" * 70)
        
        # OBSERVATION 1: Index documents (SPAN)
        print("\n  [5.1] SPAN: index_documents")
        print("  ─" * 35)
        print("  Action: Embed all documents, store in vector DB")
        
        with langfuse.start_as_current_observation(
            name="index_documents",
            as_type="span",  # "span" = sub-operation, tracked in trace
            input={"doc_count": len(docs)},
            output={"status": "initialized"},
        ):
            # Embed & store each document
            for i, doc in enumerate(docs, 1):
                vec = embedding.embed(doc)  # text → vector
                store.add(doc, vec)         # store vector + doc
                print(f"    ✓ Doc {i}: embedded & stored")

        print("  ✓ SPAN completed")

        # OBSERVATION 2: Retrieve documents (SPAN)
        print("\n  [5.2] SPAN: retrieve")
        print("  ─" * 35)
        print("  Action: Vector search for top-2 similar documents")
        
        with langfuse.start_as_current_observation(
            name="retrieve",
            as_type="span",  # another sub-operation
            input={"query": query, "top_k": 2},
        ):
            query_vec = embedding.embed(query)              # text → vector
            retrieved = store.search(query_vec, top_k=2)    # vector search
            retrieved_docs = [doc for doc, _ in retrieved]

            print(f"    ✓ Searched for: '{query}'")
            print(f"    ✓ Found {len(retrieved_docs)} similar documents")
            for i, (doc, score) in enumerate(retrieved, 1):
                print(f"      {i}. [score={score:.3f}] {doc}")

        print("  ✓ SPAN completed")

        # OBSERVATION 3: Generate answer (GENERATION)
        print("\n  [5.3] GENERATION: generate_answer")
        print("  ─" * 35)
        print("  Action: Create answer from retrieved context")
        
        answer = fake_llm_answer(query, retrieved_docs)  # mock LLM
        
        with langfuse.start_as_current_observation(
            name="generate_answer",
            as_type="generation",  # "generation" = LLM output
            model="fake-llm",       # model name
            input={"query": query, "context": retrieved_docs},
            output={"answer": answer},
        ):
            pass
        
        print(f"    ✓ Answer generated")
        print("  ✓ GENERATION completed")

        # ─────── STEP 6: Flush traces to Langfuse cloud ───────
        print("\n" + "=" * 70)
        print("[STEP 6] Flushing traces to Langfuse")
        print("=" * 70)
        
        langfuse.flush()  # Send all observations to Langfuse cloud
        print("✓ Traces sent successfully to Langfuse cloud!")

        # ─────── STEP 7: Print final results ───────
        print("\n" + "=" * 70)
        print("[RESULTS]")
        print("=" * 70)
        
        print(f"\n📝 Query:\n  {query}")
        
        print(f"\n📚 Retrieved Documents ({len(retrieved_docs)}):")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  {i}. {doc}")
        
        print(f"\n💡 Generated Answer:\n  {answer}")
        
        print("\n" + "=" * 70)
        print("✓ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nYou can now check your Langfuse dashboard:")
        print(f"  → https://cloud.langfuse.com")
        print(f"  → Project: {public_key}")
        print(f"  → Look for trace: 'index_documents' → 'retrieve' → 'generate_answer'")
        print()

    except Exception as e:
        print(f"\n✗ ERROR during trace execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
