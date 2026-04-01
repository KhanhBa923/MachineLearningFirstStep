"""
RAG (Retrieval-Augmented Generation) and Vector Search Example
Demonstrates how to:
1. Create text embeddings using vector representations
2. Store and search through vectors
3. Retrieve relevant documents based on semantic similarity
4. Use retrieved documents for generation (RAG)
"""

import numpy as np
from typing import List, Tuple
import math


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration"""
    
    def __init__(self):
        self.documents = []
        self.vectors = []
    
    def add_document(self, text: str, vector: np.ndarray):
        """Add a document with its embedding vector"""
        self.documents.append(text)
        self.vectors.append(vector)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for top-k most similar documents"""
        similarities = []
        
        for doc, vec in zip(self.documents, self.vectors):
            similarity = self.cosine_similarity(query_vector, vec)
            similarities.append((doc, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class SimpleEmbedding:
    """Simple embedding generator (using word frequency - for demonstration only)"""
    
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        # Simple word-to-index mapping
        self.word_index = {}
        self.index = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def _get_word_index(self, word: str) -> int:
        """Get or create word index"""
        if word not in self.word_index:
            if self.index < self.vocab_size:
                self.word_index[word] = self.index
                self.index += 1
            else:
                return -1
        return self.word_index[word]
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        vector = np.zeros(self.vocab_size)
        words = self._tokenize(text)
        
        for word in words:
            idx = self._get_word_index(word)
            if idx >= 0:
                vector[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class RAGSystem:
    """Simple RAG system combining retrieval and generation"""
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.embedding_model = SimpleEmbedding()
        self.knowledge_base = []
    
    def add_knowledge(self, documents: List[str]):
        """Add documents to knowledge base"""
        self.knowledge_base = documents
        
        for doc in documents:
            vector = self.embedding_model.embed(doc)
            self.vector_store.add_document(doc, vector)
        
        print(f"✓ Added {len(documents)} documents to knowledge base")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents for a query"""
        query_vector = self.embedding_model.embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        
        retrieved_docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        return retrieved_docs, scores
    
    def generate_answer(self, query: str, top_k: int = 3) -> str:
        """Generate answer using RAG approach"""
        retrieved_docs, scores = self.retrieve(query, top_k)
        
        # Simple generation: combine retrieved documents
        answer = f"Query: {query}\n\n"
        answer += "Retrieved Context:\n"
        
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
            answer += f"{i}. [Similarity: {score:.3f}] {doc}\n"
        
        answer += f"\nSummary: Based on the retrieved documents, "
        answer += f"the most relevant information to '{query}' has been found above."
        
        return answer


# ============= DEMO =============

def main():
    print("=" * 60)
    print("RAG & Vector Search Example")
    print("=" * 60)
    
    # Step 1: Initialize knowledge base
    knowledge_base = [
        "Python is a high-level programming language known for simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Vector embeddings convert text into numerical representations for semantic similarity.",
        "Retrieval-augmented generation combines information retrieval with language model generation.",
        "Natural language processing helps computers understand and generate human language.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Data science involves extracting insights from data using statistical and computational methods.",
        "Neural networks are inspired by the structure and function of biological brains."
    ]
    
    # Step 2: Create RAG system and add knowledge
    print("\n1. Initializing RAG System...")
    rag_system = RAGSystem()
    rag_system.add_knowledge(knowledge_base)
    
    # Step 3: Test retrieval and generation
    print("\n2. Testing Retrieval & Generation...\n")
    
    test_queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How do embeddings work?"
    ]
    
    for query in test_queries:
        print("-" * 60)
        answer = rag_system.generate_answer(query, top_k=2)
        print(answer)
        print()
    
    # Step 4: Direct vector search example
    print("=" * 60)
    print("3. Direct Vector Search Example")
    print("=" * 60)
    
    query = "programming and coding"
    print(f"\nQuery: '{query}'")
    print("Most similar documents:")
    
    retrieved_docs, scores = rag_system.retrieve(query, top_k=3)
    for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
        print(f"{i}. [Score: {score:.3f}] {doc}")


if __name__ == "__main__":
    main()
