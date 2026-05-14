"""
Knowledge Base — ChromaDB + Embeddings
=========================================
The CPU/RAM tier of the heterogeneous compute architecture.

Uses ChromaDB for vector storage and sentence-transformers for embedding.
The Lead Orchestrator must explicitly call search_knowledge_base() —
context is never auto-appended to prompts (PRD Section 4.4).
"""

import os
from typing import List, Dict, Any, Optional


class KnowledgeBase:
    """
    Vector knowledge base backed by ChromaDB and sentence-transformers.

    Designed to run on CPU/RAM (Intel Core Ultra 9 + 32GB DDR5).
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None,
                 embedding_model: str = None):
        """
        Initialize the knowledge base.

        Args:
            persist_dir: Directory for persistent ChromaDB storage.
                         If None, uses in-memory storage.
            collection_name: Name of the ChromaDB collection.
            embedding_model: HuggingFace model ID for embeddings.
        """
        self.persist_dir = persist_dir or os.environ.get("RAG_PERSIST_DIR", None)
        self.collection_name = collection_name or os.environ.get(
            "RAG_COLLECTION_NAME", "ctf_knowledge"
        )
        self.embedding_model_name = embedding_model or os.environ.get(
            "RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )

        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB client and embedding function."""
        if self._client is not None:
            return

        import chromadb

        # Initialize ChromaDB
        if self.persist_dir:
            # Resolve relative path
            if not os.path.isabs(self.persist_dir):
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                self.persist_dir = os.path.join(project_root, self.persist_dir)

            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            print(f"ChromaDB initialized (persistent): {self.persist_dir}")
        else:
            self._client = chromadb.Client()
            print("ChromaDB initialized (in-memory)")

        # Initialize embedding function
        self._embedding_fn = self._create_embedding_fn()

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        doc_count = self._collection.count()
        print(f"Collection '{self.collection_name}': {doc_count} documents")

    def _create_embedding_fn(self):
        """Create a ChromaDB-compatible embedding function using sentence-transformers."""
        import chromadb.utils.embedding_functions as ef

        return ef.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device="cpu",  # Always CPU — GPU is reserved for LLM
        )

    def add_documents(self, documents: List[str], metadatas: List[dict] = None,
                      ids: List[str] = None):
        """
        Add documents to the knowledge base.

        Args:
            documents: List of text chunks to add.
            metadatas: Optional list of metadata dicts per document.
            ids: Optional list of unique IDs. Auto-generated if not provided.
        """
        self._ensure_initialized()

        if not documents:
            return

        if ids is None:
            # Generate deterministic IDs from content hash
            import hashlib
            ids = [
                hashlib.sha256(doc.encode()).hexdigest()[:16]
                for doc in documents
            ]

        if metadatas is None:
            metadatas = [{}] * len(documents)

        # Batch upsert (ChromaDB handles deduplication by ID)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            self._collection.upsert(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )

        print(f"Added/updated {len(documents)} documents. "
              f"Total: {self._collection.count()}")

    def search(self, query: str, top_k: int = None,
               category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant chunks.

        Args:
            query: The search query (natural language).
            top_k: Number of results to return.
            category_filter: Optional category to filter by.

        Returns:
            List of result dicts with 'text', 'metadata', and 'distance'.
        """
        self._ensure_initialized()

        if top_k is None:
            top_k = int(os.environ.get("RAG_TOP_K", "5"))

        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            print(f"Knowledge base query failed: {e}")
            return []

        # Format results
        formatted = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0,
                }
                formatted.append(result)

        return formatted

    def search_formatted(self, query: str, top_k: int = None) -> str:
        """
        Search and return results as a formatted string for LLM consumption.

        This is what the search_knowledge_base tool returns to the orchestrator.
        """
        results = self.search(query, top_k)

        if not results:
            return "No relevant knowledge found for this query."

        lines = [f"Knowledge Base Results ({len(results)} matches):"]
        for i, r in enumerate(results, 1):
            source = r['metadata'].get('source', 'unknown')
            category = r['metadata'].get('category', '')
            lines.append(f"\n--- Result {i} [source: {source}, category: {category}] ---")
            lines.append(r['text'])

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get collection statistics."""
        self._ensure_initialized()
        return {
            "collection": self.collection_name,
            "document_count": self._collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir,
        }


# =============================================================================
# Module-level singleton for shared access
# =============================================================================

_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the singleton knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
