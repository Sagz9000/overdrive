"""
Knowledge Base Ingestion CLI
===============================
Walks a directory of .md, .txt, .json, .yaml files, chunks them using
the micro-chunker, and upserts into ChromaDB.

Usage:
    python -m rag.ingest --source data/knowledge/ --persist-dir data/chromadb/
    python -m rag.ingest --source data/knowledge/gtfobins/ --category gtfobins
"""

import os
import sys
import json
import argparse
import hashlib
from typing import List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from rag.chunker import chunk_document
from rag.knowledge_base import KnowledgeBase


# File extensions we can ingest
SUPPORTED_EXTENSIONS = {'.md', '.txt', '.json', '.yaml', '.yml', '.rst'}


def _detect_category(filepath: str, default_category: str = "") -> str:
    """Detect the knowledge category from the file path."""
    path_lower = filepath.lower()

    category_map = {
        'gtfobins': 'privilege_escalation',
        'payloads': 'payloads',
        'exploitdb': 'exploits',
        'cheatsheets': 'cheatsheets',
        'cheatsheet': 'cheatsheets',
        'web': 'web',
        'sqli': 'web',
        'xss': 'web',
        'lfi': 'web',
        'crypto': 'crypto',
        'forensics': 'forensics',
        'binary': 'binary',
        'reverse': 'binary',
        'network': 'network',
        'nmap': 'network',
    }

    for keyword, category in category_map.items():
        if keyword in path_lower:
            return category

    return default_category or 'general'


def _read_file(filepath: str) -> str:
    """Read a file's contents, handling different formats."""
    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    if ext == '.json':
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # Flatten JSON to text
                parts = []
                for key, value in data.items():
                    if isinstance(value, str):
                        parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                return "\n".join(parts)
            elif isinstance(data, list):
                return "\n\n".join(
                    json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                    for item in data
                )
        except json.JSONDecodeError:
            pass

    return content


def walk_and_chunk(source_dir: str, category: str = "",
                   chunk_size: int = 200) -> List[Tuple[str, dict, str]]:
    """
    Walk a directory, read files, and chunk them.

    Returns:
        List of (chunk_text, metadata, chunk_id) tuples.
    """
    results = []

    if not os.path.exists(source_dir):
        print(f"Warning: Source directory not found: {source_dir}")
        return results

    for root, dirs, files in os.walk(source_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, source_dir)

            try:
                content = _read_file(filepath)
                if not content.strip():
                    continue

                file_category = _detect_category(rel_path, category)

                metadata = {
                    "source": rel_path,
                    "filename": filename,
                    "category": file_category,
                }

                chunks = chunk_document(content, metadata=metadata,
                                        target_size=chunk_size)

                for chunk_text, chunk_meta in chunks:
                    # Generate deterministic ID
                    chunk_id = hashlib.sha256(
                        f"{rel_path}:{chunk_meta['chunk_index']}:{chunk_text[:100]}".encode()
                    ).hexdigest()[:16]

                    results.append((chunk_text, chunk_meta, chunk_id))

            except Exception as e:
                print(f"Warning: Failed to process {filepath}: {e}")

    return results


def ingest(source_dir: str, persist_dir: str = None,
           collection_name: str = None, category: str = "",
           chunk_size: int = 200):
    """
    Full ingestion pipeline: walk -> chunk -> embed -> store.
    """
    print(f"Ingesting from: {source_dir}")
    print(f"Persist dir:    {persist_dir or 'in-memory'}")
    print(f"Chunk size:     {chunk_size} tokens")

    # Walk and chunk
    chunks = walk_and_chunk(source_dir, category=category, chunk_size=chunk_size)

    if not chunks:
        print("No documents found to ingest.")
        return

    print(f"Generated {len(chunks)} chunks from source files.")

    # Initialize knowledge base
    kb = KnowledgeBase(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    # Separate into parallel lists
    documents = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    ids = [c[2] for c in chunks]

    # Upsert
    kb.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    # Print stats
    stats = kb.get_stats()
    print(f"\nIngestion complete:")
    print(f"  Collection:  {stats['collection']}")
    print(f"  Documents:   {stats['document_count']}")
    print(f"  Embedding:   {stats['embedding_model']}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--source", required=True, help="Source directory to ingest")
    parser.add_argument("--persist-dir", default=None, help="ChromaDB persistence directory")
    parser.add_argument("--collection", default=None, help="Collection name")
    parser.add_argument("--category", default="", help="Force a category for all documents")
    parser.add_argument("--chunk-size", type=int, default=200, help="Target chunk size in tokens")
    args = parser.parse_args()

    ingest(
        source_dir=args.source,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        category=args.category,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
