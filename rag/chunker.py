"""
Micro-Chunker
===============
Token-aware text chunking for the RAG knowledge base.

Per the PRD, chunks must be strictly 150-250 tokens to deliver precise
code snippets without flooding the 8GB GPU VRAM. This module splits
documents at sentence boundaries within that window and preserves
code blocks intact.
"""

import re
from typing import List, Tuple


def _count_tokens(text: str) -> int:
    """
    Approximate token count using tiktoken (cl100k_base) if available,
    otherwise fall back to word-count heuristic (1 token ~ 0.75 words).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: ~1.33 tokens per word is a reasonable approximation
        return int(len(text.split()) * 1.33)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, preserving code blocks as atomic units.
    Code blocks (``` ... ```) are never split mid-block.
    """
    # Extract code blocks first, replace with placeholders
    code_blocks = []
    code_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)

    def _replace_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text_with_placeholders = code_pattern.sub(_replace_code, text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text_with_placeholders)

    # Also split on newlines for list items, headers, etc.
    expanded = []
    for s in sentences:
        parts = s.split('\n')
        for part in parts:
            part = part.strip()
            if part:
                expanded.append(part)

    # Restore code blocks
    result = []
    for s in expanded:
        for i, block in enumerate(code_blocks):
            s = s.replace(f"__CODE_BLOCK_{i}__", block)
        result.append(s)

    return result


def chunk_text(text: str, target_size: int = 200, min_size: int = 150,
               max_size: int = 250, overlap: int = 30) -> List[str]:
    """
    Chunk text into micro-chunks within the token budget.

    Args:
        text: The text to chunk.
        target_size: Target tokens per chunk (default 200).
        min_size: Minimum tokens per chunk (default 150).
        max_size: Maximum tokens per chunk (default 250).
        overlap: Overlap tokens between consecutive chunks (default 30).

    Returns:
        List of text chunks, each within [min_size, max_size] tokens.
    """
    if not text or not text.strip():
        return []

    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If a single sentence exceeds max_size, force-split it
        if sentence_tokens > max_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split the oversized sentence by words
            words = sentence.split()
            sub_chunk = []
            sub_tokens = 0
            for word in words:
                word_tokens = _count_tokens(word)
                if sub_tokens + word_tokens > max_size and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    # Keep overlap
                    overlap_words = sub_chunk[-max(1, overlap // 4):]
                    sub_chunk = list(overlap_words)
                    sub_tokens = _count_tokens(" ".join(sub_chunk))
                sub_chunk.append(word)
                sub_tokens += word_tokens
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        # Check if adding this sentence would exceed max
        if current_tokens + sentence_tokens > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Build overlap from the tail of the previous chunk
            if overlap > 0:
                overlap_text = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = _count_tokens(s)
                    if overlap_tokens + s_tokens > overlap:
                        break
                    overlap_text.insert(0, s)
                    overlap_tokens += s_tokens
                current_chunk = overlap_text
                current_tokens = overlap_tokens
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Flush remaining
    if current_chunk:
        chunk_text_str = " ".join(current_chunk)
        # If the last chunk is too small, merge with previous
        if chunks and _count_tokens(chunk_text_str) < min_size:
            chunks[-1] = chunks[-1] + " " + chunk_text_str
        else:
            chunks.append(chunk_text_str)

    return chunks


def chunk_document(text: str, metadata: dict = None,
                   target_size: int = 200) -> List[Tuple[str, dict]]:
    """
    Chunk a document and pair each chunk with metadata.

    Args:
        text: Document text.
        metadata: Base metadata dict (source, category, etc.).
        target_size: Target tokens per chunk.

    Returns:
        List of (chunk_text, chunk_metadata) tuples.
    """
    if metadata is None:
        metadata = {}

    chunks = chunk_text(text, target_size=target_size)
    result = []

    for i, chunk in enumerate(chunks):
        chunk_meta = {
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "token_count": _count_tokens(chunk),
        }
        result.append((chunk, chunk_meta))

    return result
