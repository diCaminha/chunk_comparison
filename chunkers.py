from re import split
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

from semantic_chunker import chunk_docs_via_llm

def apply_chunker(
    docs: List[Document],
    method: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split a list of Documents into chunks based on the specified method.

    Args:
        docs: List of LangChain Document objects to be chunked.
        method: 'fixed' for CharacterTextSplitter or 'recursive' for RecursiveCharacterTextSplitter.
        chunk_size: Maximum size of each chunk (in characters or tokens depending on splitter).
        chunk_overlap: Number of characters/tokens to overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with the original metadata preserved.
    """
    if method == "semantic":
         # Use LangChain's split_documents to handle metadata and splitting
        return chunk_docs_via_llm(docs)
    else:
        if method == "fixed":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Chunker method '{method}' not supported. Use 'fixed' or 'recursive'.")

        # Use LangChain's split_documents to handle metadata and splitting
        chunks: List[Document] = splitter.split_documents(docs)
        return chunks
