from re import split
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document
from typing import List, Literal


Splitter = Literal["fixed", "recursive"]

def apply_chunker(
    docs: List[Document],
    method: Splitter,
    chunk_size: int,
    chunk_overlap: int,
):
    if method == "fixed":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
            
        )
        
    return splitter.split_documents(docs)