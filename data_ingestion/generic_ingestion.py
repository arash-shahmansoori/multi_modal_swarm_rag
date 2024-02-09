from typing import Union

from langchain.document_loaders.base import BaseLoader
from langchain.pydantic_v1 import BaseModel
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain_core.documents import BaseDocumentTransformer

from data_loader import doc_loader
from type_extensions import T
from utils import doc_splitter


def ingestion(
    name: str,
    source_loader: BaseLoader,
    splitter: BaseDocumentTransformer,
    chunk_size: int,
    chunk_overlap: int,
    persist_directory: str,
    vec_store: VectorStore,
    embedding: Union[BaseModel, Embeddings],
) -> VectorStore:
    """Document ingestion into the vector store

    Args:
        name (str): Source document directory, url, or resources
        source_loader (BaseLoader): The source document loader
        splitter (BaseDocumentTransformer): The splitter
        chunk_size (int): Chunk size for embedding
        chunk_overlap (int): Chunk overlap
        persist_directory (str): Directory for data persistence
        vec_store (VectorStore): The vector store
        embedding (Union[BaseModel, Embeddings]): Embedding for vector storage

    Returns:
        VectorStore: The vector store containing the documents
    """
    docs = doc_loader(name, source_loader)

    split = doc_splitter(splitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = split.split_documents(docs)

    vectordb = vec_store.from_documents(
        documents=splits, embedding=embedding, persist_directory=persist_directory
    )

    return vectordb
