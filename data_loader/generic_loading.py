from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def doc_loader(name: str, source_loader: BaseLoader, **kwargs) -> List[Document]:
    """A generic function to load the documents

    Args:
        name (str): The path name of documents to be loaded
        source_loader (BaseLoader): The source document loader

    Returns:
        List[Document]: List of documents
    """
    loader = source_loader(name, **kwargs)
    docs = loader.load()
    return docs
