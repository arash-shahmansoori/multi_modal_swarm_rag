from langchain.schema import BaseDocumentTransformer


def doc_splitter(
    doc_transform: BaseDocumentTransformer, **kwargs
) -> BaseDocumentTransformer:
    """A generic function to split the documents

    Args:
        doc_transform (BaseDocumentTransformer): Apply the transformation to the documents

    Returns:
        BaseDocumentTransformer: The transformed documents object
    """
    transform = doc_transform(**kwargs)
    return transform
