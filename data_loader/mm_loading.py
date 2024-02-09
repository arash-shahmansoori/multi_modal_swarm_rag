import qdrant_client
from llama_index import StorageContext, load_index_from_storage
from llama_index.indices.multi_modal.retriever import MultiModalVectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore


def load_mm_data(
    ingest_dir_path: str, image_similarity_top_k, indx_id: str | None = None
) -> MultiModalVectorIndexRetriever:
    """Load multi-modal data from the persisted directory

    Args:
        ingest_dir_path (str): Path to the ingested data directory (persisted)
        image_similarity_top_k (int): Top-k relevant images
        indx_id (Optional[str]): ID of the index to load.
        Defaults to None, which assumes there's only a single index in the index store and load it.

    Returns:
        MultiModalVectorIndexRetriever: Multi-modal retriever engine
    """

    client = qdrant_client.QdrantClient(path=ingest_dir_path)

    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store, persist_dir=ingest_dir_path
    )

    index = load_index_from_storage(
        storage_context, indx_id, text_store=text_store, image_store=image_store
    )

    retriever_engine = index.as_retriever(image_similarity_top_k=image_similarity_top_k)

    return retriever_engine
