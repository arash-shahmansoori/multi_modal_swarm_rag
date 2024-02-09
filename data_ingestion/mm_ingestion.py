import qdrant_client
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.indices.multi_modal.retriever import MultiModalVectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore


def ingest_mm_data(
    data_dir_path: str, ingest_dir_path: str, image_similarity_top_k: int
) -> MultiModalVectorIndexRetriever:
    """Ingest multi-modal data with persistance

    Args:
        data_dir_path (str): Path to the data directory to be ingested
        ingest_dir_path (str): Path to the ingested data directory
        image_similarity_top_k (int): Top-k relevant images

    Returns:
        MultiModalVectorIndexRetriever: Multi-modal retriever engine
    """

    # Read the images
    documents_images = SimpleDirectoryReader(data_dir_path).load_data()

    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path=ingest_dir_path)

    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    # Create the MultiModal index
    index = MultiModalVectorStoreIndex.from_documents(
        documents_images, storage_context=storage_context, show_progress=True
    )

    # Save it
    index.storage_context.persist(persist_dir=ingest_dir_path)

    retriever_engine = index.as_retriever(image_similarity_top_k=image_similarity_top_k)

    return retriever_engine
