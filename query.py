from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from ingest import configure_embedding_settings

from config import HF_EMBEDDING_MODEL

configure_embedding_settings()
def query():
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    # Always a good idea to pass embedding model to avoid running into default traps
    index = load_index_from_storage(storage_context=storage_context, embed_model=HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL))

    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query("What is this document about?")
    print(response.response)

    for node in response.source_nodes:
        print(node.metadata, node.score)

if __name__ == "__main__":
    query()
