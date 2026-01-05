from llama_index.core import StorageContext, load_index_from_storage

def query():
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("What is this document about?")

    for node in response.source_nodes:
        print(node.metadata, node.score)

if __name__ == "__main__":
    query()
