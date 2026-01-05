from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext

def ingest():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")

if __name__ == "__main__":
    ingest()
