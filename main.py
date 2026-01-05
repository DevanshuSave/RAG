from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

print(f"Loaded {len(documents)} documents")

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.chunk_size = 512
Settings.chunk_overlap = 50

Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)


from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
import faiss

# Create FAISS index
dimension = 1536  # embedding dim
faiss_index = faiss.IndexFlatL2(dimension)

vector_store = FaissVectorStore(faiss_index=faiss_index)

# Build index
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)

# Persist to disk
index.storage_context.persist(persist_dir="./storage")


from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)


query_engine = index.as_query_engine(
    similarity_top_k=5
)


response = query_engine.query(
    "What does the document say about transformer attention?"
)

print(response)
