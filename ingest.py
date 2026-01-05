from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
import faiss

from config import CHUNK_SIZE, CHUNK_OVERLAP, HF_EMBEDDING_MODEL, EMBEDDING_DIM
def configure_embedding_settings():

    # Configure chunking
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    # Configure embedding
    Settings.node_parser = SentenceSplitter()

    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL)

    # Configure Local LLM
    Settings.llm = Ollama(model="llama3", request_timeout=60)


def ingest():
    configure_embedding_settings()
    documents = SimpleDirectoryReader("data").load_data()
    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    index.storage_context.persist(persist_dir="storage")
    print('Completed configuration and data storage.')

if __name__ == "__main__":
    ingest()
