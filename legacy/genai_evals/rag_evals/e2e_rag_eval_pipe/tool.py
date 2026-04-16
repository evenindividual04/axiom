import os
from agno.knowledge import Knowledge
from agno.vectordb.llamaindex import LlamaIndexVectorDb
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.core import VectorStoreIndex, StorageContext, Settings
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient

load_dotenv(find_dotenv())

# Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001",
#                                             embedding_config=EmbedContentConfig(output_dimensionality=768))
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

qdrant_connector = QdrantClient(url="http://localhost:6333", api_key="th3s3cr3tk3y")

def retrieve_leukemia_knowledge_base(query: str) -> str:
    """
    Use this tool to retrieve knowledge about leukemia.

    @type query: str
    @return: str
    """
    if qdrant_connector.collection_exists(collection_name=os.environ.get("COLLECTION_NAME")):
        vector_store = QdrantVectorStore(client=qdrant_connector, collection_name=os.environ.get("COLLECTION_NAME")) # collection name should match the collection name while ingesting
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        retriever = index.as_retriever(top_k=15)
        knowledge = Knowledge(
            vector_db=LlamaIndexVectorDb(knowledge_retriever=retriever)
        )
        context = ''
        documents = knowledge.search(query=query, max_results=10)
        for document in documents:
            context += document.content

        print(f"Context retrieved: {context}")
        return context
    else:
        # handle the fallback if not qdrant
        pass
