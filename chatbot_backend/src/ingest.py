import os
import sys
import logging

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rag_utils import load_documents, split_documents
from dotenv import load_dotenv

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables at module level
load_dotenv()

# --- Configuration ---
DATA_PATH = "data"
# Pinecone specific configurations
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_DIMENSION = 768

def create_vector_db():
    """
    Loads documents, splits them, creates embeddings, and stores them in Pinecone.
    """
    logger.info(f"Starting document ingestion and Pinecone index creation/update...")
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not pinecone_api_key or not pinecone_environment or not PINECONE_INDEX_NAME:
        logger.error("Pinecone API Key, Environment, or Index Name not found in .env. Aborting.")
        return
    
    if not google_api_key:
        logger.error("Google API Key not found in .env. Aborting.")
        return
    
    # Loading documents
    documents = load_documents(DATA_PATH)
    if not documents:
        logger.error("No documents loaded. Pinecone ingestion aborted.")
        return
    
    # Splitting documents into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    if not chunks:
        logger.error("No chunks created from documents. Pinecone ingestion aborted.")
        return
    
    # Initializing Embedding Model
    logger.info(f"Initializing Google Generative AI embedding model for Pinecone ingestion...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        logger.info("Google Generative AI embedding model initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}", exc_info=True)
        return
    
    # Initializing Pinecone connection
    logger.info(f"Initializing Pinecone with index: {PINECONE_INDEX_NAME}...")
    try:
        # PineconeVectorStore automatically initializes the Pinecone client internally
        # It will also create the index if it doesn't exist and add documents
        vector_db = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=pinecone_api_key
        )
        logger.info(f"Documents ingested into Pinecone index '{PINECONE_INDEX_NAME}' successfully.")
        logger.info(f"Total documents processed: {len(chunks)}")
        # Pinecone handles persistence automatically on its server
    except Exception as e:
        logger.error(f"Error creating/updating Pinecone index '{PINECONE_INDEX_NAME}': {e}", exc_info=True)
        return

if __name__ == "__main__":
    create_vector_db()