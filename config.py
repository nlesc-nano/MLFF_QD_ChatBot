import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DB_PATH = "./chroma_db_store"
    # Models: "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SEARCH_K = 10
    RERANK_TOP_N = 5
    
    DATA_DIR = "data"
    DATA_FILES = [
        "manual_user_guide.txt", 
        "qa_pairs_new.txt", 
        "paper.docx"
    ]

    @property
    def get_data_paths(self):
        """Returns full paths for data files."""
        return [os.path.join(self.DATA_DIR, f) for f in self.DATA_FILES]

config = Config()