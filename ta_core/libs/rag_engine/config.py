from dotenv import load_dotenv
import os

load_dotenv()

from config import models  # noqa: E402  (shared LLM endpoint, Ollama by default)


class Config:
    API_KEY = models.llm_api_key
    BASE_URL = models.llm_base_url

    LLM_MODEL = os.getenv("LLM_MODEL_NAME") or models.financial_model
    JUDGE_MODEL = os.getenv("JUDGE_MODEL_NAME") or models.financial_model
    REASONER_MODEL = os.getenv("REASONER_MODEL") or models.t4_cio_model
    # Embeddings run locally through sentence-transformers (no API needed).
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    GPT_MODEL = os.getenv("GPT_MODEL") or models.primary_model
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K = 20

    BASE_WORKDIR = os.getenv("WORKDIR", "./rag_storage")
    MAX_RPM = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 100))
    MAX_CONCURRENCY = models.llm_concurrency
    LLM_TIMEOUT_SECONDS = float(os.getenv("RAG_LLM_TIMEOUT_SECONDS", models.llm_timeout_seconds))
    STORAGE_INIT_TIMEOUT_SECONDS = float(os.getenv("RAG_STORAGE_INIT_TIMEOUT_SECONDS", 120))
    QUERY_TIMEOUT_SECONDS = float(os.getenv("RAG_QUERY_TIMEOUT_SECONDS", 660))
    QUESTION_TIMEOUT_SECONDS = float(os.getenv("RAG_QUESTION_TIMEOUT_SECONDS", 2400))

    CHUNK_SIZE = 4096
    CHUNK_OVERLAP = 300

settings = Config()