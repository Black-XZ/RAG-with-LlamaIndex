"""
RAG with LlamaIndex
===================
A Document Question Answering system with configurable chunking and LLM backends.
"""

__version__ = "0.1.0"
__author__ = "RAG-LlamaIndex Project"

# Core modules
from src.loaders import DocumentLoader, load_documents
from src.indexing import ChunkingConfig, IndexBuilder, build_index_from_config
from src.retriever import RAGRetriever, RetrieverConfig, create_retriever
from src.llm_backends import (
    BaseLLMBackend,
    Mistral7BBackend,
    T5BaseBackend,
    MockLLMBackend,
    create_llm_backend,
    LLMConfig,
    GenerationResult
)
from src.rag_pipeline import RAGPipeline, RAGConfig, RAGResponse, create_rag_pipeline
from src.eval_protocol import EvaluationProtocol, EvalQuery, EvalResult
from src.metrics import MetricsCalculator, MetricSummary, calculate_statistics

__all__ = [
    # Loaders
    "DocumentLoader",
    "load_documents",
    # Indexing
    "ChunkingConfig",
    "IndexBuilder",
    "build_index_from_config",
    # Retriever
    "RAGRetriever",
    "RetrieverConfig",
    "create_retriever",
    # LLM
    "BaseLLMBackend",
    "Mistral7BBackend",
    "T5BaseBackend",
    "MockLLMBackend",
    "create_llm_backend",
    "LLMConfig",
    "GenerationResult",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
    "RAGResponse",
    "create_rag_pipeline",
    # Evaluation
    "EvaluationProtocol",
    "EvalQuery",
    "EvalResult",
    # Metrics
    "MetricsCalculator",
    "MetricSummary",
    "calculate_statistics",
]
