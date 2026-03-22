"""
RAG Pipeline Module
===================
End-to-end RAG pipeline: query -> retrieve -> generate.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

# LlamaIndex imports - handle version compatibility
try:
    from llama_index.core.indices import VectorStoreIndex
except ImportError:
    from llama_index import VectorStoreIndex

from src.llm_backends import BaseLLMBackend, GenerationResult, create_llm_backend
from src.retriever import RAGRetriever, RetrieverConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    cited_chunk_ids: List[str]
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    model_name: str
    llm_config: Dict[str, Any]
    query: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/saving."""
        return {
            'query': self.query,
            'answer': self.answer,
            'cited_chunk_ids': self.cited_chunk_ids,
            'retrieval_latency': self.retrieval_latency,
            'generation_latency': self.generation_latency,
            'total_latency': self.total_latency,
            'model_name': self.model_name,
            'num_retrieved_chunks': len(self.retrieved_chunks),
            'error': self.error
        }


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    llm_backend: str = "flant5"  # "mistral7b", "flant5", "t5base", "mock"
    top_k: int = 3
    similarity_threshold: float = 0.0
    max_new_tokens: int = 256
    temperature: float = 0.1
    use_quantization: bool = False
    prompt_template: Optional[str] = None
    device: str = "auto"

    # System prompts
    default_system_prompt: str = """You are a document-grounded assistant. Use ONLY the provided context chunks from the retrieved documents to answer. If the answer is not found in the context, say "I cannot find sufficient evidence in the provided documents." Cite the chunk IDs you used."""

    default_user_template: str = """Question: {question}

Context Chunks:
{context}

Provide a concise, well-structured answer. Add citations like [C1], [C2] that map to chunk IDs."""


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Takes a query, retrieves relevant chunks, and generates an answer.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            index: The vector store index.
            config: Optional pipeline configuration.
        """
        self.index = index
        self.config = config or RAGConfig()

        # Initialize retriever
        retriever_config = RetrieverConfig(
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold
        )
        self.retriever = RAGRetriever(index, retriever_config)

        # Initialize LLM backend
        self.llm = self._setup_llm()

        logger.info(f"RAG Pipeline initialized with {self.config.llm_backend} backend")

    def _setup_llm(self) -> BaseLLMBackend:
        """Set up the LLM backend."""
        from src.llm_backends import LLMConfig

        llm_config = LLMConfig(
            model_name=self.config.llm_backend,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            use_quantization=self.config.use_quantization
        )

        return create_llm_backend(self.config.llm_backend, llm_config)

    def query(
        self,
        question: str,
        return_chunks: bool = True
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.

        Args:
            question: The user's question.
            return_chunks: Whether to include retrieved chunks in response.

        Returns:
            RAGResponse with answer and metadata.
        """
        total_start = time.time()
        retrieval_start = time.time()

        try:
            # Step 1: Retrieve relevant chunks
            retrieved_nodes = self.retriever.retrieve(question)
            retrieval_latency = time.time() - retrieval_start

            if not retrieved_nodes:
                logger.warning(f"No relevant chunks found for query: {question[:50]}...")
                return RAGResponse(
                    answer="I cannot find sufficient evidence in the provided documents.",
                    retrieved_chunks=[],
                    cited_chunk_ids=[],
                    retrieval_latency=retrieval_latency,
                    generation_latency=0.0,
                    total_latency=time.time() - total_start,
                    model_name=self.config.llm_backend,
                    llm_config=self.config.__dict__,
                    query=question,
                    error="No relevant chunks found"
                )

            # Step 2: Prepare context
            context_parts = []
            chunk_metadata = []

            for i, node_with_score in enumerate(retrieved_nodes):
                node = node_with_score.node
                chunk_id = node.metadata.get('chunk_id', f"C{i+1}")

                context_parts.append(f"[{chunk_id}]\n{node.text}")
                chunk_metadata.append({
                    'chunk_id': chunk_id,
                    'source_file': node.metadata.get('source_file', 'unknown'),
                    'score': node_with_score.score,
                    'text_preview': node.text[:200]
                })

            context = "\n\n".join(context_parts)

            # Step 3: Generate answer
            generation_start = time.time()
            result = self.llm.generate(question, context)
            generation_latency = time.time() - generation_start

            # Step 4: Extract cited chunk IDs (basic extraction)
            cited_ids = self._extract_citations(result.text, chunk_metadata)

            return RAGResponse(
                answer=result.text,
                retrieved_chunks=chunk_metadata if return_chunks else [],
                cited_chunk_ids=cited_ids,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=time.time() - total_start,
                model_name=result.model_name,
                llm_config={
                    'max_new_tokens': self.config.max_new_tokens,
                    'temperature': self.config.temperature
                },
                query=question
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer="An error occurred while processing your query.",
                retrieved_chunks=[],
                cited_chunk_ids=[],
                retrieval_latency=0.0,
                generation_latency=0.0,
                total_latency=time.time() - total_start,
                model_name=self.config.llm_backend,
                llm_config=self.config.__dict__,
                query=question,
                error=str(e)
            )

    def _extract_citations(
        self,
        answer: str,
        chunk_metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract cited chunk IDs from the answer."""
        import re

        cited_ids = []
        citation_pattern = r'\[C(\d+)\]'

        matches = re.findall(citation_pattern, answer)

        for match in matches:
            idx = int(match) - 1
            if 0 <= idx < len(chunk_metadata):
                cited_ids.append(chunk_metadata[idx]['chunk_id'])

        return list(set(cited_ids))

    def batch_query(
        self,
        questions: List[str],
        verbose: bool = True
    ) -> List[RAGResponse]:
        """
        Process multiple queries.

        Args:
            questions: List of questions to process.
            verbose: Whether to print progress.

        Returns:
            List of RAGResponse objects.
        """
        responses = []

        for i, question in enumerate(questions):
            if verbose:
                print(f"Processing {i+1}/{len(questions)}: {question[:50]}...")

            response = self.query(question)
            responses.append(response)

        return responses


def create_rag_pipeline(
    index: VectorStoreIndex,
    llm_backend: str = "flant5",
    top_k: int = 3,
    **kwargs
) -> RAGPipeline:
    """
    Convenience function to create a RAG pipeline.

    Args:
        index: Vector store index.
        llm_backend: LLM backend type.
        top_k: Number of chunks to retrieve.
        **kwargs: Additional configuration options.

    Returns:
        Configured RAGPipeline instance.
    """
    config = RAGConfig(
        llm_backend=llm_backend,
        top_k=top_k,
        **kwargs
    )
    return RAGPipeline(index, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from loaders import load_documents
    from indexing import build_index_from_config

    # Load documents and build index
    print("Loading documents...")
    result = load_documents("data")
    print(f"Loaded {len(result.documents)} documents")

    # Build index
    config_path = "configs/chunking_256.yaml"
    if Path(config_path).exists():
        print("Building index...")
        index, config = build_index_from_config(config_path, result.documents)

        # Create RAG pipeline with mock backend for testing
        print("\nCreating RAG pipeline...")
        pipeline = create_rag_pipeline(
            index,
            llm_backend="mock",  # Use mock for quick testing
            top_k=3
        )

        # Test query
        print("\n--- Testing RAG Pipeline ---")
        query = "What is retrieval-augmented generation?"
        response = pipeline.query(query)

        print(f"\nQuery: {response.query}")
        print(f"Answer: {response.answer}")
        print(f"\nRetrieved {len(response.retrieved_chunks)} chunks:")
        for chunk in response.retrieved_chunks:
            print(f"  [{chunk['chunk_id']}] Score: {chunk['score']:.4f} | {chunk['source_file']}")

        print(f"\nLatency breakdown:")
        print(f"  Retrieval: {response.retrieval_latency:.3f}s")
        print(f"  Generation: {response.generation_latency:.3f}s")
        print(f"  Total: {response.total_latency:.3f}s")
