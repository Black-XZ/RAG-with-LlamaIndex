"""
Retriever Module
================
Provides retrieval functionality for the RAG pipeline.
"""

import logging
from typing import List, Optional, Dict, Any

# LlamaIndex imports - handle version compatibility
try:
    from llama_index.core.indices import VectorStoreIndex
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
    from llama_index.core.postprocessor import SimilarityPostprocessor
except ImportError:
    from llama_index import VectorStoreIndex
    from llama_index.schema import NodeWithScore, QueryBundle
    from llama_index.retrievers import VectorIndexRetriever, BaseRetriever
    from llama_index.postprocessor import SimilarityPostprocessor

logger = logging.getLogger(__name__)


class RetrieverConfig:
    """Configuration for the retriever."""

    def __init__(
        self,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
        alpha: float = 1.0,  # Balance between keyword and semantic
        mode: str = "default"  # "default", "mmr" (maximal marginal relevance)
    ):
        """
        Initialize retriever configuration.

        Args:
            top_k: Number of top results to retrieve.
            similarity_threshold: Minimum similarity score threshold.
            alpha: Balance between keyword (0) and semantic (1) retrieval.
            mode: Retrieval mode ("default" or "mmr").
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.mode = mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'alpha': self.alpha,
            'mode': self.mode
        }


class RAGRetriever:
    """Retriever wrapper for RAG pipeline."""

    def __init__(
        self,
        index: VectorStoreIndex,
        config: Optional[RetrieverConfig] = None
    ):
        """
        Initialize the retriever.

        Args:
            index: The vector store index.
            config: Optional retriever configuration.
        """
        self.index = index
        self.config = config or RetrieverConfig()

        # Create the base retriever
        self._setup_retriever()

        logger.info(f"RAGRetriever initialized with top_k={self.config.top_k}")

    def _setup_retriever(self):
        """Set up the internal retriever."""
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config.top_k,
            vector_store_query_mode="default",
            alpha=self.config.alpha,
            filters=None,
            node_ids=None,
            doc_ids=None,
            sparse_top_k=None,
        )

        # Add post-processor for similarity filtering
        self.postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.config.similarity_threshold
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.

        Args:
            query: The query string.
            k: Optional override for top_k.

        Returns:
            List of nodes with scores.
        """
        k = k or self.config.top_k

        # Temporarily adjust top_k if needed
        original_k = self.config.top_k
        if k != original_k:
            self.config.top_k = k
            self._setup_retriever()

        try:
            query_bundle = QueryBundle(query_str=query)
            nodes = self.retriever.retrieve(query_bundle)

            # Apply post-processing
            nodes = self.postprocessor.postprocess_nodes(nodes)

            logger.debug(f"Retrieved {len(nodes)} nodes for query: {query[:50]}...")

            return nodes

        finally:
            # Restore original top_k
            if k != original_k:
                self.config.top_k = original_k
                self._setup_retriever()

    def retrieve_with_metadata(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve nodes with formatted metadata for evaluation.

        Args:
            query: The query string.
            k: Optional override for top_k.

        Returns:
            List of node dictionaries with metadata.
        """
        nodes = self.retrieve(query, k)

        results = []
        for i, node in enumerate(nodes):
            results.append({
                'rank': i + 1,
                'node_id': node.node.node_id,
                'chunk_id': node.node.metadata.get('chunk_id', node.node.node_id),
                'source_file': node.node.metadata.get('source_file', 'unknown'),
                'score': node.score,
                'text': node.node.text,
                'text_preview': node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
            })

        return results

    def get_retrieval_stats(
        self,
        queries: List[str]
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.

        Args:
            queries: List of queries to analyze.

        Returns:
            Dictionary with retrieval statistics.
        """
        all_scores = []
        nodes_per_query = []

        for query in queries:
            nodes = self.retrieve(query)
            if nodes:
                all_scores.extend([n.score for n in nodes])
                nodes_per_query.append(len(nodes))

        return {
            'num_queries': len(queries),
            'avg_nodes_per_query': sum(nodes_per_query) / len(nodes_per_query) if nodes_per_query else 0,
            'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
            'min_score': min(all_scores) if all_scores else 0,
            'max_score': max(all_scores) if all_scores else 0,
            'score_std': self._calculate_std(all_scores) if all_scores else 0
        }

    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        import statistics
        return statistics.stdev(values) if len(values) > 1 else 0


def create_retriever(
    index: VectorStoreIndex,
    top_k: int = 3,
    similarity_threshold: float = 0.0
) -> RAGRetriever:
    """
    Convenience function to create a retriever.

    Args:
        index: Vector store index.
        top_k: Number of results to retrieve.
        similarity_threshold: Minimum similarity threshold.

    Returns:
        Configured RAGRetriever instance.
    """
    config = RetrieverConfig(
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    return RAGRetriever(index, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from loaders import load_documents
    from indexing import build_index_from_config

    # Load documents and build index
    result = load_documents("data")
    print(f"Loaded {len(result.documents)} documents")

    # Build index
    config_path = "configs/chunking_256.yaml"
    if Path(config_path).exists():
        index, config = build_index_from_config(config_path, result.documents)

        # Create retriever
        retriever = create_retriever(index, top_k=3)

        # Test retrieval
        query = "What is RAG?"
        results = retriever.retrieve_with_metadata(query)

        print(f"\nTop 3 results for: '{query}'")
        for r in results:
            print(f"  [{r['rank']}] Score: {r['score']:.4f} | Source: {r['source_file']}")
            print(f"      Preview: {r['text_preview'][:100]}...")
