"""
Evaluation Runner CLI
====================
Command-line interface for running RAG evaluations.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline, RAGConfig
from src.eval_protocol import EvaluationProtocol, EvalResult
from src.indexing import ChunkingConfig
from src.utils import setup_logging, get_timestamp, ensure_dir, set_seed

logger = logging.getLogger("RAG.Eval")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # LLM configuration
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        choices=["mistral7b", "flant5", "t5base", "mock"],
        help="LLM backend to use"
    )

    # Chunking configuration
    parser.add_argument(
        "--chunk",
        type=int,
        required=True,
        choices=[256, 512],
        help="Chunk size (must match a config file)"
    )

    # Index configuration
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Path to pre-built index directory"
    )

    # Data configuration
    parser.add_argument(
        "--queries",
        type=str,
        default="data/queries.jsonl",
        help="Path to queries JSONL file"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )

    # Run configuration
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve"
    )

    # LLM parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature"
    )

    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use 4-bit quantization for Mistral"
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    parser.add_argument(
        "--create-human-template",
        action="store_true",
        help="Also create human evaluation template"
    )

    return parser.parse_args()


def load_index(index_dir: str, chunk_size: int):
    """
    Load or build the index.

    Args:
        index_dir: Path to index directory.
        chunk_size: Expected chunk size.

    Returns:
        Tuple of (index, actual_chunk_size)
    """
    # LlamaIndex imports - handle version compatibility
    try:
        from llama_index.core import load_index_from_storage
        from llama_index.core.storage import StorageContext
        from llama_index.core import Settings
    except ImportError:
        from llama_index import load_index_from_storage
        from llama_index.storage import StorageContext
        from llama_index import Settings

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    index_path = Path(index_dir)

    if not index_path.exists():
        logger.warning(f"Index not found at {index_dir}, building new index...")
        return None, None

    # Load index metadata
    metadata_path = index_path / "metadata.json"
    actual_chunk = None
    embed_model_name = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        actual_chunk = metadata.get('config', {}).get('chunk_size', None)
        embed_model_name = metadata.get('config', {}).get('embedding_model', None)
        logger.info(f"Loaded index with chunk size: {actual_chunk}, embedding: {embed_model_name}")

    # Set up embedding model from metadata
    if embed_model_name:
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        Settings.embed_model = embed_model

    # Load the index
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model if embed_model_name else None
        )
        return index, actual_chunk
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return None, None


def run_evaluation(
    pipeline: RAGPipeline,
    queries: List[Dict[str, str]],
    llm_name: str,
    chunking: str
) -> List[EvalResult]:
    """
    Run evaluation on queries.

    Args:
        pipeline: RAG pipeline instance.
        queries: List of query dictionaries.
        llm_name: Name of LLM backend.
        chunking: Chunking configuration name.

    Returns:
        List of EvalResult objects.
    """
    results = []

    for i, query_data in enumerate(queries):
        query_id = query_data.get('id', f'q{i+1}')
        question = query_data.get('question', '')

        if not question:
            logger.warning(f"Empty question for {query_id}, skipping")
            continue

        logger.info(f"[{i+1}/{len(queries)}] Processing: {question[:50]}...")

        try:
            # Run RAG query
            response = pipeline.query(question)

            # Convert to eval result
            result = EvalResult(
                query_id=query_id,
                question=question,
                llm=llm_name,
                chunking=chunking,
                answer=response.answer,
                cited_docs=response.cited_chunk_ids,
                response_length_chars=len(response.answer),
                response_length_words=len(response.answer.split()),
                retrieval_latency=response.retrieval_latency,
                generation_latency=response.generation_latency,
                total_latency=response.total_latency,
                num_retrieved_chunks=len(response.retrieved_chunks),
                notes=response.error
            )

            results.append(result)

            if response.error:
                logger.warning(f"Query {query_id} had error: {response.error}")

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            # Create error result
            results.append(EvalResult(
                query_id=query_id,
                question=question,
                llm=llm_name,
                chunking=chunking,
                answer="",
                cited_docs=[],
                error=str(e)
            ))

    return results


def main():
    """Main entry point for evaluation."""
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)

    # Set random seed
    set_seed(args.seed)

    # Generate run ID
    run_id = args.run_id or get_timestamp()

    logger.info("=" * 60)
    logger.info("RAG EVALUATION RUNNER")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"LLM Backend: {args.llm}")
    logger.info(f"Chunk Size: {args.chunk}")
    logger.info(f"Top-K: {args.top_k}")

    try:
        # Determine index directory
        if args.index_dir:
            index_dir = args.index_dir
        else:
            index_dir = f"results/runs/run_{args.chunk}/index"

        # Load or build index
        logger.info(f"Loading index from: {index_dir}")
        index, actual_chunk = load_index(index_dir, args.chunk)

        if index is None:
            logger.error("Could not load or build index. Please run build_index first.")
            logger.error(f"Expected index at: {index_dir}")
            logger.error("Or provide --index-dir with a valid path")
            return 1

        # Load queries
        logger.info(f"Loading queries from: {args.queries}")
        protocol = EvaluationProtocol(output_dir=args.output_dir or "results/runs")
        queries = protocol.load_queries(args.queries)
        logger.info(f"Loaded {len(queries)} queries")

        # Create LLM config
        llm_config = RAGConfig(
            llm_backend=args.llm,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_quantization=args.use_quantization
        )

        # Create pipeline
        logger.info(f"Creating RAG pipeline with {args.llm}...")
        pipeline = RAGPipeline(index, llm_config)

        # Create config name for results
        config_name = f"{args.llm}_chunk{args.chunk}"

        # Run evaluation
        logger.info("Starting evaluation...")
        query_list = [
            {"id": q.id, "question": q.question}
            for q in queries
        ]

        results = run_evaluation(
            pipeline,
            query_list,
            llm_name=args.llm,
            chunking=f"chunk{args.chunk}"
        )

        # Save results
        logger.info("Saving results...")
        result_filename = f"eval_{config_name}.csv"
        result_path = protocol.save_results(results, result_filename)

        # Create human evaluation template if requested
        if args.create_human_template:
            human_path = protocol.create_human_eval_template(results)
            logger.info(f"Human evaluation template saved to: {human_path}")

        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Configuration: {config_name}")
        logger.info(f"Queries processed: {len(results)}")
        logger.info(f"Results saved to: {result_path}")

        # Calculate and print quick stats
        total_latency = sum(r.total_latency for r in results)
        avg_latency = total_latency / len(results) if results else 0
        avg_length = sum(r.response_length_words for r in results) / len(results) if results else 0

        logger.info(f"Average latency: {avg_latency:.2f}s")
        logger.info(f"Average response length: {avg_length:.1f} words")

        return 0

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
