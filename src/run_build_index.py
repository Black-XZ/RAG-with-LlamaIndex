"""
Build Index CLI
===============
Command-line interface for building vector indices.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import DocumentLoader
from src.indexing import ChunkingConfig, IndexBuilder, build_index_from_config
from src.utils import setup_logging, get_timestamp, ensure_dir

logger = logging.getLogger("RAG.BuildIndex")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build vector index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to chunking configuration YAML file"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing corpus"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID (defaults to timestamp)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip loading documents (use with existing document list)"
    )

    return parser.parse_args()


def main():
    """Main entry point for index building."""
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)

    logger.info("=" * 60)
    logger.info("RAG INDEX BUILDER")
    logger.info("=" * 60)

    # Generate run ID
    run_id = args.run_id or get_timestamp()

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = ChunkingConfig(args.config)

        logger.info(f"Chunking strategy: {config.strategy}")
        logger.info(f"Chunk size: {config.chunk_size}")
        logger.info(f"Overlap ratio: {config.overlap_ratio}")
        logger.info(f"Embedding model: {config.embedding_model}")
        logger.info(f"Index type: {config.index_type}")

        # Load documents
        if args.skip_load:
            logger.info("Skipping document loading (--skip-load specified)")
            documents = []
        else:
            logger.info("Loading documents...")
            loader = DocumentLoader(args.data_dir)
            result = loader.load_all_documents()

            logger.info(f"Loaded {len(result.documents)} documents")
            logger.info(f"Estimated chunks: {result.total_chunks_estimate}")

            if result.failed_files:
                logger.warning(f"Failed to load {len(result.failed_files)} files:")
                for f in result.failed_files:
                    logger.warning(f"  - {f}")

            if not result.documents:
                logger.error("No documents loaded! Exiting.")
                sys.exit(1)

            documents = result.documents

        # Build index
        logger.info("Building index...")
        builder = IndexBuilder(config, run_id)

        # Override output directory if specified
        if args.output_dir:
            builder.output_dir = Path(args.output_dir)
            builder.output_dir.mkdir(parents=True, exist_ok=True)

        index = builder.build_index(documents)

        logger.info("Index built successfully!")
        logger.info(f"Index saved to: {builder.output_dir}")

        # Save metadata
        metadata_path = builder.output_dir / "build_info.json"
        import json
        build_info = {
            "run_id": run_id,
            "timestamp": get_timestamp(),
            "config_file": args.config,
            "num_documents": len(documents),
            "chunk_size": config.chunk_size,
            "embedding_model": config.embedding_model,
            "index_output_dir": str(builder.output_dir)
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(build_info, f, indent=2)

        logger.info(f"Build info saved to: {metadata_path}")
        logger.info("=" * 60)
        logger.info("INDEX BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
