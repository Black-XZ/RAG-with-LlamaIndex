"""
Indexing Module
===============
Handles text chunking, embedding generation, and index construction.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

import yaml

# LlamaIndex imports - handle version compatibility
try:
    from llama_index.core import Document, Settings
    from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
    from llama_index.core.schema import BaseNode
    from llama_index.core.indices import VectorStoreIndex
except ImportError:
    from llama_index import Document, Settings
    from llama_index.node_parser import SentenceSplitter, TokenTextSplitter
    from llama_index.schema import BaseNode
    from llama_index import VectorStoreIndex

# FAISS imports - optional
try:
    from llama_index.core.storage import StorageContext
    from llama_index.vector_stores.faiss import FaissVectorStore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    StorageContext = None
    FaissVectorStore = None

# Embedding imports
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    try:
        from llama_index.embeddings import HuggingFaceEmbedding
    except ImportError:
        HuggingFaceEmbedding = None

# FAISS direct import
try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

logger = logging.getLogger(__name__)


class ChunkingConfig:
    """Configuration for text chunking."""

    def __init__(self, config_path: str):
        """
        Load chunking configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.chunk_size = config['chunking']['chunk_size']
        self.overlap_ratio = config['chunking']['overlap_ratio']
        self.overlap_tokens = config['chunking'].get('overlap_tokens', int(self.chunk_size * self.overlap_ratio))
        self.strategy = config['chunking']['strategy']

        self.embedding_model = config['embedding']['model_name']
        self.embedding_dim = config['embedding']['dimension']
        self.embedding_device = config['embedding']['device']

        self.index_type = config['index']['type']

        self.metadata = config.get('metadata', {})

        logger.info(f"Loaded chunking config: size={self.chunk_size}, "
                   f"overlap={self.overlap_ratio}, strategy={self.strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'chunk_size': self.chunk_size,
            'overlap_ratio': self.overlap_ratio,
            'overlap_tokens': self.overlap_tokens,
            'strategy': self.strategy,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type
        }


class IndexBuilder:
    """Builds vector indices from documents."""

    def __init__(self, config: ChunkingConfig, run_id: Optional[str] = None):
        """
        Initialize the index builder.

        Args:
            config: Chunking configuration.
            run_id: Unique identifier for this build run.
        """
        self.config = config

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.output_dir = Path(f"results/runs/{run_id}/index")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self._setup_embed_model()

        # Initialize node parser
        self._setup_node_parser()

        logger.info(f"IndexBuilder initialized for run: {run_id}")

    def _setup_embed_model(self):
        """Set up the embedding model."""
        if HuggingFaceEmbedding is None:
            raise ImportError(
                "HuggingFaceEmbedding not available. "
                "Install with: pip install llama-index-embeddings-huggingface"
            )

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            embed_batch_size=32
        )
        Settings.embed_model = self.embed_model
        logger.info(f"Embedding model loaded: {self.config.embedding_model}")

    def _setup_node_parser(self):
        """Set up the node parser based on chunking strategy."""
        if self.config.strategy == "token":
            self.node_parser = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.overlap_tokens,
                separator=" "
            )
        else:  # Default to sentence-based
            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.overlap_tokens,
                separator="."
            )
        logger.info(f"Node parser configured: {self.config.strategy}")

    def build_index(
        self,
        documents: List[Document],
        save_index: bool = True
    ) -> VectorStoreIndex:
        """
        Build a vector index from documents.

        Args:
            documents: List of LlamaIndex Documents.
            save_index: Whether to save the index to disk.

        Returns:
            VectorStoreIndex object.
        """
        logger.info(f"Building index with {len(documents)} documents...")

        # Parse documents into nodes
        nodes = self._parse_documents(documents)
        logger.info(f"Created {len(nodes)} nodes from documents")

        # Build index
        if self.config.index_type == "faiss" and FAISS_AVAILABLE:
            index = self._build_faiss_index(nodes)
        else:
            index = self._build_default_index(nodes)

        # Save index and metadata
        if save_index:
            self._save_index_and_metadata(index, nodes)

        return index

    def _parse_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Parse documents into nodes using the configured chunking strategy.

        Args:
            documents: List of documents to parse.

        Returns:
            List of parsed nodes.
        """
        all_nodes = []

        for doc in documents:
            try:
                nodes = self.node_parser.get_nodes_from_documents([doc])

                # Add chunk metadata
                for i, node in enumerate(nodes):
                    node.metadata["chunk_id"] = f"{doc.doc_id}_chunk_{i}"
                    node.metadata["chunk_index"] = i
                    node.metadata["total_chunks"] = len(nodes)
                    node.metadata["source_file"] = doc.metadata.get("file_name", "unknown")

                all_nodes.extend(nodes)

            except Exception as e:
                logger.error(f"Error parsing document {doc.doc_id}: {e}")

        return all_nodes

    def _build_default_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Build index using default vector store."""
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embed_model,
            show_progress=True
        )
        return index

    def _build_faiss_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Build index using FAISS vector store."""
        if not FAISS_AVAILABLE or faiss is None:
            logger.warning("FAISS not available, falling back to default index")
            return self._build_default_index(nodes)

        try:
            # Create FAISS index
            d = self.config.embedding_dim
            faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity

            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )

            return index
        except Exception as e:
            logger.warning(f"FAISS index build failed: {e}, falling back to default")
            return self._build_default_index(nodes)

    def _save_index_and_metadata(
        self,
        index: VectorStoreIndex,
        nodes: List[BaseNode]
    ):
        """
        Save index and metadata to disk.

        Args:
            index: The built vector index.
            nodes: List of nodes in the index.
        """
        logger.info(f"Saving index to {self.output_dir}")

        # Save index
        index.storage_context.persist(persist_dir=str(self.output_dir))

        # Save metadata
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "num_documents": len(set(n.metadata.get("source_file", "unknown") for n in nodes)),
            "num_nodes": len(nodes),
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "overlap_ratio": self.config.overlap_ratio
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Save node info (for evaluation traceability)
        node_info = []
        for node in nodes:
            node_info.append({
                "chunk_id": node.metadata.get("chunk_id", node.node_id),
                "node_id": node.node_id,
                "source_file": node.metadata.get("source_file", "unknown"),
                "chunk_index": node.metadata.get("chunk_index", 0),
                "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
            })

        node_info_path = self.output_dir / "node_info.json"
        with open(node_info_path, 'w', encoding='utf-8') as f:
            json.dump(node_info, f, indent=2)

        logger.info(f"Index and metadata saved successfully")

    def load_index(self) -> VectorStoreIndex:
        """
        Load a previously saved index.

        Returns:
            Loaded VectorStoreIndex.
        """
        index_path = self.output_dir

        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}")

        if FAISS_AVAILABLE:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(index_path),
                    vector_store=FaissVectorStore.from_persist_dir(str(index_path))
                )
            except Exception:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(index_path)
                )
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(index_path)
            )

        index = VectorStoreIndex.from_existing_index(
            storage_context=storage_context,
            embed_model=self.embed_model
        )

        logger.info(f"Loaded index from {index_path}")
        return index


def build_index_from_config(
    config_path: str,
    documents: List[Document],
    run_id: Optional[str] = None
) -> tuple[VectorStoreIndex, ChunkingConfig]:
    """
    Convenience function to build index from config file.

    Args:
        config_path: Path to YAML configuration.
        documents: Documents to index.
        run_id: Optional run identifier.

    Returns:
        Tuple of (index, config).
    """
    config = ChunkingConfig(config_path)
    builder = IndexBuilder(config, run_id)
    index = builder.build_index(documents)
    return index, config


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from loaders import load_documents

    logging.basicConfig(level=logging.INFO)

    # Load documents
    result = load_documents("data")
    print(f"Loaded {len(result.documents)} documents")

    # Build index
    config_path = "configs/chunking_256.yaml"
    if Path(config_path).exists():
        index, config = build_index_from_config(config_path, result.documents)
        print(f"Index built with {config.chunk_size} chunk size")
