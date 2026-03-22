"""
Document Loaders Module
=======================
Handles loading and cleaning of documents from various formats (PDF, TXT).
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# LlamaIndex imports - handle version compatibility
try:
    from llama_index.core import Document
except ImportError:
    from llama_index import Document

# PDF Reader imports - handle version compatibility
try:
    from llama_index.readers.file import PDFReader
except ImportError:
    try:
        from llama_index.readers import PDFReader
    except ImportError:
        PDFReader = None

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of document loading operation."""
    documents: List[Document]
    failed_files: List[str]
    total_chunks_estimate: int


class DocumentLoader:
    """Loads documents from various file formats with cleaning."""

    def __init__(self, data_dir: str):
        """
        Initialize the document loader.

        Args:
            data_dir: Path to the data directory containing documents.
        """
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "corpus"

        if not self.corpus_dir.exists():
            logger.warning(f"Corpus directory not found: {self.corpus_dir}")
            self.corpus_dir.mkdir(parents=True, exist_ok=True)

    def load_all_documents(self, file_patterns: Optional[List[str]] = None) -> LoadResult:
        """
        Load all documents from the corpus directory.

        Args:
            file_patterns: List of glob patterns to match files.
                          Defaults to ["*.txt", "*.pdf"].

        Returns:
            LoadResult containing loaded documents and metadata.
        """
        if file_patterns is None:
            file_patterns = ["*.txt", "*.pdf"]

        documents = []
        failed_files = []
        total_chars = 0

        for pattern in file_patterns:
            for file_path in self.corpus_dir.glob(pattern):
                try:
                    if file_path.suffix.lower() == ".txt":
                        doc = self._load_txt(file_path)
                    elif file_path.suffix.lower() == ".pdf":
                        doc = self._load_pdf(file_path)
                    else:
                        logger.warning(f"Unsupported file type: {file_path}")
                        continue

                    if doc:
                        doc.metadata["source_file"] = str(file_path)
                        doc.metadata["file_type"] = file_path.suffix.lower()
                        documents.append(doc)
                        total_chars += len(doc.text)
                        logger.info(f"Loaded: {file_path.name} ({len(doc.text)} chars)")

                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    failed_files.append(str(file_path))

        # Estimate chunks (rough estimate: avg 500 chars per chunk)
        estimated_chunks = total_chars // 500

        result = LoadResult(
            documents=documents,
            failed_files=failed_files,
            total_chunks_estimate=estimated_chunks
        )

        logger.info(f"Loaded {len(documents)} documents, {len(failed_files)} failed, "
                   f"~{estimated_chunks} estimated chunks")

        return result

    def _load_txt(self, file_path: Path) -> Optional[Document]:
        """
        Load a text file with basic cleaning.

        Args:
            file_path: Path to the text file.

        Returns:
            LlamaIndex Document or None if loading fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned_text = self._clean_text(text)
            doc_id = file_path.stem

            return Document(
                text=cleaned_text,
                doc_id=doc_id,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path)
                }
            )
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            return None

    def _load_pdf(self, file_path: Path) -> Optional[Document]:
        """
        Load a PDF file using PDFReader.

        Args:
            file_path: Path to the PDF file.

        Returns:
            LlamaIndex Document or None if loading fails.
        """
        if PDFReader is None:
            logger.warning(f"PDFReader not available, skipping: {file_path}")
            return None

        try:
            reader = PDFReader()
            docs = reader.load_data(file_path=file_path)

            if not docs:
                logger.warning(f"No content extracted from PDF: {file_path}")
                return None

            # Combine all pages into a single document
            combined_text = "\n\n".join([doc.text for doc in docs])
            cleaned_text = self._clean_text(combined_text)

            doc_id = file_path.stem

            return Document(
                text=cleaned_text,
                doc_id=doc_id,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "num_pages": len(docs)
                }
            )
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing noise and normalizing whitespace.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Remove very short lines that are likely noise
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line) > 10 or line == '']
        text = '\n'.join(cleaned_lines)

        # Remove common PDF artifacts
        text = re.sub(r'-\s*\n\s*', '', text)  # Hyphenated line breaks

        # Final cleanup
        text = text.strip()

        return text

    def get_document_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of available documents without loading full content.

        Returns:
            List of document metadata dictionaries.
        """
        doc_list = []

        for pattern in ["*.txt", "*.pdf"]:
            for file_path in self.corpus_dir.glob(pattern):
                doc_list.append({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size
                })

        return doc_list


def load_documents(
    data_dir: str,
    file_patterns: Optional[List[str]] = None
) -> LoadResult:
    """
    Convenience function to load all documents.

    Args:
        data_dir: Path to the data directory.
        file_patterns: File patterns to match.

    Returns:
        LoadResult with loaded documents.
    """
    loader = DocumentLoader(data_dir)
    return loader.load_all_documents(file_patterns)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    result = load_documents("data")
    print(f"\nLoaded {len(result.documents)} documents")
    print(f"Failed: {len(result.failed_files)} files")
    print(f"Estimated chunks: {result.total_chunks_estimate}")
