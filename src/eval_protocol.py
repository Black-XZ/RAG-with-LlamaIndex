"""
Evaluation Protocol Module
==========================
Handles evaluation of RAG systems including human evaluation templates.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A query for evaluation."""
    id: str
    question: str
    doc_hint: Optional[str] = None


@dataclass
class EvalResult:
    """Result of evaluating a single query."""
    query_id: str
    question: str
    llm: str
    chunking: str
    answer: str
    cited_docs: List[str]
    relevance_score: Optional[int] = None  # 1-5
    task_completion: Optional[int] = None  # 0/1
    response_length_chars: int = 0
    response_length_words: int = 0
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    num_retrieved_chunks: int = 0
    notes: Optional[str] = None


class EvaluationProtocol:
    """
    Evaluation protocol for RAG systems.

    Supports automated metrics and provides templates for human evaluation.
    """

    def __init__(self, output_dir: str = "results/runs"):
        """
        Initialize evaluation protocol.

        Args:
            output_dir: Directory for saving evaluation results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluation run initialized: {self.run_timestamp}")

    def load_queries(self, query_file: str) -> List[EvalQuery]:
        """
        Load queries from JSONL file.

        Args:
            query_file: Path to queries.jsonl file.

        Returns:
            List of EvalQuery objects.
        """
        queries = []

        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    queries.append(EvalQuery(
                        id=data['id'],
                        question=data['question'],
                        doc_hint=data.get('doc_hint')
                    ))

        logger.info(f"Loaded {len(queries)} queries from {query_file}")
        return queries

    def evaluate_response(
        self,
        response_dict: Dict[str, Any],
        gold_answer: Optional[str] = None
    ) -> EvalResult:
        """
        Create an EvalResult from a RAG response.

        Args:
            response_dict: Dictionary containing response data.
            gold_answer: Optional gold standard answer for comparison.

        Returns:
            EvalResult object.
        """
        # Calculate response length
        answer = response_dict.get('answer', '')
        length_chars = len(answer)
        length_words = len(answer.split())

        result = EvalResult(
            query_id=response_dict.get('query_id', 'unknown'),
            question=response_dict.get('query', ''),
            llm=response_dict.get('model_name', 'unknown'),
            chunking=response_dict.get('chunking', 'unknown'),
            answer=answer,
            cited_docs=response_dict.get('cited_chunk_ids', []),
            response_length_chars=length_chars,
            response_length_words=length_words,
            retrieval_latency=response_dict.get('retrieval_latency', 0.0),
            generation_latency=response_dict.get('generation_latency', 0.0),
            total_latency=response_dict.get('total_latency', 0.0),
            num_retrieved_chunks=response_dict.get('num_retrieved_chunks', 0)
        )

        # If gold answer provided, do weak supervision for task completion
        if gold_answer:
            result.task_completion = self._weak_supervision_check(answer, gold_answer)

        return result

    def _weak_supervision_check(self, answer: str, gold_answer: str) -> int:
        """
        Weak supervision check for task completion.

        Looks for keyword overlap between answer and gold answer.

        Args:
            answer: Generated answer.
            gold_answer: Reference answer.

        Returns:
            1 if there's significant overlap, 0 otherwise.
        """
        # Simple keyword-based check
        answer_lower = answer.lower()
        gold_lower = gold_answer.lower()

        # Extract key terms (words > 4 chars)
        answer_words = set(w for w in answer_lower.split() if len(w) > 4)
        gold_words = set(w for w in gold_lower.split() if len(w) > 4)

        # Calculate overlap
        if gold_words:
            overlap = len(answer_words & gold_words) / len(gold_words)
            return 1 if overlap > 0.3 else 0

        return 0

    def save_results(self, results: List[EvalResult], filename: str):
        """
        Save evaluation results to CSV.

        Args:
            results: List of EvalResult objects.
            filename: Output filename.
        """
        output_path = self.run_dir / filename

        data = []
        for r in results:
            data.append({
                'query_id': r.query_id,
                'question': r.question,
                'llm': r.llm,
                'chunking': r.chunking,
                'answer': r.answer,
                'cited_docs': ', '.join(r.cited_docs),
                'relevance_score': r.relevance_score,
                'task_completion': r.task_completion,
                'response_length_chars': r.response_length_chars,
                'response_length_words': r.response_length_words,
                'retrieval_latency': r.retrieval_latency,
                'generation_latency': r.generation_latency,
                'total_latency': r.total_latency,
                'num_retrieved_chunks': r.num_retrieved_chunks,
                'notes': r.notes
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Results saved to {output_path}")
        return output_path

    def create_human_eval_template(
        self,
        results: List[EvalResult],
        filename: str = "human_eval_template.csv"
    ) -> Path:
        """
        Create a template for human evaluation.

        Args:
            results: Evaluation results to evaluate.
            filename: Output filename.

        Returns:
            Path to the created template file.
        """
        output_path = self.run_dir / filename

        data = []
        for r in results:
            data.append({
                'query_id': r.query_id,
                'question': r.question,
                'llm': r.llm,
                'chunking': r.chunking,
                'answer': r.answer,
                'cited_docs': ', '.join(r.cited_docs),
                'relevance(1-5)': '',  # To be filled by human
                'completion(0/1)': '',  # To be filled by human
                'notes': ''  # To be filled by human
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Human evaluation template saved to {output_path}")
        return output_path

    def generate_comparison_table(
        self,
        all_results: Dict[str, List[EvalResult]]
    ) -> pd.DataFrame:
        """
        Generate a comparison table across different configurations.

        Args:
            all_results: Dictionary mapping config names to result lists.
                        e.g., {"mistral7b_256": [...], "flant5_256": [...]}

        Returns:
            DataFrame with comparison statistics.
        """
        rows = []

        for config_name, results in all_results.items():
            # Calculate statistics
            avg_relevance = sum(r.relevance_score or 0 for r in results) / len(results)
            task_completion_rate = sum(r.task_completion or 0 for r in results) / len(results)
            avg_length = sum(r.response_length_words for r in results) / len(results)
            avg_latency = sum(r.total_latency for r in results) / len(results)
            avg_retrieval_latency = sum(r.retrieval_latency for r in results) / len(results)
            avg_generation_latency = sum(r.generation_latency for r in results) / len(results)

            rows.append({
                'Configuration': config_name,
                'Num Queries': len(results),
                'Avg Relevance (1-5)': round(avg_relevance, 2),
                'Task Completion Rate': round(task_completion_rate, 2),
                'Avg Response Length (words)': round(avg_length, 1),
                'Avg Total Latency (s)': round(avg_latency, 2),
                'Avg Retrieval Latency (s)': round(avg_retrieval_latency, 3),
                'Avg Generation Latency (s)': round(avg_generation_latency, 2)
            })

        df = pd.DataFrame(rows)
        return df

    def save_comparison_table(
        self,
        all_results: Dict[str, List[EvalResult]],
        filename: str = "comparison_table.csv"
    ) -> Path:
        """
        Generate and save comparison table.

        Args:
            all_results: Dictionary of results by configuration.
            filename: Output filename.

        Returns:
            Path to saved table.
        """
        df = self.generate_comparison_table(all_results)
        output_path = self.run_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Comparison table saved to {output_path}")
        return output_path


def load_evaluation_queries(query_file: str) -> List[EvalQuery]:
    """
    Convenience function to load evaluation queries.

    Args:
        query_file: Path to queries.jsonl.

    Returns:
        List of EvalQuery objects.
    """
    protocol = EvaluationProtocol()
    return protocol.load_queries(query_file)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test loading queries
    if Path("data/queries.jsonl").exists():
        queries = load_evaluation_queries("data/queries.jsonl")
        print(f"Loaded {len(queries)} queries:")
        for q in queries[:3]:
            print(f"  [{q.id}] {q.question}")
