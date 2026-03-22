"""
Metrics Module
==============
Statistical analysis and metric calculations for RAG evaluation.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    count: int


class MetricsCalculator:
    """Calculate and analyze metrics for RAG evaluation."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def calculate_summary(
        self,
        values: List[float]
    ) -> MetricSummary:
        """
        Calculate summary statistics for a list of values.

        Args:
            values: List of numeric values.

        Returns:
            MetricSummary with statistics.
        """
        if not values:
            return MetricSummary(
                mean=0.0, std=0.0, min_val=0.0,
                max_val=0.0, median=0.0, count=0
            )

        arr = np.array(values)

        return MetricSummary(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            median=float(np.median(arr)),
            count=len(arr)
        )

    def calculate_latency_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, MetricSummary]:
        """
        Calculate latency metrics from results.

        Args:
            results: List of result dictionaries.

        Returns:
            Dictionary mapping metric names to summaries.
        """
        retrieval_times = [r.get('retrieval_latency', 0) for r in results]
        generation_times = [r.get('generation_latency', 0) for r in results]
        total_times = [r.get('total_latency', 0) for r in results]

        return {
            'retrieval': self.calculate_summary(retrieval_times),
            'generation': self.calculate_summary(generation_times),
            'total': self.calculate_summary(total_times)
        }

    def calculate_response_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, MetricSummary]:
        """
        Calculate response length metrics.

        Args:
            results: List of result dictionaries.

        Returns:
            Dictionary mapping metric names to summaries.
        """
        char_lengths = [r.get('response_length_chars', 0) for r in results]
        word_lengths = [r.get('response_length_words', 0) for r in results]

        return {
            'char_length': self.calculate_summary(char_lengths),
            'word_length': self.calculate_summary(word_lengths)
        }

    def calculate_relevance_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate relevance-related metrics.

        Args:
            results: List of result dictionaries.

        Returns:
            Dictionary with relevance statistics.
        """
        relevance_scores = [r.get('relevance_score') for r in results
                          if r.get('relevance_score') is not None]

        if not relevance_scores:
            return {
                'mean_relevance': 0.0,
                'relevance_distribution': {},
                'num_scored': 0
            }

        return {
            'mean_relevance': float(np.mean(relevance_scores)),
            'std_relevance': float(np.std(relevance_scores)),
            'median_relevance': float(np.median(relevance_scores)),
            'relevance_distribution': self._calculate_distribution(relevance_scores),
            'num_scored': len(relevance_scores)
        }

    def calculate_completion_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate task completion metrics.

        Args:
            results: List of result dictionaries.

        Returns:
            Dictionary with completion statistics.
        """
        completion_scores = [r.get('task_completion') for r in results
                           if r.get('task_completion') is not None]

        if not completion_scores:
            return {
                'completion_rate': 0.0,
                'num_scored': 0
            }

        return {
            'completion_rate': float(np.mean(completion_scores)),
            'num_completed': int(sum(completion_scores)),
            'num_scored': len(completion_scores)
        }

    def _calculate_distribution(
        self,
        values: List[float],
        bins: Optional[List[int]] = None
    ) -> Dict[str, int]:
        """
        Calculate frequency distribution.

        Args:
            values: List of values.
            bins: Optional custom bins for integer distributions.

        Returns:
            Dictionary mapping values to counts.
        """
        if not values:
            return {}

        if bins is None and all(isinstance(v, int) for v in values):
            # Integer distribution
            bins = sorted(set(values))
            return {str(b): values.count(b) for b in bins}
        else:
            # Create histogram bins
            arr = np.array(values)
            hist, _ = np.histogram(arr, bins=5)
            bin_edges = np.histogram(arr, bins=5)[1]

            distribution = {}
            for i in range(len(hist)):
                key = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
                distribution[key] = int(hist[i])

            return distribution

    def generate_full_report(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report.

        Args:
            results: List of result dictionaries.

        Returns:
            Dictionary containing all metrics.
        """
        report = {
            'num_queries': len(results),
            'latency_metrics': self.calculate_latency_metrics(results),
            'response_metrics': self.calculate_response_metrics(results),
            'relevance_metrics': self.calculate_relevance_metrics(results),
            'completion_metrics': self.calculate_completion_metrics(results)
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """
        Print a formatted metrics report.

        Args:
            report: Report dictionary from generate_full_report.
        """
        print("\n" + "=" * 60)
        print("RAG EVALUATION METRICS REPORT")
        print("=" * 60)

        print(f"\nTotal Queries: {report['num_queries']}")

        # Latency metrics
        print("\n--- LATENCY METRICS ---")
        latency = report['latency_metrics']
        for name, metrics in latency.items():
            print(f"\n{name.upper()}:")
            print(f"  Mean: {metrics.mean:.3f}s (std: {metrics.std:.3f}s)")
            print(f"  Min:  {metrics.min_val:.3f}s | Max: {metrics.max_val:.3f}s")
            print(f"  Median: {metrics.median:.3f}s")

        # Response metrics
        print("\n--- RESPONSE LENGTH METRICS ---")
        response = report['response_metrics']
        for name, metrics in response.items():
            print(f"\n{name.upper()}:")
            print(f"  Mean: {metrics.mean:.1f} (std: {metrics.std:.1f})")
            print(f"  Min: {metrics.min_val:.0f} | Max: {metrics.max_val:.0f}")

        # Relevance metrics
        print("\n--- RELEVANCE METRICS ---")
        rel = report['relevance_metrics']
        if rel['num_scored'] > 0:
            print(f"Mean Relevance: {rel['mean_relevance']:.2f}/5.0 (std: {rel['std_relevance']:.2f})")
            print(f"Median Relevance: {rel['median_relevance']:.2f}")
            print(f"Scored Queries: {rel['num_scored']}")
            print(f"Distribution: {rel['relevance_distribution']}")
        else:
            print("No relevance scores available (requires human evaluation)")

        # Completion metrics
        print("\n--- TASK COMPLETION METRICS ---")
        comp = report['completion_metrics']
        if comp['num_scored'] > 0:
            print(f"Completion Rate: {comp['completion_rate']:.2%}")
            print(f"Completed: {comp['num_completed']}/{comp['num_scored']}")
        else:
            print("No completion scores available")

        print("\n" + "=" * 60)


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to calculate all statistics.

    Args:
        results: List of result dictionaries.

    Returns:
        Dictionary with statistics.
    """
    calculator = MetricsCalculator()
    return calculator.generate_full_report(results)


if __name__ == "__main__":
    # Example usage
    sample_results = [
        {
            'query_id': 'q001',
            'response_length_chars': 150,
            'response_length_words': 25,
            'retrieval_latency': 0.05,
            'generation_latency': 0.8,
            'total_latency': 0.85,
            'relevance_score': 4,
            'task_completion': 1
        },
        {
            'query_id': 'q002',
            'response_length_chars': 200,
            'response_length_words': 35,
            'retrieval_latency': 0.06,
            'generation_latency': 1.0,
            'total_latency': 1.06,
            'relevance_score': 3,
            'task_completion': 1
        },
        {
            'query_id': 'q003',
            'response_length_chars': 100,
            'response_length_words': 18,
            'retrieval_latency': 0.04,
            'generation_latency': 0.7,
            'total_latency': 0.74,
            'relevance_score': 5,
            'task_completion': 1
        }
    ]

    calculator = MetricsCalculator()
    report = calculator.generate_full_report(sample_results)
    calculator.print_report(report)
