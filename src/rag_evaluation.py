"""Evaluation framework for RAG using trec_eval metrics."""

import os
import json
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class QueryResult:
    """Query result with relevance judgments."""
    query_id: str
    query_text: str
    retrieved_docs: List[Tuple[str, float]]  # (doc_id, score)
    relevant_docs: List[str]  # List of relevant doc_ids
    qrels: Dict[str, int]  # doc_id -> relevance score (0-4)


@dataclass
class EvaluationMetrics:
    """IR evaluation metrics."""
    mrr: float  # Mean Reciprocal Rank
    ndcg_10: float  # nDCG@10
    ndcg_100: float  # nDCG@100
    map: float  # Mean Average Precision
    recall_10: float  # Recall@10
    recall_100: float  # Recall@100
    precision_10: float  # Precision@10
    precision_100: float  # Precision@100


class TRECEvaluator:
    """Wrapper for trec_eval evaluation."""
    
    def __init__(self, trec_eval_path: Optional[str] = None):
        """
        Initialize trec_eval evaluator.
        
        Args:
            trec_eval_path: Path to trec_eval binary. If None, tries to find it.
        """
        self.trec_eval_path = trec_eval_path or self._find_trec_eval()
        if not self.trec_eval_path:
            print("Warning: trec_eval not found. Will use manual computation.")
    
    def _find_trec_eval(self) -> Optional[str]:
        """Try to find trec_eval binary."""
        possible_paths = [
            "/usr/bin/trec_eval",
            "/usr/local/bin/trec_eval",
            "trec_eval",
            "./trec_eval",
        ]
        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, "-h"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        return None
    
    def compute_metrics_manual(
        self,
        query_results: List[QueryResult]
    ) -> EvaluationMetrics:
        """
        Compute IR metrics manually (without trec_eval).
        
        Args:
            query_results: List of query results with relevance judgments
            
        Returns:
            Evaluation metrics
        """
        mrr_scores = []
        map_scores = []
        ndcg_10_scores = []
        ndcg_100_scores = []
        recall_10_scores = []
        recall_100_scores = []
        precision_10_scores = []
        precision_100_scores = []
        
        for qr in query_results:
            # Sort retrieved docs by score
            sorted_docs = sorted(qr.retrieved_docs, key=lambda x: x[1], reverse=True)
            retrieved_ids = [doc_id for doc_id, _ in sorted_docs]
            
            # Get relevant docs
            relevant_set = set(qr.relevant_docs)
            num_relevant = len(relevant_set)
            
            if num_relevant == 0:
                continue
            
            # MRR (Mean Reciprocal Rank)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:100], 1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)
            
            # MAP (Mean Average Precision)
            ap = 0.0
            num_retrieved_relevant = 0
            for rank, doc_id in enumerate(retrieved_ids[:100], 1):
                if doc_id in relevant_set:
                    num_retrieved_relevant += 1
                    precision_at_rank = num_retrieved_relevant / rank
                    ap += precision_at_rank
            ap = ap / num_relevant if num_relevant > 0 else 0.0
            map_scores.append(ap)
            
            # nDCG@10 and nDCG@100
            def dcg_at_k(relevances: List[int], k: int) -> float:
                """Compute DCG at k."""
                relevances = relevances[:k]
                dcg = 0.0
                for i, rel in enumerate(relevances, 1):
                    dcg += (2**rel - 1) / np.log2(i + 1)
                return dcg
            
            def ndcg_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
                """Compute nDCG at k."""
                relevances = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids[:k]]
                dcg = dcg_at_k(relevances, k)
                
                # Ideal DCG
                ideal_relevances = [1] * min(num_relevant, k)
                ideal_dcg = dcg_at_k(ideal_relevances, k)
                
                return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            
            ndcg_10 = ndcg_at_k(retrieved_ids, relevant_set, 10)
            ndcg_100 = ndcg_at_k(retrieved_ids, relevant_set, 100)
            ndcg_10_scores.append(ndcg_10)
            ndcg_100_scores.append(ndcg_100)
            
            # Recall@10 and Recall@100
            retrieved_relevant_10 = len([d for d in retrieved_ids[:10] if d in relevant_set])
            retrieved_relevant_100 = len([d for d in retrieved_ids[:100] if d in relevant_set])
            
            recall_10 = retrieved_relevant_10 / num_relevant if num_relevant > 0 else 0.0
            recall_100 = retrieved_relevant_100 / num_relevant if num_relevant > 0 else 0.0
            recall_10_scores.append(recall_10)
            recall_100_scores.append(recall_100)
            
            # Precision@10 and Precision@100
            precision_10 = retrieved_relevant_10 / min(10, len(retrieved_ids)) if retrieved_ids else 0.0
            precision_100 = retrieved_relevant_100 / min(100, len(retrieved_ids)) if retrieved_ids else 0.0
            precision_10_scores.append(precision_10)
            precision_100_scores.append(precision_100)
        
        return EvaluationMetrics(
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            ndcg_10=np.mean(ndcg_10_scores) if ndcg_10_scores else 0.0,
            ndcg_100=np.mean(ndcg_100_scores) if ndcg_100_scores else 0.0,
            map=np.mean(map_scores) if map_scores else 0.0,
            recall_10=np.mean(recall_10_scores) if recall_10_scores else 0.0,
            recall_100=np.mean(recall_100_scores) if recall_100_scores else 0.0,
            precision_10=np.mean(precision_10_scores) if precision_10_scores else 0.0,
            precision_100=np.mean(precision_100_scores) if precision_100_scores else 0.0
        )
    
    def compute_metrics_trec_eval(
        self,
        query_results: List[QueryResult],
        output_file: Optional[str] = None
    ) -> Tuple[EvaluationMetrics, Dict[str, float]]:
        """
        Compute metrics using trec_eval.
        
        Args:
            query_results: List of query results
            output_file: Optional file to save detailed metrics
            
        Returns:
            Tuple of (EvaluationMetrics, detailed_metrics_dict)
        """
        if not self.trec_eval_path:
            print("trec_eval not available, using manual computation")
            return self.compute_metrics_manual(query_results), {}
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.qrels') as qrels_file:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as run_file:
                qrels_path = qrels_file.name
                run_path = run_file.name
                
                # Write qrels file
                for qr in query_results:
                    for doc_id, rel_score in qr.qrels.items():
                        qrels_file.write(f"{qr.query_id} 0 {doc_id} {rel_score}\n")
                
                # Write run file
                for qr in query_results:
                    for rank, (doc_id, score) in enumerate(qr.retrieved_docs[:1000], 1):
                        run_file.write(f"{qr.query_id} Q0 {doc_id} {rank} {score} run\n")
        
        try:
            # Run trec_eval
            metrics_to_compute = [
                'recip_rank',  # MRR
                'ndcg_cut_10',
                'ndcg_cut_100',
                'map',
                'recall_10',
                'recall_100',
                'P_10',
                'P_100'
            ]
            
            cmd = [self.trec_eval_path, '-m', 'all_trec', qrels_path, run_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"trec_eval error: {result.stderr}")
                return self.compute_metrics_manual(query_results), {}
            
            # Parse output
            detailed_metrics = {}
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        metric_name = parts[0]
                        query_id = parts[1]
                        value = float(parts[2])
                        detailed_metrics[f"{metric_name}_{query_id}"] = value
            
            # Extract average metrics
            avg_metrics = {}
            for metric in metrics_to_compute:
                values = [v for k, v in detailed_metrics.items() if k.startswith(metric)]
                if values:
                    avg_metrics[metric] = np.mean(values)
            
            # Map to our metrics
            eval_metrics = EvaluationMetrics(
                mrr=avg_metrics.get('recip_rank', 0.0),
                ndcg_10=avg_metrics.get('ndcg_cut_10', 0.0),
                ndcg_100=avg_metrics.get('ndcg_cut_100', 0.0),
                map=avg_metrics.get('map', 0.0),
                recall_10=avg_metrics.get('recall_10', 0.0),
                recall_100=avg_metrics.get('recall_100', 0.0),
                precision_10=avg_metrics.get('P_10', 0.0),
                precision_100=avg_metrics.get('P_100', 0.0)
            )
            
            # Save detailed metrics if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(detailed_metrics, f, indent=2)
            
            return eval_metrics, detailed_metrics
            
        finally:
            # Clean up temp files
            if os.path.exists(qrels_path):
                os.unlink(qrels_path)
            if os.path.exists(run_path):
                os.unlink(run_path)
    
    def evaluate(
        self,
        query_results: List[QueryResult],
        use_trec_eval: bool = True,
        output_file: Optional[str] = None
    ) -> Tuple[EvaluationMetrics, Dict[str, float]]:
        """
        Evaluate retrieval results.
        
        Args:
            query_results: List of query results with relevance judgments
            use_trec_eval: Whether to use trec_eval (if available)
            output_file: Optional file to save detailed metrics
            
        Returns:
            Tuple of (EvaluationMetrics, detailed_metrics_dict)
        """
        if use_trec_eval and self.trec_eval_path:
            return self.compute_metrics_trec_eval(query_results, output_file)
        else:
            metrics = self.compute_metrics_manual(query_results)
            return metrics, {}


def print_metrics(metrics: EvaluationMetrics, title: str = "Evaluation Results"):
    """Print evaluation metrics in a readable format."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"MRR (Mean Reciprocal Rank):     {metrics.mrr:.4f}")
    print(f"MAP (Mean Average Precision):  {metrics.map:.4f}")
    print(f"nDCG@10:                        {metrics.ndcg_10:.4f}")
    print(f"nDCG@100:                       {metrics.ndcg_100:.4f}")
    print(f"Recall@10:                      {metrics.recall_10:.4f}")
    print(f"Recall@100:                     {metrics.recall_100:.4f}")
    print(f"Precision@10:                   {metrics.precision_10:.4f}")
    print(f"Precision@100:                  {metrics.precision_100:.4f}")
    print("="*60 + "\n")


