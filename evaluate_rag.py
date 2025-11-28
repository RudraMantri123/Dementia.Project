"""Evaluate the enhanced RAG system."""

import os
import json
from typing import List, Dict
from src.rag_enhanced import EnhancedKnowledgeBase
from src.rag_evaluation import TRECEvaluator, QueryResult, print_metrics


def create_test_queries() -> List[Dict[str, any]]:
    """Create test queries with relevance judgments."""
    # This is a simplified example - in practice, you'd have a proper test set
    return [
        {
            "query_id": "q1",
            "query": "What are the early signs of dementia?",
            "relevant_docs": ["doc1", "doc2"],  # Document IDs that are relevant
            "qrels": {"doc1": 2, "doc2": 1}  # Relevance scores (0-4)
        },
        {
            "query_id": "q2",
            "query": "How can I help someone with memory problems?",
            "relevant_docs": ["doc3", "doc4"],
            "qrels": {"doc3": 2, "doc4": 1}
        },
        {
            "query_id": "q3",
            "query": "What foods are good for brain health?",
            "relevant_docs": ["doc5"],
            "qrels": {"doc5": 2}
        }
    ]


def build_document_mapping(kb: EnhancedKnowledgeBase) -> Dict[str, int]:
    """
    Create mapping from document IDs to indices.
    
    In practice, you'd have proper document IDs. For now, we'll use indices.
    """
    mapping = {}
    for i, doc in enumerate(kb.documents):
        # Use source metadata or create ID
        doc_id = doc.metadata.get('source', f'doc{i}')
        mapping[doc_id] = i
    return mapping


def evaluate_rag_system(
    kb: EnhancedKnowledgeBase,
    test_queries: List[Dict],
    methods: List[str] = ['dense', 'bm25', 'hybrid']
):
    """Evaluate RAG system with different retrieval methods."""
    
    evaluator = TRECEvaluator()
    doc_mapping = build_document_mapping(kb)
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating {method.upper()} retrieval")
        print(f"{'='*60}\n")
        
        query_results = []
        
        for test_query in test_queries:
            # Retrieve documents
            retrieved_docs = kb.similarity_search(
                test_query['query'],
                k=100,
                method=method
            )
            
            # Get scores (simplified - in practice, get actual scores)
            if method == 'hybrid':
                hybrid_results = kb.hybrid_search(test_query['query'], k=100)
                retrieved_with_scores = [
                    (f"doc{i}", r.score)
                    for i, r in enumerate(hybrid_results)
                ]
            elif method == 'bm25' and kb.bm25:
                scores = kb.bm25.get_scores(test_query['query'])
                retrieved_with_scores = [
                    (f"doc{i}", score)
                    for i, score in enumerate(scores)
                ]
                # Sort and take top 100
                retrieved_with_scores = sorted(
                    retrieved_with_scores,
                    key=lambda x: x[1],
                    reverse=True
                )[:100]
            else:
                # For dense, use indices
                retrieved_with_scores = [
                    (f"doc{i}", 1.0 - (i / len(kb.documents)))
                    for i in range(min(100, len(retrieved_docs)))
                ]
            
            # Create QueryResult
            query_result = QueryResult(
                query_id=test_query['query_id'],
                query_text=test_query['query'],
                retrieved_docs=retrieved_with_scores,
                relevant_docs=test_query['relevant_docs'],
                qrels=test_query['qrels']
            )
            query_results.append(query_result)
        
        # Evaluate
        metrics, detailed = evaluator.evaluate(
            query_results,
            use_trec_eval=True,
            output_file=f"evaluation_{method}.json"
        )
        
        results[method] = {
            'metrics': metrics,
            'detailed': detailed
        }
        
        print_metrics(metrics, f"{method.upper()} Retrieval Results")
    
    return results


def main():
    """Main evaluation function."""
    print("="*60)
    print("Enhanced RAG System Evaluation")
    print("="*60)
    
    # Initialize knowledge base
    print("\n1. Initializing Enhanced Knowledge Base...")
    kb = EnhancedKnowledgeBase(
        query_encoder_model="sentence-transformers/ms-marco-MiniLM-L-6-v2",
        doc_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        use_bm25=True
    )
    
    # Load or build knowledge base
    vector_store_path = "data/vector_store_enhanced"
    
    if os.path.exists(vector_store_path) and os.path.exists(
        os.path.join(vector_store_path, "documents.pkl")
    ):
        print(f"Loading existing knowledge base from {vector_store_path}...")
        kb.load(vector_store_path)
    else:
        print("Building knowledge base from articles...")
        articles_dir = "data/articles"
        article_files = [
            os.path.join(articles_dir, f)
            for f in os.listdir(articles_dir)
            if f.endswith('.txt')
        ]
        
        kb.add_documents_from_files(article_files)
        kb.save(vector_store_path)
        print(f"Knowledge base saved to {vector_store_path}")
    
    print(f"Loaded {len(kb.documents)} document chunks")
    
    # Create test queries
    print("\n2. Creating test queries...")
    test_queries = create_test_queries()
    print(f"Created {len(test_queries)} test queries")
    
    # Note: In a real evaluation, you'd need proper relevance judgments
    # For now, this is a framework that can be used with proper test data
    
    print("\n3. Running evaluation...")
    print("Note: This is a demonstration. For real evaluation, you need:")
    print("  - Proper test queries with relevance judgments")
    print("  - Document IDs that match your corpus")
    print("  - QREL files in standard format")
    
    # Evaluate
    results = evaluate_rag_system(
        kb,
        test_queries,
        methods=['dense', 'bm25', 'hybrid']
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method, result in results.items():
        m = result['metrics']
        print(f"\n{method.upper()}:")
        print(f"  MRR:    {m.mrr:.4f}")
        print(f"  MAP:    {m.map:.4f}")
        print(f"  nDCG@10: {m.ndcg_10:.4f}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()


