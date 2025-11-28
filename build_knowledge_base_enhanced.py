"""Build the enhanced knowledge base with multiple encoders."""

import os
from src.rag_enhanced import EnhancedKnowledgeBase

def main():
    """Build and save the enhanced knowledge base."""
    print("="*80)
    print("Enhanced Knowledge Base Builder")
    print("="*80)
    
    print("\nInitializing enhanced knowledge base...")
    print("  - Query Encoder: ms-marco-MiniLM-L-6-v2 (optimized for queries)")
    print("  - Document Encoder: all-MiniLM-L6-v2 (optimized for documents)")
    print("  - BM25: Manual implementation with vector reconstruction")
    
    kb = EnhancedKnowledgeBase(
        query_encoder_model="sentence-transformers/ms-marco-MiniLM-L-6-v2",
        doc_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        use_bm25=True
    )

    # Get all article files
    articles_dir = "data/articles"
    if not os.path.exists(articles_dir):
        print(f"Error: Articles directory not found: {articles_dir}")
        return
    
    article_files = [
        os.path.join(articles_dir, f)
        for f in os.listdir(articles_dir)
        if f.endswith('.txt')
    ]

    print(f"\nFound {len(article_files)} articles")

    # Add documents to knowledge base
    print("\nLoading and processing articles...")
    print("This will:")
    print("  1. Split documents into chunks")
    print("  2. Encode with query encoder")
    print("  3. Encode with document encoder")
    print("  4. Build BM25 index")
    
    kb.add_documents_from_files(article_files)

    # Save vector store
    vector_store_path = "data/vector_store_enhanced"
    print(f"\nSaving enhanced vector store to {vector_store_path}...")
    kb.save(vector_store_path)

    print("\n" + "="*80)
    print("Enhanced Knowledge Base Built Successfully!")
    print("="*80)
    print(f"\nTotal document chunks: {len(kb.documents)}")
    if kb.query_vector_store:
        try:
            query_count = kb.query_vector_store.index.ntotal
        except:
            query_count = len(kb.documents)
        print(f"Query vector store: {query_count} vectors")
    else:
        print(f"Query vector store: 0 vectors")
    
    if kb.doc_vector_store:
        try:
            doc_count = kb.doc_vector_store.index.ntotal
        except:
            doc_count = len(kb.documents)
        print(f"Document vector store: {doc_count} vectors")
    else:
        print(f"Document vector store: 0 vectors")
    
    print(f"BM25 index: {'Ready' if kb.bm25 else 'Not available'}")

    # Test retrieval with different methods
    print("\n" + "="*80)
    print("Testing Retrieval Methods")
    print("="*80)

    test_queries = [
        "What are the early signs of dementia?",
        "How can I help someone with memory problems?",
        "What foods are good for brain health?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Test dense retrieval
        dense_results = kb.similarity_search(query, k=2, method='dense')
        print(f"Dense (Query Encoder): {len(dense_results)} results")
        if dense_results:
            print(f"  Top result: {dense_results[0].page_content[:100]}...")
        
        # Test BM25 retrieval
        bm25_results = kb.similarity_search(query, k=2, method='bm25')
        print(f"BM25 (Sparse): {len(bm25_results)} results")
        if bm25_results:
            print(f"  Top result: {bm25_results[0].page_content[:100]}...")
        
        # Test hybrid retrieval
        hybrid_results = kb.hybrid_search(query, k=2)
        print(f"Hybrid (Dense + BM25): {len(hybrid_results)} results")
        if hybrid_results:
            print(f"  Top result (score: {hybrid_results[0].score:.4f}): {hybrid_results[0].document.page_content[:100]}...")
        
        # Test BM25 manual computation
        if kb.bm25:
            print("\nBM25 Manual Computation:")
            doc_idx = 0
            doc_vector = kb.bm25.get_doc_vector(doc_idx)
            query_vector = kb.bm25.get_query_vector(query)
            dot_product = kb.bm25.compute_dot_product(doc_idx, query)
            print(f"  Document vector size: {len(doc_vector)} terms")
            print(f"  Query vector size: {len(query_vector)} terms")
            print(f"  Dot product (BM25 score): {dot_product:.4f}")

    print("\n" + "="*80)
    print("Enhanced Knowledge Base is ready!")
    print("="*80)
    print("\nYou can now use this knowledge base with:")
    print("  - Dense retrieval (query encoder)")
    print("  - Sparse retrieval (BM25)")
    print("  - Hybrid retrieval (combining both)")
    print("\nTo evaluate the system, run: python evaluate_rag.py")

if __name__ == "__main__":
    main()

