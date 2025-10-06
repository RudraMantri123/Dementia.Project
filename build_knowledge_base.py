"""Build the knowledge base from articles."""

import os
from src.knowledge_base import KnowledgeBase

def main():
    """Build and save the knowledge base."""
    print("Initializing knowledge base...")
    kb = KnowledgeBase()

    # Get all article files
    articles_dir = "data/articles"
    article_files = [
        os.path.join(articles_dir, f)
        for f in os.listdir(articles_dir)
        if f.endswith('.txt')
    ]

    print(f"Found {len(article_files)} articles")

    # Add documents to knowledge base
    print("Loading and processing articles...")
    kb.add_documents_from_files(article_files)

    # Save vector store
    vector_store_path = "data/vector_store"
    print(f"Saving vector store to {vector_store_path}...")
    kb.save(vector_store_path)

    print("Knowledge base built successfully!")

    # Test retrieval
    print("\n" + "="*80)
    print("Testing retrieval...")
    print("="*80)

    test_queries = [
        "What are the early signs of dementia?",
        "How can I help someone with memory problems?",
        "What foods are good for brain health?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = kb.similarity_search(query, k=2)
        print(f"Found {len(results)} relevant chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\n  Chunk {i}:")
            print(f"  {doc.page_content[:200]}...")

    print("\n" + "="*80)
    print("Knowledge base is ready!")
    print("="*80)

if __name__ == "__main__":
    main()
