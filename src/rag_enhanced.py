"""Enhanced RAG system with multiple encoders, BM25, and evaluation."""

import os
import json
import pickle
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict, Counter
import math
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# For BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not installed. BM25 will use manual implementation.")


@dataclass
class RetrievalResult:
    """Result from retrieval with scores."""
    document: Document
    score: float
    method: str  # 'dense', 'sparse', 'bm25', 'hybrid'


class BM25Manual:
    """Manual BM25 implementation with document vector reconstruction."""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with manual computation.
        
        Args:
            corpus: List of document texts
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        self.doc_len = []
        self.vocab = set()
        
        # Tokenize and build vocabulary
        self.tokenized_corpus = []
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.tokenized_corpus.append(tokens)
            self.vocab.update(tokens)
            self.doc_len.append(len(tokens))
        
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Build document frequency vectors
        self._build_doc_freqs()
        self._build_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _build_doc_freqs(self):
        """Build document frequency vectors for each document."""
        self.doc_freqs = []
        for doc_tokens in self.tokenized_corpus:
            doc_freq = Counter(doc_tokens)
            self.doc_freqs.append(doc_freq)
    
    def _build_idf(self):
        """Build inverse document frequency dictionary."""
        num_docs = len(self.tokenized_corpus)
        for term in self.vocab:
            df = sum(1 for doc_freq in self.doc_freqs if term in doc_freq)
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    def get_doc_vector(self, doc_idx: int) -> Dict[str, float]:
        """
        Reconstruct BM25 document vector.
        
        Args:
            doc_idx: Index of document in corpus
            
        Returns:
            Dictionary mapping terms to their BM25 weights in this document
        """
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        doc_vector = {}
        
        for term, tf in doc_freq.items():
            # BM25 term weight = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avgdl)))
            idf = self.idf.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            doc_vector[term] = idf * (numerator / denominator)
        
        return doc_vector
    
    def get_query_vector(self, query: str) -> Dict[str, float]:
        """
        Reconstruct BM25 query vector.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary mapping terms to their IDF values
        """
        query_tokens = self._tokenize(query)
        query_vector = {}
        for term in query_tokens:
            if term in self.idf:
                query_vector[term] = self.idf[term]
        return query_vector
    
    def compute_dot_product(self, doc_idx: int, query: str) -> float:
        """
        Compute BM25 score by manual dot product of document and query vectors.
        
        Args:
            doc_idx: Index of document
            query: Query text
            
        Returns:
            BM25 score
        """
        doc_vector = self.get_doc_vector(doc_idx)
        query_vector = self.get_query_vector(query)
        
        # Dot product: sum of (doc_weight * query_weight) for matching terms
        score = 0.0
        for term, query_weight in query_vector.items():
            if term in doc_vector:
                score += doc_vector[term] * query_weight
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Query text
            
        Returns:
            List of scores for each document
        """
        scores = []
        for i in range(len(self.corpus)):
            score = self.compute_dot_product(i, query)
            scores.append(score)
        return scores


class EnhancedKnowledgeBase:
    """Enhanced knowledge base with multiple encoders and BM25."""
    
    def __init__(
        self,
        query_encoder_model: str = "sentence-transformers/ms-marco-MiniLM-L-6-v2",
        doc_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        splade_model: Optional[str] = None,
        use_bm25: bool = True
    ):
        """
        Initialize enhanced knowledge base.
        
        Args:
            query_encoder_model: Model for query encoding
            doc_encoder_model: Model for document encoding
            splade_model: Optional SPLADE model for sparse retrieval
            use_bm25: Whether to use BM25
        """
        # Query encoder (optimized for queries)
        print(f"Loading query encoder: {query_encoder_model}")
        self.query_encoder = SentenceTransformer(query_encoder_model)
        
        # Document encoder (optimized for documents)
        print(f"Loading document encoder: {doc_encoder_model}")
        self.doc_encoder = SentenceTransformer(doc_encoder_model)
        
        # SPLADE encoder (sparse retrieval)
        self.splade_encoder = None
        if splade_model:
            try:
                print(f"Loading SPLADE encoder: {splade_model}")
                self.splade_tokenizer = AutoTokenizer.from_pretrained(splade_model)
                self.splade_model = AutoModel.from_pretrained(splade_model)
                self.splade_model.eval()
            except Exception as e:
                print(f"Warning: Could not load SPLADE model: {e}")
                self.splade_encoder = None
        
        # Vector stores
        self.query_vector_store: Optional[FAISS] = None
        self.doc_vector_store: Optional[FAISS] = None
        
        # BM25
        self.bm25: Optional[BM25Manual] = None
        self.use_bm25 = use_bm25
        
        # Document storage
        self.documents: List[Document] = []
        self.doc_texts: List[str] = []
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Add documents to the knowledge base."""
        all_chunks = []
        all_metadatas = []
        
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
            
            if metadatas and i < len(metadatas):
                all_metadatas.extend([metadatas[i]] * len(chunks))
            else:
                all_metadatas.extend([{}] * len(chunks))
        
        # Create Document objects
        docs = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(all_chunks, all_metadatas)
        ]
        
        self.documents.extend(docs)
        self.doc_texts.extend([doc.page_content for doc in docs])
        
        # Encode documents
        print("Encoding documents with query encoder...")
        query_embeddings = self.query_encoder.encode(
            [doc.page_content for doc in docs],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("Encoding documents with document encoder...")
        doc_embeddings = self.doc_encoder.encode(
            [doc.page_content for doc in docs],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create or update vector stores
        # Use a custom embedding class that wraps our encoders
        class CustomEmbeddings:
            def __init__(self, encoder):
                self.encoder = encoder
            
            def embed_query(self, text: str):
                return self.encoder.encode(text, convert_to_numpy=True).tolist()
            
            def embed_documents(self, texts: List[str]):
                return self.encoder.encode(texts, convert_to_numpy=True).tolist()
        
        query_embeddings_wrapper = CustomEmbeddings(self.query_encoder)
        doc_embeddings_wrapper = CustomEmbeddings(self.doc_encoder)
        
        if self.query_vector_store is None:
            self.query_vector_store = FAISS.from_documents(docs, query_embeddings_wrapper)
        else:
            self.query_vector_store.add_documents(docs)
        
        if self.doc_vector_store is None:
            self.doc_vector_store = FAISS.from_documents(docs, doc_embeddings_wrapper)
        else:
            self.doc_vector_store.add_documents(docs)
        
        # Initialize BM25
        if self.use_bm25:
            print("Building BM25 index...")
            self.bm25 = BM25Manual(self.doc_texts)
    
    def add_documents_from_files(self, file_paths: List[str]) -> None:
        """Load documents from text files."""
        documents = []
        metadatas = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({'source': file_path})
        
        if documents:
            self.add_documents(documents, metadatas)
    
    def _encode_query_splade(self, query: str) -> torch.Tensor:
        """Encode query using SPLADE."""
        if self.splade_encoder is None:
            return None
        
        inputs = self.splade_tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.splade_model(**inputs)
            # SPLADE outputs sparse vectors
            sparse_vec = torch.max(torch.log(1 + torch.relu(outputs.logits)), dim=1)[0]
        return sparse_vec
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        use_bm25: bool = True
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            k: Number of results
            alpha: Weight for dense vs sparse (0.5 = equal weight)
            use_bm25: Whether to use BM25 for sparse retrieval
            
        Returns:
            List of retrieval results with scores
        """
        results_dict = {}
        
        # Dense retrieval (query encoder)
        if self.query_vector_store:
            query_embedding = self.query_encoder.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Get top 2k for reranking
            docs_with_scores = self.query_vector_store.similarity_search_with_score(
                query, k=min(2*k, len(self.documents))
            )
            
            for doc, score in docs_with_scores:
                doc_id = id(doc)
                if doc_id not in results_dict:
                    results_dict[doc_id] = RetrievalResult(
                        document=doc,
                        score=0.0,
                        method='hybrid'
                    )
                # Normalize score (FAISS returns distance, convert to similarity)
                normalized_score = 1.0 / (1.0 + score)
                results_dict[doc_id].score += alpha * normalized_score
        
        # Sparse retrieval (BM25)
        if use_bm25 and self.bm25:
            bm25_scores = self.bm25.get_scores(query)
            max_bm25 = max(bm25_scores) if bm25_scores else 1.0
            
            for i, score in enumerate(bm25_scores):
                if i < len(self.documents):
                    doc = self.documents[i]
                    doc_id = id(doc)
                    if doc_id not in results_dict:
                        results_dict[doc_id] = RetrievalResult(
                            document=doc,
                            score=0.0,
                            method='hybrid'
                        )
                    # Normalize BM25 score
                    normalized_bm25 = score / max_bm25 if max_bm25 > 0 else 0
                    results_dict[doc_id].score += (1 - alpha) * normalized_bm25
        
        # Sort by score and return top k
        sorted_results = sorted(results_dict.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        method: str = 'hybrid'
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            method: 'dense', 'sparse', 'bm25', or 'hybrid'
            
        Returns:
            List of similar documents
        """
        if method == 'hybrid':
            results = self.hybrid_search(query, k=k)
            return [r.document for r in results]
        elif method == 'dense' and self.query_vector_store:
            # Encode query and search
            query_embedding = self.query_encoder.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)
            docs_with_scores = self.query_vector_store.similarity_search_with_score(query, k=k)
            return [doc for doc, _ in docs_with_scores]
        elif method == 'bm25' and self.bm25:
            scores = self.bm25.get_scores(query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [self.documents[i] for i in top_indices]
        else:
            return []
    
    def save(self, path: str) -> None:
        """Save the knowledge base to disk."""
        os.makedirs(path, exist_ok=True)
        
        if self.query_vector_store:
            self.query_vector_store.save_local(os.path.join(path, "query_vector_store"))
        if self.doc_vector_store:
            self.doc_vector_store.save_local(os.path.join(path, "doc_vector_store"))
        
        # Save BM25
        if self.bm25:
            with open(os.path.join(path, "bm25.pkl"), 'wb') as f:
                pickle.dump(self.bm25, f)
        
        # Save documents
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, "doc_texts.pkl"), 'wb') as f:
            pickle.dump(self.doc_texts, f)
    
    def load(self, path: str) -> None:
        """Load the knowledge base from disk."""
        # Create custom embeddings wrappers for loading
        class CustomEmbeddings:
            def __init__(self, encoder):
                self.encoder = encoder
            
            def embed_query(self, text: str):
                return self.encoder.encode(text, convert_to_numpy=True).tolist()
            
            def embed_documents(self, texts: List[str]):
                return self.encoder.encode(texts, convert_to_numpy=True).tolist()
        
        query_embeddings_wrapper = CustomEmbeddings(self.query_encoder)
        doc_embeddings_wrapper = CustomEmbeddings(self.doc_encoder)
        
        if os.path.exists(os.path.join(path, "query_vector_store")):
            self.query_vector_store = FAISS.load_local(
                os.path.join(path, "query_vector_store"),
                query_embeddings_wrapper,
                allow_dangerous_deserialization=True
            )
        
        if os.path.exists(os.path.join(path, "doc_vector_store")):
            self.doc_vector_store = FAISS.load_local(
                os.path.join(path, "doc_vector_store"),
                doc_embeddings_wrapper,
                allow_dangerous_deserialization=True
            )
        
        if os.path.exists(os.path.join(path, "bm25.pkl")):
            with open(os.path.join(path, "bm25.pkl"), 'rb') as f:
                self.bm25 = pickle.load(f)
        
        if os.path.exists(os.path.join(path, "documents.pkl")):
            with open(os.path.join(path, "documents.pkl"), 'rb') as f:
                self.documents = pickle.load(f)
        
        if os.path.exists(os.path.join(path, "doc_texts.pkl")):
            with open(os.path.join(path, "doc_texts.pkl"), 'rb') as f:
                self.doc_texts = pickle.load(f)

