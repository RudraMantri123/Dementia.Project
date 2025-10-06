"""Knowledge base management using vector stores."""

import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class KnowledgeBase:
    """Manages the vector store and document retrieval for dementia-related information."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the knowledge base.

        Args:
            embedding_model: Name of the HuggingFace embedding model to use
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries for each document
        """
        # Split documents into chunks
        all_chunks = []
        all_metadatas = []

        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)

            # Add metadata for each chunk
            if metadatas and i < len(metadatas):
                all_metadatas.extend([metadatas[i]] * len(chunks))
            else:
                all_metadatas.extend([{}] * len(chunks))

        # Create Document objects
        docs = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(all_chunks, all_metadatas)
        ]

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

    def add_documents_from_files(self, file_paths: List[str]) -> None:
        """
        Load documents from text files and add to knowledge base.

        Args:
            file_paths: List of paths to text files
        """
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

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents in the knowledge base.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            return []

        return self.vector_store.similarity_search(query, k=k)

    def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is not None:
            self.vector_store.save_local(path)

    def load(self, path: str) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Directory path to load the vector store from
        """
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
