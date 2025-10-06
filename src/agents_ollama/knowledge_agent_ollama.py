"""Knowledge agent using Ollama and RAG."""

from typing import Dict, Any, Optional
from .base_agent_ollama import BaseAgentOllama


class KnowledgeAgentOllama(BaseAgentOllama):
    """Answers questions using RAG with free Ollama models."""

    def __init__(self, knowledge_base, model_name: str = "llama3.2"):
        super().__init__(model_name, temperature=0.7)
        self.knowledge_base = knowledge_base

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Retrieve relevant context from knowledge base
        if self.knowledge_base and self.knowledge_base.vector_store:
            docs = self.knowledge_base.vector_store.similarity_search(user_input, k=3)
            context_text = "\n\n".join([doc.page_content for doc in docs])
        else:
            context_text = "No knowledge base available."

        prompt = f"""You are a knowledgeable assistant specializing in dementia and cognitive health.
Answer the question using ONLY the context provided. If you don't know, say so.

Context: {context_text}

Question: {user_input}

Helpful Answer:"""

        response = self.llm.invoke(prompt)
        
        return {
            'response': response,
            'agent': 'knowledge',
            'num_sources': len(docs) if self.knowledge_base else 0
        }
