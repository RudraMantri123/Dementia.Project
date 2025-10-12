"""Knowledge agent for answering factual questions about dementia."""

from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from .base_agent import BaseAgent


class KnowledgeAgent(BaseAgent):
    """Answers factual questions using the RAG knowledge base."""

    def __init__(
        self,
        knowledge_base,
        api_key: str,
        model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the knowledge agent.

        Args:
            knowledge_base: KnowledgeBase instance
            api_key: OpenAI API key
            model_name: Name of the OpenAI model
        """
        super().__init__(api_key, model_name, temperature=0.7)
        self.knowledge_base = knowledge_base

        # Enhanced prompt for knowledge queries
        self.prompt_template = """You are a knowledgeable and compassionate expert on dementia and caregiving.
Use the following context to answer the question CONCISELY.

IMPORTANT GUIDELINES:
- Keep answers SHORT (3-5 sentences maximum)
- Provide accurate, evidence-based information
- Use clear, accessible language
- Focus on the most important 2-3 key points only
- If unsure, say so briefly
- For medical advice, briefly mention consulting healthcare professionals

Context from knowledge base:
{context}

Question: {question}

Brief, Helpful Answer (3-5 sentences max):"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain if knowledge base is available
        self.qa_chain = None
        if knowledge_base and knowledge_base.vector_store:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=knowledge_base.vector_store.as_retriever(
                    search_kwargs={"k": 5}  # Retrieve top 5 chunks
                ),
                chain_type_kwargs={"prompt": self.PROMPT},
                return_source_documents=True
            )

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process factual question using RAG.

        Args:
            user_input: User's question
            context: Optional context information

        Returns:
            Dictionary with answer and sources
        """
        self.add_to_history('user', user_input)

        if not self.qa_chain:
            response_text = "I apologize, but my knowledge base is not currently available. Please try again later."
            self.add_to_history('assistant', response_text)
            return {
                'response': response_text,
                'agent': 'knowledge',
                'sources': []
            }

        try:
            # Get response from QA chain
            result = self.qa_chain({"query": user_input})
            response_text = result.get('result', '')
            source_docs = result.get('source_documents', [])

            # Extract source information
            sources = []
            for doc in source_docs:
                sources.append({
                    'content': doc.page_content[:200] + "...",
                    'metadata': doc.metadata
                })

            self.add_to_history('assistant', response_text)

            return {
                'response': response_text,
                'agent': 'knowledge',
                'sources': sources,
                'num_sources': len(sources)
            }

        except Exception as e:
            error_msg = f"I encountered an error while searching my knowledge base: {str(e)}"
            self.add_to_history('assistant', error_msg)
            return {
                'response': error_msg,
                'agent': 'knowledge',
                'sources': [],
                'error': str(e)
            }
