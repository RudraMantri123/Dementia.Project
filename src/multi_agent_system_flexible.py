"""Multi-agent system supporting both OpenAI and Ollama models."""

from typing import Dict, Any, Optional
from src.knowledge_base import KnowledgeBase


class MultiAgentChatbotFlexible:
    """Main chatbot system coordinating multiple specialized agents with flexible model support."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model_type: str = "ollama",
        api_key: Optional[str] = None,
        model_name: str = "llama3.2"
    ):
        """
        Initialize the multi-agent system.

        Args:
            knowledge_base: KnowledgeBase instance
            model_type: Type of model ('openai' or 'ollama')
            api_key: API key (required only for OpenAI)
            model_name: Name of the model to use
        """
        self.model_type = model_type
        self.api_key = api_key
        self.model_name = model_name

        # Import agents dynamically based on model type
        if model_type == "openai":
            from src.agents.orchestrator import AgentOrchestrator
            from src.agents.knowledge_agent import KnowledgeAgent
            from src.agents.empathy_agent import EmpathyAgent
            from src.agents.cognitive_agent import CognitiveAgent

            # Initialize with OpenAI
            self.orchestrator = AgentOrchestrator(api_key, model_name)
            self.knowledge_agent = KnowledgeAgent(knowledge_base, api_key, model_name)
            self.empathy_agent = EmpathyAgent(api_key, model_name)
            self.cognitive_agent = CognitiveAgent(api_key, model_name)

        else:  # ollama
            from src.agents_ollama.orchestrator_ollama import AgentOrchestratorOllama
            from src.agents_ollama.knowledge_agent_ollama import KnowledgeAgentOllama
            from src.agents_ollama.empathy_agent_ollama import EmpathyAgentOllama
            from src.agents_ollama.cognitive_agent_ollama import CognitiveAgentOllama

            # Initialize with Ollama
            self.orchestrator = AgentOrchestratorOllama(model_name)
            self.knowledge_agent = KnowledgeAgentOllama(knowledge_base, model_name)
            self.empathy_agent = EmpathyAgentOllama(model_name)
            self.cognitive_agent = CognitiveAgentOllama(model_name)

        # Map agent names to agent instances
        self.agents = {
            'knowledge': self.knowledge_agent,
            'empathy': self.empathy_agent,
            'cognitive': self.cognitive_agent
        }

        # Track conversation state
        self.conversation_state = {
            'current_agent': None,
            'exercise_state': None,
            'exercise_data': None
        }

        # Conversation history for analytics
        self.conversation_log = []

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input through the multi-agent system.

        Args:
            user_input: User's message

        Returns:
            Dictionary with response and metadata
        """
        # Check if we're in the middle of a cognitive exercise
        if self.conversation_state.get('exercise_state'):
            result = self._continue_exercise(user_input)
        else:
            # Use orchestrator to determine which agent to route to
            routing = self.orchestrator.process(user_input)
            agent_name = routing['route_to']
            intent = routing['intent']
            reasoning = routing.get('reasoning', '')

            # Get appropriate agent
            agent = self.agents.get(agent_name)

            if not agent:
                return {
                    'response': "I'm here to help! You can ask me about dementia, share your feelings, or try a cognitive exercise.",
                    'agent': 'system',
                    'intent': intent
                }

            # Process with selected agent
            context = {
                'intent': intent,
                'exercise_state': self.conversation_state.get('exercise_state'),
                'exercise_data': self.conversation_state.get('exercise_data')
            }

            result = agent.process(user_input, context)

            # Update conversation state if cognitive exercise started
            if agent_name == 'cognitive' and result.get('exercise_state'):
                self.conversation_state['current_agent'] = 'cognitive'
                self.conversation_state['exercise_state'] = result['exercise_state']
                self.conversation_state['exercise_data'] = result.get('exercise_data')

            # Add routing information
            result['intent'] = intent

        # Log conversation for analytics
        self.conversation_log.append({
            'user_input': user_input,
            'agent': result.get('agent'),
            'intent': result.get('intent'),
            'timestamp': self._get_timestamp()
        })

        return result

    def _continue_exercise(self, user_input: str) -> Dict[str, Any]:
        """
        Continue an ongoing cognitive exercise.

        Args:
            user_input: User's response

        Returns:
            Exercise continuation or completion
        """
        if not self.conversation_state.get('exercise_state'):
            return {
                'response': """I apologize, but it seems the exercise state was lost. This can happen if the connection was interrupted.

Let's start fresh! You can ask me for:
• A memory exercise
• A story recall exercise
• Help with dementia-related questions
• Emotional support

What would you like to try?""",
                'agent': 'cognitive',
                'intent': 'COGNITIVE'
            }

        cognitive_agent = self.agents.get('cognitive')
        if not cognitive_agent:
            return {
                'response': "I apologize, but the cognitive agent is not available right now. Please try asking a different question.",
                'agent': 'system',
                'intent': 'ERROR'
            }

        context = {
            'exercise_state': self.conversation_state['exercise_state'],
            'exercise_data': self.conversation_state.get('exercise_data', {})
        }

        result = cognitive_agent.process(user_input, context)

        if result.get('response'):
            result['response'] = f"*We're continuing your cognitive exercise.*\n\n{result['response']}"

        if result.get('exercise_state'):
            self.conversation_state['exercise_state'] = result['exercise_state']
            self.conversation_state['exercise_data'] = result.get('exercise_data')
        else:
            self.conversation_state['current_agent'] = None
            self.conversation_state['exercise_state'] = None
            self.conversation_state['exercise_data'] = None

        return result

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation.

        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_log:
            return {
                'total_messages': 0,
                'agent_distribution': {},
                'intent_distribution': {}
            }

        # Count agent usage
        agent_counts = {}
        intent_counts = {}

        for entry in self.conversation_log:
            agent = entry.get('agent', 'unknown')
            intent = entry.get('intent', 'unknown')

            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            'total_messages': len(self.conversation_log),
            'agent_distribution': agent_counts,
            'intent_distribution': intent_counts,
            'conversation_log': self.conversation_log[-10:]  # Last 10 entries
        }

    def reset_conversation(self):
        """Reset conversation state and history."""
        self.conversation_state = {
            'current_agent': None,
            'exercise_state': None,
            'exercise_data': None
        }
        # Clear agent histories
        for agent in self.agents.values():
            agent.conversation_history = []

    @staticmethod
    def _get_timestamp():
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
