"""Cognitive agent using Ollama with AI-generated exercises."""

from typing import Dict, Any, Optional
import random
import json
from .base_agent_ollama import BaseAgentOllama


class CognitiveAgentOllama(BaseAgentOllama):
    """Provides AI-generated cognitive exercises using free models."""

    def __init__(self, model_name: str = "llama3.2"):
        super().__init__(model_name, temperature=0.9)  # Higher temp for creativity
        self.active_exercise = None
        self.exercise_data = {}

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if context and context.get('exercise_state'):
            return self._continue_exercise(user_input, context['exercise_state'])
        return self._start_new_exercise(user_input)

    def _start_new_exercise(self, user_input: str) -> Dict[str, Any]:
        """Generate a dynamic exercise using LLM."""
        user_lower = user_input.lower()

        # Determine exercise type
        if 'story' in user_lower or 'recall' in user_lower:
            return self._generate_story_exercise()
        elif 'pattern' in user_lower or 'sequence' in user_lower:
            return self._generate_pattern_exercise()
        elif 'orientation' in user_lower:
            return self._generate_orientation_exercise()
        else:
            # Randomly choose for variety
            exercise_types = [
                self._generate_memory_exercise,
                self._generate_story_exercise,
                self._generate_pattern_exercise
            ]
            return random.choice(exercise_types)()

    def _generate_memory_exercise(self) -> Dict[str, Any]:
        """Generate a fresh memory list exercise using AI."""
        prompt = """Generate a memory exercise for dementia patients. Create a list of 5 related items from ONE category.

Choose from categories like: grocery items, household objects, garden flowers, family activities, musical instruments, cities, foods, hobbies, or seasons.

Respond ONLY with JSON in this exact format (no extra text):
{
  "category": "category_name",
  "items": ["item1", "item2", "item3", "item4", "item5"],
  "memory_tip": "A helpful mnemonic strategy"
}

Make it engaging and age-appropriate for elderly adults."""

        try:
            response = self.llm.invoke(prompt)
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])

                self.exercise_data = {
                    'type': 'memory_list',
                    'category': data['category'],
                    'items': data['items'],
                    'memory_tip': data.get('memory_tip', 'Create a mental story connecting these items!')
                }

                items_text = '\n'.join([f"{i+1}. {item.title()}" for i, item in enumerate(data['items'])])

                response_text = f"""🧠 Memory Challenge: {data['category'].title()}

I'm going to show you {len(data['items'])} items. Here's a memory tip: {data.get('memory_tip', 'Try to create a story!')}

{items_text}

💡 Take 30-45 seconds to memorize them. When you're ready, type 'ready' and I'll test your memory!"""

                return {
                    'response': response_text,
                    'agent': 'cognitive',
                    'exercise_type': 'memory_list',
                    'exercise_state': 'waiting_for_ready',
                    'exercise_data': self.exercise_data
                }
        except Exception as e:
            # Fallback to simple exercise
            return self._fallback_memory_exercise()

        return self._fallback_memory_exercise()

    def _generate_story_exercise(self) -> Dict[str, Any]:
        """Generate a fresh story recall exercise using AI."""
        prompt = """Create a SHORT story (3-4 sentences) for a dementia memory exercise. Include specific details: names, colors, numbers, times, places.

Then create 4 simple recall questions about the story.

Respond ONLY with JSON (no extra text):
{
  "title": "story_title",
  "story": "the short story text",
  "questions": ["question1?", "question2?", "question3?", "question4?"]
}

Make it warm, positive, and relatable for elderly adults."""

        try:
            response = self.llm.invoke(prompt)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])

                self.exercise_data = {
                    'type': 'story_recall',
                    'story': data['story'],
                    'title': data['title'],
                    'questions': data['questions']
                }

                response_text = f"""📖 Story Recall: "{data['title']}"

Read this story carefully and remember the details:

{data['story']}

💡 Visualize it like a movie in your mind!

When ready to answer questions, type 'ready'."""

                return {
                    'response': response_text,
                    'agent': 'cognitive',
                    'exercise_type': 'story_recall',
                    'exercise_state': 'waiting_for_ready',
                    'exercise_data': self.exercise_data
                }
        except:
            return self._fallback_story_exercise()

    def _generate_pattern_exercise(self) -> Dict[str, Any]:
        """Generate a pattern recognition exercise."""
        prompt = """Create a simple pattern recognition exercise for dementia patients.

Generate ONE sequence with a missing element. Use emojis, numbers, or simple words.

Examples:
- Colors: 🔵 🔴 🔵 🔴 🔵 ?
- Numbers: 2 4 6 8 ?
- Days: Monday Tuesday ? Thursday

Respond ONLY with JSON:
{
  "sequence": ["item1", "item2", "item3", "item4", "?"],
  "answer": "missing_item",
  "hint": "pattern description"
}"""

        try:
            response = self.llm.invoke(prompt)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])

                sequence_str = ' '.join(data['sequence'])

                response_text = f"""🧩 Pattern Recognition

What comes next in this sequence?

{sequence_str}

💡 Hint: {data['hint']}

Type your answer!"""

                return {
                    'response': response_text,
                    'agent': 'cognitive',
                    'exercise_type': 'pattern',
                    'answer': data['answer']
                }
        except:
            return self._fallback_pattern_exercise()

    def _generate_orientation_exercise(self) -> Dict[str, Any]:
        """Generate orientation questions."""
        import datetime
        now = datetime.datetime.now()

        response_text = f"""📅 Orientation Exercise

These questions help you stay connected to the present:

1. What day of the week is today?
2. What month are we in?
3. What season is it?
4. What time of day is it (morning/afternoon/evening)?

💡 Take your time and answer thoughtfully!"""

        return {
            'response': response_text,
            'agent': 'cognitive',
            'exercise_type': 'orientation'
        }

    def _fallback_memory_exercise(self) -> Dict[str, Any]:
        """Fallback if LLM generation fails."""
        categories = {
            'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry'],
            'animals': ['dog', 'cat', 'elephant', 'lion', 'bird'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple']
        }
        category = random.choice(list(categories.keys()))
        items = random.sample(categories[category], 5)

        items_text = '\n'.join([f"{i+1}. {item.title()}" for i, item in enumerate(items)])

        return {
            'response': f"""🧠 Memory Exercise: {category.title()}

{items_text}

Type 'ready' when you want to recall them.""",
            'agent': 'cognitive',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': {'type': 'memory_list', 'category': category, 'items': items}
        }

    def _fallback_story_exercise(self) -> Dict[str, Any]:
        """Fallback story exercise."""
        return {
            'response': """📖 Story: "A Pleasant Afternoon"

Maria visited the park on Tuesday. She wore her blue hat and brought sandwiches. Her friend Tom joined her, and they watched the ducks swim.

Type 'ready' for questions.""",
            'agent': 'cognitive',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': {'type': 'story_recall'}
        }

    def _fallback_pattern_exercise(self) -> Dict[str, Any]:
        """Fallback pattern exercise."""
        return {
            'response': """🧩 Pattern: 🔵 🔴 🔵 🔴 🔵 ?

What comes next?""",
            'agent': 'cognitive',
            'answer': '🔴'
        }

    def _continue_exercise(self, user_input: str, exercise_state: str) -> Dict[str, Any]:
        if 'ready' in user_input.lower():
            exercise_type = self.exercise_data.get('type', 'unknown')

            if exercise_type == 'memory_list':
                category = self.exercise_data.get('category', 'items')
                return {
                    'response': f"Excellent! Now try to recall as many {category} as you can. List them separated by commas.",
                    'agent': 'cognitive',
                    'exercise_state': 'evaluating'
                }
            elif exercise_type == 'story_recall':
                questions = self.exercise_data.get('questions', [])
                questions_text = '\n'.join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                return {
                    'response': f"Great! Now answer these questions:\n\n{questions_text}",
                    'agent': 'cognitive',
                    'exercise_state': 'evaluating'
                }

        return {
            'response': "Type 'ready' to continue the exercise.",
            'agent': 'cognitive'
        }
