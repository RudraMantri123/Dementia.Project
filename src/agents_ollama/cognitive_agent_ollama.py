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
            # Restore exercise data from context
            if context.get('exercise_data'):
                self.exercise_data = context['exercise_data']
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

                response_text = f"""Memory Exercise: {data['category'].title()}

{items_text}

Memorize these items. Type 'ready' when done."""

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
        prompt = """Create a DETAILED narrative incident (8-12 sentences) for a dementia memory exercise. Tell a complete story about a specific event or experience.

Include rich, specific details:
- Character names, ages, and relationships
- Specific times (e.g., "9:30 AM", "Tuesday afternoon")
- Colors, numbers, and quantities
- Places and locations (street names, building names)
- Sequence of events with a beginning, middle, and end
- Sensory details and emotions

Make it warm, positive, and relatable - like something that could happen in everyday life.

Then create a summary prompt asking the user to recall the story.

Respond ONLY with JSON (no extra text):
{
  "title": "The Story Title",
  "story": "A detailed narrative with multiple paragraphs describing the incident in rich detail...",
  "summary_prompt": "Describe what happened in this story. Include the main characters, setting, and key events."
}

Example themes: a neighborhood gathering, a helpful encounter, a family visit, shopping experience, or solving a small problem."""

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
                    'summary_prompt': data.get('summary_prompt', 'Please summarize the story in your own words.')
                }

                response_text = f"""Story Recall: "{data['title']}"

{data['story']}

Read and remember the details. Type 'ready' when done."""

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

                # Store exercise data for evaluation
                self.exercise_data = {
                    'type': 'pattern',
                    'answer': data['answer'].lower().strip(),
                    'hint': data['hint']
                }

                response_text = f"""Pattern Exercise

{sequence_str}

What comes next? (Hint: {data['hint']})"""

                return {
                    'response': response_text,
                    'agent': 'cognitive',
                    'exercise_type': 'pattern',
                    'exercise_state': 'evaluating',  # CRITICAL: Must set this!
                    'exercise_data': self.exercise_data
                }
        except:
            return self._fallback_pattern_exercise()

    def _generate_orientation_exercise(self) -> Dict[str, Any]:
        """Generate orientation questions."""
        import datetime
        now = datetime.datetime.now()

        response_text = f"""Orientation Questions:

1. What day of the week is today?
2. What month are we in?
3. What season is it?
4. What time of day is it?"""

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
            'response': f"""Memory Exercise: {category.title()}

{items_text}

Type 'ready' when done.""",
            'agent': 'cognitive',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': {'type': 'memory_list', 'category': category, 'items': items}
        }

    def _fallback_story_exercise(self) -> Dict[str, Any]:
        """Fallback story exercise with detailed narrative."""
        story = """It was a sunny Tuesday morning when Maria decided to visit Riverside Park. She put on her favorite blue sun hat and packed two turkey sandwiches wrapped in wax paper, along with a bottle of lemonade in her canvas tote bag.

When she arrived at the park around 10:30 AM, she found her longtime friend Tom already sitting on their usual green bench near the pond. Tom had brought a bag of bread crumbs to feed the ducks. Together, they spent the next hour watching seven ducks swimming gracefully in the clear water.

Around noon, they shared the sandwiches Maria had brought, enjoying the warm sunshine and gentle breeze. A young girl in a yellow dress rode by on her bicycle, ringing her bell cheerfully. Before leaving at 1 PM, they agreed to meet again next Tuesday at the same spot."""

        self.exercise_data = {
            'type': 'story_recall',
            'story': story,
            'title': 'A Pleasant Afternoon',
            'summary_prompt': 'Please summarize what happened in the story. Include the main characters and key details.'
        }

        return {
            'response': f"""Story Recall: "A Pleasant Afternoon"

{story}

Read and remember. Type 'ready' when done.""",
            'agent': 'cognitive',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': self.exercise_data
        }

    def _fallback_pattern_exercise(self) -> Dict[str, Any]:
        """Fallback pattern exercise."""
        self.exercise_data = {
            'type': 'pattern',
            'answer': 'Red',
            'hint': 'The pattern alternates between colors'
        }

        return {
            'response': """Pattern: Blue Red Blue Red Blue ?

What comes next?""",
            'agent': 'cognitive',
            'exercise_type': 'pattern',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }

    def _continue_exercise(self, user_input: str, exercise_state: str) -> Dict[str, Any]:
        if not self.exercise_data:
            return {
                'response': """Exercise state lost. Try:
• Memory exercise
• Story recall
• Pattern recognition""",
                'agent': 'cognitive',
                'exercise_complete': True
            }

        # Handle "ready" state - user is ready to be tested
        if exercise_state == 'waiting_for_ready' and 'ready' in user_input.lower():
            exercise_type = self.exercise_data.get('type', 'unknown')

            if exercise_type == 'memory_list':
                category = self.exercise_data.get('category', 'items')
                items = self.exercise_data.get('items', [])
                num_items = len(items)

                return {
                    'response': f"""Recall the {num_items} {category} (separated by commas):""",
                    'agent': 'cognitive',
                    'exercise_state': 'evaluating',
                    'exercise_data': self.exercise_data
                }
            elif exercise_type == 'story_recall':
                summary_prompt = self.exercise_data.get('summary_prompt', 'Please summarize the story.')
                title = self.exercise_data.get('title', 'The Story')

                return {
                    'response': f"""Recall "{title}" - {summary_prompt}""",
                    'agent': 'cognitive',
                    'exercise_state': 'evaluating',
                    'exercise_data': self.exercise_data
                }
            else:
                return {
                    'response': "Exercise type unknown. Start a new one?",
                    'agent': 'cognitive',
                    'exercise_complete': True
                }

        elif exercise_state == 'evaluating':
            return self._evaluate_response(user_input)

        return {
            'response': "Type 'ready' to continue.",
            'agent': 'cognitive'
        }

    def _evaluate_response(self, user_response: str) -> Dict[str, Any]:
        """Evaluate user's response to the exercise."""
        if not self.exercise_data:
            return {
                'response': """Exercise lost. Try: memory, story, or pattern exercise?""",
                'agent': 'cognitive',
                'exercise_complete': True
            }

        exercise_type = self.exercise_data.get('type', 'unknown')

        if exercise_type == 'memory_list':
            items = self.exercise_data.get('items', [])
            category = self.exercise_data.get('category', 'items')
            items_list = '\n'.join([f"  {i+1}. {item.title()}" for i, item in enumerate(items)])

            response = f"""Original {category.title()}:
{items_list}

Try another exercise?"""

        elif exercise_type == 'story_recall':
            story = self.exercise_data.get('story', '')
            title = self.exercise_data.get('title', 'The Story')

            response = f"""Original: "{title}"

{story}

Try another exercise?"""

        elif exercise_type == 'pattern':
            correct_answer = self.exercise_data.get('answer', '').lower().strip()
            user_answer = user_response.lower().strip()
            hint = self.exercise_data.get('hint', '')

            is_correct = (user_answer == correct_answer or
                         correct_answer in user_answer or
                         user_answer in correct_answer)

            if is_correct:
                response = f"""Correct! Answer: {correct_answer}

Try another exercise?"""
            else:
                response = f"""Correct answer: {correct_answer}
Hint: {hint}

Try another exercise?"""

        else:
            response = f"""Exercise complete. Try another?"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_complete': True
        }
