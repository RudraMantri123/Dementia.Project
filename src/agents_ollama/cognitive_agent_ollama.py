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

                response_text = f"""[Brain] Memory Challenge: {data['category'].title()}

Memorize these {len(data['items'])} items. Tip: {data.get('memory_tip', 'Try to create a story!')}

{items_text}

Take 30-45 seconds. Type 'ready' when you're done!"""

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

                response_text = f"""[Story] Story Recall: "{data['title']}"

Read carefully and remember as much detail as you can:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{data['story']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Tip] Visualize it like a movie - focus on names, times, and events.

Take 1-2 minutes to read. Type 'ready' when you're done."""

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

                response_text = f"""[Pattern] Pattern Recognition

What comes next in this sequence?

{sequence_str}

[Tip] Hint: {data['hint']}

Type your answer!"""

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

        response_text = f"""[Calendar] Orientation Exercise

These questions help you stay connected to the present:

1. What day of the week is today?
2. What month are we in?
3. What season is it?
4. What time of day is it (morning/afternoon/evening)?

[Tip] Take your time and answer thoughtfully!"""

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
            'response': f"""[Brain] Memory Exercise: {category.title()}

{items_text}

Type 'ready' when you want to recall them.""",
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
            'response': f"""[Story] Story Recall: "A Pleasant Afternoon"

Read carefully and remember as much detail as you can:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{story}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Tip] Visualize it like a movie - focus on names, times, and events.

Take 1-2 minutes to read. Type 'ready' when you're done.""",
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
            'response': """[Pattern] Pattern: Blue Red Blue Red Blue ?

What comes next?""",
            'agent': 'cognitive',
            'exercise_type': 'pattern',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }

    def _continue_exercise(self, user_input: str, exercise_state: str) -> Dict[str, Any]:
        # DEFENSIVE: Validate exercise_data exists
        if not self.exercise_data:
            return {
                'response': """I apologize, but I seem to have lost track of the exercise. Let's start fresh!

Would you like to try:
• A memory exercise
• A story recall exercise
• A pattern recognition exercise

Just let me know what you'd like to try!""",
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
                    'response': f"""Perfect! Time to test your memory!

[Important] Don't scroll up - recall from memory alone!

You were shown {num_items} {category}. Type the items you remember, separated by commas (any order).

Example: apple, banana, orange""",
                    'agent': 'cognitive',
                    'exercise_state': 'evaluating',
                    'exercise_data': self.exercise_data
                }
            elif exercise_type == 'story_recall':
                summary_prompt = self.exercise_data.get('summary_prompt', 'Please summarize the story.')
                title = self.exercise_data.get('title', 'The Story')

                return {
                    'response': f"""Excellent! Now recall "{title}" without looking back.

{summary_prompt}

Try to include: characters, setting, key events, and specific details you remember.

[Tip] Start with what you remember most clearly.""",
                    'agent': 'cognitive',
                    'exercise_state': 'evaluating',
                    'exercise_data': self.exercise_data
                }
            else:
                # Unknown exercise type
                return {
                    'response': "I'm not sure what exercise we were doing. Let's start a new one! What would you like to try?",
                    'agent': 'cognitive',
                    'exercise_complete': True
                }

        # Handle "evaluating" state - user has provided their answer
        elif exercise_state == 'evaluating':
            return self._evaluate_response(user_input)

        return {
            'response': "Type 'ready' to continue the exercise.",
            'agent': 'cognitive'
        }

    def _evaluate_response(self, user_response: str) -> Dict[str, Any]:
        """Evaluate user's response to the exercise."""
        # DEFENSIVE: Check if exercise_data exists
        if not self.exercise_data:
            return {
                'response': """I apologize, but I seem to have lost track of the exercise. This can happen if the connection was interrupted.

Let's start fresh! Would you like to try:
• A memory exercise
• A story recall exercise
• A pattern recognition exercise

Just let me know what you'd like to try!""",
                'agent': 'cognitive',
                'exercise_complete': True
            }

        exercise_type = self.exercise_data.get('type', 'unknown')

        # Build response based on exercise type
        if exercise_type == 'memory_list':
            items = self.exercise_data.get('items', [])
            category = self.exercise_data.get('category', 'items')

            # Show the original list
            items_list = '\n'.join([f"  {i+1}. {item.title()}" for i, item in enumerate(items)])

            response = f"""Thank you for trying!

[List] The Original List ({category.title()}):
{items_list}

Great effort! Memory recall can be challenging - the important thing is engaging with the exercise. Regular mental exercises support brain health and strengthen neural pathways.

Would you like to try another exercise or continue our conversation?"""

        elif exercise_type == 'story_recall':
            story = self.exercise_data.get('story', '')
            title = self.exercise_data.get('title', 'The Story')

            response = f"""Thank you for trying!

[Story] Original Story: "{title}"

{story}

I appreciate you working through this exercise! Story recall with specific details can be challenging - don't worry about getting everything perfect. The effort itself matters and helps keep your mind active.

Would you like to try a different exercise or continue our conversation?"""

        elif exercise_type == 'pattern':
            correct_answer = self.exercise_data.get('answer', '').lower().strip()
            user_answer = user_response.lower().strip()
            hint = self.exercise_data.get('hint', '')

            # Check if answer is correct (flexible matching)
            is_correct = (user_answer == correct_answer or
                         correct_answer in user_answer or
                         user_answer in correct_answer)

            if is_correct:
                response = f"""Excellent work! That's correct - the answer was: {correct_answer}

Great job recognizing the pattern! This helps strengthen cognitive skills and mental flexibility.

Would you like to try another exercise or continue our conversation?"""
            else:
                response = f"""Thank you for trying!

The correct answer was: {correct_answer} (Hint: {hint})

Pattern recognition can be tricky - the important thing is engaging with the exercise. Every attempt helps strengthen your brain!

Would you like to try another exercise or continue our conversation?"""

        else:
            # Generic response for other exercise types
            response = f"""Thank you for participating!

Great job engaging with this exercise! Every cognitive activity is valuable for brain health.

Would you like to try another exercise or continue our conversation?"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_complete': True
        }
