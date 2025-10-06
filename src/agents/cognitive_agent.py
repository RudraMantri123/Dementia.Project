"""Cognitive agent for memory exercises and brain training."""

from typing import Dict, Any, Optional, List
import random
from .base_agent import BaseAgent


class CognitiveAgent(BaseAgent):
    """Provides cognitive exercises and memory training activities."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the cognitive agent.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model
        """
        super().__init__(api_key, model_name, temperature=0.7)
        self.active_exercise = None
        self.exercise_data = {}

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process request for cognitive exercise.

        Args:
            user_input: User's request
            context: Optional context with exercise state

        Returns:
            Dictionary with exercise and response
        """
        # Check if continuing an active exercise
        if context and context.get('exercise_state'):
            return self._continue_exercise(user_input, context['exercise_state'])

        # Start new exercise
        return self._start_new_exercise(user_input)

    def _start_new_exercise(self, user_input: str) -> Dict[str, Any]:
        """
        Start a new cognitive exercise.

        Args:
            user_input: User's request

        Returns:
            Exercise details and instructions
        """
        # Determine exercise type
        user_lower = user_input.lower()

        if 'memory' in user_lower or 'remember' in user_lower or 'list' in user_lower:
            return self._memory_list_exercise()
        elif 'word' in user_lower or 'association' in user_lower:
            return self._word_association_exercise()
        elif 'story' in user_lower or 'recall' in user_lower:
            return self._story_recall_exercise()
        elif 'pattern' in user_lower or 'sequence' in user_lower:
            return self._pattern_recognition_exercise()
        elif 'orientation' in user_lower or 'awareness' in user_lower:
            return self._orientation_exercise()
        else:
            # Randomly select an exercise type for variety
            exercises = [
                self._memory_list_exercise,
                self._story_recall_exercise,
                self._pattern_recognition_exercise,
                self._orientation_exercise
            ]
            return random.choice(exercises)()

    def _memory_list_exercise(self) -> Dict[str, Any]:
        """Create an enhanced memory list exercise with mnemonic strategies."""
        categories = {
            'grocery_items': {
                'items': ['milk', 'bread', 'eggs', 'cheese', 'butter', 'yogurt', 'apples', 'bananas', 'chicken', 'rice'],
                'tip': 'Try grouping them: dairy products, fruits, and staples!'
            },
            'household_objects': {
                'items': ['keys', 'wallet', 'glasses', 'phone', 'watch', 'pen', 'notebook', 'umbrella', 'cup', 'book'],
                'tip': 'Imagine placing each item in a different room of your house!'
            },
            'family_activities': {
                'items': ['birthday', 'picnic', 'wedding', 'graduation', 'reunion', 'vacation', 'holiday', 'barbecue', 'concert', 'game night'],
                'tip': 'Think of a story connecting these happy moments!'
            },
            'garden_items': {
                'items': ['rose', 'tulip', 'daisy', 'sunflower', 'lily', 'orchid', 'daffodil', 'carnation', 'lavender', 'marigold'],
                'tip': 'Picture a colorful garden in your mind with each flower!'
            },
            'musical_instruments': {
                'items': ['piano', 'guitar', 'violin', 'drums', 'flute', 'trumpet', 'saxophone', 'harp', 'cello', 'clarinet'],
                'tip': 'Imagine the sound each one makes!'
            }
        }

        category = random.choice(list(categories.keys()))
        category_data = categories[category]
        num_items = random.choice([4, 5, 6])  # Vary difficulty
        items = random.sample(category_data['items'], num_items)

        # Create a visual representation with emojis
        emoji_map = {
            'grocery_items': '🛒',
            'household_objects': '🏠',
            'family_activities': '👨‍👩‍👧‍👦',
            'garden_items': '🌸',
            'musical_instruments': '🎵'
        }

        self.exercise_data = {
            'type': 'memory_list',
            'category': category,
            'items': items,
            'started': True,
            'num_items': num_items
        }

        category_name = category.replace('_', ' ')
        response = f"""{emoji_map.get(category, '📝')} Memory Challenge: {category_name.title()}

I'm going to show you {num_items} {category_name}. Use this memory tip: {category_data['tip']}

{chr(10).join([f"{i+1}. {item.title()} {self._get_item_emoji(item)}" for i, item in enumerate(items)])}

💡 Memory Strategy: Try creating a mental image or story connecting these items!

Take 30-45 seconds to memorize them. When you're ready, type 'ready' and I'll test your memory!"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_type': 'memory_list',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': self.exercise_data
        }

    def _get_item_emoji(self, item: str) -> str:
        """Get emoji for common items."""
        emoji_dict = {
            'milk': '🥛', 'bread': '🍞', 'eggs': '🥚', 'cheese': '🧀', 'butter': '🧈',
            'apple': '🍎', 'banana': '🍌', 'chicken': '🍗', 'rose': '🌹', 'tulip': '🌷',
            'daisy': '🌼', 'sunflower': '🌻', 'piano': '🎹', 'guitar': '🎸', 'drums': '🥁',
            'phone': '📱', 'keys': '🔑', 'glasses': '👓', 'book': '📖', 'birthday': '🎂',
            'wedding': '💒', 'vacation': '✈️', 'picnic': '🧺'
        }
        return emoji_dict.get(item.lower(), '')

    def _word_association_exercise(self) -> Dict[str, Any]:
        """Create a word association exercise."""
        starter_words = [
            'summer', 'ocean', 'music', 'garden', 'childhood',
            'family', 'celebration', 'journey', 'home', 'friendship'
        ]

        word = random.choice(starter_words)

        response = f"""Let's do a word association exercise! This helps with creativity and memory.

I'll give you a word: "{word.title()}"

Take a moment and share with me:
1. What's the first thing that comes to mind?
2. A memory associated with this word
3. Three other words you associate with it

This exercise helps strengthen neural connections and can be a pleasant way to reminisce!"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_type': 'word_association',
            'starter_word': word
        }

    def _story_recall_exercise(self) -> Dict[str, Any]:
        """Create an enhanced story recall exercise with varied themes."""
        stories = [
            {
                'title': 'The Garden Visit',
                'story': "Every Saturday morning, Margaret visited the community garden on Elm Street. She wore her favorite yellow hat and brought her red watering can. Her neighbor Tom was already there, tending to his tomato plants. They picked three ripe tomatoes and shared them over tea at 10 o'clock.",
                'questions': [
                    "What day did Margaret visit the garden?",
                    "What color was her hat?",
                    "What was her neighbor's name?",
                    "What vegetables did they pick?",
                    "What time did they have tea?"
                ],
                'answers': ['Saturday', 'yellow', 'Tom', 'tomatoes', '10 o\'clock'],
                'difficulty': 'medium'
            },
            {
                'title': 'The Birthday Surprise',
                'story': "Sarah decided to surprise her grandmother Helen on her 80th birthday. She baked a chocolate cake and decorated it with purple flowers. Her brother James arrived at 3 PM with a photo album. Together they sang Happy Birthday and gave Helen a silver locket.",
                'questions': [
                    "Whose birthday was it?",
                    "How old was she turning?",
                    "What flavor was the cake?",
                    "What did James bring?",
                    "What gift did they give Helen?"
                ],
                'answers': ['Helen', '80', 'chocolate', 'photo album', 'silver locket'],
                'difficulty': 'hard'
            },
            {
                'title': 'Morning Walk',
                'story': "Robert took his golden retriever Buddy for a walk in the park. They passed by the pond where ducks were swimming. Buddy played with a tennis ball near the fountain. They rested on a green bench before heading home.",
                'questions': [
                    "What was the dog's name?",
                    "What kind of dog was Buddy?",
                    "What were the ducks doing?",
                    "What toy did Buddy play with?",
                    "What color was the bench?"
                ],
                'answers': ['Buddy', 'golden retriever', 'swimming', 'tennis ball', 'green'],
                'difficulty': 'medium'
            },
            {
                'title': 'The Library Book',
                'story': "Emma visited the downtown library on Tuesday to return a mystery novel. The librarian, Ms. Chen, recommended a book about gardening. Emma checked out two books and stopped at the coffee shop next door for a cappuccino.",
                'questions': [
                    "When did Emma visit the library?",
                    "What type of book did she return?",
                    "What was the librarian's name?",
                    "How many books did Emma check out?",
                    "What drink did she order?"
                ],
                'answers': ['Tuesday', 'mystery', 'Ms. Chen', 'two', 'cappuccino'],
                'difficulty': 'hard'
            },
            {
                'title': 'Family Dinner',
                'story': "On Sunday evening, the Lopez family gathered for dinner. Grandpa Carlos made his famous chicken soup. The table was set with blue napkins, and fresh bread was served warm. Everyone enjoyed the meal and shared stories from their week.",
                'questions': [
                    "What day was the family dinner?",
                    "Who made the soup?",
                    "What kind of soup was it?",
                    "What color were the napkins?",
                    "What was served with the soup?"
                ],
                'answers': ['Sunday', 'Grandpa Carlos', 'chicken', 'blue', 'bread'],
                'difficulty': 'easy'
            }
        ]

        story_data = random.choice(stories)
        self.exercise_data = {
            'type': 'story_recall',
            'story': story_data['story'],
            'title': story_data['title'],
            'questions': story_data['questions'],
            'answers': story_data['answers'],
            'difficulty': story_data['difficulty']
        }

        difficulty_emoji = {'easy': '⭐', 'medium': '⭐⭐', 'hard': '⭐⭐⭐'}

        response = f"""📖 Story Recall Exercise: "{story_data['title']}"
Difficulty: {difficulty_emoji[story_data['difficulty']]}

I'll tell you a short story with specific details. Pay attention to names, numbers, colors, and actions!

{story_data['story']}

💡 Memory Tip: As you read, try to visualize the scene like a movie in your mind. Picture the people, colors, and actions!

Take your time to read it carefully. When you feel ready to answer questions, type 'ready'."""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_type': 'story_recall',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': self.exercise_data
        }

    def _pattern_recognition_exercise(self) -> Dict[str, Any]:
        """Create a pattern recognition exercise."""
        patterns = [
            {
                'sequence': ['🔵', '🔴', '🔵', '🔴', '🔵', '?'],
                'answer': '🔴',
                'type': 'alternating colors',
                'difficulty': 'easy'
            },
            {
                'sequence': ['2', '4', '6', '8', '?'],
                'answer': '10',
                'type': 'even numbers',
                'difficulty': 'easy'
            },
            {
                'sequence': ['🌙', '⭐', '⭐', '🌙', '⭐', '⭐', '?'],
                'answer': '🌙',
                'type': 'repeating pattern',
                'difficulty': 'medium'
            },
            {
                'sequence': ['Monday', 'Wednesday', 'Friday', '?'],
                'answer': 'Sunday',
                'type': 'days of week (skipping one)',
                'difficulty': 'medium'
            },
            {
                'sequence': ['1', '4', '9', '16', '?'],
                'answer': '25',
                'type': 'square numbers',
                'difficulty': 'hard'
            }
        ]

        pattern = random.choice(patterns)
        self.exercise_data = {
            'type': 'pattern_recognition',
            'sequence': pattern['sequence'],
            'answer': pattern['answer'],
            'pattern_type': pattern['type']
        }

        difficulty_emoji = {'easy': '⭐', 'medium': '⭐⭐', 'hard': '⭐⭐⭐'}

        response = f"""🧩 Pattern Recognition Exercise
Difficulty: {difficulty_emoji[pattern['difficulty']]}

Look at this sequence and figure out what comes next:

{' '.join(pattern['sequence'])}

💡 Hint: This is a {pattern['type']} pattern!

What comes next in the sequence? Type your answer when ready!"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_type': 'pattern_recognition',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }

    def _orientation_exercise(self) -> Dict[str, Any]:
        """Create an orientation/awareness exercise."""
        import datetime
        now = datetime.datetime.now()

        questions_sets = [
            {
                'title': 'Time Orientation',
                'questions': [
                    "What day of the week is it today?",
                    "What month are we in?",
                    "What year is it?",
                    "Approximately what time is it now (morning, afternoon, or evening)?"
                ],
                'tips': "Take a moment to think about today's date and time.",
                'icon': '📅'
            },
            {
                'title': 'Personal Awareness',
                'questions': [
                    "What is your full name?",
                    "What city or town do you live in?",
                    "Can you name three people who are important to you?",
                    "What did you have for your last meal?"
                ],
                'tips': "These questions help you connect with your personal identity and recent memories.",
                'icon': '👤'
            },
            {
                'title': 'Environmental Awareness',
                'questions': [
                    "What season is it now?",
                    "What's the weather like today?",
                    "Name three things you can see around you right now",
                    "What room are you in?"
                ],
                'tips': "Look around and use your senses to connect with your environment.",
                'icon': '🌍'
            }
        ]

        question_set = random.choice(questions_sets)

        response = f"""{question_set['icon']} {question_set['title']} Exercise

{question_set['tips']}

Please answer these questions:

{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(question_set['questions'])])}

💡 Take your time and answer thoughtfully. These exercises help maintain awareness and connection to the present!"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_type': 'orientation',
            'question_set': question_set['title']
        }

    def _continue_exercise(self, user_input: str, exercise_state: str) -> Dict[str, Any]:
        """
        Continue an ongoing exercise.

        Args:
            user_input: User's response
            exercise_state: Current state of exercise

        Returns:
            Next step or evaluation
        """
        if exercise_state == 'waiting_for_ready' and 'ready' in user_input.lower():
            if self.exercise_data.get('type') == 'memory_list':
                return self._ask_memory_recall()
            elif self.exercise_data.get('type') == 'story_recall':
                return self._ask_story_questions()

        return {
            'response': "I'm here to help with the exercise. Type 'ready' when you'd like to continue, or ask for a different exercise!",
            'agent': 'cognitive'
        }

    def _ask_memory_recall(self) -> Dict[str, Any]:
        """Ask user to recall the memory list."""
        items = self.exercise_data.get('items', [])
        category = self.exercise_data.get('category', 'items')

        response = f"""Excellent! Now, try to recall as many {category} from the list as you can.
Don't worry if you can't remember them all - just do your best!

Type the {category} you remember, separated by commas."""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }

    def _ask_story_questions(self) -> Dict[str, Any]:
        """Ask questions about the story."""
        questions = self.exercise_data.get('questions', [])

        response = f"""Great! Now let me ask you some questions about the story:

{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}

Take your time and answer as many as you can remember!"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }
