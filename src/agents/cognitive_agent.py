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
            # Restore exercise data from context
            if context.get('exercise_data'):
                self.exercise_data = context['exercise_data']
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

        # Create a visual representation with text labels
        label_map = {
            'grocery_items': '[Shopping]',
            'household_objects': '[Home]',
            'family_activities': '[Family]',
            'garden_items': '[Garden]',
            'musical_instruments': '[Music]'
        }

        self.exercise_data = {
            'type': 'memory_list',
            'category': category,
            'items': items,
            'started': True,
            'num_items': num_items
        }

        category_name = category.replace('_', ' ')
        response = f"""{label_map.get(category, '[Note]')} Memory Challenge: {category_name.title()}

Memorize these {num_items} {category_name}. Tip: {category_data['tip']}

{chr(10).join([f"{i+1}. {item.title()}" for i, item in enumerate(items)])}

Take 30-45 seconds. Type 'ready' when you're done!"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_type': 'memory_list',
            'exercise_state': 'waiting_for_ready',
            'exercise_data': self.exercise_data
        }


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
        """Create an enhanced story recall exercise with detailed narrative incidents."""
        stories = [
            {
                'title': 'The Unexpected Visitor',
                'story': """It was a rainy Tuesday afternoon when Eleanor heard a knock at her front door. She had been reading her favorite mystery novel in the living room, sipping chamomile tea from her blue china cup. When she opened the door, she found a young woman in a bright red raincoat standing on the porch, holding a basket covered with a checkered cloth.

"Hello, Mrs. Patterson," the woman said with a warm smile. "I'm Jennifer, your new neighbor from number 47. I just moved in last week and wanted to introduce myself."

Eleanor invited her in and learned that Jennifer was a baker who had just opened a small shop on Main Street called 'Sweet Memories Bakery'. The basket contained fresh blueberry muffins, still warm from the oven, and a small jar of homemade strawberry jam. They sat at the kitchen table for nearly an hour, chatting about the neighborhood. Jennifer mentioned she had a golden retriever named Charlie who loved to play in the nearby park every morning at 7 AM.

Before leaving, Jennifer wrote her phone number on a yellow sticky note and placed it on Eleanor's refrigerator. "Call me anytime," she said. "I'd love to have you visit the bakery. We open at 6 AM every day except Sundays."

As Jennifer left, Eleanor noticed the rain had stopped and a rainbow stretched across the sky.""",
                'summary_prompt': "Please summarize this story in 3-4 sentences. Include the main characters, what happened, and the key details.",
                'key_points': [
                    "Eleanor received an unexpected visitor on a rainy Tuesday",
                    "Jennifer, a new neighbor from number 47, introduced herself",
                    "Jennifer is a baker who owns Sweet Memories Bakery on Main Street",
                    "She brought blueberry muffins and strawberry jam",
                    "Jennifer has a golden retriever named Charlie",
                    "She left her phone number on a yellow sticky note"
                ],
                'difficulty': 'hard'
            },
            {
                'title': 'The Lost Keys Adventure',
                'story': """On a bright Saturday morning at 9:30 AM, David realized he couldn't find his car keys. He had an important doctor's appointment at 11 o'clock and was starting to panic. He checked his brown leather jacket hanging by the door - no keys. He looked on the kitchen counter where he usually kept them - nothing there either.

His daughter Lisa, who was visiting for the weekend, suggested they retrace his steps from the previous evening. "Dad, you went to the grocery store yesterday around 5 PM, right?" she asked. David remembered stopping at Thompson's Market to buy milk, eggs, and his favorite coffee beans.

They called the store, and the manager, Mr. Garcia, said he'd check the lost and found. While waiting, David checked the front porch and found them right there, sitting on the white wicker chair next to his gardening gloves! He must have set them down when he brought in the groceries yesterday evening.

Relieved, David called the doctor's office to confirm he would make it on time. Lisa laughed and suggested they celebrate with pancakes at the Corner Diner, their favorite breakfast spot. They arrived at the diner at 10 AM, ordered chocolate chip pancakes with maple syrup, and David even had time to read the morning newspaper before his appointment.

The waitress, Mary, who had served them for years, brought extra whipped cream and wished David good luck at his appointment.""",
                'summary_prompt': "Please summarize this story in your own words. What was the main problem and how was it resolved?",
                'key_points': [
                    "David lost his car keys on Saturday morning before an 11 AM appointment",
                    "His daughter Lisa helped him search",
                    "He had been to Thompson's Market the previous evening",
                    "Keys were found on the front porch white wicker chair",
                    "They celebrated at Corner Diner with chocolate chip pancakes",
                    "They arrived at the diner at 10 AM"
                ],
                'difficulty': 'medium'
            },
            {
                'title': 'The Community Picnic',
                'story': """The annual Oak Street Community Picnic was scheduled for the first Sunday in June at Riverside Park. Maria had volunteered to organize the event, which would start at noon and run until 5 PM.

On the day of the picnic, Maria arrived early at 10:30 AM with her husband Roberto. They set up six large blue picnic tables under the oak trees near the playground. Roberto brought his portable grill to make his famous turkey burgers, while Maria prepared her special potato salad with the recipe she inherited from her grandmother.

By 12:30 PM, over forty neighbors had gathered. The Johnson family brought a watermelon and homemade lemonade. Mrs. Chen, the retired teacher from number 28, organized activities for the children, including a three-legged race and a scavenger hunt. The prize for the scavenger hunt was a colorful kite from the local toy store.

Around 2 PM, someone suggested a softball game. They divided into two teams: the "Oak Street Oaks" wearing red bandanas and the "Riverside Runners" wearing blue caps. The game lasted an hour, with the Riverside Runners winning 12 to 9.

As the sun began to set around 4:30 PM, everyone gathered for a group photo by the fountain. Maria's son Tommy, who was 8 years old, had brought his camera and took several pictures. Before leaving, they all agreed to meet again the following month for a movie night in the park, scheduled for the third Friday in July.""",
                'summary_prompt': "Describe what happened at the community picnic. Include who organized it, what activities took place, and any important details.",
                'key_points': [
                    "Oak Street Community Picnic at Riverside Park, first Sunday in June",
                    "Maria organized it with her husband Roberto",
                    "Event from noon to 5 PM, they arrived at 10:30 AM",
                    "Roberto made turkey burgers, Maria made potato salad",
                    "Over forty neighbors attended by 12:30 PM",
                    "Activities included three-legged race and scavenger hunt",
                    "Softball game at 2 PM - Riverside Runners won 12 to 9",
                    "Group photo at fountain around 4:30 PM"
                ],
                'difficulty': 'hard'
            },
            {
                'title': 'The Helpful Neighbor',
                'story': """It was a snowy Wednesday morning in January when 78-year-old Mr. Peterson looked out his window and saw about six inches of fresh snow covering his driveway. He knew shoveling would be difficult with his back pain, but he needed to get to the pharmacy by 2 PM to pick up his prescription.

Just as he was putting on his thick winter coat and wool scarf, the doorbell rang. It was Kevin, the 16-year-old from next door, standing there with a snow shovel in his hand and a friendly smile on his face.

"Good morning, Mr. Peterson! I'm going around offering to shovel driveways for $15. I'm saving up for a new bicycle," Kevin explained, his breath forming little clouds in the cold air.

Mr. Peterson gladly accepted and invited Kevin in for hot chocolate when he finished. Twenty minutes later, Kevin had cleared both the driveway and the front walkway. Mr. Peterson not only paid him $20 (five dollars more than asked), but also gave him a plate of his wife's homemade oatmeal cookies wrapped in aluminum foil.

"My wife Martha used to make these every winter," Mr. Peterson said with a warm smile. "She passed away two years ago, but I still have her recipes."

Kevin thanked him and mentioned he'd be happy to help with snow removal anytime. Before leaving, he also cleared Mr. Peterson's mailbox and sprinkled salt on the front steps to prevent ice from forming. Mr. Peterson made it to the pharmacy on time and felt grateful for such thoughtful neighbors.""",
                'summary_prompt': "Tell me what happened in this story. Who were the main characters and what kind deed took place?",
                'key_points': [
                    "Snowy Wednesday morning in January with six inches of snow",
                    "Mr. Peterson, 78 years old, needed to get to pharmacy by 2 PM",
                    "Kevin, 16-year-old neighbor, offered to shovel for $15",
                    "Kevin cleared driveway and walkway in twenty minutes",
                    "Mr. Peterson paid $20 and gave oatmeal cookies",
                    "Kevin also cleared mailbox and salted front steps"
                ],
                'difficulty': 'medium'
            },
            {
                'title': 'The Farmers Market Discovery',
                'story': """Every Saturday morning, the downtown farmers market opened at 8 AM in the town square. Linda had been walking past it for months but never stopped to browse. This particular Saturday in late September, the smell of fresh apples and cinnamon pulled her in.

The market had twelve vendors set up in two rows. The first booth she visited belonged to a farmer named Jack who grew organic vegetables. He was offering a special deal: three bunches of carrots for five dollars. Linda bought them and also picked up a pound of green beans and two red bell peppers.

At the next booth, a woman named Sophia was selling honey from her backyard beehives. She had five different varieties: wildflower, clover, orange blossom, lavender, and buckwheat. Sophia explained that the lavender honey was perfect for evening tea because it helped with relaxation. Linda purchased a small jar of the lavender honey for eight dollars.

As she continued walking, Linda noticed a musician playing guitar near the fountain at the center of the square. He was performing folk songs, and a small crowd had gathered. A young girl, maybe seven years old, was dancing to the music while wearing a yellow sundress.

Linda's final stop was at a bakery booth run by the Martinez family. They had fresh-baked sourdough bread, chocolate croissants, and peach cobbler. Linda bought two croissants for tomorrow's breakfast and added them to her canvas shopping bag.

By 9:30 AM, Linda had spent a total of twenty-eight dollars but felt it was worth every penny. She made a mental note to return next Saturday and maybe bring her friend Carol along. As she walked home, she thought about how she'd been missing out on this wonderful community tradition all these months.""",
                'summary_prompt': "Summarize Linda's experience at the farmers market. What did she buy and who did she meet?",
                'key_points': [
                    "Linda visited the downtown farmers market on Saturday morning in late September",
                    "Market opened at 8 AM with twelve vendors in two rows",
                    "Bought carrots, green beans, and red bell peppers from Jack",
                    "Purchased lavender honey from Sophia for eight dollars",
                    "Saw a musician playing guitar and a young girl in yellow sundress dancing",
                    "Bought two chocolate croissants from Martinez family bakery",
                    "Spent twenty-eight dollars total and left at 9:30 AM"
                ],
                'difficulty': 'hard'
            }
        ]

        story_data = random.choice(stories)
        self.exercise_data = {
            'type': 'story_recall',
            'story': story_data['story'],
            'title': story_data['title'],
            'summary_prompt': story_data['summary_prompt'],
            'key_points': story_data['key_points'],
            'difficulty': story_data['difficulty']
        }

        difficulty_text = {'easy': 'Easy (*)', 'medium': 'Medium (**)', 'hard': 'Hard (***)'}

        response = f"""[Story] Story Recall: "{story_data['title']}" - {difficulty_text[story_data['difficulty']]}

Read carefully and remember as much detail as you can:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{story_data['story']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Tip] Visualize it like a movie - focus on names, times, and events.

Take 1-2 minutes to read. Type 'ready' when you're done."""

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
                'sequence': ['Blue', 'Red', 'Blue', 'Red', 'Blue', '?'],
                'answer': 'Red',
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
                'sequence': ['Moon', 'Star', 'Star', 'Moon', 'Star', 'Star', '?'],
                'answer': 'Moon',
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

        difficulty_text = {'easy': 'Easy (*)', 'medium': 'Medium (**)', 'hard': 'Hard (***)'}

        response = f"""[Pattern] Pattern Recognition Exercise
Difficulty: {difficulty_text[pattern['difficulty']]}

Look at this sequence and figure out what comes next:

{' '.join(pattern['sequence'])}

[Tip] Hint: This is a {pattern['type']} pattern!

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
                'icon': '[Calendar]'
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
                'icon': '[Person]'
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
                'icon': '[Environment]'
            }
        ]

        question_set = random.choice(questions_sets)

        response = f"""{question_set['icon']} {question_set['title']} Exercise

{question_set['tips']}

Please answer these questions:

{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(question_set['questions'])])}

[Tip] Take your time and answer thoughtfully. These exercises help maintain awareness and connection to the present!"""

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
            exercise_type = self.exercise_data.get('type')
            if exercise_type == 'memory_list':
                return self._ask_memory_recall()
            elif exercise_type == 'story_recall':
                return self._ask_story_questions()
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
            'response': "I'm here to help with the exercise. Type 'ready' when you'd like to continue, or ask for a different exercise!",
            'agent': 'cognitive'
        }

    def _evaluate_response(self, user_response: str) -> Dict[str, Any]:
        """
        Evaluate user's response to the exercise.

        Args:
            user_response: User's answer

        Returns:
            Encouraging feedback and next steps
        """
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

[List] The Original List ({category.replace('_', ' ').title()}):
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

        elif exercise_type == 'pattern_recognition':
            correct_answer = self.exercise_data.get('answer', '').lower().strip()
            user_answer = user_response.lower().strip()
            pattern_type = self.exercise_data.get('pattern_type', '')

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

The correct answer was: {correct_answer} (Pattern: {pattern_type})

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

    def _ask_memory_recall(self) -> Dict[str, Any]:
        """Ask user to recall the memory list."""
        items = self.exercise_data.get('items', [])
        category = self.exercise_data.get('category', 'items')
        num_items = len(items)

        category_display = category.replace('_', ' ')

        response = f"""Perfect! Time to test your memory!

[Important] Don't scroll up - recall from memory alone!

You were shown {num_items} {category_display}. Type the items you remember, separated by commas (any order).

Example: apple, banana, orange"""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }

    def _ask_story_questions(self) -> Dict[str, Any]:
        """Ask user to summarize the story."""
        summary_prompt = self.exercise_data.get('summary_prompt', 'Please summarize the story.')
        title = self.exercise_data.get('title', 'The Story')
        key_points = self.exercise_data.get('key_points', [])

        response = f"""Excellent! Now recall "{title}" without looking back.

{summary_prompt}

Try to include: characters, setting, key events, and specific details you remember.

[Tip] Start with what you remember most clearly."""

        return {
            'response': response,
            'agent': 'cognitive',
            'exercise_state': 'evaluating',
            'exercise_data': self.exercise_data
        }
