"""
Evaluation metrics for cognitive exercise quality.

This module provides tools to evaluate:
1. Exercise appropriateness
2. Clarity and readability
3. Therapeutic value
4. Engagement level
5. Difficulty calibration
"""

from typing import Dict, List, Any
import re
from textblob import TextBlob


class ExerciseEvaluator:
    """Evaluates quality and appropriateness of cognitive exercises."""

    def __init__(self):
        """Initialize evaluator with scoring criteria."""
        self.min_length = 50  # Minimum characters for exercise
        self.max_length = 2000  # Maximum characters
        self.target_readability = 8.0  # Grade level (8th grade = accessible)

    def evaluate_exercise(self, exercise_text: str) -> Dict[str, Any]:
        """
        Comprehensively evaluate an exercise.

        Args:
            exercise_text: The generated exercise text

        Returns:
            Dictionary with scores and feedback
        """
        scores = {
            'length_score': self._evaluate_length(exercise_text),
            'readability_score': self._evaluate_readability(exercise_text),
            'structure_score': self._evaluate_structure(exercise_text),
            'encouragement_score': self._evaluate_encouragement(exercise_text),
            'clarity_score': self._evaluate_clarity(exercise_text),
        }

        # Calculate overall score (weighted average)
        weights = {
            'length_score': 0.15,
            'readability_score': 0.25,
            'structure_score': 0.20,
            'encouragement_score': 0.20,
            'clarity_score': 0.20
        }

        overall_score = sum(scores[k] * weights[k] for k in scores.keys())

        # Generate feedback
        feedback = self._generate_feedback(scores)

        return {
            'overall_score': round(overall_score, 2),
            'scores': scores,
            'feedback': feedback,
            'grade': self._get_grade(overall_score)
        }

    def _evaluate_length(self, text: str) -> float:
        """
        Evaluate text length appropriateness.

        Returns:
            Score from 0-100
        """
        length = len(text)

        if length < self.min_length:
            return max(0, 50 * (length / self.min_length))
        elif length > self.max_length:
            penalty = min(50, (length - self.max_length) / 100)
            return max(0, 100 - penalty)
        else:
            return 100

    def _evaluate_readability(self, text: str) -> float:
        """
        Evaluate readability using Flesch Reading Ease.

        Returns:
            Score from 0-100
        """
        try:
            # Calculate Flesch Reading Ease
            blob = TextBlob(text)
            sentences = len(blob.sentences)
            words = len(blob.words)
            syllables = sum(self._count_syllables(str(word)) for word in blob.words)

            if sentences == 0 or words == 0:
                return 50

            # Flesch Reading Ease formula
            fre = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

            # Convert to 0-100 scale
            # FRE 60-70 = 8th grade (target)
            # FRE 70-80 = 7th grade (good)
            # FRE 80-90 = 6th grade (excellent)
            if fre >= 60:
                return min(100, 60 + (fre - 60) * 1.5)
            else:
                return max(0, fre)

        except Exception as e:
            print(f"Readability evaluation error: {e}")
            return 50

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple heuristic)."""
        word = word.lower()
        vowels = 'aeiou'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1

        return max(1, syllable_count)

    def _evaluate_structure(self, text: str) -> float:
        """
        Evaluate exercise structure and organization.

        Checks for:
        - Clear title/heading
        - Numbered or bulleted lists
        - Sections/breaks
        - Emojis for visual appeal

        Returns:
            Score from 0-100
        """
        score = 0

        # Check for title/heading (usually has ** or # or all caps)
        if re.search(r'\*\*[^*]+\*\*|^[A-Z\s]{10,}', text):
            score += 25

        # Check for lists (numbered or bulleted)
        if re.search(r'^\d+\.|\n\d+\.|- |\* ', text, re.MULTILINE):
            score += 25

        # Check for sections/breaks
        if '━' in text or '---' in text or '\n\n' in text:
            score += 25

        # Check for emojis (visual appeal)
        if re.search(r'[\U0001F300-\U0001F9FF]', text):
            score += 25

        return score

    def _evaluate_encouragement(self, text: str) -> float:
        """
        Evaluate presence of encouraging and supportive language.

        Returns:
            Score from 0-100
        """
        encouraging_phrases = [
            'you can', 'take your time', 'no rush', 'no wrong answer',
            'great job', 'well done', 'excellent', 'good effort',
            'it\'s okay', 'don\'t worry', 'remember', 'important',
            'you\'re doing', 'keep going', 'nice work', 'wonderful'
        ]

        text_lower = text.lower()
        found_count = sum(1 for phrase in encouraging_phrases if phrase in text_lower)

        # Score based on number of encouraging phrases found
        # 0 phrases = 0, 1-2 = 50, 3+ = 100
        if found_count == 0:
            return 0
        elif found_count <= 2:
            return 50
        else:
            return min(100, 50 + (found_count - 2) * 15)

    def _evaluate_clarity(self, text: str) -> float:
        """
        Evaluate clarity and specificity.

        Checks for:
        - Clear instructions
        - Specific details
        - Avoids ambiguity

        Returns:
            Score from 0-100
        """
        score = 0

        # Check for clear instruction words
        instruction_words = ['when', 'type', 'answer', 'share', 'tell', 'describe', 'ready']
        if any(word in text.lower() for word in instruction_words):
            score += 30

        # Check for specific numbers/quantities
        if re.search(r'\d+', text):
            score += 20

        # Check for examples
        if 'example' in text.lower() or 'for instance' in text.lower():
            score += 20

        # Check for clear formatting indicators
        if '💡' in text or 'tip:' in text.lower() or 'hint:' in text.lower():
            score += 30

        return score

    def _generate_feedback(self, scores: Dict[str, float]) -> List[str]:
        """Generate human-readable feedback based on scores."""
        feedback = []

        if scores['length_score'] < 70:
            feedback.append("Consider adjusting exercise length for better engagement")

        if scores['readability_score'] < 60:
            feedback.append("Simplify language for better accessibility")

        if scores['structure_score'] < 50:
            feedback.append("Improve structure with clear sections and lists")

        if scores['encouragement_score'] < 50:
            feedback.append("Add more encouraging and supportive language")

        if scores['clarity_score'] < 60:
            feedback.append("Provide clearer instructions and examples")

        if not feedback:
            feedback.append("Exercise meets quality standards!")

        return feedback

    def _get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Acceptable)'
        elif score >= 60:
            return 'D (Needs Improvement)'
        else:
            return 'F (Poor)'

    def batch_evaluate(
        self,
        exercises: List[str],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate multiple exercises and provide statistics.

        Args:
            exercises: List of exercise texts
            verbose: Whether to print detailed results

        Returns:
            Dictionary with aggregate statistics
        """
        results = []

        for i, exercise in enumerate(exercises, 1):
            result = self.evaluate_exercise(exercise)
            results.append(result)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Exercise {i}/{len(exercises)}")
                print(f"{'='*60}")
                print(f"Overall Score: {result['overall_score']}/100 ({result['grade']})")
                print(f"\nDetailed Scores:")
                for key, value in result['scores'].items():
                    print(f"  {key}: {value:.1f}/100")
                print(f"\nFeedback:")
                for item in result['feedback']:
                    print(f"  • {item}")

        # Calculate statistics
        overall_scores = [r['overall_score'] for r in results]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        return {
            'num_exercises': len(exercises),
            'average_score': round(avg_score, 2),
            'min_score': min(overall_scores) if overall_scores else 0,
            'max_score': max(overall_scores) if overall_scores else 0,
            'results': results
        }


def compare_models(
    base_exercises: List[str],
    finetuned_exercises: List[str]
) -> Dict[str, Any]:
    """
    Compare exercises from base model vs fine-tuned model.

    Args:
        base_exercises: Exercises from base model
        finetuned_exercises: Exercises from fine-tuned model

    Returns:
        Comparison statistics
    """
    evaluator = ExerciseEvaluator()

    print("Evaluating Base Model Exercises...")
    base_results = evaluator.batch_evaluate(base_exercises, verbose=False)

    print("\nEvaluating Fine-Tuned Model Exercises...")
    finetuned_results = evaluator.batch_evaluate(finetuned_exercises, verbose=False)

    improvement = finetuned_results['average_score'] - base_results['average_score']
    improvement_pct = (improvement / base_results['average_score'] * 100) if base_results['average_score'] > 0 else 0

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nBase Model Average Score: {base_results['average_score']:.2f}/100")
    print(f"Fine-Tuned Model Average Score: {finetuned_results['average_score']:.2f}/100")
    print(f"\nImprovement: {improvement:+.2f} points ({improvement_pct:+.1f}%)")

    if improvement > 0:
        print(f"\n✓ Fine-tuned model shows improvement!")
    elif improvement == 0:
        print(f"\n→ Models perform similarly")
    else:
        print(f"\n⚠ Fine-tuned model needs further training")

    return {
        'base_results': base_results,
        'finetuned_results': finetuned_results,
        'improvement': improvement,
        'improvement_percentage': improvement_pct
    }


def main():
    """Example usage of evaluator."""
    evaluator = ExerciseEvaluator()

    # Example exercise
    sample_exercise = """**Memory Challenge: Household Items** 🏠

I'm going to show you 5 common household items:

1. Reading Glasses 👓
2. House Keys 🔑
3. TV Remote 📺
4. Coffee Mug ☕
5. Telephone 📞

💡 **Memory Tip**: Imagine walking through your home and placing each item in a specific room.

Take 30-45 seconds to memorize them. When you're ready, type 'ready'!"""

    print("Evaluating Sample Exercise...")
    result = evaluator.evaluate_exercise(sample_exercise)

    print(f"\nOverall Score: {result['overall_score']}/100 ({result['grade']})")
    print(f"\nDetailed Scores:")
    for key, value in result['scores'].items():
        print(f"  {key}: {value:.1f}/100")
    print(f"\nFeedback:")
    for item in result['feedback']:
        print(f"  • {item}")


if __name__ == "__main__":
    main()
