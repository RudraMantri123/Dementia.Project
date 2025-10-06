"""Image-based cognitive exercises module."""

import os
import io
import base64
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np


class ImageExerciseGenerator:
    """Generates image-based cognitive exercises."""

    def __init__(self, output_dir: str = "data/exercise_images"):
        """
        Initialize image exercise generator.

        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_pattern_recognition_exercise(
        self,
        difficulty: int = 3
    ) -> Dict[str, Any]:
        """
        Generate pattern recognition exercise.

        Args:
            difficulty: Difficulty level (1-5)

        Returns:
            Dictionary with exercise data and image
        """
        # Create image with pattern
        width, height = 600, 400
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)

        # Number of shapes based on difficulty
        num_shapes = 3 + difficulty

        # Generate pattern
        shapes = ['circle', 'square', 'triangle']
        colors = ['red', 'blue', 'green', 'yellow', 'purple']

        pattern = []
        for i in range(num_shapes):
            shape = random.choice(shapes)
            color = random.choice(colors)
            pattern.append({'shape': shape, 'color': color})

        # Draw pattern
        x_offset = 50
        y_center = height // 2

        for i, item in enumerate(pattern):
            x = x_offset + (i * 100)
            self._draw_shape(draw, x, y_center, item['shape'], item['color'])

        # Generate options with one correct continuation
        correct_next = {
            'shape': random.choice(shapes),
            'color': random.choice(colors)
        }

        # Create question
        question = f"What comes next in this pattern?"

        # Save image
        img_path = self._save_exercise_image(image, 'pattern')

        return {
            'type': 'pattern_recognition',
            'difficulty': difficulty,
            'question': question,
            'image_path': img_path,
            'image_base64': self._image_to_base64(image),
            'pattern': pattern,
            'correct_answer': correct_next,
            'explanation': f"The pattern continues with {correct_next['color']} {correct_next['shape']}"
        }

    def generate_memory_matching_exercise(
        self,
        difficulty: int = 3
    ) -> Dict[str, Any]:
        """
        Generate memory matching exercise with images.

        Args:
            difficulty: Difficulty level (1-5)

        Returns:
            Dictionary with exercise data
        """
        # Number of pairs based on difficulty
        num_pairs = 2 + difficulty

        # Common objects for elderly patients
        objects = [
            'apple', 'banana', 'car', 'house', 'tree', 'flower',
            'cup', 'book', 'chair', 'clock', 'phone', 'glasses'
        ]

        selected_objects = random.sample(objects, min(num_pairs, len(objects)))

        # Create image grid
        width, height = 600, 400
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)

        # Draw grid
        cols = 4
        rows = (num_pairs * 2 + cols - 1) // cols
        cell_width = width // cols
        cell_height = height // rows

        # Shuffle positions
        positions = list(range(num_pairs * 2))
        random.shuffle(positions)

        items = selected_objects + selected_objects  # Duplicate for pairs
        random.shuffle(items)

        # Draw items (in memory phase, show all)
        for i, item in enumerate(items):
            row = i // cols
            col = i % cols
            x = col * cell_width + cell_width // 2
            y = row * cell_height + cell_height // 2

            # Draw simple representation
            self._draw_object_representation(draw, x, y, item, cell_width // 2)

        img_path = self._save_exercise_image(image, 'memory')

        return {
            'type': 'memory_matching',
            'difficulty': difficulty,
            'question': "Remember these items. You'll need to match them in the next step.",
            'image_path': img_path,
            'image_base64': self._image_to_base64(image),
            'num_pairs': num_pairs,
            'objects': selected_objects,
            'instructions': "Study this image for 30 seconds, then type 'ready' when you're prepared to match the items."
        }

    def generate_find_difference_exercise(
        self,
        difficulty: int = 3
    ) -> Dict[str, Any]:
        """
        Generate 'find the differences' exercise.

        Args:
            difficulty: Difficulty level (1-5)

        Returns:
            Dictionary with two images and differences
        """
        width, height = 400, 400
        image1 = Image.new('RGB', (width, height), 'white')
        draw1 = ImageDraw.Draw(image1)

        # Draw base scene
        # Sky
        draw1.rectangle([0, 0, width, height//2], fill='lightblue')
        # Ground
        draw1.rectangle([0, height//2, width, height], fill='lightgreen')

        # Add objects
        objects = [
            {'type': 'circle', 'pos': (100, 100), 'size': 40, 'color': 'yellow'},  # Sun
            {'type': 'rectangle', 'pos': (200, 250), 'size': (60, 80), 'color': 'brown'},  # House
            {'type': 'circle', 'pos': (350, 150), 'size': 30, 'color': 'white'}  # Cloud
        ]

        for obj in objects:
            self._draw_object(draw1, obj)

        # Create second image with differences
        image2 = image1.copy()
        draw2 = ImageDraw.Draw(image2)

        # Add differences based on difficulty
        num_differences = min(difficulty, 5)
        differences = []

        if num_differences >= 1:
            # Change sun color
            draw2.ellipse([100-40, 100-40, 100+40, 100+40], fill='orange')
            differences.append("Sun color changed from yellow to orange")

        if num_differences >= 2:
            # Move cloud
            draw2.ellipse([300-30, 150-30, 300+30, 150+30], fill='white')
            differences.append("Cloud moved to the left")

        if num_differences >= 3:
            # Add a tree
            draw2.rectangle([320, 280, 340, 320], fill='brown')
            draw2.ellipse([310, 260, 350, 290], fill='green')
            differences.append("Tree added on the right")

        img1_path = self._save_exercise_image(image1, 'diff1')
        img2_path = self._save_exercise_image(image2, 'diff2')

        return {
            'type': 'find_differences',
            'difficulty': difficulty,
            'question': f"Find {num_differences} differences between these two images.",
            'image1_path': img1_path,
            'image2_path': img2_path,
            'image1_base64': self._image_to_base64(image1),
            'image2_base64': self._image_to_base64(image2),
            'num_differences': num_differences,
            'differences': differences
        }

    def generate_sequence_exercise(
        self,
        difficulty: int = 3
    ) -> Dict[str, Any]:
        """
        Generate sequence ordering exercise with images.

        Args:
            difficulty: Difficulty level (1-5)

        Returns:
            Dictionary with exercise data
        """
        # Common sequences
        sequences = [
            ['morning', 'afternoon', 'evening', 'night'],
            ['seed', 'sprout', 'plant', 'flower'],
            ['baby', 'child', 'adult', 'elder'],
            ['spring', 'summer', 'fall', 'winter']
        ]

        selected_sequence = random.choice(sequences)

        # Limit based on difficulty
        sequence_length = min(3 + (difficulty // 2), len(selected_sequence))
        sequence = selected_sequence[:sequence_length]

        # Shuffle for user to reorder
        shuffled = sequence.copy()
        random.shuffle(shuffled)

        # Create image showing shuffled sequence
        width, height = 600, 200
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)

        cell_width = width // len(shuffled)

        for i, item in enumerate(shuffled):
            x = i * cell_width + cell_width // 2
            y = height // 2
            self._draw_text(draw, x, y, item, font_size=24)

        img_path = self._save_exercise_image(image, 'sequence')

        return {
            'type': 'sequence_ordering',
            'difficulty': difficulty,
            'question': "Put these items in the correct order:",
            'image_path': img_path,
            'image_base64': self._image_to_base64(image),
            'shuffled_sequence': shuffled,
            'correct_sequence': sequence,
            'instructions': "Type the items in the correct order, separated by commas."
        }

    def _draw_shape(self, draw: ImageDraw, x: int, y: int, shape: str, color: str, size: int = 40):
        """Draw a shape at specified position."""
        if shape == 'circle':
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color, outline='black')
        elif shape == 'square':
            draw.rectangle([x-size, y-size, x+size, y+size], fill=color, outline='black')
        elif shape == 'triangle':
            points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
            draw.polygon(points, fill=color, outline='black')

    def _draw_object(self, draw: ImageDraw, obj: Dict[str, Any]):
        """Draw an object based on its specification."""
        if obj['type'] == 'circle':
            x, y = obj['pos']
            size = obj['size']
            draw.ellipse([x-size, y-size, x+size, y+size], fill=obj['color'], outline='black')
        elif obj['type'] == 'rectangle':
            x, y = obj['pos']
            w, h = obj['size']
            draw.rectangle([x, y, x+w, y+h], fill=obj['color'], outline='black')

    def _draw_object_representation(self, draw: ImageDraw, x: int, y: int, obj_name: str, size: int):
        """Draw simple representation of an object."""
        # Simple colored circles with text labels
        colors = {
            'apple': 'red', 'banana': 'yellow', 'car': 'blue',
            'house': 'brown', 'tree': 'green', 'flower': 'pink',
            'cup': 'white', 'book': 'beige', 'chair': 'brown',
            'clock': 'gray', 'phone': 'black', 'glasses': 'gray'
        }

        color = colors.get(obj_name, 'gray')
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color, outline='black', width=2)

        # Draw label
        self._draw_text(draw, x, y, obj_name[:3], font_size=16)

    def _draw_text(self, draw: ImageDraw, x: int, y: int, text: str, font_size: int = 20):
        """Draw text at specified position."""
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.text((x - text_width//2, y - text_height//2), text, fill='black', font=font)

    def _save_exercise_image(self, image: Image.Image, prefix: str) -> str:
        """Save exercise image and return path."""
        import uuid
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        image.save(filepath)
        return filepath

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
