"""
Fine-tuning pipeline for cognitive agent exercise generation.

This module provides tools to:
1. Prepare training data
2. Fine-tune models using OpenAI's API or local models
3. Evaluate fine-tuned models
4. Deploy improved models to production
"""

import os
import json
import openai
from typing import List, Dict, Optional
import time
from pathlib import Path


class CognitiveFine Tuner:
    """Handles fine-tuning of cognitive exercise generation models."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the fine-tuner.

        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key

        self.training_file_id = None
        self.fine_tuned_model = None

    def prepare_training_data(
        self,
        input_file: str,
        output_file: str = None,
        validate: bool = True
    ) -> str:
        """
        Prepare and validate training data for fine-tuning.

        Args:
            input_file: Path to JSONL training file
            output_file: Optional path to save validated data
            validate: Whether to validate the data format

        Returns:
            Path to prepared training file
        """
        print(f"[Training] Preparing training data from: {input_file}")

        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]

        print(f"[Success] Loaded {len(data)} training examples")

        if validate:
            print("[Validating] Validating data format...")
            self._validate_training_data(data)
            print("[Success] Data validation passed")

        if output_file:
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            print(f"[Success] Saved validated data to: {output_file}")
            return output_file

        return input_file

    def _validate_training_data(self, data: List[Dict]) -> None:
        """
        Validate training data format.

        Args:
            data: List of training examples

        Raises:
            ValueError: If data format is invalid
        """
        required_keys = ['messages']

        for i, example in enumerate(data):
            # Check required keys
            if 'messages' not in example:
                raise ValueError(f"Example {i} missing 'messages' key")

            messages = example['messages']

            # Check messages structure
            if not isinstance(messages, list):
                raise ValueError(f"Example {i}: 'messages' must be a list")

            if len(messages) < 2:
                raise ValueError(f"Example {i}: need at least system + user/assistant")

            # Check message roles
            roles = [msg.get('role') for msg in messages]
            if 'system' not in roles:
                raise ValueError(f"Example {i}: missing system message")

            # Check message content
            for msg in messages:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError(f"Example {i}: message missing role or content")

                if not isinstance(msg['content'], str) or len(msg['content']) == 0:
                    raise ValueError(f"Example {i}: invalid message content")

    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI.

        Args:
            file_path: Path to training JSONL file

        Returns:
            File ID for use in fine-tuning

        Raises:
            Exception: If upload fails
        """
        print(f"[Uploading] Uploading training file to OpenAI...")

        try:
            with open(file_path, 'rb') as f:
                response = openai.files.create(
                    file=f,
                    purpose='fine-tune'
                )

            self.training_file_id = response.id
            print(f"[Success] File uploaded successfully")
            print(f"  File ID: {self.training_file_id}")

            return self.training_file_id

        except Exception as e:
            print(f"[Error] Upload failed: {e}")
            raise

    def start_fine_tuning(
        self,
        training_file_id: str = None,
        model: str = "gpt-3.5-turbo",
        suffix: str = "cognitive-exercises",
        n_epochs: int = 3,
        batch_size: int = None,
        learning_rate_multiplier: float = None
    ) -> str:
        """
        Start fine-tuning job.

        Args:
            training_file_id: ID of uploaded training file
            model: Base model to fine-tune
            suffix: Suffix for fine-tuned model name
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate_multiplier: Learning rate multiplier

        Returns:
            Fine-tuning job ID
        """
        file_id = training_file_id or self.training_file_id

        if not file_id:
            raise ValueError("No training file ID provided")

        print(f"[Starting] Starting fine-tuning job...")
        print(f"  Base model: {model}")
        print(f"  Training file: {file_id}")
        print(f"  Epochs: {n_epochs}")

        try:
            # Create fine-tuning job
            hyperparameters = {"n_epochs": n_epochs}
            if batch_size:
                hyperparameters["batch_size"] = batch_size
            if learning_rate_multiplier:
                hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

            response = openai.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
                suffix=suffix,
                hyperparameters=hyperparameters
            )

            job_id = response.id
            print(f"[Success] Fine-tuning job created")
            print(f"  Job ID: {job_id}")
            print(f"  Status: {response.status}")

            return job_id

        except Exception as e:
            print(f"[Error] Fine-tuning failed to start: {e}")
            raise

    def monitor_fine_tuning(self, job_id: str, check_interval: int = 60) -> str:
        """
        Monitor fine-tuning job progress.

        Args:
            job_id: Fine-tuning job ID
            check_interval: Seconds between status checks

        Returns:
            Fine-tuned model ID when complete
        """
        print(f"[Monitoring] Monitoring fine-tuning job: {job_id}")
        print(f"  Checking status every {check_interval} seconds...")
        print(f"  (This may take 10-30 minutes depending on data size)")

        while True:
            try:
                response = openai.fine_tuning.jobs.retrieve(job_id)
                status = response.status

                print(f"  Status: {status}", end='')

                if status == "succeeded":
                    self.fine_tuned_model = response.fine_tuned_model
                    print(f"\n[Success] Fine-tuning completed successfully!")
                    print(f"  Fine-tuned model: {self.fine_tuned_model}")
                    return self.fine_tuned_model

                elif status == "failed":
                    print(f"\n[Error] Fine-tuning failed")
                    if hasattr(response, 'error'):
                        print(f"  Error: {response.error}")
                    raise Exception("Fine-tuning job failed")

                elif status == "cancelled":
                    print(f"\n[Warning] Fine-tuning was cancelled")
                    raise Exception("Fine-tuning job cancelled")

                else:
                    # Still running
                    if hasattr(response, 'trained_tokens'):
                        print(f" (trained tokens: {response.trained_tokens})", end='')
                    print()  # New line

                time.sleep(check_interval)

            except KeyboardInterrupt:
                print(f"\n[Warning] Monitoring interrupted")
                print(f"  Job is still running. Check status with job ID: {job_id}")
                raise

            except Exception as e:
                if "Fine-tuning job" not in str(e):
                    print(f"\n[Error] Error checking status: {e}")
                raise

    def test_fine_tuned_model(
        self,
        model_id: str = None,
        test_prompts: List[str] = None
    ) -> List[Dict]:
        """
        Test fine-tuned model with example prompts.

        Args:
            model_id: Fine-tuned model ID (uses self.fine_tuned_model if not provided)
            test_prompts: List of test prompts

        Returns:
            List of test results
        """
        model = model_id or self.fine_tuned_model

        if not model:
            raise ValueError("No fine-tuned model ID provided")

        if not test_prompts:
            test_prompts = [
                "Generate a memory list exercise about fruits, difficulty easy",
                "Create a story recall about a birthday party, moderate dementia",
                "Design a pattern recognition with numbers, medium difficulty",
            ]

        print(f"[Testing] Testing fine-tuned model: {model}")
        print()

        results = []

        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}/{len(test_prompts)}: {prompt}")
            print("-" * 70)

            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cognitive therapist specializing in dementia care."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=800
                )

                generated = response.choices[0].message.content
                print(generated)
                print()

                results.append({
                    "prompt": prompt,
                    "response": generated,
                    "model": model
                })

            except Exception as e:
                print(f"[Error] Error: {e}")
                print()
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "model": model
                })

        return results

    def save_model_info(self, output_file: str = "data/fine_tuning/model_info.json"):
        """
        Save fine-tuned model information.

        Args:
            output_file: Path to save model info
        """
        if not self.fine_tuned_model:
            print("[Warning] No fine-tuned model to save")
            return

        model_info = {
            "fine_tuned_model": self.fine_tuned_model,
            "training_file_id": self.training_file_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"[Success] Model info saved to: {output_file}")

    def full_pipeline(
        self,
        training_file: str,
        base_model: str = "gpt-3.5-turbo",
        n_epochs: int = 3,
        test: bool = True
    ) -> str:
        """
        Run complete fine-tuning pipeline.

        Args:
            training_file: Path to training data JSONL
            base_model: Base model to fine-tune
            n_epochs: Number of training epochs
            test: Whether to test the model after training

        Returns:
            Fine-tuned model ID
        """
        print("=" * 70)
        print("COGNITIVE AGENT FINE-TUNING PIPELINE")
        print("=" * 70)
        print()

        # Step 1: Prepare data
        training_file = self.prepare_training_data(training_file)

        # Step 2: Upload
        file_id = self.upload_training_file(training_file)

        # Step 3: Start fine-tuning
        job_id = self.start_fine_tuning(
            training_file_id=file_id,
            model=base_model,
            n_epochs=n_epochs
        )

        # Step 4: Monitor
        model_id = self.monitor_fine_tuning(job_id)

        # Step 5: Test
        if test:
            print()
            self.test_fine_tuned_model(model_id)

        # Step 6: Save info
        self.save_model_info()

        print()
        print("=" * 70)
        print("[Success] FINE-TUNING PIPELINE COMPLETED")
        print("=" * 70)
        print(f"Fine-tuned model: {model_id}")
        print()
        print("Next steps:")
        print("1. Update cognitive_agent.py to use this model")
        print("2. Add model ID to .env file")
        print("3. Test in production environment")

        return model_id


def main():
    """Run fine-tuning pipeline from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune cognitive agent")
    parser.add_argument(
        "--training-file",
        default="data/fine_tuning/cognitive_training_data.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test model after training"
    )

    args = parser.parse_args()

    # Initialize fine-tuner
    fine_tuner = CognitiveFine Tuner()

    # Run pipeline
    fine_tuner.full_pipeline(
        training_file=args.training_file,
        base_model=args.model,
        n_epochs=args.epochs,
        test=args.test
    )


if __name__ == "__main__":
    main()
