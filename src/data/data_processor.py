import json
from typing import List, Dict, Optional
from datasets import Dataset 
from transformers import PreTrainedTokenizer 
import os

class DataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        """
        Initializes the DataProcessor with a tokenizer and maximum sequence length.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to preprocess the text data.
            max_length (int): Maximum sequence length for tokenized inputs.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, file_path: str) -> Dataset:
        """
        Loads data from a JSON file and converts it into a Hugging Face Dataset.

        Args:
            file_path (str): Path to the JSON file containing the data.

        Returns:
            Dataset: A Hugging Face Dataset object created from the loaded data.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Load JSON data from the file
        return Dataset.from_list(data)  # Convert the data into a Dataset object

    def preprocess_function(self, dataset: Dict) -> Dict:
        """
        Preprocesses the dataset by tokenizing input-output pairs.

        Args:
            dataset (Dict): A dictionary containing 'input' and 'output' keys.

        Returns:
            Dict: A dictionary containing tokenized inputs with keys 'input_ids', 'attention_mask', and 'labels'.
        """
        # Combine input and output into a single string for each example
        texts = [f"Input: {inp}\nOutput: {out}" for inp, out in zip(dataset['input'], dataset['output'])]

        # Tokenize the combined text with padding and truncation
        tokenized_inputs = self.tokenizer(
            texts,
            padding="max_length",  # Pad sequences to the maximum length
            truncation=True,       # Truncate sequences longer than max_length
            max_length=self.max_length,  # Maximum sequence length
            return_tensors="pt"    # Return PyTorch tensors
        )
        # Return a dict with keys 'input_ids', 'attention_mask', and 'labels'
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': tokenized_inputs['input_ids']  # For causal LM, labels are the same as input_ids
        }

    def create_sample_data(self, file_path: str):
        """
        Creates a sample data file with input-output pairs if the file does not exist.
        Args:
            file_path (str): Path to the JSON file to create.
        """
        if not os.path.exists(file_path):
            sample_data = [
                {"input": "What is LoRA?", "output": "LoRA stands for Low-Rank Adaptation, a technique to efficiently fine-tune large models."},
                {"input": "What is the capital of France?", "output": "Paris."}
            ]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2)

    def prepare_dataset(self, file_path: str):
        """
        Loads and preprocesses the dataset from the given file path.
        Args:
            file_path (str): Path to the JSON file containing the data.
        Returns:
            Dataset: A Hugging Face Dataset object with tokenized inputs.
        """
        dataset = self.load_data(file_path)
        tokenized_dataset = self.preprocess_function(dataset)
        # Convert the dict into a Dataset object so it can be indexed by integers
        return Dataset.from_dict(tokenized_dataset)