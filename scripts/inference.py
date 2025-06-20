import os
import torch
import sys
import logging
from typing import Optional
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config.config import LoRAConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRAInference:
    def __init__(self, config: LoRAConfig, model_path: Optional[str] = None):
        """
        Initialize the LoRA inference model.
        
        Args:
            config: LoRA configuration
            model_path: Path to the trained model (defaults to output/final_model)
        """
        self.config = config
        self.model_path = model_path or os.path.join(config.output_dir, "final_model")
        
        # Create logs directory
        self.logs_dir = os.path.join(config.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup log file
        self.log_file = os.path.join(self.logs_dir, "qa_log.txt")
        
        # Check if trained model exists
        if not os.path.exists(self.model_path):
            logger.error(f"Trained model not found at {self.model_path}")
            logger.info("Please train the model first using: python scripts/train.py")
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            offload_folder=os.path.join(config.output_dir, "offload") if self.device != "cuda" else None
        )
        
        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from {self.model_path}")
        self.model = PeftModel.from_pretrained(self.base_model, self.model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Q&A logs will be saved to: {self.log_file}")
    
    def save_qa_to_file(self, question: str, answer: str):
        """
        Save question and answer to log file with timestamp.
        
        Args:
            question: The input question
            answer: The model's response
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write(f"{'='*60}\n")
        
        logger.info(f"Q&A saved to: {self.log_file}")
    
    def generate_response(self, question: str, max_length: int = 200) -> str:
        """
        Generate a response for the given question.
        
        Args:
            question: The input question
            max_length: Maximum length of the generated response
            
        Returns:
            Generated response
        """
        if not hasattr(self, 'model'):
            return "Model not loaded. Please check the model path."
        
        # Format the input
        input_text = f"Input: {question}\nOutput:"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "Output:")
        if "Output:" in response:
            response = response.split("Output:")[1].strip()
        
        # Save to file
        self.save_qa_to_file(question, response)
        
        return response
    
    def interactive_mode(self):
        """Run interactive question-answering mode."""
        if not hasattr(self, 'model'):
            logger.error("Model not loaded. Cannot start interactive mode.")
            return
        
        print("\n" + "="*50)
        print("🤖 LoRA Model Interactive Mode")
        print("="*50)
        print("Type your questions and press Enter.")
        print("Type 'quit' or 'exit' to stop.")
        print(f"All Q&A will be saved to: {self.log_file}")
        print("="*50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("🤔 Thinking...")
                response = self.generate_response(question)
                print(f"💡 Answer: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print("❌ Sorry, there was an error generating the response.")

def main():
    """Main function to run inference."""
    logger.info("Starting LoRA inference script")
    
    # Load configuration
    config = LoRAConfig()
    
    # Initialize inference model
    inference_model = LoRAInference(config)
    
    # Check if model was loaded successfully
    if not hasattr(inference_model, 'model'):
        return
    
    # Test with some sample questions
    test_questions = [
        "What is Machine Learning?",
        "Explain the concept of Neural Networks.",
        "What is the difference between supervised and unsupervised learning?"
    ]
    
    print("\n🧪 Testing with sample questions:")
    print("="*50)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        response = inference_model.generate_response(question)
        print(f"💡 Answer: {response}")
        print("-" * 30)
    
    # Start interactive mode
    inference_model.interactive_mode()

if __name__ == "__main__":
    main() 