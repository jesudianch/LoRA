#!/usr/bin/env python3
"""
Quick test script to check if the trained LoRA model is working.
"""

import os
import sys
import torch
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config.config import LoRAConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def save_test_result(question: str, answer: str, config: LoRAConfig):
    """Save test question and answer to a file."""
    logs_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    test_log_file = os.path.join(logs_dir, "test_results.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(test_log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Test Timestamp: {timestamp}\n")
        f.write(f"Test Question: {question}\n")
        f.write(f"Model Answer: {answer}\n")
        f.write(f"{'='*60}\n")
    
    print(f"📝 Test result saved to: {test_log_file}")

def test_model():
    """Test the trained model with a simple question."""
    
    config = LoRAConfig()
    model_path = os.path.join(config.output_dir, "final_model")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print(f"Expected path: {model_path}")
        print("Please train the model first using: python scripts/train.py")
        return False
    
    print("✅ Trained model found!")
    
    try:
        # Load tokenizer
        print("📚 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        print("🤖 Loading base model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Load LoRA adapter
        print("🔧 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Test question
        test_question = "What is Machine Learning?"
        print(f"\n🧪 Testing with question: '{test_question}'")
        
        # Format input
        input_text = f"Input: {test_question}\nOutput:"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        print("🤔 Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Output:" in response:
            answer = response.split("Output:")[1].strip()
        else:
            answer = response
        
        print(f"💡 Model Answer: {answer}")
        
        # Save test result
        save_test_result(test_question, answer, config)
        
        print("\n✅ Model is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model() 