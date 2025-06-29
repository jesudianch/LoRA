import os
import torch
import sys 
import logging
import argparse

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.config.config import LoRAConfig
    from src.models.lora_trainer import LoRATrainer
    from src.data.data_processor import DataProcessor
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LoRA model')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with minimal training')
    args = parser.parse_args()
    
    logger.info("Starting LoRA training script")
    
    # Load configuration
    config = LoRAConfig()
    
    # Modify config for test mode
    if args.test_mode:
        logger.info("Running in test mode - using minimal configuration")
        config.num_epochs = 1
        config.batch_size = 1
        config.max_length = 128
        config.use_wandb = False  # Disable wandb in test mode
        config.warmup_steps = 1
    
    # Create output Directory
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Output directory created at {config.output_dir}")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer for model {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for causal LM
    
    # Initialize data processor
    logger.info("Initializing Data Processor")
    data_processor = DataProcessor(tokenizer, config.max_length)
    
    # Prepare training dataset
    logger.info(f"Preparing training dataset from {config.train_file}")
    os.makedirs(os.path.dirname(config.train_file), exist_ok=True)
    data_processor.create_sample_data(config.train_file)  # Create sample data if not exists
    train_dataset = data_processor.prepare_dataset(file_path=config.train_file) # generate the dataset from the file
    logger.info(f"Training dataset prepared with {len(train_dataset)} samples")

    # If eval_file is provided, prepare evaluation dataset
    eval_dataset = None
    if eval_file := config.eval_file:
        logger.info(f"Preparing evaluation dataset from {eval_file}")
        eval_dataset = data_processor.prepare_dataset(eval_file)
        logger.info(f"Evaluation dataset prepared with {len(eval_dataset)} samples")

    # Initialize LoRA trainer
    logger.info("Initializing LoRA Trainer")
    trainer = LoRATrainer(config)
    
    # Start training
    logger.info("Starting training")
    trainer.train(train_dataset)

    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
    logger.info("LoRA training script finished")