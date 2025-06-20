from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LoRAConfig:
    # Model configuration
    model_name: str = "facebook/opt-125m"  # Smaller model for testing
    max_length: int = 256  # Reduced max length
    batch_size: int = 4  # Smaller batch size

    # LoRA (Low-Rank Adaptation) parameters
    lora_r: int = 8  # Rank of the LoRA decomposition
    lora_alpha: int = 16  # Scaling factor for LoRA updates
    lora_dropout: float = 0.1  # Dropout rate for LoRA layers
    lora_target_modules: list = None  # List of target modules to apply LoRA (e.g., specific layers)

    # Training parameters
    num_epochs: int = 1  # Reduced for testing
    learning_rate: float = 2e-5  # Learning rate for the optimizer
    weight_decay: float = 0.01  # Weight decay for regularization
    warmup_steps: int = 10  # Reduced warmup steps

    # Data parameters
    train_file: str = "data/train.json"  # Path to the training dataset
    eval_file: Optional[str] = None  # Path to the evaluation dataset (optional)

    # Output directory for saving model checkpoints and results
    output_dir: str = "output"  # Directory to save model outputs and checkpoints

    # Mixed precision training
    fp16: bool = False  # Use fp16 precision if supported

    # Weights & Biases configuration
    use_wandb: bool = True  # Set to True to enable wandb logging