# LoRA Demo - Low-Rank Adaptation for Language Models

This project demonstrates the implementation of LoRA (Low-Rank Adaptation) for efficient fine-tuning of large language models using the Hugging Face Transformers library.

## 🚀 Overview

LoRA is a technique that enables efficient fine-tuning of large models by introducing low-rank updates to the model's weights. Instead of updating all parameters, LoRA adds small rank decomposition matrices to existing layers, significantly reducing the number of trainable parameters while maintaining performance.

### Key Features:
- **Efficient Fine-tuning**: Only 0.24% of parameters are trainable (786,432 out of 331,982,848)
- **Weights & Biases Integration**: Automatic experiment tracking and model versioning
- **Docker Support**: Containerized environment for reproducible training
- **Flexible Configuration**: Easy-to-modify training parameters
- **Sample Data Generation**: Automatic creation of training datasets

## 📁 Project Structure

```
Lora-Demo/
├── src/
│   ├── config/
│   │   └── config.py          # LoRA configuration parameters
│   ├── data/
│   │   └── data_processor.py  # Data loading and preprocessing
│   ├── models/
│   │   └── lora_trainer.py    # LoRA training implementation
│   └── utils/                 # Utility functions
├── scripts/
│   └── train.py              # Main training script
├── data/
│   └── train.json            # Training dataset
├── output/                   # Model checkpoints and logs
├── wandb/                    # Weights & Biases logs
├── Dockerfile               # Docker configuration
├── docker-compose.yaml      # Docker Compose setup
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (API keys)
```

## 🛠️ Setup

### Option 1: Local Environment

1. **Create Conda Environment**:
   ```bash
   conda create --prefix ./envs python=3.10 -y
   conda activate ./envs
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases** (Optional):
   ```bash
   echo "WANDB_API_KEY=your_api_key_here" > .env
   ```

### Option 2: Docker (Recommended)

1. **Build and Run with Docker Compose**:
   ```bash
   # Build the image
   docker compose build lora-demo
   
   # Run training
   docker compose up lora-demo
   ```

2. **Run with Jupyter Notebook** (for development):
   ```bash
   docker compose up jupyter
   # Access at http://localhost:8888
   ```

## 🚀 Usage

### Training

#### Using Docker (Recommended):
```bash
# Start training with wandb logging
docker compose up lora-demo

# Run in background
docker compose up -d lora-demo

# View logs
docker compose logs -f lora-demo
```

#### Using GitHub Container Registry:
```bash
# Pull the latest image from GitHub Container Registry
docker pull ghcr.io/jesudianchallapalli/lora-demo:latest

# Run training with the pre-built image
docker run -v $(pwd)/output:/app/output -v $(pwd)/data:/app/data ghcr.io/jesudianchallapalli/lora-demo:latest

# Run inference with the pre-built image
docker run -v $(pwd)/output:/app/output ghcr.io/jesudianchallapalli/lora-demo:latest python scripts/inference.py
```

#### Using Local Environment:
```bash
# Activate environment
conda activate ./envs

# Run training
python scripts/train.py
```

### Testing the Trained Model

After training, you can test how well your model answers questions:

#### Quick Test (Docker):
```bash
# Test if the model works
docker compose run --rm test-model
```

#### Interactive Mode (Docker):
```bash
# Start interactive question-answering
docker compose run --rm inference
```

#### Local Testing:
```bash
# Quick test
python scripts/test_model.py

# Interactive mode
python scripts/inference.py
```

#### Example Usage:
```python
from scripts.inference import LoRAInference
from src.config.config import LoRAConfig

# Load configuration and model
config = LoRAConfig()
inference = LoRAInference(config)

# Ask questions
response = inference.generate_response("What is Machine Learning?")
print(response)
```

## 🔄 GitHub Actions

This repository includes automated workflows for building, testing, and releasing the LoRA model:

### **Available Workflows:**

1. **Docker Hub CI/CD** (`.github/workflows/docker-hub-ci.yml`) - **NEW!**
   - Builds and tests on every push/PR
   - Pushes Docker images to Docker Hub
   - Includes security scanning with Trivy
   - Multi-platform support (amd64, arm64)

2. **Docker Hub Release** (`.github/workflows/release-docker-hub.yml`) - **NEW!**
   - Creates releases when tags are pushed
   - Pushes versioned images to Docker Hub
   - Creates GitHub releases with assets

3. **Security Scan** (`.github/workflows/security-scan.yml`) - **NEW!**
   - Scheduled weekly vulnerability scans
   - Manual trigger support
   - Uploads results to GitHub Security tab

4. **GitHub Container Registry** (`.github/workflows/docker-build.yml`)
   - Builds Docker image on every push/PR
   - Pushes to GitHub Container Registry
   - Supports multiple platforms (amd64, arm64)

5. **Model Testing** (`.github/workflows/test-model.yml`)
   - Tests the model training and inference
   - Uploads test results as artifacts
   - Can be triggered manually

6. **Release** (`.github/workflows/release.yml`)
   - Creates release when a new tag is published
   - Builds and pushes release-specific Docker image
   - Creates downloadable model assets

### **Using Docker Hub Images:**

```bash
# Pull the latest development build from Docker Hub
docker pull your-dockerhub-username/lora-demo:main

# Pull a specific release from Docker Hub
docker pull your-dockerhub-username/lora-demo:v1.0.0

# Pull the latest stable version
docker pull your-dockerhub-username/lora-demo:latest

# Run with Docker Hub image
docker run -v $(pwd)/output:/app/output your-dockerhub-username/lora-demo:latest python scripts/inference.py
```

### **Using GitHub Container Registry Images:**

```bash
# Pull the latest development build
docker pull ghcr.io/jesudianchallapalli/lora-demo:main

# Pull a specific release
docker pull ghcr.io/jesudianchallapalli/lora-demo:v1.0.0

# Run with the automated build
docker run -v $(pwd)/output:/app/output ghcr.io/jesudianchallapalli/lora-demo:main python scripts/inference.py
```

### **Setting up Secrets:**

#### For Docker Hub CI/CD:
1. Go to your repository Settings → Secrets and variables → Actions
2. Add these secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token (not password)

#### For Weights & Biases:
1. Add a new secret named `WANDB_API_KEY`
2. Set the value to your Weights & Biases API key

#### Creating Docker Hub Access Token:
1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Give it a name (e.g., "GitHub Actions")
5. Copy the token and add it as `DOCKERHUB_TOKEN` secret

### **CI/CD Pipeline Features:**

- **Automated Testing**: Runs tests before building images
- **Multi-platform Builds**: Supports both AMD64 and ARM64 architectures
- **Security Scanning**: Automated vulnerability scanning with Trivy
- **Caching**: Uses GitHub Actions cache for faster builds
- **Release Management**: Automatic release creation with assets
- **Notifications**: Success/failure notifications in workflow logs

### Configuration

Modify training parameters in `src/config/config.py`:

```python
@dataclass
class LoRAConfig:
    # Model configuration
    model_name: str = "facebook/opt-350m"
    max_length: int = 512
    batch_size: int = 8
    
    # LoRA parameters
    lora_r: int = 8                    # Rank of LoRA decomposition
    lora_alpha: int = 16               # Scaling factor
    lora_dropout: float = 0.1          # Dropout rate
    
    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Weights & Biases
    use_wandb: bool = True
    
    # Mixed precision
    fp16: bool = False
```

### Data Format

Training data should be in JSON format with `input` and `output` fields:

```json
[
  {
    "input": "What is Machine Learning?",
    "output": "Machine Learning is a subset of artificial intelligence..."
  },
  {
    "input": "Explain neural networks",
    "output": "Neural networks are computational models..."
  }
]
```

## 📊 Monitoring

### Weights & Biases Dashboard

When `use_wandb=True`, training metrics are automatically logged to:
- **Project**:
- **Real-time Metrics**: Loss, learning rate, training progress
- **Model Artifacts**: Checkpoints and final model
- **System Metrics**: GPU/CPU usage, memory consumption

### Local Logs

Training logs are saved in:
- `output/logs/` - Training logs
- `output/final_model/` - Trained model
- `wandb/` - Weights & Biases local files

## 🔧 Advanced Configuration

### GPU Support

To enable GPU training in Docker:

```yaml
# In docker-compose.yaml
services:
  lora-demo:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### Custom Datasets

1. **Replace training data**:
   ```bash
   # Copy your dataset to data/train.json
   cp your_dataset.json data/train.json
   ```

2. **Add evaluation data**:
   ```python
   # In config.py
   eval_file: str = "data/eval.json"
   ```

### Model Selection

Change the base model in `config.py`:

```python
# Smaller model for faster training
model_name: str = "facebook/opt-125m"

# Larger model for better performance
model_name: str = "facebook/opt-1.3b"
```

## 📈 Performance

### Current Configuration:
- **Model**: facebook/opt-350m
- **Trainable Parameters**: 786,432 (0.24%)
- **Total Parameters**: 331,982,848
- **Memory Usage**: ~2GB (CPU training)
- **Training Time**: ~10-15 minutes (3 epochs, 3 samples)

### Optimization Tips:
- **Increase batch size** for better GPU utilization
- **Enable fp16** for faster training on supported hardware
- **Adjust LoRA rank** (`lora_r`) for performance vs. parameter trade-off
- **Use gradient accumulation** for larger effective batch sizes

## 🐛 Troubleshooting

### Common Issues:

1. **Wandb Authentication Error**:
   ```bash
   # Ensure API key is set
   echo "WANDB_API_KEY=your_key" > .env
   ```

2. **Memory Issues**:
   ```python
   # Reduce batch size
   batch_size: int = 4
   
   # Enable gradient checkpointing
   gradient_checkpointing: bool = True
   ```

3. **Docker Build Failures**:
   ```bash
   # Clean build
   docker compose build --no-cache lora-demo
   ```
