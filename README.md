# LoRA Demo - Low-Rank Adaptation for Language Models

[![CI/CD Pipeline](https://github.com/jesudianch/LoRA/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/jesudianch/LoRA/actions/workflows/ci-cd-pipeline.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/jesudianch/lora-training-api)](https://hub.docker.com/r/jesudianch/lora-training-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates the implementation of LoRA (Low-Rank Adaptation) for efficient fine-tuning of large language models using the Hugging Face Transformers library.

## 🚀 Overview

LoRA is a technique that enables efficient fine-tuning of large models by introducing low-rank updates to the model's weights. Instead of updating all parameters, LoRA adds small rank decomposition matrices to existing layers, significantly reducing the number of trainable parameters while maintaining performance.

### Key Features:
- **Efficient Fine-tuning**: Only 0.24% of parameters are trainable (786,432 out of 331,982,848)
- **Weights & Biases Integration**: Automatic experiment tracking and model versioning
- **Docker Support**: Containerized environment for reproducible training
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Flexible Configuration**: Easy-to-modify training parameters
- **Sample Data Generation**: Automatic creation of training datasets

## 📁 Project Structure

```
LoRA/
├── .github/
│   └── workflows/
│       └── ci-cd-pipeline.yml    # CI/CD automation
├── src/
│   ├── config/
│   │   └── config.py             # LoRA configuration parameters
│   ├── data/
│   │   └── data_processor.py     # Data loading and preprocessing
│   ├── models/                   # Model implementations
│   └── utils/                    # Utility functions
├── scripts/
│   ├── train.py                  # Main training script
│   ├── test_model.py            # Model testing script
│   └── inference.py             # Inference script
├── data/
│   └── train.json               # Training dataset
├── output/                      # Model checkpoints and logs
├── Dockerfile                   # Docker configuration
├── docker-compose.yaml          # Docker Compose setup
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Setup

### Option 1: Local Environment

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jesudianch/LoRA.git
   cd LoRA
   ```

2. **Create Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker (Recommended)

1. **Using Docker Compose**:
   ```bash
   # Build and run
   docker-compose up --build
   ```

2. **Using Pre-built Image**:
   ```bash
   # Pull from Docker Hub
   docker pull jesudianch/lora-training-api:latest
   
   # Run training
   docker run -v $(pwd)/output:/app/output jesudianch/lora-training-api:latest
   ```

## 🚀 Usage

### Training

#### Using Docker (Recommended):
```bash
# Start training
docker-compose up lora-training

# Run in background
docker-compose up -d lora-training

# View logs
docker-compose logs -f lora-training
```

#### Using Local Environment:
```bash
# Activate environment
source venv/bin/activate

# Run training
python scripts/train.py

# Run with test mode
python scripts/train.py --test-mode
```

### Testing the Model

```bash
# Test the trained model
python scripts/test_model.py

# Run inference
python scripts/inference.py
```

### Interactive Usage:
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

## 🔄 CI/CD Pipeline

This repository includes a comprehensive CI/CD pipeline that automatically:

### **Pipeline Stages:**

1. **🧪 Testing Stage**:
   - Sets up Python 3.11 environment
   - Installs dependencies
   - Runs model tests (`test_model.py`)
   - Validates training process with test mode

2. **🐳 Build & Push Stage**:
   - Builds Docker image with commit hash tagging
   - Pushes to Docker Hub with `latest` and commit-specific tags
   - Uses GitHub Actions caching for faster builds
   - Only runs on main branch pushes

3. **🚀 Deploy Stage**:
   - Deploys the application
   - Shows deployment status and image information

### **Automated Features:**
- ✅ **Automated Testing**: Every push and PR is tested
- ✅ **Docker Image Building**: Multi-stage builds with caching
- ✅ **Version Tagging**: Commit-based and latest tagging
- ✅ **Slack Notifications**: Optional build status notifications
- ✅ **Branch Protection**: Only main branch deployments

### **Available Images:**

```bash
# Latest stable version
docker pull jesudianch/lora-training-api:latest

# Specific commit version
docker pull jesudianch/lora-training-api:abc1234

# Run with specific version
docker run -v $(pwd)/output:/app/output jesudianch/lora-training-api:latest
```

### **Setting up CI/CD:**

#### Required Secrets:
Add these secrets in your GitHub repository settings (Settings → Secrets and variables → Actions):

1. **`DOCKER_USERNAME`**: Your Docker Hub username
2. **`DOCKER_PASSWORD`**: Your Docker Hub access token
3. **`SLACK_WEBHOOK_URL`**: (Optional) For build notifications

#### Creating Docker Hub Access Token:
1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Name it "GitHub Actions" and copy the token
5. Add it as `DOCKER_PASSWORD` secret in GitHub

## ⚙️ Configuration

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
    
    # Weights & Biases (optional)
    use_wandb: bool = False
    
    # Mixed precision
    fp16: bool = False
```

## 📊 Data Format

Training data should be in JSON format with `input` and `output` fields:

```json
[
  {
    "input": "What is Machine Learning?",
    "output": "Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
  },
  {
    "input": "Explain neural networks",
    "output": "Neural networks are computational models inspired by biological neural networks that consist of interconnected nodes (neurons) organized in layers to process and learn from data."
  }
]
```

## 📈 Performance Metrics

### Current Configuration:
- **Model**: facebook/opt-350m
- **Trainable Parameters**: 786,432 (0.24% of total)
- **Total Parameters**: 331,982,848
- **Memory Usage**: ~2GB (CPU training)
- **Training Time**: ~10-15 minutes (3 epochs)

### Optimization Tips:
- **Increase batch size** for better GPU utilization
- **Enable fp16** for faster training on supported hardware
- **Adjust LoRA rank** (`lora_r`) for performance vs. parameter trade-off
- **Use gradient accumulation** for larger effective batch sizes

## 🔧 Advanced Usage

### GPU Support

Enable GPU training in Docker:

```yaml
# In docker-compose.yaml
services:
  lora-training:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### Custom Datasets

1. **Replace training data**:
   ```bash
   cp your_dataset.json data/train.json
   ```

2. **Add validation data**:
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

## 🐛 Troubleshooting

### Common Issues:

1. **Memory Issues**:
   ```python
   # Reduce batch size in config.py
   batch_size: int = 4
   
   # Enable gradient checkpointing
   gradient_checkpointing: bool = True
   ```

2. **Docker Build Failures**:
   ```bash
   # Clean build
   docker-compose build --no-cache
   ```

3. **CI/CD Pipeline Failures**:
   - Check that all required secrets are set
   - Verify Docker Hub credentials
   - Review GitHub Actions logs for specific errors

4. **Model Loading Issues**:
   ```bash
   # Ensure output directory exists
   mkdir -p output
   
   # Check model file permissions
   ls -la output/
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

The CI/CD pipeline will automatically test your changes!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the excellent ML library
- [LoRA Paper](https://arxiv.org/abs/2106.09685) for the original research
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Docker](https://docker.com/) for containerization support

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/jesudianch/LoRA/issues) page
2. Create a new issue with detailed information
3. The CI/CD pipeline logs can help diagnose build problems

---

**Happy Training! 🚀**
