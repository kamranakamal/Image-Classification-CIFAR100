# CIFAR-100 Image Classification

A deep learning project for classifying images from the CIFAR-100 dataset using EfficientNet-B0 with a Gradio web interface.

## Project Structure

```
CIFAR100/
├── app/
│   ├── gradio_app.py          # Gradio web interface
│   └── classes.txt            # CIFAR-100 class labels
├── src/
│   ├── config.py              # Configuration and transforms
│   ├── effnet_model.py        # EfficientNet model definition
│   ├── model_trainer.py       # Training script
│   ├── Helper.py              # Utility functions
│   └── __init__.py
├── models/
│   └── best_model_*.pth       # Pre-trained model weights
├── data/
│   └── cifar-100-python/      # CIFAR-100 dataset (downloaded)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, CPU supported)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CIFAR100
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Web Interface

Start the Gradio app for interactive predictions:
```bash
python app/gradio_app.py
```

The app will launch at `http://127.0.0.1:7860`

**Features:**
- Upload or draw an image
- Get top-3 predictions with confidence scores
- View prediction time

### Train the Model

To retrain the model on CIFAR-100:
```bash
python src/model_trainer.py
```

The training script:
- Downloads CIFAR-100 if not present
- Uses EfficientNet-B0 with ImageNet pretrained weights
- Applies data augmentation (RandAugment)
- Saves best model based on validation accuracy

**Configuration:**
Edit `src/config.py` to adjust:
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate

## Model Details

- **Architecture**: EfficientNet-B0
- **Pre-training**: ImageNet
- **Classes**: 100 (CIFAR-100)
- **Input Size**: 224×224 RGB images
- **Normalization**: ImageNet mean/std

## Docker

### Run Inference (Gradio Web App)

Build the lightweight CPU-only inference image:

```bash
docker build -t cifar100-app .
docker run -p 7860:7860 cifar100-app
```

Access the app at `http://localhost:7860`

### Train the Model in Docker (with GPU support)

Build the training image with CUDA support:

```bash
docker build -f Dockerfile.train -t cifar100-train .
docker run --gpus all -v $(pwd)/models:/app/models cifar100-train
```

**Note:** Requires NVIDIA Docker runtime for GPU access.

For CPU training:
```bash
docker run -v $(pwd)/models:/app/models cifar100-train
```

Volume mount explanation:
- `-v $(pwd)/models:/app/models` - Save trained models to your local machine

## Dependencies

Key packages (see `requirements.txt`):
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `gradio` - Web interface
- `torchmetrics` - Evaluation metrics
- `Pillow` - Image processing

## Performance

The pre-trained model achieves:
- **Accuracy**: ~82.67%
- **F1-Score**: ~82.64%

## License

MIT
