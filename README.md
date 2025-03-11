# Vehicle Classification System

## Overview

This project implements a deep learning-based vehicle classification system that can identify 10 different types of vehicles in images. The system uses a pre-trained ResNet50 model fine-tuned on a custom vehicle dataset and provides both a command-line interface for training and evaluation, as well as a user-friendly Streamlit web application for real-time predictions.

## Features

- **Transfer Learning**: Utilizes a pre-trained ResNet50 model fine-tuned for vehicle classification
- **High Accuracy**: Achieves 96.5% accuracy on the validation set
- **User-friendly Interface**: Interactive Streamlit web application for image uploads and predictions
- **Comprehensive Tools**: Includes scripts for training, evaluation, and inference
- **Model Analysis**: Generates confusion matrices, classification reports, and visualizations

## Vehicle Classes

The model can classify vehicles into the following 10 categories:
1. SUV
2. Bus
3. Family sedan
4. Fire engine
5. Heavy truck
6. Jeep
7. Minibus
8. Racing car
9. Taxi
10. Truck

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended but not required for Streamlit web application)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/<username>/vehicle-classification.git
cd vehicle-classification
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model (if not training your own):
```bash
mkdir -p models
# Download model file to models/best_model_resnet50.pth
# For example, you might use wget or curl here
```

## Usage

### Running the Web Application

To start the Streamlit web application:

```bash
streamlit run app.py
```

Then open your browser and navigate to http://localhost:8501

### Training a Model

To train the model on your own dataset:

```bash
python train.py --data_dir ./dataset --epochs 20 --batch_size 32
```

Additional training options:
```bash
python train.py --help
```

### Evaluating the Model

To evaluate a trained model on test data:

```bash
python test.py --test_dir ./dataset/test --model_file best_model_resnet50.pth
```

For labeled test data with performance metrics:
```bash
python test.py --test_dir ./dataset/test --labeled
```

## Dataset Preparation

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── SUV/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── bus/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── SUV/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each class should have its own folder under the train and validation directories. The test directory can be organized either with or without class folders, depending on whether you're using labeled test data.

## Training Process

The training script uses the following techniques:

1. **Transfer Learning**: Uses a pre-trained ResNet50 model initialized with ImageNet weights
2. **Layer Freezing**: Freezes most of the network layers, only training the final fully connected layer
3. **Learning Rate Scheduling**: Reduces learning rate during training to fine-tune model performance
4. **Early Stopping**: Stops training if validation loss stops improving to prevent overfitting
5. **Best Model Saving**: Saves the model with the highest validation accuracy

## Model Architecture

The architecture is based on ResNet50, a 50-layer deep convolutional neural network with residual connections. The network consists of:

- Initial convolutional and pooling layers
- 4 blocks with multiple bottleneck layers
- Global average pooling layer
- Custom fully connected layer (modified to output 10 classes)

The model uses the following preprocessing:
- Images are resized to 224×224 pixels
- Pixel values are normalized using ImageNet means and standard deviations

## Performance

On the validation dataset, the model achieves:
- Accuracy: 96.5%
- Fast inference time: ~20ms per image on GPU

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ResNet architecture was developed by Microsoft Research
- PyTorch and torchvision for model implementation
- Streamlit for the web application framework
