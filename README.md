# Deepfake Detection using Meso4 and Xception

A deepfake detection system implementing and benchmarking the Meso4 and Xception architectures for identifying manipulated facial videos and images.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Benchmarking](#benchmarking)
- [Requirements](#requirements)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project was developed as part of the EE656 course and focuses on detecting deepfake videos and images using deep learning techniques. The implementation is based on the IEEE research paper **"Towards Benchmarking and Evaluating Deepfake Detection"**, which provides a comprehensive framework for evaluating deepfake detection models.

The implementation includes two state-of-the-art architectures:

1. **Meso4**: A lightweight CNN architecture specifically designed for deepfake detection
2. **Xception**: A deep convolutional neural network with depthwise separable convolutions

The project includes comprehensive benchmarking tools to evaluate model performance across multiple metrics including AUC-ROC, FLOPS, parameter count, perturbation robustness, and inference time, following the benchmarking methodology proposed in the reference paper.

## ✨ Features

- **Face Detection & Extraction**: Automatic face detection and extraction from video frames using face_recognition library
- **Multiple Model Support**: Implementation of both Meso4 and Xception architectures
- **Comprehensive Benchmarking**: Evaluation across multiple metrics:
  - Area Under Curve (AUC) of ROC plots
  - Floating Point Operations Per Second (FLOPS)
  - Model parameter count
  - Robustness to perturbations
  - Inference time analysis
- **Data Pipeline**: Efficient video processing and face extraction pipeline
- **Training & Evaluation Notebooks**: Interactive Jupyter notebooks for training and benchmarking

## 📁 Project Structure

```
Deepfake-detection (EE656)/
│
├── Benchmarking Code/
│   ├── benchmarking.ipynb      # Comprehensive model benchmarking
│   ├── train.ipynb              # Model training notebook
│   ├── classifiers.py           # Model architectures (Meso4, Xception)
│   └── pipeline.py              # Video processing and face extraction
│
├── Report.pdf                   # Project report and findings
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── LICENSE                      # License information
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for training)
- FFmpeg (for video processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Rudraksh919/Benchmarking-Deepfake-Detection.git
cd Benchmarking-Deepfake-Detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows PowerShell: venv\Scripts\Activate.ps1
# On Windows CMD: venv\Scripts\activate.bat
# On macOS/Linux: source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 💻 Usage

### Training Models

Open and run the training notebook:
```bash
jupyter notebook "Benchmarking Code/train.ipynb"
```

The notebook includes:
- Data loading and preprocessing
- Model initialization
- Training loop
- Model saving

### Benchmarking

Run the benchmarking notebook to evaluate models:
```bash
jupyter notebook "Benchmarking Code/benchmarking.ipynb"
```

This will generate:
- ROC curves and AUC scores
- FLOPS calculations
- Parameter counts
- Perturbation analysis
- Inference time measurements

### Using the Face Extraction Pipeline

```python
from pipeline import FaceFinder

# Extract faces from a video
finder = FaceFinder('path/to/video.mp4')
finder.find_faces(resize=0.5, stop=100)

# Access extracted faces
faces = finder.faces
```

### Making Predictions

```python
from classifiers import Meso4, XceptionClassifier
import numpy as np

# Load a model
model = Meso4()
model.load('path/to/weights.h5')

# Make predictions
predictions = model.predict(face_images)
```

## 🧠 Models

### Meso4

A lightweight architecture with 4 convolutional layers designed specifically for mesoscopic properties of deepfakes:
- **Parameters**: ~36K
- **Input Size**: 256x256x3
- **Architecture**: 4 Conv blocks + 2 Dense layers
- **Activation**: ReLU for conv layers, LeakyReLU and Sigmoid for dense layers

### Xception

A deep convolutional neural network based on depthwise separable convolutions:
- **Parameters**: ~22M (when using full architecture)
- **Input Size**: 256x256x3
- **Architecture**: Modified Xception with custom classification head
- **Activation**: Sigmoid output for binary classification

## 📊 Benchmarking

The benchmarking suite evaluates models on:

1. **AUC-ROC**: Area under the receiver operating characteristic curve
2. **FLOPS**: Computational complexity measurement
3. **Parameters**: Total trainable parameters
4. **Perturbation Robustness**: Performance under various image perturbations:
   - Gaussian blur
   - Brightness adjustment
   - Contrast adjustment
   - JPEG compression
5. **Inference Time**: Average prediction time per image

## 📦 Requirements

Key dependencies:
- tensorflow >= 2.4.0
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- face-recognition >= 1.3.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- imageio >= 2.9.0
- imageio-ffmpeg >= 0.4.0
- scipy >= 1.5.0
- Pillow >= 8.0.0

See `requirements.txt` for complete list.

## 📄 References

For detailed information about the research foundation of this project, see [REFERENCES.md](REFERENCES.md).

**Primary Paper**: "Towards Benchmarking and Evaluating Deepfake Detection" (IEEE)

## 📈 Results

Detailed results and analysis can be found in `Report.pdf`, including:
- Comparative performance metrics following the benchmarking framework
- ROC curves and AUC scores
- Confusion matrices
- Computational efficiency analysis (FLOPS and parameters)
- Robustness analysis under perturbations (blur, brightness, contrast, compression)

The evaluation methodology follows the comprehensive benchmarking approach outlined in the IEEE paper "Towards Benchmarking and Evaluating Deepfake Detection".

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course: EE656
- Implementation based on the research paper: **"Towards Benchmarking and Evaluating Deepfake Detection"** (IEEE)
- Face detection powered by the face_recognition library
- TensorFlow/Keras framework for deep learning implementation

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Ensure you have appropriate permissions and comply with ethical guidelines when working with facial recognition and deepfake detection technologies.
