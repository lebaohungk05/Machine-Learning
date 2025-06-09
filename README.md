# Emotion Recognition using Deep Learning and Traditional Machine Learning

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Đồ án môn học: CSE703020-1-3-24(N02) - Học Máy**  
**Giảng viên hướng dẫn:** Phạm Tiến Lâm  
**Nhóm thực hiện:**
- Lê Bảo Hưng (Trưởng nhóm)
- Nguyễn Văn Thái
- Nguyễn Quang Hiệp

</div>

---

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Models](#models)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Evaluation](#evaluation)
11. [Conclusions](#conclusions)

---

## 🎯 Introduction

This project implements and compares two different approaches for facial emotion recognition:
- **Deep Learning Approach**: Convolutional Neural Network (CNN)
- **Traditional ML Approach**: Support Vector Machine with HOG features (SVM+HOG)

The system can classify facial expressions into 7 basic emotions: angry, disgust, fear, happy, sad, surprise, and neutral.

---

## 📝 Problem Statement

### Context
Facial emotion recognition is a crucial component in human-computer interaction, psychological analysis, and various AI applications. The challenge lies in accurately identifying emotions from facial expressions, which can vary significantly across individuals and cultures.

### Objectives
- Develop an automated system for emotion classification from facial images
- Compare deep learning vs traditional machine learning approaches
- Achieve high accuracy while maintaining computational efficiency
- Provide insights into model performance and failure cases

### Input/Output Specification
- **Input**: Grayscale facial images (48×48 pixels)
- **Output**: Predicted emotion label from 7 categories

---

## 📊 Dataset

### Overview
- **Source**: Dataset structure similar to FER2013 (Facial Expression Recognition 2013)
- **Format**: Grayscale images, 48×48 pixels
- **Classes**: 7 emotion categories based on Ekman's model
- **Structure**:
  ```
  data/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── sad/
  │   ├── surprise/
  │   └── neutral/
  └── test/
      └── [same structure]
  ```

### Data Preprocessing
1. **Image Loading**: Read images and convert to grayscale
2. **Normalization**: Scale pixel values to [0,1] range
3. **Label Encoding**: One-hot encoding for CNN, integer labels for SVM

---

## 🔬 Methodology

### Literature Review

#### Traditional Methods
- **Feature Extraction**: HOG, LBP, SIFT, Gabor filters
- **Classifiers**: SVM, Random Forest, KNN, Naive Bayes
- **Advantages**: Fast, interpretable, low resource requirements
- **Limitations**: Manual feature engineering, limited performance

#### Deep Learning Methods
- **Architectures**: CNN, ResNet, VGG, DenseNet
- **Techniques**: Transfer learning, attention mechanisms
- **Advantages**: Automatic feature learning, state-of-the-art performance
- **Limitations**: Requires large datasets, computationally intensive

### Our Approach
We implement both paradigms to provide a comprehensive comparison:
1. **CNN**: Custom architecture with regularization techniques
2. **SVM+HOG**: Classical pipeline with optimized hyperparameters

---

## 🤖 Models

### Model 1: Convolutional Neural Network (CNN)

#### Architecture
```
Input (48×48×1)
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv Block 3: Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Conv Block 4: Conv2D(256) → BatchNorm → GlobalAveragePooling
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(7, softmax)
```

#### Key Features
- **Regularization**: L2 penalty (0.001), Dropout (0.25-0.5)
- **Optimization**: Adam optimizer (lr=0.0001)
- **Data Augmentation**: Rotation, shifting, zooming, flipping
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#### Hyperparameters
- Batch size: 32
- Epochs: 100 (with early stopping)
- Learning rate: 0.0001 (adaptive)
- Total parameters: ~500K

### Model 2: SVM with HOG Features

#### Feature Extraction
- **HOG Parameters**:
  - Orientations: 9
  - Pixels per cell: 8×8
  - Cells per block: 2×2
  - Block normalization: L2-Hys
  - Feature vector size: 900 dimensions

#### Classification
- **SVM Configuration**:
  - Kernel: RBF (optimized via GridSearch)
  - C: [0.1, 1, 10, 100]
  - Gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
  - Cross-validation: 5-fold

---

## 📈 Results

### Performance Comparison

| Metric | CNN | SVM+HOG |
|--------|-----|---------|
| **Test Accuracy** | 0.6842 | 0.5124 |
| **Training Time** | 45.2 min | 12.8 min |
| **Model Complexity** | 512K params | 900 features |
| **Inference Speed** | 30 FPS | 120 FPS |

### Per-Emotion Performance

| Emotion | CNN Accuracy | SVM Accuracy | Best Model |
|---------|--------------|--------------|------------|
| Angry | 0.6234 | 0.4821 | CNN |
| Disgust | 0.5912 | 0.3654 | CNN |
| Fear | 0.6421 | 0.4532 | CNN |
| Happy | 0.8765 | 0.7234 | CNN |
| Sad | 0.6123 | 0.4876 | CNN |
| Surprise | 0.7234 | 0.5432 | CNN |
| Neutral | 0.7156 | 0.5287 | CNN |

### Key Findings
1. CNN outperforms SVM+HOG by 17.18% in accuracy
2. Happy emotion is easiest to classify (87.65% CNN, 72.34% SVM)
3. Disgust is most challenging (59.12% CNN, 36.54% SVM)
4. Similar emotions often confused: fear/surprise, sad/neutral
5. SVM trains 3.5× faster but CNN is more accurate

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended for CNN)
- 8GB RAM minimum

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/EMO-AffectNetModel.git
cd EMO-AffectNetModel
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
Place your dataset in the `data/` directory following the structure mentioned above.

---

## 💻 Usage

### Training Models

#### Train CNN Model
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train CNN with data augmentation
- Save model to `models/emotion_model_cnn.h5`
- Generate training plots and reports

#### Train SVM+HOG Model
```bash
python svm_hog.py
```
This will:
- Extract HOG features from images
- Perform GridSearch for optimal parameters
- Save model to `models/emotion_model_svm.pkl`
- Generate confusion matrix and reports

### Real-time Emotion Detection
```bash
python detect.py
```
This will open your webcam and perform real-time emotion detection.

### Model Comparison Analysis
```bash
python analyze_results.py
```
This generates comprehensive comparison reports and visualizations.

---

## 📁 Project Structure

```
EMO-AffectNetModel/
├── data/                      # Dataset directory
│   ├── train/                 # Training images
│   └── test/                  # Test images
├── models/                    # Saved models and results
│   ├── emotion_model_cnn.h5   # Trained CNN model
│   ├── emotion_model_svm.pkl  # Trained SVM model
│   ├── cnn_results.json       # CNN evaluation results
│   └── svm_results.json       # SVM evaluation results
├── plots/                     # Visualizations
│   ├── training_history_*.png # Training curves
│   ├── confusion_matrix_*.png # Confusion matrices
│   └── comparison_report.html # Detailed comparison
├── train.py                   # CNN training script
├── svm_hog.py                # SVM+HOG training script
├── model.py                   # CNN architecture definition
├── utils.py                   # Utility functions
├── detect.py                  # Real-time detection
├── analyze_results.py         # Results analysis
└── requirements.txt           # Dependencies
```

---

## 📊 Evaluation

### Metrics Used
1. **Accuracy**: Overall classification accuracy
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Detailed error analysis

### Evaluation Strategy
- Train/Validation/Test split: 60/20/20
- Stratified sampling to maintain class distribution
- 5-fold cross-validation for SVM hyperparameter tuning
- Early stopping to prevent overfitting

### Visualization
- Training/validation curves
- Confusion matrices with heatmaps
- Per-class performance comparison
- HOG feature visualization

---

## 🎓 Conclusions

### Summary
1. **CNN achieves superior accuracy** (68.42%) compared to SVM+HOG (51.24%)
2. **SVM+HOG offers faster training** and inference with acceptable performance
3. **Data augmentation crucial** for preventing overfitting in CNN
4. **Both models struggle** with similar emotions (fear/surprise, sad/neutral)

### Recommendations

**Use CNN when:**
- Accuracy is paramount
- Sufficient computational resources available
- Real-time performance not critical

**Use SVM+HOG when:**
- Quick prototyping needed
- Limited computational resources
- Model interpretability important
- Training time is constrained

### Future Improvements
1. **Ensemble Methods**: Combine CNN and SVM predictions
2. **Transfer Learning**: Use pre-trained models (VGGFace, FaceNet)
3. **Data Enhancement**: Collect more samples for underrepresented emotions
4. **Advanced Architectures**: Implement attention mechanisms, Vision Transformers
5. **Multi-modal Approach**: Incorporate facial landmarks, audio features

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Thanks to Prof. Phạm Tiến Lâm for guidance
- FER2013 dataset creators
- Open-source community for libraries and tools

---

<div align="center">
  <i>For questions or contributions, please open an issue or submit a pull request.</i>
</div> 