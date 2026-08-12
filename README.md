# 🛡️ DeepShield-MF

### Multidomain Deepfake Detection System using CNNs, Vision Transformers, Frequency Analysis & Multi-Fusion Strategies

## 📌 Project Overview

**DeepShield-MF** is a deep learning-based **multidomain deepfake detection system** designed to identify manipulated and synthetic media across different domains.

The system combines **Convolutional Neural Networks (CNNs)**, **Vision Transformers (ViTs)**, and **frequency-domain analysis** to capture both spatial and frequency-based artifacts introduced by deepfake generation techniques.

Multiple **feature and decision fusion strategies** are used to improve the robustness and generalization of the detection system across different datasets and manipulation methods.

##  Key Features

* Multidomain deepfake detection
* CNN-based spatial feature extraction
* Vision Transformer (ViT) based global feature extraction
* Frequency-domain artifact analysis
* Multiple feature-fusion strategies
* Decision-level fusion
* Support for different deepfake datasets
* Model evaluation across multiple domains

##  System Architecture

```text
                    Input Media
                         │
              ┌──────────┴──────────┐
              │                     │
        Spatial Domain        Frequency Domain
              │                     │
        ┌─────┴─────┐          FFT / DCT
        │           │              │
       CNN          ViT       Frequency Features
        │           │              │
        └─────┬─────┘              │
              │                     │
              └──────────┬──────────┘
                         ↓
                  Feature Fusion
                         ↓
                 Classification Head
                         ↓
                Real / Deepfake
```

##  Methodology

### 1. Spatial Feature Extraction

CNN models are used to capture local spatial artifacts such as:

* Facial inconsistencies
* Texture abnormalities
* Blending artifacts
* Boundary distortions

### 2. Vision Transformer

Vision Transformers analyze relationships between different image patches and capture **global contextual information** that may not be easily detected by conventional CNNs.

### 3. Frequency Analysis

Deepfake generation can introduce abnormal patterns in the frequency domain.

Frequency-based representations such as **FFT** or **DCT** are analyzed to extract additional manipulation-related features.

### 4. Multi-Fusion Strategies

The system experiments with multiple fusion approaches:

```text
CNN Features ─────┐
                  ├──→ Feature Fusion → Classifier
ViT Features ─────┤
                  │
Frequency ────────┘
```

Possible fusion strategies include:

* Early Fusion
* Feature-Level Fusion
* Late Fusion
* Decision-Level Fusion
* Weighted Fusion

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **CNNs**
* **Vision Transformers**
* **FFT / DCT**
* **Deep Learning**

##  Project Structure

```text
DeepShield-MF/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── cnn/
│   ├── vit/
│   └── fusion/
│
├── src/
│   ├── preprocessing/
│   ├── cnn_model.py
│   ├── vit_model.py
│   ├── frequency_analysis.py
│   ├── fusion.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── metrics/
│   └── visualizations/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DeepShield-MF.git
cd DeepShield-MF
```

Install dependencies:

```bash
pip install -r requirements.txt
```

##  Training

Train the model using:

```bash
python src/train.py
```
##  Evaluation

Evaluate the trained model:

```bash
python src/evaluate.py
```

The system can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

##  Objective

The primary objective of **DeepShield-MF** is to develop a robust deepfake detection system that can generalize across **multiple domains, datasets, and manipulation techniques** by combining complementary spatial, global, and frequency-domain representations.

##  Future Improvements

* Audio-visual deepfake detection
* Video-level temporal modeling
* Self-supervised pretraining
* Cross-dataset generalization
* Explainable AI for deepfake detection
* Real-time deepfake detection
* Deployment using FastAPI and Docker
* Integration with web and mobile applications





This is the complete architecture of the whole project


                 ┌─────────────────────┐
                 │     IMAGE / VIDEO   │
                 └──────────┬──────────┘
                            ↓
                    PREPROCESSING
                            ↓
                  Face / Frame Extraction
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   CNN Branch           ViT Branch       Frequency Branch
   ResNet50/             ViT-B/16          FFT + DCT
   EfficientNet
        ↓                   ↓                   ↓
        └───────────────────┬───────────────────┘
                            ↓
                    FEATURE FUSION
                            ↓
                ┌───────────┴───────────┐
                ↓                       ↓
          Feature Fusion          Decision Fusion
                ↓                       ↓
                └───────────┬───────────┘
                            ↓
                     CLASSIFIER
                            ↓
                     Real / Fake
                            ↓
                    Confidence Score
                            ↓
                  ┌─────────┴─────────┐
                  ↓                   ↓
               IMAGE                VIDEO
                  ↓                   ↓
              Grad-CAM            BiLSTM
                                      ↓
                              Temporal Prediction
                                      ↓
                              Video Real/Fake
