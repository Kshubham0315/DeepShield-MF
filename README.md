# DeepShield-MF

## Deepfake Video Detection using CNN, Vision Transformer, FFT and BiLSTM

DeepShield-MF is an AI-powered deepfake video detection system that analyzes videos using spatial, frequency-domain, and temporal features to classify content as **Real** or **Fake**.

The project contains two model approaches:

* **ViT Baseline:** Frame-level deepfake detection using Vision Transformer.
* **Hybrid Model:** Video-level detection using ResNet50 + ViT + FFT + BiLSTM with feature fusion.


---

## Overview

Deepfake videos can contain different types of artifacts that may not always be visible in individual frames.

DeepShield-MF analyzes the video from three different perspectives:

1. **Spatial Domain** - Visual features from individual frames.
2. **Frequency Domain** - Frequency artifacts using FFT.
3. **Temporal Domain** - Frame-to-frame relationships using BiLSTM.

These features are combined to improve deepfake detection.

---

## Architecture

The hybrid architecture follows:

```text
Input Video
     |
Frame Extraction
     |
Sequence Builder (224x224)
     |
     +----------------+----------------+
     |                |                |
     v                v                v
   FFT             ResNet50          ViT-B/16
 Branch          Spatial Encoder   Spatial Encoder
     |                |                |
     +----------------+----------------+
                      |
                Feature Fusion
                      |
                BiLSTM Temporal
                   Modeling
                      |
                Temporal Pooling
                      |
                  Classifier
                      |
                +-----+-----+
                |           |
              Real        Fake
```

## Model Components

### 1. ResNet50

ResNet50 is used as a CNN-based spatial feature extractor.

It captures local visual information such as:

* Facial textures
* Local artifacts
* Blending inconsistencies
* Fine-grained visual patterns

### 2. Vision Transformer

The project uses the Vision Transformer architecture:

```text
vit_base_patch16_224
```

ViT processes the image as a sequence of patches and captures global relationships between different regions of the frame.

The model is initialized using ImageNet pretrained weights.

### 3. FFT Frequency Branch

The FFT branch extracts frequency-domain information from video frames.

Deepfake generation techniques can introduce unusual frequency patterns that may not be easily visible in normal RGB images.

The FFT features are therefore combined with spatial features to provide additional information to the classifier.

### 4. BiLSTM Temporal Modeling

A video contains a sequence of frames, and deepfake artifacts can appear as temporal inconsistencies.

The BiLSTM processes the fused features from consecutive frames and learns temporal relationships across the sequence.

---

## Model Variants

### Production Model

The current deployed backend uses a frame-level ViT classifier.

```python
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=2
)
```

Configuration:

* Model: ViT-B/16
* Input Size: 224 × 224
* Number of Classes: 2
* Pretrained Weights: ImageNet
* Prediction Type: Frame-level

### Hybrid Research Model

The latest experimental model uses:

```text
ResNet50 + ViT-B/16 + FFT + BiLSTM
```

Configuration:

```text
Sequence Length = 8
Sequence Stride = 2
Image Size = 224 × 224
Classes = 2
```

---

## Dataset

The model uses a frame-based dataset containing real and fake video frames.

Recommended structure:

```text
data/
└── FF_frames/
    ├── fake/
    │   ├── video001_00001.jpg
    │   ├── video001_00002.jpg
    │   └── ...
    │
    └── real/
        ├── video101_00001.jpg
        ├── video101_00002.jpg
        └── ...
```

Frames are extracted from videos at approximately 1 FPS.

Frame names preserve video identity and frame order.

Example:

```text
video123_00001.jpg
video123_00002.jpg
video123_00003.jpg
```

This ordering is important for temporal sequence modeling.

---

## Data Splitting

The hybrid model performs the train/validation split at the **video level** instead of the individual frame level.

This helps prevent data leakage.

The pipeline is:

```text
Original Videos
      |
      v
Train / Validation Split
      |
      v
Sequence Generation
      |
      v
Model Training
```

This prevents frames from the same source video from appearing in both training and validation sets.

---

## Sequence Generation

The hybrid model creates temporal windows from ordered frames.

Default configuration:

```python
SEQ_LEN = 8
SEQ_STRIDE = 2
```

For example:

```text
Frame 1
Frame 2
Frame 3
Frame 4
Frame 5
Frame 6
Frame 7
Frame 8
       |
       v
Sequence 1
```

With a stride of 2, the next sequence begins after two frames.

---

## Training Pipeline

The training pipeline consists of the following stages:

```text
Video Records
      |
Video-Level Split
      |
Sequence Generation
      |
Frame Preprocessing
      |
CNN + ViT + FFT Feature Extraction
      |
Feature Fusion
      |
BiLSTM Temporal Modeling
      |
Temporal Pooling
      |
Classification
      |
Best Validation Checkpoint
```

Example:

```python
video_records, class_names, class_to_idx = build_video_records(
    DATA_PATH,
    min_frames=SEQ_LEN
)

train_videos, val_videos = split_video_records(
    video_records,
    val_ratio=VAL_RATIO,
    seed=SEED
)

train_samples = make_sequence_samples(
    train_videos,
    seq_len=SEQ_LEN,
    seq_stride=SEQ_STRIDE
)

val_samples = make_sequence_samples(
    val_videos,
    seq_len=SEQ_LEN,
    seq_stride=SEQ_STRIDE
)
```

---

## Optimizer

The hybrid model uses AdamW:

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY
)
```

## Learning Rate Scheduler

The project uses `ReduceLROnPlateau`:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)
```

---

## Training Stability

The training pipeline includes:

* Class-weighted loss
* Label smoothing
* Gradient clipping
* AdamW optimizer
* Learning-rate scheduling
* Early stopping
* Best validation checkpoint
* Frozen pretrained backbones for initial experiments

---

## Evaluation Metrics

The project evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Classification Report

Example:

```python
precision = precision_score(
    labels_all,
    preds_all,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    labels_all,
    preds_all,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    labels_all,
    preds_all,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    labels_all,
    preds_all
)
```

---

## Results

### ViT Baseline

| Metric              |  Result |
| ------------------- | ------: |
| Training Accuracy   | ~89.71% |
| Validation Accuracy | ~87.77% |

The hybrid CNN + ViT + FFT + BiLSTM model is currently an experimental research track under active tuning. Its latest accuracy and F1-score should be taken from the final evaluation results of the training notebook.

---

## Video Prediction

The production backend performs frame-level inference.

The process is:

```text
Input Video
     |
Open Video using OpenCV
     |
Extract Frames
     |
Convert BGR -> RGB
     |
Resize to 224x224
     |
ViT Prediction
     |
Count Real/Fake Frames
     |
Calculate Prediction Percentage
     |
Return Result
```

Simplified implementation:

```python
cap = cv2.VideoCapture(video_path)

real_count = 0
manipulated_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    image = transform(
        Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    real_count += predicted.item() == 0
    manipulated_count += predicted.item() == 1

cap.release()
```

The frontend displays the final real/fake prediction based on the processed frames.

---

## System Architecture

```text
                    React Frontend
                          |
                          v
                   Video Upload
                          |
                          v
                    Flask Backend
                          |
                          v
                    Video Processing
                          |
                          v
                    ViT Model
                          |
                          v
                    Frame Predictions
                          |
                          v
                  Real / Fake Result
```

---

## Project Structure

```text
DeepShield-MF/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│       └── best_vit_model.pth
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── data/
│   └── FF_frames/
│       ├── fake/
│       └── real/
│
├── notebooks/
│   └── frac_df_cnnvit_fft_temporal.ipynb
│
├── docs/
│   └── architecture.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Kshubham0315/DeepShield-MF.git
cd DeepShield-MF
```

### Backend Setup

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

For Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask backend:

```bash
python app.py
```

---

## Frontend Setup

Go to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm start
```

---

## CUDA Verification

To check whether PyTorch can access the GPU:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"
```

Expected output on a CUDA-enabled system:

```text
CUDA Available: True
```

---

## Model Weights

The production model is stored at:

```text
backend/models/best_vit_model.pth
```

The trained model is also available through Hugging Face:

https://huggingface.co/RohRoos84/deepshield-model

The backend can optionally download the model during startup using the `MODEL_URL` environment variable.

---

## Environment Variables

Example:

```env
MODEL_URL=https://huggingface.co/...
PORT=5000
```

Do not commit API keys, tokens, passwords or other sensitive credentials to GitHub.

---

## Technologies Used

### Machine Learning

* Python
* PyTorch
* TorchVision
* timm
* NumPy
* scikit-learn
* Pillow

### Computer Vision

* OpenCV
* ResNet50
* Vision Transformer
* FFT

### Deep Learning

* CNN
* Vision Transformer
* BiLSTM
* Feature Fusion

### Backend

* Flask
* REST API

### Frontend

* Streamlit


---

## Applications

DeepShield-MF can be useful for:

* Deepfake detection
* Video forensics
* Digital media verification
* Computer vision research
* AI security research
* Video classification
* Multimedia analysis
* Deep learning experimentation

---

## Limitations

Deepfake detection is an evolving problem. Model performance can vary depending on:

* Dataset quality
* Video resolution
* Compression
* Lighting conditions
* Face quality
* Deepfake generation technique
* Unseen manipulation methods

Therefore, the prediction should be treated as an AI-assisted result rather than absolute proof.

---

## License

This project is intended for educational, research and experimental purposes.

Shubham Kumar
