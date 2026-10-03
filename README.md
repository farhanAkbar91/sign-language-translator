# 🤟 Real-Time Sign Language Translation Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic_Tracking-0075FF?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras_LSTM-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

An end-to-end Computer Vision and Deep Learning pipeline for real-time sign language recognition and translation. Built upon **MediaPipe Holistic** multi-landmark extraction and **LSTM recurrent neural networks (TensorFlow/Keras)**, this architecture performs temporal sequence classification over continuous gestural inputs.

The system is designed with a **language-agnostic architecture**: initially prototyped and validated on Indonesian Sign Language (**BISINDO**), with active extensibility toward large-scale open sign language corpora such as **ASL (American Sign Language / WLASL)** to scale vocabulary coverage and robust continuous gesture translation.

---

## 📌 Architectural Overview

```
Webcam Feed (30 FPS)
       │
       ▼
MediaPipe Holistic Pipeline  ───► Landmark Extraction (Pose, Left Hand, Right Hand)
       │
       ▼
Feature Vector (258 coordinates per frame)
       │
       ▼
Temporal Sequence Buffer (30 consecutive frames)
       │
       ▼
Stacked LSTM Network (TensorFlow/Keras)
       │
       ▼
Softmax Classification & Real-Time Sentence Synthesis
```

### Key Technical Highlights
- **Spatial Landmark Extraction**: Captures 33 pose landmarks (`(x, y, z, visibility)`) and 21 landmark points per hand (`(x, y, z)`), resulting in a compact 258-dimensional feature representation per frame.
- **Temporal Sequence Modeling**: 30-frame sliding window captures movement trajectory, hand dynamics, and facial orientation for fine-grained gesture disambiguation.
- **Lightweight Inference**: Runs comfortably on consumer CPUs / standard webcams without requiring discrete GPU acceleration during inference.

---

## 🚀 Getting Started

### 1. Prerequisites
- **OS**: Windows 10/11
- **Python**: Python 3.11 (`py -3.11`)
- **Hardware**: Standard USB / Integrated Webcam

### 2. Environment Setup
Double-click `setup_env.bat` or run:
```cmd
setup_env.bat
```
This script initializes a clean virtual environment (`.venv`) based on Python 3.11 and installs all necessary dependencies (`opencv-python`, `mediapipe`, `tensorflow`, `scikit-learn`).

### 3. Data Collection Pipeline
To record custom sign language samples:
```cmd
run_collect.bat
```
*or manually:*
```cmd
.venv\Scripts\python collect_data.py
```
- Interactive CLI allows selecting specific target words or capturing complete lexicons sequentially.
- Features automatic countdown timers and visual landmark overlay feedback.

### 4. Model Training
Once sequence data is collected:
```cmd
run_train.bat
```
*or manually:*
```cmd
.venv\Scripts\python train_model.py
```
- Trains a stacked LSTM network and exports the trained model weights to `action_bisindo.h5` and class labels to `actions.npy`.

### 5. Real-Time Translation & Inference
Launch real-time webcam inference:
```cmd
run_predict.bat
```
*or manually:*
```cmd
.venv\Scripts\python predict_realtime.py
```
- Streams webcam video, extracts features in real-time, displays live landmark overlays, and renders recognized sentences directly onto the video interface.
- Press **`q`** to safely terminate the video stream.

---

## 🗺️ Roadmap & Scaling to ASL

- [x] BISINDO Dukcapil domain-specific prototype & landmark extractor.
- [x] Real-time inference visualizer with smoothed sequence buffers.
- [ ] Integration of large-scale open datasets (**WLASL / ASL Alphabet**).
- [ ] Transformer / Conformer-based spatial-temporal architecture for expanded vocabulary (>100 words).
- [ ] Export optimized inference graphs (ONNX / TFLite) for browser and edge deployment.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
