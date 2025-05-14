# ECG Signal Classification with Custom Neural Network

This project provides a complete workflow for classifying ECG (Electrocardiogram) signals using a custom deep neural network architecture. It is designed to support both educational and production use, with modular notebooks for data preprocessing, experimentation, and final model training.



---

## 📌 Project Objective

Classify ECG signals to identify patterns and anomalies (e.g., arrhythmias) using a custom-designed neural network. The system processes raw ECG waveforms, extracts heartbeat-level segments, and applies a 1D CNN to perform supervised classification.

---

## 🧠 Data Description

- **Source**: ECG signals ( MIT-BIH Arrhythmia Database)
- **Sampling Rate**: ~360 Hz
- **Preprocessing**:
  - Baseline wander and noise removal (e.g., filtering)
  - R-peak detection for segmenting heartbeats
  - Normalization and padding of beats
- **Output**: Labeled heartbeat segments ready for training

---

## 📒 Notebooks Overview

### 1. `Dataset Preparation.ipynb`
- Loads raw ECG data
- Applies filtering and normalization
- Segments data into individual beats using R-peak detection
- Visualizes signals and distributions
- Outputs processed dataset

### 2. `Model Training Experiments.ipynb`
- Tests different training configurations
- Evaluates hyperparameters (batch size, learning rate, etc.)
- Optionally compares model variants
- Includes cross-validation and class balancing

### 3. `Model Training.ipynb`
- Implements a custom CNN architecture
- Trains the model on processed ECG beats
- Evaluates model using accuracy, F1-score, confusion matrix
- Saves trained model for reuse

---

## 🧩 Custom CNN Architecture

The model is a 1D Convolutional Neural Network tailored for time-series ECG data:

- **Conv1D Layers**: Extract localized patterns (e.g., QRS complexes)
- **Batch Normalization + ReLU**: Improve training stability
- **MaxPooling**: Reduce temporal resolution
- **Dropout**: Prevent overfitting
- **Optional BiLSTM Layer**: Capture temporal dependencies
- **Dense Layers**: Fully connected layers for final classification
- **Softmax Output**: Outputs class probabilities

This architecture balances performance and generalization while remaining interpretable and flexible.

---

## 🎯 Use Cases

- **Educational**: A practical example of signal processing, deep learning, and biomedical ML.
- **Production**: Adaptable for real-time ECG monitoring systems or clinical diagnostic support.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please:

- Open issues for bugs or suggestions  
- Submit PRs for feature enhancements or fixes

---

## 📚 References

- [PhysioNet MIT-BIH Arrhythmia Dataset](https://physionet.org/content/mitdb/1.0.0/)
- ECG Classification using CNNs: Acharya et al. (2017), Yildirim et al. (2018), and others.

---


