# Facial-Expression-Classification-using-MobileNet
😀Real-time emotion detection using lightweight deep learning for smarter human-computer interaction.

📌 Introduction

This project develops a facial emotion recognition system using MobileNet, a lightweight Convolutional Neural Network (CNN). The system classifies facial expressions into seven categories:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

The model is optimized for deployment on mobile and embedded systems, making it suitable for real-time predictions on low-power devices.

Applications:

Healthcare: Emotion monitoring for mental health

Security: Stress/aggression detection in surveillance

Education: Adaptive e-learning based on student engagement

Customer Service & Robotics: Emotion-aware interactions

🛠️ Tech Stack

Language & Frameworks: Python, TensorFlow, Keras

Libraries: OpenCV, Matplotlib, Seaborn, NumPy, Pandas

Dataset: FER-2013 (Kaggle)

Platform: Google Colab (GPU-enabled)

Model Architecture: MobileNet (pre-trained on ImageNet)

📊 Dataset

Subset Used: 2,000 images per class → 14,000 total images

Organization: Images stored in subdirectories per emotion label

Reason for FER-2013: Large, diverse, real-world facial expressions for robust generalization

🔄 Preprocessing & Data Augmentation

Image Preprocessing:

Resized to 224×224 pixels

Grayscale → RGB conversion

Pixel normalization ([0, 1] scale)

Data Augmentation:

Horizontal flipping

Zooming & shearing

Width & height shifting

🧠 Model Architecture

Base Model: MobileNet (pre-trained, frozen layers)

Flatten Layer: Converts feature maps to 1D

Dense Layer: 7 neurons with softmax activation

Optimizer: Adam

Loss Function: Categorical cross-entropy

Training Enhancements: EarlyStopping & ModelCheckpoint

⚙️ Training Configuration

Epochs: 30

Batch Size: 32

Validation Split: 20%

Steps per Epoch: 10

📈 Results

Training Accuracy: 98.11%

Validation Accuracy: 98.83%

Rapid convergence within first few epochs

Strong generalization with minimal overfitting

Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix, Multi-class ROC Curve

Observations:

Slight confusion between similar emotions (e.g., Fear vs Surprise)

Model is highly efficient for real-time FER tasks

🖥️ GUI & Sample Predictions

Graphical interface implemented for real-time emotion detection

Sample Predictions:

Actual: Happy → Predicted: Happy

Actual: Surprise → Predicted: Surprise
