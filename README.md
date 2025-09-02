# Facial-Expression-Classification-using-MobileNet
😀Real-time emotion detection using lightweight deep learning for smarter human-computer interaction.

Introduction

This project develops a facial emotion recognition system using MobileNet, a lightweight Convolutional Neural Network (CNN). The system classifies facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise. It is designed to be efficient and deployable on mobile or embedded devices for real-time emotion recognition.

Facial expression recognition has applications in:

Healthcare: Mood and mental health monitoring

Security: Stress or aggression detection

Education: Adapting content based on student engagement

Robotics & Smart Devices: Emotion-aware interfaces

Features

Real-time facial emotion recognition

Classifies seven basic emotions

Lightweight and fast using MobileNet

GUI for live predictions

Data augmentation to improve model robustness

Dataset

FER-2013 Dataset from Kaggle: Link

Balanced subset: 2,000 images per class (14,000 total)

Images preprocessed: resized to 224×224, converted to RGB, normalized

Augmentation: flipping, zooming, shearing, shifting

Technologies Used

Deep Learning & Computer Vision: TensorFlow, Keras, OpenCV

Data Analysis & Visualization: Matplotlib, Seaborn

Development Platform: Google Colab (GPU-enabled)

Installation

Clone the repository:

git clone <repository_url>


Install required packages:

pip install tensorflow keras opencv-python matplotlib seaborn


Download the dataset and organize images into subfolders by emotion label.

Usage

Load the dataset and preprocess images.

Train the MobileNet-based CNN model.

Use the saved best_model.h5 for predictions.

Run GUI for real-time emotion detection from webcam or image input.

Results

Training Accuracy: 98.11%

Validation Accuracy: 98.83%

High performance across all seven emotion classes

Confusion matrix and ROC curves confirm robustness
