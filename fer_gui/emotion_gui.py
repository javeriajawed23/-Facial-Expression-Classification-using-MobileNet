import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("best_model.h5")

# Class labels from your training data
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Update if your classes differ

# Prediction function
def predict_emotion(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# GUI
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load image and display
        img = Image.open(file_path)
        img_resized = img.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img_resized)
        panel.config(image=img_tk)
        panel.image = img_tk
        
        # Predict
        emotion, conf = predict_emotion(file_path)
        result_label.config(text=f"Prediction: {emotion}")

# Main window
window = tk.Tk()
window.title("Emotion Detection from Image")
window.geometry("400x400")

panel = tk.Label(window)
panel.pack()

btn = tk.Button(window, text="Upload Image", command=upload_and_predict)
btn.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=20)

window.mainloop()
