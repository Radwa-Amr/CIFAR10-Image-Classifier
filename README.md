# 🧠 CIFAR-10 Image Classifier (ANN + CNN)

A deep learning project that classifies images from the **CIFAR-10** dataset using a **Convolutional Neural Network (CNN)** built with **TensorFlow Keras**.  
This project also includes a **Streamlit web app** for interactive image predictions.

---

## 📂 Project Structure

'''
CIFAR10-Classifier/
│
├── app.py # Streamlit web app for predictions
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── generate_samples.py # (Optional) Generate one sample image per class
│
├── saved_model/
│ └── cifar10_model.h5 # Trained model file
│
└── src/
├── main.py # Entry point / utility runner
├── model.py # Model architecture definition
├── train.py # Model training script
└── utils.py # Helper functions (class names, preprocessing, etc.)
'''

---

## 🧩 Model Architecture

This CNN model is designed for CIFAR-10 (32×32 RGB images).  
It uses multiple convolutional blocks with batch normalization and dropout for regularization:
'''
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
'''
---
## Dataset Info

The CIFAR-10 dataset contains 60,000 color images (32×32 pixels), divided into 10 categories:

Label | Class Name
------|------------
0 | Airplane
1 | Automobile
2 | Bird
3 | Cat
4 | Deer
5 | Dog
6 | Frog
7 | Horse
8 | Ship
9 | Truck

Each class has 6,000 images (5,000 training + 1,000 testing).

---
## Requirements

To install dependencies, create and activate a virtual environment:
'''
-m venv venv
venv\Scripts\activate  # On Windows
'''

Then install the required packages:

pip install -r requirements.txt
---
## Training the Model

To train or retrain your model:
'''
python -m src.train
'''
'''
This script:

Loads the CIFAR-10 dataset

Normalizes and augments the data

Trains the CNN model

Saves the trained model in saved_model/cifar10_model.h5
'''
---
## Run the Streamlit App

Once your model is trained and saved, launch the web app with:

streamlit run app.py

'''
You’ll see a simple web interface where you can:

Upload 32×32 RGB images

Get a prediction for which class the image belongs to

View the model’s confidence score
'''