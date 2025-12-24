🚘 License-plate-recognition-using-CNN
📌 Project Overview

This project implements an Automatic Number Plate Recognition (ANPR) system using Convolutional Neural Networks (CNNs).
The system is capable of:

Detecting vehicle license plates from images

Extracting the plate region

Recognizing alphanumeric characters using deep learning

The solution is built and tested in Jupyter Notebook / Google Colab, making it easy to reproduce and extend.

🎯 Objectives

Automate vehicle number plate detection

Use CNNs for accurate character recognition

Demonstrate a real-world application of Computer Vision & Deep Learning

Build a foundation for smart traffic and surveillance systems

🧠 Technologies Used

Python 3

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Jupyter Notebook / Google Colab

🏗️ System Architecture

Input Image

Pre-processing

Grayscale conversion

Noise removal

Edge detection

License Plate Detection

Character Segmentation

CNN-based Character Recognition

Final Plate Text Output

📂 Project Structure
ANPR-using-CNN/
│
├── notebook/
│   └── license_plate_recognition_using_cnn.ipynb
│
├── dataset/
│   ├── train/
│   ├── test/
│
├── models/
│   └── cnn_model.h5
│
├── outputs/
│   └── detected_plates/
│
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone[https://github.com/Anujjadaun97/License-plate-recognition-using-CNN/tree/main]
cd ANPR-using-CNN

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebook
jupyter notebook


Open:

notebook/license_plate_recognition_using_cnn.ipynb

📊 Dataset

Character-level labeled images

Includes digits (0–9) and alphabets (A–Z)

Organized into training and testing folders

📌 Dataset preprocessing and loading is handled inside the notebook.

🧪 Model Details

Model Type: Convolutional Neural Network (CNN)

Loss Function: Categorical Cross-Entropy

Optimizer: Adam

Evaluation Metric: Accuracy

📈 Results

Accurate detection of license plates from vehicle images up to 96%

High recognition accuracy for segmented characters

Works well under controlled lighting and image clarity

🚀 Future Improvements

YOLO / SSD based plate detection

OCR using CRNN or Transformers

Real-time video stream processing

Deployment using Streamlit / Flask

Support for Indian license plate formats

