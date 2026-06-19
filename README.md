# 🦠 COVID-19 Chest X-Ray Classification Web App

An AI-powered web application that classifies Chest X-Ray images into:

* Covid
* Normal
* Viral Pneumonia

The project uses a MobileNetV2 Deep Learning model trained on Chest X-Ray images and is deployed using FastAPI, Streamlit, Render, and Hugging Face.

## 🚀 Live Demo

### Frontend

[https://YOUR-FRONTEND-URL.onrender.com](https://covid19-xray-frontend.onrender.com)

### Backend API

[https://YOUR-BACKEND-URL.onrender.com](https://covid19-xray-classification-webapp.onrender.com)

### API Documentation

[https://YOUR-BACKEND-URL.onrender.com/docs](https://covid19-xray-frontend.onrender.com/docs)

---

## 📌 Features

* Upload Chest X-Ray Images
* AI-Based Disease Classification
* Prediction Confidence Score
* Interactive Probability Visualization
* Download Prediction Report
* FastAPI REST API
* Streamlit User Interface
* Hugging Face Model Hosting
* Render Cloud Deployment

---

## 🛠️ Tech Stack

### Frontend

* Streamlit
* Plotly
* Pandas
* Requests

### Backend

* FastAPI
* TensorFlow
* NumPy
* Pillow

### Deployment

* Render
* Hugging Face

---

## 🧠 Deep Learning Model

Model: MobileNetV2

Input Size:

* 224 × 224 × 3

Output Classes:

* Covid
* Normal
* Viral Pneumonia

---

## 📂 Project Structure

```text
COVID-19/
│
├── app.py
├── main.py
├── requirements-frontend.txt
├── requirements-backend.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/Ghanshyam6032/covid19-xray-classification-webapp.git
```

Move into project directory

```bash
cd covid19-xray-classification-webapp
```

---

## ▶️ Run Backend

```bash
uvicorn main:app --reload
```

---

## ▶️ Run Frontend

```bash
streamlit run app.py
```

---

## 📊 Model Performance

Accuracy: 71%

Classification Report:

| Class           | Precision | Recall | F1-Score |
| --------------- | --------- | ------ | -------- |
| Covid           | 0.92      | 0.85   | 0.88     |
| Normal          | 1.00      | 0.35   | 0.52     |
| Viral Pneumonia | 0.51      | 0.90   | 0.65     |

---

## 👨‍💻 Author

Ghanshyam Prajapati

GitHub:
https://github.com/Ghanshyam6032

---

## ⚠️ Disclaimer

This application is developed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
