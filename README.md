# 🧠 Brain Tumor Classification Web App (CNN-Based)

This is a **Streamlit-based web application** that classifies brain tumors from MRI images using a **Convolutional Neural Network (CNN)** trained with TensorFlow/Keras.

---

## 🎯 Features

- Upload MRI images (`.jpg`, `.jpeg`, `.png`)
- Predicts 4 tumor classes:
  - **Glioma 🔴**
  - **Meningioma 🟡**
  - **No Tumor 🟢**
  - **Pituitary 🟣**
- Displays the **predicted tumor type** with a **confidence score**
- Built with `Streamlit` + `TensorFlow`

---

## 🧠 How the Model Works

- Input image resized to **128×128**
- Normalized and passed into a **CNN**
- Model outputs softmax probabilities for 4 classes

---

## 📁 Folder Structure

```
brain_tumor_app/
├── app.py                 # Streamlit app
├── requirements.txt        # Required libraries
├── model_loader.py        # Loads trained SVM model
└── README.md              # Project documentation
```

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
streamlit
pillow
numpy
scikit-learn
gdown
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🧪 Model Training Overview

- Model: `sklearn.svm.SVC(kernel="poly", probability=True)`
- Image Size: 128 × 128 RGB
- Input Features: Flattened (49152)
- Dataset: Brain Tumor MRI Images  
  [📁 Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## 📌 Notes

- Input images must be MRI scans
- Model was trained on unnormalized pixel values (0–255)
- Streamlit frontend uses the same preprocessing as training

---

## 📄 License & Disclaimer

This project is provided under the MIT License.

> ⚠️ **Disclaimer:** This tool is for academic demonstration only. It is **not approved for clinical or diagnostic use.**

---

## Acknowledgements

- Dataset: [Masoud Nickparvar – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Streamlit team for their amazing framework

---
