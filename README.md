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
├── requirements.txt       # Required libraries
├── brain_tumor_cnn.keras  # Trained CNN model
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
tensorflow
numpy
gdown
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🧪 Model Training (Summary)

- **Framework**: TensorFlow/Keras  
- **Architecture**:
  - 2x `Conv2D` + `MaxPooling2D`
  - Flatten → Dense → Dropout → Softmax
- **Input Shape**: (128, 128, 3)
- **Output**: 4-class softmax
- **Loss Function**: `categorical_crossentropy`
- **Optimizer**: `adam`
- **Epochs**: 10 (can be increased)
- Dataset: Brain Tumor MRI Images  
  [📁 Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## 📌 Notes

- Input images must be MRI scans

---

## 📄 License & Disclaimer

This project is provided under the MIT License.

> ⚠️ **Disclaimer:** This tool is for academic demonstration only. It is **not approved for clinical or diagnostic use.**

---

## Acknowledgements

- Dataset: [Masoud Nickparvar – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Streamlit team for their amazing framework

---
