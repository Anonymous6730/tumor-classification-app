# 🧠 Brain Tumor Classification Web App

This is a Streamlit-based web application for **brain tumor classification** using **MRI images**. The model uses a Support Vector Machine (SVM) classifier trained on the Brain MRI Dataset to identify whether a tumor is present and classify it as one of the following:

- **Glioma 🔴**
- **Meningioma 🟡**
- **No Tumor 🟢**
- **Pituitary 🟣**

> ⚠️ *This project is for educational/demo purposes only and not for medical use.*

---

## 🚀 How It Works

1. Upload an MRI image (JPG/PNG)
2. The image is resized to 128×128 and flattened
3. The SVM model predicts the tumor class
4. The app shows the predicted label and confidence score

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
