import streamlit as st
from PIL import Image
import numpy as np
import time

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# ---- LOAD CNN MODEL ----
import gdown
from tensorflow import keras

url = "https://drive.google.com/uc?export=download&id=1_1HJawPvoqibkFG2TCR0updNfSwWyFOn"
output = 'brain_tumor_cnn_tfdata.h5'
gdown.download(url, output, quiet=False)
@st.cache_resource
def load_model():
    return keras.models.load_model('brain_tumor_cnn_tfdata.h5')

model = load_model()

# ---- SIDEBAR ----
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")
    st.markdown("""
    Upload an **MRI scan** to detect and classify tumor type.

    **Supported formats:** `.jpg`, `.jpeg`, `.png`

    ### 📋 Tumor Classes:
    - Glioma 🔴
    - Meningioma 🟡
    - No Tumor 🟢
    - Pituitary 🟣

    ⚠️ *For demo/educational use only*
    """)
    st.markdown("---")

# ---- MAIN HEADER ----
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload an MRI image to predict tumor type</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])
CLASSES = ['Glioma 🔴', 'Meningioma 🟡', 'No Tumor 🟢', 'Pituitary 🟣']

# ---- IMAGE PREPROCESSING ----
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    arr = np.array(image).astype(np.float32) / 255.0
    return arr


# ---- PREDICTION LOGIC ----
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded MRI Image",use_container_width=True)

    if st.button("🧠 Predict"):
        with st.spinner("Analyzing image..."):
            time.sleep(1)
            processed = preprocess_image(image)
            processed = np.expand_dims(processed, axis=0)  # (1, 128, 128, 1)
            proba = model.predict(processed)[0]
            class_index = int(np.argmax(proba))
            confidence = float(np.max(proba)) * 100

        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color:#6A5ACD;">✅ Prediction: {CLASSES[class_index]}</h2>
            <h4>Confidence: <span style="color:green;">{confidence:.2f}%</span></h4>
        </div>
        """, unsafe_allow_html=True)
