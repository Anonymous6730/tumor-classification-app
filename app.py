import streamlit as st
from PIL import Image
import numpy as np
from model_loader import load_model
import time

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

with st.spinner("Loading model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

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

st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload an MRI image to predict tumor type</h4>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])
CLASSES = ['Glioma 🔴', 'Meningioma 🟡', 'No Tumor 🟢', 'Pituitary 🟣']

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    arr = np.array(image)
    return arr.flatten().reshape(1, -1)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded MRI Image", use_container_width=True)

    if st.button("🧠 Predict Tumor"):
        with st.spinner("Analyzing image..."):
            time.sleep(1)
            processed = preprocess_image(image)
            class_index = model.predict(processed)[0]

        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color:#6A5ACD;">✅ Prediction: {CLASSES[class_index]}</h2>
        </div>
        """, unsafe_allow_html=True)
