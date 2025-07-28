import streamlit as st
import torch, timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch import nn

# Class names
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load model
@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load('vit_finetuned_on_glioma_v2.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Streamlit UI ---
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor Classifier")
st.markdown("""
This Vision Transformer (ViT)-based model analyzes brain MRI scans and classifies them into one of four categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

📌 **Note:** This application is for educational and research purposes only. It is not intended for clinical diagnosis or treatment.
""")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess and predict
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    # Display prediction
    st.markdown("### 🧪 Prediction Result")
    st.write(f"**Predicted Class:** {CLASS_NAMES[predicted_class]}")
    st.progress(confidence)
    st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")

    # Warn if confidence is low
    if confidence < 0.75:
        st.warning("⚠️ The model's confidence is relatively low. Please interpret the result cautiously.")

    # Additional info
    st.markdown("✅ This prediction is based on a fine-tuned Vision Transformer (ViT) model.")

    # Class-wise confidence
    with st.expander("🔍 Show class-wise confidence scores"):
        for i, prob in enumerate(probabilities):
            st.write(f"{CLASS_NAMES[i]}: {prob.item()*100:.2f}%")
