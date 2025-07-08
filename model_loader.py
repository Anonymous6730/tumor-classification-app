import gdown
import os
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "brain_tumor_cnn.keras"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?export=download&id=1eZ7v4dQzZ-5JEXi9KFFzDTOzI7w2z_VZ"
        gdown.download(url, model_path, quiet=False)
    
    with open(model_path, "rb") as f:
        model = pickle.loads(f)
    
    return model
