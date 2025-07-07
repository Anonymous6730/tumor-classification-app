import gdown
import os
import pickle
import streamlit as st

def load_model():
    model_path = "Prob_Poly_SVC.pkl"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/file/d/1YJaveRTqbWjB-G4dqXm8lDuFGm9Gs4QB/view?usp=drive_link"
        gdown.download(url, model_path, quiet=False)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model