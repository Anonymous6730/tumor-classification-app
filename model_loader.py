import gdown, pickle, streamlit as st

def load_model():
    url = "https://drive.google.com/uc?id=Prob_Poly_SVC.pkl"
    gdown.download(url, output="Prob_Poly_SVC.pkl", quiet=True)
    with open("Prob_Poly_SVC.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
