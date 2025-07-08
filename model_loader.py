import gdown, pickle, streamlit as st

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?export=download&id=1XbyMmGEEEhgXNxSNBholGERGqViFb5e5"
    gdown.download(url, output="Prob_Poly_SVC.pkl", quiet=True)
    with open("Prob_Poly_SVC.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
