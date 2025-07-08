import gdown
from tensorflow import keras

url = https://drive.google.com/uc?export=download&id=1eZ7v4dQzZ-5JEXi9KFFzDTOzI7w2z_VZ
output = 'brain_tumor_cnn.keras'
gdown.download(url, output, quiet=False)
model = keras.models.load_model('brain_tumor_cnn.keras')
return model
