import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.keras")

model = load_model()

class_names = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

st.title("AI Image Classifier - CIFAR10")

st.write("Upload an image and the AI will classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).resize((32,32))

    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction")
    st.write(predicted_class)

    probabilities = tf.nn.softmax(prediction[0])

    df = pd.DataFrame({
        "Class": class_names,
        "Probability": probabilities.numpy()
    })

    st.subheader("Class Probabilities")

    st.bar_chart(df.set_index("Class"))
