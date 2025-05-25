import streamlit as st
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# === CONFIG ===
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "Rice_Disease_Classification_Model.h5")

# === Kelas ===
class_names = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

# === Load Model ===
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# === Streamlit App ===
st.title("🌾 Paddy Disease Classifier")
st.write("Upload satu gambar daun padi untuk mendeteksi jenis penyakitnya.")

# Upload 1 image
uploaded_file = st.file_uploader("📤 Upload a paddy leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

    # Load & preprocess image (dengan RGB)
    img = load_img(uploaded_file, target_size=IMAGE_SIZE, color_mode="rgb")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]

    # Tampilkan hasil
    st.subheader("🧾 Predicted Disease:")
    st.success(f"🌿 {predicted_label}")

    # Confidence detail sebagai tabel
    with st.expander("🔍 Confidence Scores (Debug)"):
        confidence_scores = {
            "Class": class_names,
            "Confidence (%)": [f"{p * 100:.2f}" for p in prediction[0]]
        }
        df_confidence = pd.DataFrame(confidence_scores)
        st.table(df_confidence)