import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import random

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Load trained model
model = load_model("fer2013_emotion_model.h5",compile = False)

emotion_labels = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Reduce emotions
def reduce_emotion(emotion):
    if emotion in ["Disgust", "Contempt", "Anger"]:
        return "Angry"
    elif emotion in ["Fear", "Surprise"]:
        return "Surprise"
    elif emotion == "Happy":
        return "Happy"
    elif emotion == "Sad":
        return "Sad"
    else:
        return "Neutral"
emoji_map = {
    "Happy": ["😀"],
    "Sad": ["😢","😞"],
    "Angry": ["😠","😡"],
    "Surprise": ["😮"],
    "Neutral": ["😐","😶","😑"]
}

st.title("😊 Face to Emoji Converter")
st.write("Upload a face image to get an emoji")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_full, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]   # first detected face
        face = gray_full[y:y+h, x:x+w]
    else:
        face = gray_full

    face = cv2.resize(face, (64, 64))
    face = face.astype("float32") / 255.0

# IMPORTANT: expand dims ONLY ONCE
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    pred = model.predict(face, verbose=0)
    emotion_idx = int(np.argmax(pred))
    raw_emotion = emotion_labels[emotion_idx]

    emotion = reduce_emotion(raw_emotion)
    emoji = random.choice(emoji_map[emotion])

    st.image(img, caption=f"Detected Emotion: {emotion}")
    st.markdown(
        f"<h1 style='text-align:center'>{emoji}</h1>",
        unsafe_allow_html=True
    )