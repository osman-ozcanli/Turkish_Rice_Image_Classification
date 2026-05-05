import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

st.set_page_config(page_title="Rice Variety Classifier", page_icon="🌾", layout="centered")

st.title("🌾 Rice Variety Classifier")
st.caption("Base CNN vs Transfer Learning (MobileNetV2) — correct preprocessing")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

# -----------------------------
# UI - model choice
# -----------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.radio(
    "Model",
    ["Base CNN", "Transfer Learning (MobileNetV2)"]
)

CONF_THRESH = st.sidebar.slider("Confidence threshold (%)", 50, 99, 85)

model_path = "rice_varieties.keras" if model_choice == "Base CNN" else "rice_vgg16_transfer.keras"
model = load_model(model_path)

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("📤 Upload a rice image (jpg/png)", type=["jpg", "jpeg", "png"])

def prep_image(pil_img: Image.Image, mode: str) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)

    if mode == "Base CNN":
        arr = arr.astype("float32") / 255.0
    else:
        # MobileNetV2 expects special scaling (typically [-1, 1])
        arr = mb_preprocess(arr.astype("float32"))

    return np.expand_dims(arr, axis=0)

if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

image = Image.open(uploaded)
st.image(image, caption="Uploaded image", use_column_width=True)

x = prep_image(image, model_choice)
probs = model.predict(x, verbose=0)[0]

top1 = int(np.argmax(probs))
top1_name = CLASS_NAMES[top1]
top1_conf = float(probs[top1] * 100)

# Top-2 for sanity
top2 = int(np.argsort(probs)[-2])
top2_name = CLASS_NAMES[top2]
top2_conf = float(probs[top2] * 100)

st.markdown("### ✅ Prediction")
st.success(f"**{top1_name}**")

st.metric("Confidence", f"{top1_conf:.2f}%")

if top1_conf < CONF_THRESH:
    st.warning("Low confidence. Image may be out-of-distribution (lighting/background/zoom). Try a closer crop on the rice grains.")

st.caption(f"Top-2: {top1_name} ({top1_conf:.1f}%) · {top2_name} ({top2_conf:.1f}%)")

st.markdown("### 📊 Probabilities")
for c, p in sorted(zip(CLASS_NAMES, probs), key=lambda t: t[1], reverse=True):
    st.progress(float(p), text=f"{c}: {p*100:.1f}%")
