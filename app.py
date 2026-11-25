import base64
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Clasificador de reciclaje", page_icon="♻️")

MODEL_DIR = Path("modelo_reciclaje_mobilenet")
MODEL_PATH_H5 = MODEL_DIR / "waste_mobilenet.h5"
MODEL_PATH_KERAS = MODEL_DIR / "waste_mobilenet.keras"
CLASSES_PATH = MODEL_DIR / "history.json"
IMG_SIZE = (224, 224)

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}


def load_class_names() -> list[str]:
    if not CLASSES_PATH.exists():
        st.error(f"No se encontró el archivo de clases: {CLASSES_PATH}")
        st.stop()
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    model_path = None
    for p in (MODEL_PATH_KERAS, MODEL_PATH_H5):
        if p.exists():
            model_path = p
            break

    if model_path is None:
        st.error(f"No se encontró ningún modelo en {MODEL_DIR}/")
        st.stop()

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo ({model_path.name}): {e}")
        st.stop()


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)  # el modelo ya normaliza por dentro
    return np.expand_dims(arr, axis=0)


def predict_probs(model, image: Image.Image) -> np.ndarray:
    return model.predict(prepare_image(image), verbose=0)[0]


def show_image(image: Image.Image, caption: str = ""):
    """Muestra la imagen incrustada en el HTML (no usa el servidor de medios)."""
    preview = image.convert("RGB")
    preview.thumbnail((900, 900))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f"""
        <div style="text-align:center">
            <img src="data:image/jpeg;base64,{b64}"
                 style="max-width:100%; border-radius:8px;">
            <p style="color:gray; font-size:0.85em;">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("♻️ Resiclaje clase de IA-UTH-2026-Brayan Torres")
st.write("Sube una imagen y el modelo intentará identificar el tipo de residuo.")
st.caption("Clases: cartón, vidrio, metal, papel, plástico y basura general.")

class_names = load_class_names()
model = load_model()

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    show_image(image, "Imagen cargada")

    preds = predict_probs(model, image)
    top3_idx = np.argsort(preds)[-3:][::-1]

    principal = class_names[top3_idx[0]]
    st.subheader("Predicción principal")
    st.success(f"{LABELS_ES.get(principal, principal)}: {preds[top3_idx[0]]*100:.2f}%")

    st.subheader("Top 3")
    for idx in top3_idx:
        raw = class_names[idx]
        st.write(f"- {LABELS_ES.get(raw, raw)}: {preds[idx]*100:.2f}%")

    st.subheader("Detalle por clase")
    for i, raw in enumerate(class_names):
        st.progress(float(preds[i]), text=f"{LABELS_ES.get(raw, raw)}: {preds[i]*100:.2f}%")
else:
    st.info("Carga una imagen para comenzar.")
