import os
from pathlib import Path
import gdown
import streamlit as st
import tensorflow as tf

# =========================================================
# SAYFA VE MODEL AYARLARI
# =========================================================
st.set_page_config(
    page_title="Vida Kusur Tespiti", page_icon="🔩", layout="wide"
)

st.title("🔩 Vida Yüzey Kusuru & Anomali Tespit Sistemi")
st.caption("ResNet50 Patch-Based Anomali & Isı Haritası Çıkarım Arayüzü")

BEST_MODEL_PATH = Path("best_resnet50_nonoverlap_64.keras")

# GOOGLE DRIVE PAYLAŞIM LİNKİNİ BURAYA YAPIŞTIR:
GDRIVE_URL = "https://drive.google.com/file/d/1QbZWNr2MZErGOwDmrEsrgoXLze2atV34/view?usp=sharing"  # <-- Kendi linkini yaz


@st.cache_resource
def load_keras_model():
    # Model klasörde yoksa otomatik Drive'dan indirir
    if not BEST_MODEL_PATH.exists():
        with st.spinner(
            "Model dosyası Google Drive'dan indiriliyor (Sadece ilk açılışta 1-2 dk sürer)..."
        ):
            gdown.download(
                url=GDRIVE_URL,
                output=str(BEST_MODEL_PATH),
                quiet=False,
                fuzzy=True,
            )
    return tf.keras.models.load_model(BEST_MODEL_PATH)


try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Model yüklenemedi. Hata: {e}")
    st.stop()
