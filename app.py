import os
from pathlib import Path
import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# =========================================================
# SAYFA VE MODEL AYARLARI
# =========================================================
st.set_page_config(
    page_title="Vida Kusur Tespiti", page_icon="🔩", layout="wide"
)

st.title("🔩 Vida Yüzey Kusuru & Patch-Based Anomali Tespit Sistemi")
st.caption(
    "ResNet50 64x64 Non-Overlapping ROI Patch & Top-3 Aggregation"
)

BEST_MODEL_PATH = Path("best_resnet50_nonoverlap_64.keras")
GDRIVE_FILE_ID = "1QbZWNr2MZErGOwDmrEsrgoXLze2atV34"

# =========================================================
# EĞİTİM DOSYASINDAKİ SABİTLER
# =========================================================
PATCH_SIZE = 64
STRIDE = 64
MODEL_INPUT_SIZE = 224
SCREW_BBOX_PAD = 12
MIN_SCREW_RATIO = 0.20
PATCH_THRESHOLD = 0.615
IMAGE_THRESHOLD = 0.809


@st.cache_resource
def load_keras_model():
    if not BEST_MODEL_PATH.exists():
        with st.spinner("Model dosyası indiriliyor..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, str(BEST_MODEL_PATH), quiet=False)
    return tf.keras.models.load_model(BEST_MODEL_PATH)


try:
    model = load_keras_model()
    st.sidebar.success("✅ Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenemedi. Hata: {e}")
    st.stop()


# =========================================================
# HELPER FONKSİYONLAR (ROI VE PATCH EXTRACTION)
# =========================================================
def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = (labels == largest_label).astype(np.uint8)
    return largest

def create_screw_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    candidates = []
    for candidate in [th, 255 - th]:
        candidate = (candidate > 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidate = keep_largest_component(candidate)
        ratio = candidate.mean()
        if 0.01 <= ratio <= 0.80:
            candidates.append((candidate, ratio))
            
    if len(candidates) == 0:
        screw_mask = keep_largest_component(255 - th)
        return screw_mask.astype(np.uint8)

    candidates = sorted(candidates, key=lambda x: x[1])
    screw_mask = candidates[0][0]
    return screw_mask.astype(np.uint8)

def get_bbox_from_mask(mask, pad=0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(w, xs.max() + pad + 1)
    y2 = min(h, ys.max() + pad + 1)
    return x1, y1, x2, y2

def extract_and_prepare_patches(image_np):
    screw_mask = create_screw_mask(image_np)
    bbox = get_bbox_from_mask(screw_mask, pad=SCREW_BBOX_PAD)

    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    image_roi = image_np[y1:y2, x1:x2]
    screw_roi = screw_mask[y1:y2, x1:x2]

    h, w = image_roi.shape[:2]
    new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
    new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
    pad_h = new_h - h
    pad_w = new_w - w

    image_pad = np.pad(image_roi, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    screw_pad = np.pad(screw_roi, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    h_pad, w_pad = image_pad.shape[:2]
    patches = []
    coords = []

    for y in range(0, h_pad, STRIDE):
        for x in range(0, w_pad, STRIDE):
            screw_patch = screw_pad[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            screw_pixels = int(screw_patch.sum())
            screw_ratio = screw_pixels / (PATCH_SIZE * PATCH_SIZE)

            # Arka plan oranı düşük olan yamaları atla
            if screw_ratio < MIN_SCREW_RATIO:
                continue

            img_patch = image_pad[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Modeli eğitirken kullanılan 224x224 boyutuna getir (0-255 değer aralığı korunur)
            img_patch_pil = Image.fromarray(img_patch)
            img_224 = img_patch_pil.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), resample=Image.BILINEAR)
            
            patches.append(np.array(img_224, dtype=np.float32))
            # Kutuları orjinal resim üzerinde çizebilmek için global x,y değerleri
            coords.append((x1 + x, y1 + y))

    if not patches:
        return None, None
        
    return np.array(patches), coords


# =========================================================
# ARAYÜZ VE ÇIKARIM
# =========================================================
st.sidebar.header("⚙️ Ayarlar & Dosya Yükleme")

uploaded_file = st.sidebar.file_uploader(
    "Test Edilecek Vida Görselini Seçin", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Yüklenen Orijinal Görsel")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Analiz Sonucu")

        with st.spinner("Yamalar (Patches) çıkarılıyor ve analiz ediliyor..."):
            patches, coords = extract_and_prepare_patches(image_np)

            if patches is None or len(patches) == 0:
                st.warning("Bu görselde incelenecek uygun vida bölgesi bulunamadı.")
            else:
                # Modele pikseller verilir
                probs = model.predict(patches, verbose=0).flatten()

                # Top-3 Aggregation hesaplama
                sorted_probs = np.sort(probs)[::-1]
                topk_mean = sorted_probs[:min(3, len(sorted_probs))].mean()
                
                # Resim seviyesindeki karar (Image Level Threshold)
                is_defective = topk_mean >= IMAGE_THRESHOLD

                if is_defective:
                    st.error("⚠️ **HATALI / KUSURLU VİDA TESPİT EDİLDİ**")
                    st.metric("Top-3 Ortalama Anomali Skoru", f"{topk_mean:.4f}")
                else:
                    st.success("✅ **NORMAL / SAĞLAM VİDA**")
                    st.metric("Top-3 Ortalama Anomali Skoru", f"{topk_mean:.4f}")

                st.caption(f"Analiz Edilen Toplam Yama Sayısı: `{len(patches)}`")

                # Kusurlu yamaların görselleştirilmesi
                st.divider()
                st.subheader("🗺️ Kusur Tespit Haritası")

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_np)

                # Sadece vida hatalıysa çizim yap
                if is_defective:
                    # 1. Tüm resim boyutunda siyah bir maske oluştur
                    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                    has_defect_patch = False
                    
                    # 2. Eşiği geçen yamaları maske üzerinde beyaza (255) boya
                    for i, (x, y) in enumerate(coords):
                        score = probs[i]
                        if score >= PATCH_THRESHOLD:
                            has_defect_patch = True
                            mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255
                            
                    if has_defect_patch:
                        # 3. Birbirine bağlı olan alanları (connected components) bul
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                        
                        if num_labels > 1:
                            # Arka plan (0. etiket) hariç en büyük alanı seç
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            bx, by, bw, bh, area = stats[largest_label]
                            
                            # 4. En büyük alanın etrafına tek bir kutu çiz
                            rect = plt.Rectangle(
                                (bx, by), bw, bh,
                                linewidth=3, edgecolor="r", facecolor="none"
                            )
                            ax.add_patch(rect)
                            ax.text(
                                bx, by - 10, "Kusur",
                                color="red", fontsize=12, weight="bold",
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                            )
                    else:
                        st.info("Görsel hatalı olarak sınıflandırıldı ancak eşiği geçen belirgin bir yama bulunamadı.")
                else:
                    st.info("Görsel sağlam olduğu için işaretleme yapılmadı.")

                ax.axis("off")
                st.pyplot(fig)
else:
    st.info("Lütfen sol taraftaki menüden analiz edilecek bir vida resmi yükleyin.")