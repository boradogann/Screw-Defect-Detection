from pathlib import Path
import cv2
import gradio as gr
import numpy as np

# Matplotlib headless mode (sunucu çökmesini önler)
import matplotlib

matplotlib.use("Agg")
from PIL import Image, ImageOps
import tensorflow as tf

# =========================================================
# AYARLAR (GÖRELİ DOSYA YOLLARI)
# =========================================================

# Model dosyası app.py ile aynı klasörde olmalıdır
BEST_MODEL_PATH = Path("best_resnet50_nonoverlap_64.keras")

PATCH_SIZE = 64
STRIDE = 64
MODEL_INPUT_SIZE = 224
BATCH_SIZE = 32

MIN_SCREW_RATIO = 0.20
SCREW_BBOX_PAD = 12

# Image-level karar eşiği
IMAGE_THRESHOLD = 0.8093571662902832

# Heatmap/localization threshold
PATCH_THRESHOLD_FOR_HEATMAP = 0.615128350257874

# Hata bounding box padding
DEFECT_BBOX_PAD = 6

print("Model yükleniyor...")
model = tf.keras.models.load_model(BEST_MODEL_PATH)
print("Model başarıyla yüklendi!")


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================


def pil_to_rgb_array(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


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
        return keep_largest_component(255 - th).astype(np.uint8)

    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[0][0].astype(np.uint8)


def refine_defect_region(
    heatmap,
    threshold=PATCH_THRESHOLD_FOR_HEATMAP,
    min_area=250,
    morph_kernel=7,
    dilate_iter=1,
    bbox_pad=6,
    core_percentile=35,
):
    binary = (heatmap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        empty = np.zeros_like(binary)
        return empty, empty, None

    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if binary.sum() == 0:
        empty = np.zeros_like(binary)
        return empty, empty, None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        empty = np.zeros_like(binary)
        return empty, empty, None

    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp_vals = heatmap[labels == lbl]
        score = 0.7 * float(comp_vals.max()) + 0.3 * float(comp_vals.mean())
        candidates.append((lbl, score, area))

    if not candidates:
        empty = np.zeros_like(binary)
        return empty, empty, None

    best_label = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)[
        0
    ][0]
    best_comp = (labels == best_label).astype(np.uint8)

    support_mask = cv2.dilate(
        best_comp, np.ones((5, 5), np.uint8), iterations=dilate_iter
    )
    sup_vals = heatmap[support_mask == 1]
    core_thr = max(threshold, float(np.percentile(sup_vals, core_percentile)))
    core_mask = ((heatmap >= core_thr) * support_mask).astype(np.uint8)
    core_mask = cv2.morphologyEx(
        core_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    core_mask = cv2.morphologyEx(
        core_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    if core_mask.sum() == 0:
        core_mask = support_mask.copy()

    ys, xs = np.where(core_mask > 0)
    h, w = core_mask.shape[:2]
    x1 = max(0, int(xs.min()) - bbox_pad)
    y1 = max(0, int(ys.min()) - bbox_pad)
    x2 = min(w, int(xs.max()) + bbox_pad + 1)
    y2 = min(h, int(ys.max()) + bbox_pad + 1)
    bbox = (x1, y1, x2, y2)

    return support_mask.astype(np.uint8), core_mask.astype(np.uint8), bbox


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


def build_heatmap_from_image(image_rgb):
    H, W = image_rgb.shape[:2]
    screw_mask = create_screw_mask(image_rgb)
    screw_bbox = get_bbox_from_mask(screw_mask, pad=SCREW_BBOX_PAD)

    if screw_bbox is None:
        raise ValueError("Vida bulunamadı.")

    x1, y1, x2, y2 = screw_bbox
    image_roi = image_rgb[y1:y2, x1:x2]
    screw_roi = screw_mask[y1:y2, x1:x2]
    roi_h, roi_w = image_roi.shape[:2]

    new_h = int(np.ceil(roi_h / PATCH_SIZE) * PATCH_SIZE)
    new_w = int(np.ceil(roi_w / PATCH_SIZE) * PATCH_SIZE)

    pad_h = new_h - roi_h
    pad_w = new_w - roi_w

    image_pad = np.pad(
        image_roi, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"
    )
    screw_pad = np.pad(
        screw_roi,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )

    patches = []
    coords = []

    for yy in range(0, new_h, STRIDE):
        for xx in range(0, new_w, STRIDE):
            patch = image_pad[yy : yy + PATCH_SIZE, xx : xx + PATCH_SIZE]
            screw_patch = screw_pad[yy : yy + PATCH_SIZE, xx : xx + PATCH_SIZE]
            screw_ratio = screw_patch.sum() / (PATCH_SIZE * PATCH_SIZE)

            if screw_ratio < MIN_SCREW_RATIO:
                continue

            patch_224 = np.array(
                Image.fromarray(patch).resize(
                    (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                    resample=Image.BILINEAR,
                )
            )
            patches.append(patch_224)
            coords.append((xx, yy))

    if len(patches) == 0:
        raise ValueError("Geçerli vida patch'i bulunamadı.")

    patches = np.array(patches).astype(np.float32)
    probs = model.predict(patches, batch_size=BATCH_SIZE, verbose=0).reshape(
        -1
    )

    heatmap_roi = np.zeros((new_h, new_w), dtype=np.float32)
    count_roi = np.zeros((new_h, new_w), dtype=np.float32)

    for (xx, yy), prob in zip(coords, probs):
        heatmap_roi[yy : yy + PATCH_SIZE, xx : xx + PATCH_SIZE] += prob
        count_roi[yy : yy + PATCH_SIZE, xx : xx + PATCH_SIZE] += 1

    heatmap_roi = heatmap_roi / np.maximum(count_roi, 1)
    heatmap_roi = heatmap_roi[:roi_h, :roi_w]

    heatmap_full = np.zeros((H, W), dtype=np.float32)
    heatmap_full[y1:y2, x1:x2] = heatmap_roi
    heatmap_full = heatmap_full * screw_mask

    sorted_probs = np.sort(probs)[::-1]
    top3_mean = sorted_probs[: min(3, len(sorted_probs))].mean()
    pred_label = "DEFECTED" if top3_mean >= IMAGE_THRESHOLD else "SOLID"

    return heatmap_full, screw_mask, top3_mean, pred_label


def make_focused_heatmap_image(heatmap, core_mask, bbox=None, expand=15):
    H, W = heatmap.shape[:2]
    if core_mask.sum() == 0 or bbox is None:
        return np.zeros((H, W, 3), dtype=np.uint8)

    focused = heatmap * core_mask.astype(np.float32)
    vals = focused[core_mask == 1]
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax > vmin:
        focused[core_mask == 1] = (vals - vmin) / (vmax - vmin + 1e-8)

    hm = (np.clip(focused, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    mask3 = np.stack([core_mask] * 3, axis=-1).astype(bool)
    heatmap_color[~mask3] = 0
    return heatmap_color


def make_overlay(image_rgb, core_mask, pred_label, score):
    overlay = image_rgb.copy()
    if core_mask.sum() > 0:
        red_layer = np.zeros_like(image_rgb)
        red_layer[:, :, 0] = 255
        alpha = 0.50
        mask3 = np.stack([core_mask] * 3, axis=-1).astype(bool)
        overlay = np.where(
            mask3,
            (overlay * (1 - alpha) + red_layer * alpha).astype(np.uint8),
            overlay,
        )

        contours, _ = cv2.findContours(
            core_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    text = f"score={score:.3f}"
    color = (255, 60, 60) if pred_label == "DEFECTED" else (60, 200, 60)
    cv2.putText(
        overlay,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    return overlay


def make_component_bbox_image(image_rgb, core_mask, bbox):
    out = image_rgb.copy()
    if core_mask.sum() == 0 or bbox is None:
        cv2.putText(
            out,
            "No defect detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (60, 200, 60),
            2,
            cv2.LINE_AA,
        )
        return out

    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 3)

    label_y = max(25, y1 - 10)
    cv2.putText(
        out,
        "Defect Detected",
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def predict_demo(input_image):
    if input_image is None:
        return None, None, None

    image_rgb = pil_to_rgb_array(input_image)
    heatmap, screw_mask, top3_mean, pred_label = build_heatmap_from_image(
        image_rgb
    )

    support_mask, core_mask, defect_bbox = refine_defect_region(
        heatmap=heatmap,
        threshold=PATCH_THRESHOLD_FOR_HEATMAP,
        min_area=250,
        morph_kernel=7,
        dilate_iter=1,
        bbox_pad=DEFECT_BBOX_PAD,
    )

    if pred_label == "SOLID":
        core_mask = np.zeros_like(core_mask)
        defect_bbox = None

    overlay_img = make_overlay(image_rgb, core_mask, pred_label, top3_mean)
    heatmap_img = make_focused_heatmap_image(
        heatmap, core_mask, defect_bbox, expand=15
    )
    component_img = make_component_bbox_image(
        image_rgb, core_mask, defect_bbox
    )

    return overlay_img, heatmap_img, component_img


# =========================================================
# GRADIO ARAYÜZ
# =========================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Vida Kusur Tespiti") as demo:
    gr.Markdown("## 🔩 Vida Yüzey Kusuru & Anomali Tespit Sistemi")
    gr.Markdown(
        "Vida görseli yükleyin. Model otomatik olarak kusur bölgesi (overlay), ısı haritası (heatmap) ve sınır kutusunu (bounding box) çıkarsın."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Vida Görseli Yükle")
            with gr.Row():
                clear_btn = gr.ClearButton(
                    value="Temizle", components=[input_img]
                )
                submit_btn = gr.Button("Analiz Et", variant="primary")

        with gr.Column(scale=1):
            overlay_out = gr.Image(type="numpy", label="Overlay Sonucu")
            heatmap_out = gr.Image(type="numpy", label="Heatmap Sonucu")
            bbox_out = gr.Image(
                type="numpy", label="Bounding Box / Kusur Bölgesi"
            )

    submit_btn.click(
        fn=predict_demo,
        inputs=input_img,
        outputs=[overlay_out, heatmap_out, bbox_out],
    )

# Sunucu üzerinde yayına alma çalıştırması
if __name__ == "__main__":
    demo.launch()
