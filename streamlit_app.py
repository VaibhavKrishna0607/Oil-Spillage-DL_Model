import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml_model.infer import (
    extract_geometric_properties,
    krestenitis_oil_mask,
    load_checkpoint,
    load_norm_stats,
    mask_iou,
    overlay_rgb,
    predict_array,
)

SOS_IMG = ROOT / "Dataset-1" / "Images" / "images"
SOS_MASK = ROOT / "Dataset-1" / "Masks" / "masks"
KREST = ROOT / "Oil Spill Detection Dataset"
CKPT = ROOT / "ml_model" / "checkpoints" / "best_model.pth"


def list_png(folder: Path):
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.glob("*.png"))


def list_jpg(folder: Path):
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.glob("*.jpg")) + sorted(p.name for p in folder.glob("*.png"))


@st.cache_resource
def get_model():
    if not CKPT.exists():
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(str(CKPT), device)
    stats = load_norm_stats(ROOT)
    return model, meta, device


st.set_page_config(
    page_title="SIH26143 Detection Lab",
    page_icon=":material/oil_barrel:",
    layout="wide",
)

st.title("Oil spill detection lab")
st.caption("SIH 26143 — inspect the U-Net on SOS patches and Krestenitis scenes before drift / AIS work.")

model, meta, device = get_model()
stats = load_norm_stats(ROOT)

with st.sidebar:
    st.subheader("How to read this")
    st.markdown(
        "The model only answers **where is dark oil-like water in this SAR image?** "
        "It does **not** know latitude, wind, or ships. Those come later from geometry + drift + AIS."
    )
    if model is None:
        st.badge("No checkpoint", icon=":material/warning:", color="orange")
        st.markdown(
            "Train first:\n\n"
            "`python ml_model/train.py --quick`\n\n"
            "GPU full run:\n\n"
            "`python ml_model/train.py --epochs 15 --batch-size 16`"
        )
    else:
        st.badge("Model loaded", icon=":material/check_circle:", color="green")
        st.write(meta)
    st.divider()
    source = st.radio(
        "Image source",
        ["SOS patch (Dataset-1)", "Krestenitis scene", "Upload"],
        help="SOS is the training domain (256×256). Krestenitis is a harder full-scene test.",
    )
    threshold = st.slider("Probability threshold", 0.1, 0.9, 0.5, 0.05)
    pixel_res = st.number_input("Pixel size (m)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

gray = None
gt = None
note = ""
image_name = ""

if source == "SOS patch (Dataset-1)":
    split = st.selectbox("Split", ["val", "train"])
    files = list_png(SOS_IMG / split)
    if not files:
        st.error("Dataset-1 images not found.")
    else:
        image_name = st.selectbox("File", files, index=0)
        sensor = "PALSAR / L-band" if image_name.startswith("palsar") else "Sentinel-1 / C-band"
        st.caption("This patch was used (or held out) for binary oil vs water training. Sensor: **{}**.".format(sensor))
        gray = np.array(Image.open(SOS_IMG / split / image_name).convert("L"), dtype=np.float32)
        gt_path = SOS_MASK / split / image_name
        if gt_path.exists():
            gt = (np.array(Image.open(gt_path).convert("L")) > 127).astype(np.uint8)
        note = "SOS 256×256 patch. Compare prediction to the ground-truth mask on the right."

elif source == "Krestenitis scene":
    split = st.selectbox("Split", ["test", "train"])
    files = list_jpg(KREST / split / "images")
    st.info(
        "Do **not** treat this as the training set. Labels are 5-class "
        "(sea, oil, look-alike, ship, land). We score **oil = class 1 only**."
    )
    if not files:
        st.error("Oil Spill Detection Dataset images not found.")
    else:
        image_name = st.selectbox("File", files, index=0)
        gray = np.array(Image.open(KREST / split / "images" / image_name).convert("L"), dtype=np.float32)
        lab = KREST / split / "labels_1D" / Path(image_name).with_suffix(".png").name
        if lab.exists():
            gt = krestenitis_oil_mask(lab)
        note = (
            "Full scene (~1250×650). Inference tiles 256×256 windows with overlap. "
            "Look-alikes (class 2) are a common false-positive trap — check them."
        )

else:
    uploaded = st.file_uploader("Upload a SAR PNG/JPG", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is not None:
        image_name = uploaded.name
        gray = np.array(Image.open(uploaded).convert("L"), dtype=np.float32)
        note = "Uploaded image is normalized with SOS train stats. If radiometry differs, scores will drop."

if gray is None:
    st.stop()

if model is None:
    st.warning("No trained weights yet. Run `python ml_model/train.py --quick` then refresh.")
    c1, c2 = st.columns(2)
    with c1:
        st.image(gray.astype(np.uint8), caption="Input (grayscale)", use_container_width=True)
    with c2:
        if gt is not None:
            st.image((gt * 255).astype(np.uint8), caption="Ground truth oil", use_container_width=True)
    st.stop()

cache_key = "{}|{}|{}".format(source, image_name, gray.shape)
if st.session_state.get("pred_key") != cache_key:
    with st.status("Running U-Net…", expanded=True) as status:
        if gray.shape[0] > 256 or gray.shape[1] > 256:
            st.write("Scene is {}×{} — sliding-window tiling.".format(gray.shape[1], gray.shape[0]))
        pred = predict_array(model, gray, stats, device)
        st.session_state["pred"] = pred
        st.session_state["pred_key"] = cache_key
        status.update(label="Inference done", state="complete")
else:
    pred = st.session_state["pred"]

binary = (pred > threshold).astype(np.uint8)
slicks = extract_geometric_properties(binary * 255, pixel_resolution_m=float(pixel_res))
overlay = overlay_rgb(gray, binary, slicks)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Spill detected", "Yes" if slicks else "No")
k2.metric("Slicks", len(slicks))
k3.metric("Total area (km²)", round(sum(s["area_km2"] for s in slicks), 4) if slicks else 0.0)
if gt is not None:
    k4.metric("IoU vs ground truth", round(mask_iou(binary.astype(bool), gt.astype(bool)), 3))
else:
    k4.metric("IoU vs ground truth", "—")

st.caption(note)

t1, t2, t3 = st.tabs(["Compare", "Probability", "Geometry (feeds the drift model)"])

with t1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(gray.astype(np.uint8), caption="SAR (grayscale)", use_container_width=True)
    with c2:
        st.image(overlay, caption="Prediction overlay (blue fill, orange outline, yellow centroid)", use_container_width=True)
    with c3:
        if gt is not None:
            st.image((gt * 255).astype(np.uint8), caption="Ground-truth oil", use_container_width=True)
        else:
            st.info("No ground-truth mask for this image.")

with t2:
    st.markdown("Bright = model is confident the pixel is oil. Drag the sidebar threshold and watch the overlay change — you are not re-running the network.")
    heat = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    st.image(heat, caption="Oil probability", use_container_width=True)

with t3:
    st.markdown(
        "Each row is one connected slick. **Centroid** is the seed the drift engine will hindcast from. "
        "Area / eccentricity describe the slick, not the ship. Pixel metres assume Sentinel-1 GRD (~10 m) unless you change it."
    )
    if slicks:
        rows = []
        for s in slicks:
            rows.append({
                "id": s["slick_id"],
                "area_km2": s["area_km2"],
                "perimeter_m": s["perimeter_m"],
                "eccentricity": s["eccentricity"],
                "centroid_x": s["centroid"]["x"],
                "centroid_y": s["centroid"]["y"],
                "orientation_deg": s["orientation_deg"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Nothing above threshold. Lower the threshold or pick a patch that actually contains oil.")

with st.expander("What you should check yourself"):
    st.markdown(
        """
1. **SOS val patches** — prediction should roughly cover the white GT mask. If it paints the whole ocean, the model is not trained yet.
2. **Empty / tiny masks** — a good model stays quiet. False alarms here will poison AIS ranking later.
3. **Krestenitis look-alikes** — dark patches that are *not* oil. If the model lights them up, that is expected until a look-alike stage is added.
4. **Geometry** — centroid should sit inside the slick. That point is the only thing the drift model is allowed to move.
        """
    )
