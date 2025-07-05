import streamlit as st
import os
import sys
import cv2
import insightface
import numpy as np
from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis
from datetime import datetime

# === System Path Setup ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from face_enhancer import load_face_enhancer_model
from faceswap import validate_image, cpu_warning

# === Helper Functions ===
def extract_faces(img, app):
    faces = app.get(img)
    face_imgs = []
    for face in faces:
        bbox = [int(b) for b in face['bbox']]
        face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face_imgs.append((face_crop, face))
    return face_imgs, faces

def draw_boxes(image, faces):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for i, face in enumerate(faces):
        bbox = [int(b) for b in face['bbox']]
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), f"Face {i}", fill="red")
    return pil_img

def fine_face_swap_ui(img1_path, img2_path, face_idx1, face_idx2, app, swapper,
                      enhance=False, enhancer='REAL-ESRGAN 2x', device="cpu"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    facesimg1, _ = extract_faces(img1, app)
    facesimg2, _ = extract_faces(img2, app)

    face1 = facesimg1[face_idx1][1]
    face2 = facesimg2[face_idx2][1]

    img1_ = img1.copy()
    img1_ = swapper.get(img1_, face1, face2, paste_back=True)

    if enhance:
        cpu_warning(device)
        model, model_runner = load_face_enhancer_model(enhancer, device)
        img1_ = model_runner(img1_, model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    output_fn = os.path.join(output_dir, "result.png")
    cv2.imwrite(output_fn, img1_)
    return img1_, output_fn

# === Streamlit App ===
st.set_page_config(page_title="FaceFusion - Face Swap", layout="wide")
st.title("\U0001F3AD FaceFusion: AI-Powered Face Swapper")

st.sidebar.header("Usage Steps")
st.sidebar.markdown("""
1. Upload Source and Target images
2. Click 'Detect Faces'
3. Select faces to swap
4. Click 'Swap Faces' and download result
""")

st.sidebar.header("Options")
enhance_option = st.sidebar.checkbox("Apply Face Enhancement", value=False)

# Advanced UI layout
tab1, tab2 = st.tabs(["\U0001F5BC Face Swapper", "\U0001F4C1 Output History"])

with tab1:
    st.subheader("\U0001F4F7 Upload Images")
    col1, col2 = st.columns(2)
    with col1:
        source_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"], key="source")
    with col2:
        target_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="target")

    if source_file and target_file:
        os.makedirs("streamlit_app/assets", exist_ok=True)
        img1_path = f"streamlit_app/assets/source_{source_file.name}"
        img2_path = f"streamlit_app/assets/target_{target_file.name}"

        with open(img1_path, "wb") as f:
            f.write(source_file.read())
        with open(img2_path, "wb") as f:
            f.write(target_file.read())

        app_obj = FaceAnalysis(name='buffalo_l')
        app_obj.prepare(ctx_id=0, det_size=(640, 640))

        model_output_path = 'model/FaceFusion-SoC.onnx'
        if not os.path.exists(model_output_path):
            st.error("FaceFusion ONNX model not found in 'model/' folder.")
            st.stop()

        swapper = insightface.model_zoo.get_model(model_output_path, download=False, download_zip=False)

        st.success("Images uploaded successfully. Click below to detect faces.")
        if st.button("\U0001F441 Detect Faces"):
            faceimgs1, faces1 = extract_faces(cv2.imread(img1_path), app_obj)
            faceimgs2, faces2 = extract_faces(cv2.imread(img2_path), app_obj)

            if len(faceimgs1) == 0 or len(faceimgs2) == 0:
                st.error("No faces detected in one or both images. Please try with clear frontal face images.")
                st.stop()

            st.session_state.update({
                "app_obj": app_obj,
                "swapper": swapper,
                "img1_path": img1_path,
                "img2_path": img2_path,
                "facesimg1": faceimgs1,
                "facesimg2": faceimgs2
            })
            st.success("Faces detected successfully. Scroll down to select faces for swapping.")

if "facesimg1" in st.session_state and "facesimg2" in st.session_state:
    st.subheader("\U0001F464 Select Faces for Swapping")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Source Faces")
        for idx, (img, _) in enumerate(st.session_state.facesimg1):
            st.image(img[:, :, ::-1], width=150, caption=f"Source Face {idx}")
        face1_idx = st.selectbox("Select Source Face", list(range(len(st.session_state.facesimg1))), key="face1_idx")

    with col2:
        st.markdown("### Target Faces")
        for idx, (img, _) in enumerate(st.session_state.facesimg2):
            st.image(img[:, :, ::-1], width=150, caption=f"Target Face {idx}")
        face2_idx = st.selectbox("Select Target Face", list(range(len(st.session_state.facesimg2))), key="face2_idx")

    if st.button("\u2728 Swap Faces Now"):
        with st.spinner("Swapping faces, please wait..."):
            result_img, output_path = fine_face_swap_ui(
                st.session_state.img1_path,
                st.session_state.img2_path,
                face1_idx,
                face2_idx,
                st.session_state.app_obj,
                st.session_state.swapper,
                enhance=enhance_option
            )
            st.image(result_img[:, :, ::-1], caption="\U0001F4A5 Face Swapped Result", use_column_width=True)
            with open(output_path, "rb") as f:
                st.download_button("\U0001F4E5 Download Result", f, file_name="swapped_result.png", mime="image/png")

with tab2:
    st.subheader("\U0001F4DD Recent Outputs")
    if os.path.exists("outputs"):
        folders = sorted([f for f in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", f))], reverse=True)
        for folder in folders[:5]:
            result_img_path = os.path.join("outputs", folder, "result.png")
            if os.path.exists(result_img_path):
                st.image(result_img_path, caption=f"Output from {folder}", use_column_width=True)
                with open(result_img_path, "rb") as f:
                    st.download_button("Download", f, file_name=f"result_{folder}.png", mime="image/png")
    else:
        st.info("No previous outputs found.")
