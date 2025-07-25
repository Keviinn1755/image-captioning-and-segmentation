import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import os
import pandas as pd
import cv2

# Page config
st.set_page_config(page_title="Image Segmentation and Captioning", layout="centered")

MODEL_PATH = "pet_segmentation_unet.h5"

@st.cache_resource
def load_models():
    seg_model = tf.keras.models.load_model(MODEL_PATH)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return seg_model, blip_processor, blip_model

seg_model, blip_processor, blip_model = load_models()

def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return img_array

def generate_mask(image_array):
    prediction = seg_model.predict(np.expand_dims(image_array, axis=0))[0]
    mask = (prediction > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask.squeeze().astype(np.uint8))

def draw_bounding_box(image, mask):
    mask_np = np.array(mask)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_np = np.array(image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return Image.fromarray(image_np)

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# UI
st.title("Image Segmentation and Captioning")
st.markdown("---")

uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

use_yolo = st.checkbox("Enable YOLO/Mask-RCNN (placeholder only)", value=False)
opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.4, step=0.05)

if uploaded_files:
    caption_data = []

    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"{uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        img_array = preprocess_image(image)
        mask = generate_mask(img_array)
        mask_resized = mask.resize(image.size)

        overlay = Image.blend(image, mask_resized.convert("RGB"), alpha=opacity)
        boxed_image = draw_bounding_box(image, mask_resized)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(mask_resized, caption="Mask", use_container_width=True)
        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        st.image(boxed_image, caption="Bounding Box", use_container_width=True)

        caption = generate_caption(image)
        st.success(f"Caption: {caption}")
        caption_data.append({"Image": uploaded_file.name, "Caption": caption})

        def convert_to_txt(text):
            return io.BytesIO(text.encode())

        st.download_button("Download Caption", convert_to_txt(caption), file_name=f"{uploaded_file.name}_caption.txt")

    df = pd.DataFrame(caption_data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download All Captions (CSV)", data=csv, file_name="captions.csv", mime="text/csv")
