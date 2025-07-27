import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import cv2

st.title("🩺 Pneumonia Detection from Chest X-ray with Grad-CAM")


@st.cache_resource()
def load_model_with_fallback():
    model_paths = [
        "pneumonia_model.keras",
        "pneumonia_model.h5",
        os.path.join("pneumonia_app", "pneumonia_model.keras"),
        os.path.join("pneumonia_app", "pneumonia_model.h5"),
    ]
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                st.success(f"✅ Loaded model: {model_path}")
                return model
            except Exception as e:
                st.warning(f"⚠ Failed to load {model_path}: {e}")

    st.warning("⚠ Using fallback CNN model (untrained)")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = load_model_with_fallback()


def generate_gradcam(model, img_array, layer_name=None):
    try:
        if img_array.shape[0] != 1:
            img_array = np.expand_dims(img_array, axis=0)

        if not layer_name:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None


@st.cache_data(max_entries=3)
def preprocess_image(img):
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None


uploaded_file = st.file_uploader("📁 Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    img_array = preprocess_image(img)
    if img_array is not None:
        input_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("🔍 Analyzing image..."):
            pred = model.predict(input_array)[0][0]
            label = "PNEUMONIA" if pred >= 0.5 else "NORMAL"
            confidence = pred if pred >= 0.5 else 1 - pred

            st.subheader(f"Prediction: **{label}**")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")

            heatmap = generate_gradcam(model, input_array[0])
            if heatmap is not None:
                original_img = cv2.cvtColor(np.uint8(input_array[0] * 255), cv2.COLOR_RGB2BGR)
                superimposed = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
                st.image(
                    cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB),
                    caption="🔥 Grad-CAM (Important Regions Highlighted)",
                    use_container_width=True
                )