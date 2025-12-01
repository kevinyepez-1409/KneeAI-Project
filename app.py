import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from math import log
import matplotlib.cm as cm

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adamax

# =========================================================
# 1. GLOBAL CONFIG
# =========================================================
IMG_SIZE = (300, 300)

# Inference model (stable SavedModel)
MODEL_PATH = "KneeOA_GradCAM"

# Keras weights for Grad-CAM
WEIGHTS_PATH = "KneeOA_temp_weights.h5"

CLASS_NAMES = ["Mild-Mod", "Non-OA", "Severe"]
L2_REG = 0.016
DROPOUT_RATE = 0.45
LAST_CONV_LAYER_NAME = "top_conv"

st.set_page_config(
    page_title="KneeAI B3 – Osteoarthritis Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# ====== Global UI styles (CSS only, logic unchanged) ======
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    /* Title */
    .kneeai-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .kneeai-subtitle {
        font-size: 0.95rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    /* Section cards */
    .section-card {
        background-color: #111827;
        padding: 1.1rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.3rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .section-divider {
        border-top: 1px solid #1f2937;
        margin: 0.7rem 0 0.9rem 0;
    }
    /* Main metric */
    .primary-metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }
    .primary-metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    /* Footer note */
    .footer-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 1.5rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def scalar(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)

def predictive_entropy(p: np.ndarray) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe)) / log(len(p_safe))

# =========================================================
# 2A. LOAD INFERENCE MODEL (SavedModel)
# =========================================================
@st.experimental_singleton
def load_inference_model():
    try:
        loaded_model = tf.saved_model.load(MODEL_PATH)
        model_predict_fn = loaded_model.signatures["serving_default"]

        input_signature = model_predict_fn.structured_input_signature
        input_dict = input_signature[1]
        if not input_dict:
            raise ValueError("Model signature has no inputs.")
        input_key = list(input_dict.keys())[0]

        try:
            output_key = list(model_predict_fn.structured_outputs.keys())[0]
        except Exception:
            output_key = "outputs"

        print(f"✅ SavedModel loaded from: {MODEL_PATH}")
        print(f"🔑 Input key:  {input_key}")
        print(f"🔑 Output key: {output_key}")

        return loaded_model, model_predict_fn, input_key, output_key

    except Exception as e:
        st.error(f"Error loading SavedModel: {e}")
        st.warning("Verify that 'KneeOA_GradCAM' is next to app.py.")
        return None, None, None, None

inference_model, model_predict_fn, INPUT_KEY, OUTPUT_KEY = load_inference_model()

# =========================================================
# 2B. LOAD KERAS MODEL FOR GRAD-CAM
# =========================================================
def build_efficientnet_model(num_classes, l2_reg, dropout_rate, lr=1e-4):
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
        pooling="avg"
    )
    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adamax(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model

@st.experimental_singleton
def load_gradcam_model():
    try:
        keras_model = build_efficientnet_model(
            num_classes=len(CLASS_NAMES),
            l2_reg=L2_REG,
            dropout_rate=DROPOUT_RATE,
            lr=1e-4
        )
        keras_model.load_weights(WEIGHTS_PATH)
        print(f"✅ Grad-CAM Keras model loaded with weights from: {WEIGHTS_PATH}")
        return keras_model
    except Exception as e:
        st.error(f"Error loading Keras Grad-CAM model: {e}")
        st.warning(
            f"Make sure '{WEIGHTS_PATH}' is in the same folder as app.py "
            "and corresponds to the trained EfficientNetB3 model."
        )
        return None

gradcam_model = load_gradcam_model()

# =========================================================
# 3. GRAD-CAM FUNCTIONS
# =========================================================
def make_gradcam_heatmap(img_batch, keras_model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.cast(img_batch, tf.float32)
    last_conv_layer = keras_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [keras_model.inputs],
        [last_conv_layer.output, keras_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam_on_image(orig_img_array, heatmap, alpha=0.4):
    h, w, _ = orig_img_array.shape
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (h, w)
    ).numpy().squeeze()
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap_uint8)[:, :, :3]
    heatmap_color = np.uint8(255 * heatmap_color)
    superimposed_img = heatmap_color * alpha + orig_img_array.astype(np.float32) * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# =========================================================
# 4. SIDEBAR
# =========================================================
st.sidebar.title("KneeAI B3 🩺")
st.sidebar.markdown(
    """
**AI-based Knee Osteoarthritis Diagnosis**

- Backbone: `EfficientNetB3`  
- Output: 3-class softmax  
  - **Non-OA** – No radiographic OA 0-1 
  - **Mild-Mod** – KL 2 y 3 (mild/moderate)  
  - **Severe** – KL = 4 (severe)  

**Displayed panels:**
- Diagnosis & confidence
- Class probability distribution
- Clinical risk radar
- Predictive uncertainty (entropy)
- Grad-CAM attention map
"""
)
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Methodological notes", expanded=False):
    st.markdown(
        """
        - Model trained on AP knee radiographs.  
        - Prediction is based on the class with maximum softmax probability.  
        - Entropy summarizes global predictive uncertainty.  
        - Grad-CAM is used as a visual explanation tool only.
        """
    )
st.sidebar.caption("Decision-support prototype – not a substitute for clinical assessment.")

# =========================================================
# 5. MAIN LAYOUT
# =========================================================
st.markdown(
    """
    <div class="kneeai-title">KneeAI B3 – Knee Osteoarthritis Diagnosis</div>
    <div class="kneeai-subtitle">
        Automated analysis of knee radiographs using EfficientNetB3 and explainable AI (Grad-CAM).
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 1. Upload an AP knee radiograph")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and inference_model is not None and model_predict_fn is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_left, col_right = st.columns([1.1, 1.6])

    # -------- Left column: input image --------
    with col_left:
        st.markdown(
            """
            <div class="section-card">
              <div class="section-title">Input image</div>
              <div class="section-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image, caption="Original radiograph", use_column_width=True)

    # -------- Right column: analysis --------
    with col_right:
        st.markdown(
            """
            <div class="section-card">
              <div class="section-title">2. Run model inference</div>
              <div class="section-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🧠 Analyze with AI"):
            with st.spinner("Processing image, computing prediction and attention maps..."):
                # ---------- PREPROCESSING ----------
                img_resized = image.resize(IMG_SIZE)
                img_array = np.array(img_resized)
                img_normalized = scalar(img_array)
                img_batch = np.expand_dims(img_normalized, axis=0)
                input_tensor = tf.constant(img_batch, dtype=tf.float32)

                # ---------- PREDICTION (SavedModel) ----------
                # IMPORTANT FIX: pass keyword argument using ** so it matches 'input_image'
                output_dict = model_predict_fn(**{INPUT_KEY: input_tensor})
                output = output_dict[OUTPUT_KEY]
                probs = output.numpy()[0]

                predicted_index = int(np.argmax(probs))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probs[predicted_index])

                # ---------- BLOCK: MAIN RESULT ----------
                with st.container():
                    st.markdown(
                        """
                        <div class="section-card">
                          <div class="section-title">3. Model output</div>
                          <div class="section-divider"></div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if predicted_class == "Non-OA":
                        st.success(f"**Suggested diagnosis:** {predicted_class} (No radiographic OA) ✅")
                    elif predicted_class == "Mild-Mod":
                        st.warning(f"**Suggested diagnosis:** {predicted_class} (Mild/Moderate OA) ⚠")
                    else:
                        st.error(f"**Suggested diagnosis:** {predicted_class} (Severe OA) 🚨")

                    st.markdown(
                        f"""
                        <div class="primary-metric-label">Model confidence in predicted class</div>
                        <div class="primary-metric-value">{confidence:.2%}</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "The diagnosis corresponds to the class with highest softmax probability. "
                        "It should be interpreted as decision support, not as a standalone criterion."
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                # ---------- BLOCK: UNCERTAINTY ----------
                with st.container():
                    st.markdown(
                        """
                        <div class="section-card">
                          <div class="section-title">4. Predictive uncertainty (entropy)</div>
                          <div class="section-divider"></div>
                        """,
                        unsafe_allow_html=True,
                    )

                    entropy = predictive_entropy(probs)
                    st.write(
                        f"Normalized entropy: **{entropy:.2f}** "
                        "(0 = very confident, 1 = highly uncertain)."
                    )
                    if entropy < 0.3:
                        st.success("The model shows **high confidence** for this prediction.")
                    elif entropy < 0.6:
                        st.warning("The model displays **moderate uncertainty**.")
                    else:
                        st.error(
                            "The model is **highly uncertain**. Correlation with additional studies is recommended."
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                # ---------- BLOCK: PROBABILITIES & RISK RADAR ----------
                with st.container():
                    st.markdown(
                        """
                        <div class="section-card">
                          <div class="section-title">5. Probability profile and clinical risk</div>
                          <div class="section-divider"></div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 5.1 Class probability distribution
                    st.markdown("**Class probability distribution**")
                    df_probs = pd.DataFrame(probs, index=CLASS_NAMES, columns=["Probability"])
                    st.bar_chart(df_probs)

                    # 5.2 Clinical risk radar
                    st.markdown("**Clinical risk radar**")

                    idx_mild = CLASS_NAMES.index("Mild-Mod")
                    idx_non_oa = CLASS_NAMES.index("Non-OA")
                    idx_severe = CLASS_NAMES.index("Severe")

                    risk_global_oa = 1.0 - probs[idx_non_oa]
                    risk_mild_mod = probs[idx_mild]
                    risk_severe = probs[idx_severe]

                    risk_labels = ["Global OA risk", "Mild/Moderate OA", "Severe OA"]
                    risk_values = np.array([risk_global_oa, risk_mild_mod, risk_severe])

                    num_axes = len(risk_labels)
                    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
                    risk_values_closed = np.concatenate((risk_values, [risk_values[0]]))
                    angles_closed = np.concatenate((angles, [angles[0]]))

                    fig_radar, ax_radar = plt.subplots(subplot_kw={"projection": "polar"})
                    ax_radar.plot(angles_closed, risk_values_closed, marker="o")
                    ax_radar.fill(angles_closed, risk_values_closed, alpha=0.25)
                    ax_radar.set_thetagrids(angles * 180 / np.pi, risk_labels)
                    ax_radar.set_ylim(0, 1.0)
                    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
                    ax_radar.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
                    ax_radar.set_title(
                        "Clinical risk profile based on model output",
                        pad=20,
                    )
                    st.pyplot(fig_radar)

                    st.caption(
                        "Values closer to 1 indicate higher risk in each axis. "
                        "'Global OA risk' summarizes the overall probability of having "
                        "osteoarthritis (mild/moderate or severe)."
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                # ---------- BLOCK: GRAD-CAM ----------
                with st.container():
                    st.markdown(
                        """
                        <div class="section-card">
                          <div class="section-title">6. Visual explainability (Grad-CAM)</div>
                          <div class="section-divider"></div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if gradcam_model is None:
                        st.info(
                            "Grad-CAM is not available because the Keras model could not be loaded. "
                            "Please check the configured weights file."
                        )
                    else:
                        try:
                            heatmap = make_gradcam_heatmap(
                                img_batch, gradcam_model, LAST_CONV_LAYER_NAME, pred_index=predicted_index
                            )
                            orig_np = np.array(img_resized)
                            gradcam_img = overlay_gradcam_on_image(orig_np, heatmap, alpha=0.4)
                            st.image(
                                gradcam_img,
                                caption="Regions with highest contribution to the prediction (Grad-CAM)",
                                use_column_width=True,
                            )
                            st.caption(
                                "The heatmap highlights the radiographic areas that the model "
                                "relied on the most when producing the suggested diagnosis."
                            )
                        except Exception as e:
                            st.warning(
                                "Grad-CAM could not be generated. Please check that the layer "
                                f"`{LAST_CONV_LAYER_NAME}` exists in the Grad-CAM model. "
                                f"Technical detail: {e}"
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">KneeAI B3 · Academic prototype for decision support in knee OA diagnosis.</div>',
        unsafe_allow_html=True,
    )

else:
    if inference_model is None:
        st.error("The inference SavedModel could not be loaded. Please verify the model folder.")
    else:
        st.info("👆 Upload a radiograph to start the analysis.")
