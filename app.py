import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import plotly.express as px

# ======================
# CONFIG
# ======================

FASTAPI_ENDPOINT = "http://localhost:8000/predict"

st.set_page_config(
    page_title="COVID-19 AI Detector",
    page_icon="🩺",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

.prediction-box {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 15px;
    margin-top: 10px;
}

.big-font {
    font-size: 24px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================

with st.sidebar:

    st.title("🩺 Model Information")

    st.markdown("---")

    st.write("### Model")
    st.write("MobileNetV2")

    st.write("### Classes")
    st.write("• Covid")
    st.write("• Normal")
    st.write("• Viral Pneumonia")

    st.write("### Technology")
    st.write("• TensorFlow")
    st.write("• FastAPI")
    st.write("• Streamlit")
    st.write("• Render")

    st.markdown("---")

    st.info(
        "Educational Project Only. Not for Medical Diagnosis."
    )

# ======================
# HEADER
# ======================

st.title("🦠 COVID-19 Chest X-Ray Classification")

st.write(
    """
Upload a Chest X-Ray image and the AI model will classify it into:

- Covid
- Normal
- Viral Pneumonia
"""
)

st.markdown("---")

# ======================
# FILE UPLOAD
# ======================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# ======================
# PREDICTION
# ======================

if uploaded_file is not None:

    col1, col2 = st.columns([1,1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    files = {
        "file": (
            "image.jpg",
            img_bytes,
            "image/jpeg"
        )
    }

    if st.button("🔍 Predict"):

        try:

            with st.spinner(
                "Analyzing X-Ray..."
            ):

                response = requests.post(
                    FASTAPI_ENDPOINT,
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:

                result = response.json()

                predicted_class = result["predicted_class"]

                confidence = result["confidence"]

                probabilities = result["probabilities"]

                with col2:

                    st.success(
                        "Prediction Complete"
                    )

                    st.markdown(
                        f"""
                        <div class='prediction-box'>
                        <p class='big-font'>
                        Prediction:
                        {predicted_class}
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("### Confidence")

                    st.progress(confidence)

                    st.write(
                        f"{confidence:.2%}"
                    )

                st.markdown("---")

                # ======================
                # PROBABILITY TABLE
                # ======================

                class_names = [
                    "Covid",
                    "Normal",
                    "Viral Pneumonia"
                ]

                prob_df = pd.DataFrame({
                    "Class": class_names,
                    "Probability": probabilities
                })

                st.subheader(
                    "Class Probabilities"
                )

                st.dataframe(
                    prob_df,
                    use_container_width=True
                )

                # ======================
                # PLOTLY CHART
                # ======================

                fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    title="Prediction Confidence"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # ======================
                # DOWNLOAD REPORT
                # ======================

                report = f"""
COVID-19 X-Ray Classification Report

Prediction:
{predicted_class}

Confidence:
{confidence:.2%}

Class Probabilities:

Covid: {probabilities[0]:.2%}
Normal: {probabilities[1]:.2%}
Viral Pneumonia: {probabilities[2]:.2%}
"""

                st.download_button(
                    "📥 Download Report",
                    report,
                    file_name="prediction_report.txt"
                )

            else:

                st.error(
                    f"API Error: {response.text}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI server."
            )

        except Exception as e:

            st.error(str(e))

# ======================
# FOOTER
# ======================

st.markdown("---")

st.markdown(
    """
<div class='footer'>
Built with TensorFlow, FastAPI, Streamlit and MobileNetV2
</div>
""",
    unsafe_allow_html=True
)