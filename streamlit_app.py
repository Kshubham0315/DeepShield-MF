import json
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================

BACKEND_URL = "http://127.0.0.1:5000"
PREDICT_URL = f"{BACKEND_URL}/predict"
HEALTH_URL = f"{BACKEND_URL}/health"

st.set_page_config(
    page_title="DeepShield-MF",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .hero {
            padding: 2rem;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                #111827,
                #1f2937
            );
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 42px;
            margin-bottom: 5px;
        }

        .hero p {
            font-size: 17px;
            color: #d1d5db;
        }

        .metric-card {
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid #374151;
            text-align: center;
            background: #111827;
        }

        .metric-value {
            font-size: 30px;
            font-weight: bold;
        }

        .metric-label {
            font-size: 14px;
            color: #9ca3af;
        }

        .real-box {
            padding: 20px;
            border-radius: 14px;
            background: #064e3b;
            border: 1px solid #10b981;
            text-align: center;
        }

        .fake-box {
            padding: 20px;
            border-radius: 14px;
            background: #7f1d1d;
            border: 1px solid #ef4444;
            text-align: center;
        }

        .info-box {
            padding: 15px;
            border-radius: 12px;
            background: #1f2937;
            border: 1px solid #374151;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ DeepShield-MF</h1>
        <p>
            Multidomain Deepfake Detection using CNNs, Vision Transformers
            & Frequency-Domain Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write("### Backend")

    try:
        health_response = requests.get(
            HEALTH_URL,
            timeout=5
        )

        if health_response.status_code == 200:

            health = health_response.json()

            st.success("🟢 Backend Online")

            st.write(
                f"**Device:** `{health.get('device', 'Unknown')}`"
            )

            st.write(
                f"**Model Loaded:** `{health.get('model_loaded', False)}`"
            )

            st.write(
                f"**Upload Limit:** "
                f"`{health.get('max_upload_mb', 100)} MB`"
            )

        else:
            st.error("🔴 Backend Error")

    except requests.exceptions.RequestException:
        st.error("🔴 Backend Offline")

    st.divider()

    st.write("### Model")

    st.info(
        """
        **Current Model**

        Vision Transformer (ViT)

        `vit_base_patch16_224`

        Binary Classification:

        • Real  
        • Fake
        """
    )

    st.divider()

    st.write("### DeepShield-MF")

    st.caption(
        "AI-powered deepfake detection system."
    )

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Detection",
        "📊 Analysis",
        "ℹ️ About",
    ]
)

# ============================================================
# DETECTION TAB
# ============================================================

with tab1:

    st.subheader("Upload Media")

    st.write(
        "Upload an image or video to analyze whether it contains "
        "real or manipulated content."
    )

    uploaded_file = st.file_uploader(
        "Choose an image or video",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:

        st.session_state.uploaded_file = uploaded_file

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        file_type = uploaded_file.type or ""

        if file_type.startswith("image/"):

            st.image(
                uploaded_file,
                caption="Uploaded Image",
                use_container_width=True,
            )

        elif file_type.startswith("video/"):

            st.video(
                uploaded_file
            )

        st.write(
            f"**File:** `{uploaded_file.name}`"
        )

        st.write(
            f"**Size:** `{uploaded_file.size / (1024 * 1024):.2f} MB`"
        )

        st.divider()

        # ----------------------------------------------------
        # Analyze Button
        # ----------------------------------------------------

        analyze = st.button(
            "🚀 Analyze Media",
            type="primary",
            use_container_width=True,
        )

        if analyze:

            if uploaded_file.size == 0:

                st.error(
                    "Uploaded file is empty."
                )

            else:

                progress = st.progress(
                    0
                )

                status = st.empty()

                status.info(
                    "🔄 Sending file to DeepShield backend..."
                )

                try:

                    # Reset file pointer
                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type,
                        )
                    }

                    progress.progress(
                        20
                    )

                    response = requests.post(
                        PREDICT_URL,
                        files=files,
                        stream=True,
                        timeout=600,
                    )

                    if response.status_code != 200:

                        st.error(
                            f"Backend returned "
                            f"HTTP {response.status_code}"
                        )

                        st.code(
                            response.text
                        )

                    else:

                        progress.progress(
                            40
                        )

                        latest_data = None

                        # ------------------------------------------------
                        # Read SSE response
                        # ------------------------------------------------

                        for line in response.iter_lines(
                            decode_unicode=True
                        ):

                            if not line:
                                continue

                            if line.startswith("data:"):

                                raw_data = line[
                                    len("data:"):
                                ].strip()

                                try:

                                    data = json.loads(
                                        raw_data
                                    )

                                    latest_data = data

                                    total = data.get(
                                        "total_frames",
                                        1
                                    )

                                    fake = data.get(
                                        "fake_percentage",
                                        0
                                    )

                                    progress_value = min(
                                        95,
                                        40 + int(
                                            min(
                                                total,
                                                100
                                            )
                                            / 100
                                            * 55
                                        ),
                                    )

                                    progress.progress(
                                        progress_value
                                    )

                                    status.info(
                                        f"🔍 Analyzed "
                                        f"{total} frame(s)..."
                                    )

                                except json.JSONDecodeError:

                                    continue

                        # ------------------------------------------------
                        # Final Result
                        # ------------------------------------------------

                        if latest_data is not None:

                            progress.progress(
                                100
                            )

                            status.success(
                                "✅ Analysis completed."
                            )

                            st.session_state.prediction = (
                                latest_data
                            )

                            st.rerun()

                        else:

                            st.error(
                                "No prediction data received "
                                "from backend."
                            )

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ Request timed out. "
                        "The video may be too large or processing "
                        "may take longer."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Could not connect to Flask backend."
                    )

                    st.info(
                        "Make sure the backend is running on "
                        "`http://127.0.0.1:5000`."
                    )

                except Exception as e:

                    st.error(
                        "Unexpected error occurred."
                    )

                    st.exception(
                        e
                    )


# ============================================================
# RESULT
# ============================================================

with tab1:

    prediction = st.session_state.prediction

    if prediction:

        st.divider()

        st.subheader(
            "🎯 Detection Result"
        )

        total_frames = prediction.get(
            "total_frames",
            0
        )

        real_count = prediction.get(
            "real_count",
            0
        )

        fake_count = prediction.get(
            "fake_count",
            0
        )

        real_percentage = prediction.get(
            "real_percentage",
            0
        )

        fake_percentage = prediction.get(
            "fake_percentage",
            0
        )

        # ----------------------------------------------------
        # Determine result
        # ----------------------------------------------------

        if fake_percentage > real_percentage:

            final_label = "FAKE / MANIPULATED"
            confidence = fake_percentage
            result_class = "fake"

        else:

            final_label = "REAL"
            confidence = real_percentage
            result_class = "real"

        # ----------------------------------------------------
        # Main Result
        # ----------------------------------------------------

        if result_class == "fake":

            st.markdown(
                f"""
                <div class="fake-box">
                    <h1>⚠️ {final_label}</h1>
                    <h2>{confidence:.2f}%</h2>
                    <p>Manipulated content detected</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="real-box">
                    <h1>✅ {final_label}</h1>
                    <h2>{confidence:.2f}%</h2>
                    <p>No significant manipulation detected</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(
            4
        )

        with col1:

            st.metric(
                "Total Frames",
                total_frames
            )

        with col2:

            st.metric(
                "Real Frames",
                real_count
            )

        with col3:

            st.metric(
                "Fake Frames",
                fake_count
            )

        with col4:

            st.metric(
                "Fake %",
                f"{fake_percentage:.2f}%"
            )

        st.divider()

        # ----------------------------------------------------
        # Confidence Chart
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Distribution"
        )

        chart_col1, chart_col2 = st.columns(
            [1, 1]
        )

        with chart_col1:

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "Real",
                            "Fake",
                        ],
                        values=[
                            real_percentage,
                            fake_percentage,
                        ],
                        hole=0.55,
                    )
                ]
            )

            fig.update_layout(
                title="Real vs Fake",
                showlegend=True,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with chart_col2:

            result_df = pd.DataFrame(
                {
                    "Class": [
                        "Real",
                        "Fake",
                    ],
                    "Percentage": [
                        real_percentage,
                        fake_percentage,
                    ],
                }
            )

            st.bar_chart(
                result_df.set_index(
                    "Class"
                )
            )

        # ----------------------------------------------------
        # Detailed Data
        # ----------------------------------------------------

        st.subheader(
            "📋 Detailed Analysis"
        )

        result_table = pd.DataFrame(
            {
                "Metric": [
                    "Total Frames",
                    "Real Frames",
                    "Fake Frames",
                    "Real Percentage",
                    "Fake Percentage",
                    "Final Prediction",
                    "Confidence",
                ],
                "Value": [
                    total_frames,
                    real_count,
                    fake_count,
                    f"{real_percentage:.2f}%",
                    f"{fake_percentage:.2f}%",
                    final_label,
                    f"{confidence:.2f}%",
                ],
            }
        )

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ANALYSIS TAB
# ============================================================

with tab2:

    st.subheader(
        "📊 Model Analysis"
    )

    prediction = st.session_state.prediction

    if prediction:

        real_percentage = prediction.get(
            "real_percentage",
            0
        )

        fake_percentage = prediction.get(
            "fake_percentage",
            0
        )

        st.write(
            "The current inference pipeline analyzes uploaded media "
            "using the deployed Vision Transformer model."
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "Real Probability",
                f"{real_percentage:.2f}%"
            )

            st.progress(
                min(
                    1.0,
                    real_percentage / 100
                )
            )

        with col2:

            st.metric(
                "Fake Probability",
                f"{fake_percentage:.2f}%"
            )

            st.progress(
                min(
                    1.0,
                    fake_percentage / 100
                )
            )

    else:

        st.info(
            "Run an analysis first to see detailed results."
        )


# ============================================================
# ABOUT TAB
# ============================================================

with tab3:

    st.subheader(
        "🛡️ About DeepShield-MF"
    )

    st.markdown(
        """
        ### DeepShield-MF

        **DeepShield-MF** is a deep learning-based deepfake detection
        system designed to identify manipulated images and videos.

        ### Current Pipeline

        ```text
        Input Image / Video
                ↓
        Frame Extraction
                ↓
        Image Preprocessing
                ↓
        Vision Transformer
                ↓
        Binary Classification
                ↓
        Real / Fake
        ```

        ### Current Model

        - Vision Transformer (ViT)
        - Architecture: `vit_base_patch16_224`
        - Input resolution: `224 × 224`
        - Framework: PyTorch
        - GPU acceleration: CUDA
        - Classes: Real / Fake

        ### Planned DeepShield-MF Architecture

        ```text
                     INPUT
                       │
             ┌─────────┴─────────┐
             │                   │
           IMAGE               VIDEO
             │                   │
             │             Frame Extraction
             │                   │
             └─────────┬─────────┘
                       ↓
             ┌─────────────────────┐
             │   Spatial Branch    │
             │                     │
             │ CNN + Vision        │
             │ Transformer         │
             └──────────┬──────────┘
                        │
                        ↓
             ┌─────────────────────┐
             │ Frequency Branch    │
             │                     │
             │ FFT / DCT           │
             └──────────┬──────────┘
                        │
                        ↓
                Feature Fusion
                        │
                        ↓
                  Classification
                        │
                   Real / Fake
        ```

        ### Future Enhancements

        - CNN + ViT feature fusion
        - FFT / DCT frequency branch
        - Temporal modeling with BiLSTM
        - Multi-level fusion
        - Video-level prediction
        - Explainable AI
        - Cross-dataset evaluation
        - Real-time inference
        - Docker deployment
        - Cloud deployment
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DeepShield-MF • Multidomain Deepfake Detection System"
)