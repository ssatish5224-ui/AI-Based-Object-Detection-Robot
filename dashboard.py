import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import hashlib
import json

from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Object Detection Robot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()

if "spoken_keys" not in st.session_state:
    st.session_state.spoken_keys = set()

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Dashboard Settings")

dark_mode = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=False
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10,
    0.95,
    0.45,
    0.05
)

voice_alert = st.sidebar.toggle(
    "🔊 Enable Voice Alert",
    value=True
)

object_options = list(model.names.values())

selected_objects = st.sidebar.multiselect(
    "🎯 Select Objects to Detect",
    options=object_options
)

st.sidebar.divider()

st.sidebar.info(
    "AI Object Detection System\n\n"
    "Model: YOLOv8 Nano\n\n"
    "Technology: Python + OpenCV + Streamlit"
)

# ---------------- DARK MODE ----------------

if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #111827;
        color: white;
    }
    [data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("🤖 AI-Based Object Detection Robot")

st.markdown(
    "### Real-Time Object Detection Using YOLO & Computer Vision"
)

st.caption(
    "Intelligent Object Recognition and Analytics Dashboard"
)

st.divider()

# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "📷 Camera Detection",
    "📁 Image Upload",
    "📊 Detection History"
])

# ---------------- DETECTION FUNCTION ----------------

def detect_objects(image):

    img_array = np.array(image)

    selected_ids = [
        class_id
        for class_id, name in model.names.items()
        if name in selected_objects
    ]

    results = model.predict(
        img_array,
        conf=confidence,
        classes=selected_ids if selected_objects else None,
        verbose=False
    )

    result = results[0]

    detected_image = result.plot()

    detected_image = cv2.cvtColor(
        detected_image,
        cv2.COLOR_BGR2RGB
    )

    names = []
    confidences = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100

            names.append(model.names[class_id])
            confidences.append(conf)

    return detected_image, names, confidences

# ---------------- VOICE ALERT ----------------

def speak_detection(names, unique_key):

    if not names or not voice_alert:
        return

    if unique_key in st.session_state.spoken_keys:
        return

    st.session_state.spoken_keys.add(unique_key)

    message = "Detected " + ", ".join(names)
    safe_message = json.dumps(message)

    components.html(
        f"""
        <script>
        const message = {safe_message};

        window.speechSynthesis.cancel();

        const speech =
            new SpeechSynthesisUtterance(message);

        speech.lang = "en-US";
        speech.rate = 1;
        speech.pitch = 1;

        window.speechSynthesis.speak(speech);
        </script>
        """,
        height=0
    )

# ---------------- SAVE HISTORY ----------------

def save_history(file_bytes, source, names):

    image_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    unique_key = source + "_" + image_hash

    if unique_key not in st.session_state.processed_keys:

        st.session_state.processed_keys.add(unique_key)

        st.session_state.history.append({

            "Date & Time": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),

            "Source": source,

            "Total Objects": len(names),

            "Objects": ", ".join(names)

        })

    return unique_key

# ---------------- DISPLAY RESULTS ----------------

def display_results(image, names, confidences, unique_key):

    counts = Counter(names)

    total = len(names)

    average = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    st.subheader("📈 Detection Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Objects", total)

    col2.metric(
        "Object Categories",
        len(counts)
    )

    col3.metric(
        "Average Confidence",
        f"{average:.2f}%"
    )

    if total > 0:

        st.success(
            f"✅ Successfully detected {total} object(s)!"
        )

        speak_detection(names, unique_key)

    else:

        st.warning(
            "No objects detected. Try another image."
        )

    # Detected Image

    st.subheader("🎯 Detected Image")

    st.image(
        image,
        use_container_width=True
    )

    # Download Image

    image_buffer = BytesIO()

    Image.fromarray(image).save(
        image_buffer,
        format="JPEG"
    )

    st.download_button(
        "📥 Download Detected Image",
        data=image_buffer.getvalue(),
        file_name="detected_image.jpg",
        mime="image/jpeg",
        key=f"image_{unique_key}"
    )

    if counts:

        # Object Details

        st.subheader("📋 Object Details")

        data = []

        for name, count in counts.items():

            st.write(
                f"**{name.title()} : {count}**"
            )

            data.append({

                "Object": name.title(),

                "Count": count

            })

        df = pd.DataFrame(data)

        # Object Graph

        st.subheader("📊 Object Detection Analytics")

        st.bar_chart(
            df,
            x="Object",
            y="Count"
        )

        # Confidence Details

        st.subheader("🎯 Confidence Details")

        confidence_df = pd.DataFrame({

            "Object": names,

            "Confidence (%)": [
                round(c, 2) for c in confidences
            ]

        })

        st.dataframe(
            confidence_df,
            use_container_width=True
        )

        # Detailed CSV

        st.subheader("📥 Download Detection Report")

        csv = confidence_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Detailed CSV",
            data=csv,
            file_name="object_detection_report.csv",
            mime="text/csv",
            key=f"csv_{unique_key}"
        )

# ---------------- CAMERA TAB ----------------

with tab1:

    st.subheader("📷 Camera Object Detection")

    camera_image = st.camera_input(
        "Capture Image Using Laptop Camera"
    )

    if camera_image:

        file_bytes = camera_image.getvalue()

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

        with st.spinner("🤖 AI Detecting Objects..."):

            detected, names, confidences = detect_objects(image)

        unique_key = "Camera_" + hashlib.md5(
            file_bytes
        ).hexdigest()

        if names:

            save_history(
                file_bytes,
                "Camera",
                names
            )

        display_results(
            detected,
            names,
            confidences,
            unique_key
        )

# ---------------- UPLOAD TAB ----------------

with tab2:

    st.subheader("📁 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"],
        key="main_image_upload"
    )

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

        with st.spinner("Processing Image..."):

            detected, names, confidences = detect_objects(image)

        unique_key = "Upload_" + hashlib.md5(
            file_bytes
        ).hexdigest()

        if names:

            save_history(
                file_bytes,
                "Image Upload",
                names
            )

        display_results(
            detected,
            names,
            confidences,
            unique_key
        )

# ---------------- HISTORY TAB ----------------

with tab3:

    st.subheader("📊 Detection History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        total_sessions = len(history_df)

        total_objects = int(
            history_df["Total Objects"].sum()
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Detection Sessions",
            total_sessions
        )

        col2.metric(
            "Total Objects Detected",
            total_objects
        )

        st.divider()

        # Search

        search = st.text_input(
            "🔍 Search Object Name",
            key="history_search"
        )

        filtered_df = history_df.copy()

        if search:

            filtered_df = filtered_df[
                filtered_df["Objects"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.subheader("📋 Detection Records")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # Overall Statistics

        all_names = []

        for objects in history_df["Objects"]:

            all_names.extend([
                name.strip()
                for name in objects.split(",")
            ])

        object_counts = Counter(all_names)

        if object_counts:

            st.subheader("📊 Overall Object Statistics")

            stats_df = pd.DataFrame(
                list(object_counts.items()),
                columns=["Object", "Count"]
            )

            st.bar_chart(
                stats_df,
                x="Object",
                y="Count"
            )

        # Download History

        history_csv = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download History CSV",
            data=history_csv,
            file_name="detection_history.csv",
            mime="text/csv",
            key="history_csv"
        )

        # Clear History

        if st.button(
            "🗑️ Clear Detection History",
            key="clear_history"
        ):

            st.session_state.history = []

            st.session_state.processed_keys = set()

            st.session_state.spoken_keys = set()

            st.rerun()

    else:

        st.info(
            "No detection history available. "
            "Capture or upload an image first."
        )

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    """
    <center>
    🤖 AI-Based Object Detection Robot<br>
    Powered by YOLOv8, OpenCV and Streamlit
    </center>
    """,
    unsafe_allow_html=True
)