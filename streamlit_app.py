import streamlit as st
import os
from app.main import process_video

st.set_page_config(
    page_title="Intelligent Traffic Monitoring",
    layout="centered"
)

st.title("🚦 Intelligent Traffic Monitoring System")
st.write("Real-time Vehicle Detection & Counting using YOLOv8")

uploaded_video = st.file_uploader(
    "Upload traffic video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:
    os.makedirs("uploads", exist_ok=True)
    input_path = os.path.join("uploads", uploaded_video.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    st.subheader("Uploaded Video")
    st.video(input_path)

    if st.button("Run Detection"):
        with st.spinner("Running YOLOv8 detection..."):
            output_path, counts = process_video(input_path)

        st.success("Detection completed ✅")

        st.subheader("Processed Output Video")
        st.video(output_path)

        st.subheader("Vehicle Counts")
        st.write({
            "Cars": counts.car,
            "Motorcycles": counts.motorcycle,
            "Buses": counts.bus,
            "Trucks": counts.truck,
            "Total": counts.total
        })
