import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av

model = YOLO("yolo11n.pt")


class ObjectDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.track(
            img,
            persist=True,
            verbose=False
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


st.set_page_config(
    page_title="CodeAlpha Object Detection",
    page_icon="🎯"
)

st.title("🎯 CodeAlpha Object Detection and Tracking")

st.write(
    "Real-time object detection and tracking using YOLO."
)

webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)