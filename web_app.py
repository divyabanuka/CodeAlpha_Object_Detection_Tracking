import gradio as gr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")


def detect_objects(frame):
    if frame is None:
        return None

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    return results[0].plot()


# Create web app
with gr.Blocks() as demo:
    gr.Markdown("# CodeAlpha Object Detection and Tracking")
    gr.Markdown("Real-time object detection and tracking using YOLO.")

    webcam = gr.Image(
        sources=["webcam"],
        type="numpy",
        label="Webcam"
    )

    output = gr.Image(
        label="Detection & Tracking"
    )

    webcam.stream(
        detect_objects,
        inputs=webcam,
        outputs=output,
        stream_every=0.2
    )


demo.launch()