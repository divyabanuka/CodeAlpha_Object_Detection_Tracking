import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(
    page_title="CodeAlpha Object Detection & Tracking",
    page_icon="🎯"
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


st.title("🎯 CodeAlpha Object Detection & Tracking")
st.write("Upload an image or video to detect and track objects using YOLO.")


# Load YOLO model
try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load the YOLO model.")
    st.stop()


# Choose input
option = st.radio(
    "Choose input type:",
    ["🖼️ Image", "🎥 Video"],
    horizontal=True
)


# ==================================================
# IMAGE DETECTION
# ==================================================

if option == "🖼️ Image":

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)

            with st.spinner("🔍 Detecting objects..."):
                results = model.predict(
                    source=image,
                    conf=0.25,
                    verbose=False
                )

            detected_image = results[0].plot()

            st.subheader("🎯 Detected Objects")
            st.image(
                detected_image,
                channels="BGR",
                use_container_width=True
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:

                names = results[0].names

                objects = [
                    names[int(cls)]
                    for cls in results[0].boxes.cls.tolist()
                ]

                st.success(
                    "Detected: " + ", ".join(objects)
                )

            else:
                st.warning("No objects detected.")

        except Exception as e:
            st.error("❌ Could not process this image.")
            st.info("Please try another JPG, PNG, or WEBP image.")


# ==================================================
# VIDEO OBJECT TRACKING
# ==================================================

else:

    uploaded_video = st.file_uploader(
        "🎥 Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        # Save uploaded video
        input_path = None
        output_path = None

        try:
            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(uploaded_video.getbuffer())
            input_file.close()

            input_path = input_file.name

            st.subheader("🎥 Original Video")
            st.video(input_path)

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("❌ Could not open this video.")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if width <= 0 or height <= 0:
                st.error("❌ Invalid video format.")
                cap.release()
                st.stop()

            if fps <= 0:
                fps = 25

            # Output video
            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name
            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not out.isOpened():
                cap.release()
                st.error("❌ Could not create the output video.")
                st.stop()

            st.subheader("🔍 Processing & Tracking")

            progress = st.progress(0)
            status = st.empty()

            frame_count = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                # YOLO object tracking
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    verbose=False
                )

                # Draw boxes + tracking IDs
                annotated_frame = results[0].plot()

                out.write(annotated_frame)

                frame_count += 1

                if total_frames > 0:
                    percentage = min(
                        frame_count / total_frames,
                        1.0
                    )

                    progress.progress(percentage)

                    status.text(
                        f"Processing frame {frame_count} "
                        f"of {total_frames}"
                    )

            cap.release()
            out.release()

            progress.empty()
            status.empty()

            st.success("✅ Video tracking completed!")

            st.subheader("🎯 Tracked Video")
            st.video(output_path)

            # Download option
            with open(output_path, "rb") as video_file:

                st.download_button(
                    label="⬇️ Download Tracked Video",
                    data=video_file,
                    file_name="tracked_video.mp4",
                    mime="video/mp4"
                )

        except Exception as e:

            st.error("❌ Something went wrong while processing the video.")
            st.info(
                "Please try a short MP4 video and try again."
            )

        finally:

            # Remove only the uploaded input file.
            # Keep output video available for Streamlit playback.
            if input_path and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except Exception:
                    pass
