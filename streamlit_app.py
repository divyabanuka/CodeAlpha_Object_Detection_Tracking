import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="CodeAlpha Object Detection",
    page_icon="🎯"
)

st.title("🎯 CodeAlpha Object Detection")
st.write("Upload an image and YOLO will automatically detect objects.")

model = YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    results = model.predict(image)

    detected_image = results[0].plot()

    st.subheader("🔍 Detected Objects")
    st.image(
        detected_image,
        channels="BGR",
        use_container_width=True
    )

    if results[0].boxes:
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
