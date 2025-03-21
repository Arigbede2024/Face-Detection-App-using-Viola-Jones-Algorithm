import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit App Title
st.title("Face Detection App using Viola-Jones Algorithm")

# Instructions
st.markdown("""
### Instructions:
1. Upload an image.
2. Adjust detection parameters (scaleFactor and minNeighbors).
3. Choose the rectangle color.
4. Click 'Detect Faces'.
5. Download the processed image.
""")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

  # User adjustable parameters
    scaleFactor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01)
    minNeighbors = st.slider("Min Neighbors", 1, 10, 5)
    rect_color = st.color_picker("Choose rectangle color", "#FF0000")
    rect_color = tuple(int(rect_color[i:i+2], 16) for i in (1, 3, 5))

    # Detect faces button
    if st.button("Detect Faces"):
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), rect_color, 2)

        st.image(img_array, caption="Detected Faces", use_column_width=True)

# Save and download the processed image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            cv2.imwrite(tmp_file.name, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            st.download_button("Download Processed Image", tmp_file.name, file_name="detected_faces.png")
