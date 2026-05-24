import streamlit as st
import cv2
import numpy as np
import tempfile

from detector.detector import detect_products
from embeddings.embedder import get_embedding
from grouping.cluster import cluster_products
from visualization.draw import draw_boxes

st.title("Retail Product Shelf Grouping")

uploaded_file = st.file_uploader(
    "Upload Shelf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    if st.button("GO!!!"):

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp_file.write(uploaded_file.read())

        image_path = temp_file.name

        image = cv2.imread(image_path)

        detections = detect_products(image_path)

        embeddings = []

        valid_detections = []

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            embedding = get_embedding(crop)

            embeddings.append(embedding)

            valid_detections.append(det)

        if len(embeddings) > 0:

            labels = cluster_products(embeddings)

            for det, label in zip(valid_detections, labels):

                det["group_id"] = int(label)

            output_image = draw_boxes(
                image,
                valid_detections,
                labels
            )

            output_image = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_image,
                caption="Grouped Products",
                use_container_width=True
            )

            st.write(valid_detections)

        else:

            st.write("No products detected")
