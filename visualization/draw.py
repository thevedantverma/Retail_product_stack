import cv2
import random

def draw_boxes(image, detections, labels):

    colors = {}

    for label in set(labels):

        colors[label] = (
            random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255)
        )

    for det, label in zip(detections, labels):

        x1, y1, x2, y2 = det["bbox"]

        color = colors[label]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            f"Group {label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image