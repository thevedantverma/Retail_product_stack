import cv2

from detector.detector import detect_products

image = cv2.imread("test.jpg")

results = detect_products("test.jpg")

for det in results:

    x1, y1, x2, y2 = det["bbox"]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

cv2.imwrite(
    "outputs/detection_result.jpg",
    image
)

print(results)
print("Saved")