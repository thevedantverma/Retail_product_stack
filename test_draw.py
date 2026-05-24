import cv2

from visualization.draw import draw_boxes

image = cv2.imread("test.jpg")

detections = [
    {
        "bbox": [50, 50, 300, 300]
    },
    {
        "bbox": [350, 100, 600, 400]
    }
]

labels = [0, 1]

output = draw_boxes(
    image,
    detections,
    labels
)

cv2.imwrite("outputs/result.jpg", output)

print("Done")