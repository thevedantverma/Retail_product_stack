from ultralytics import YOLO

model = YOLO("C:/Users/theve/OneDrive/Desktop/CNN/runs/detect/train-2/weights/best.pt")

def detect_products(image_path):

    results = model(image_path, conf =0.4)

    detections = []

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):

            x1, y1, x2, y2 = map(int, box)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf)
            })

    return detections