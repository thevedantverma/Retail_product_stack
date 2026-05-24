from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=3, #less number of epoch because was training on CPU, LOL
    imgsz=416,
    batch=4
)