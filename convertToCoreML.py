from ultralytics import YOLO


model_signs = YOLO("models/traffic_signs_detection_model.pt")
model_signs.export(format="coreml")  # nms=True — если хочешь, чтобы NMS был внутри модели

model_zebra = YOLO("models/zebra_segmentation_model.pt")
model_zebra.export(format="coreml")  # nms=True — если хочешь, чтобы NMS был внутри модели
