from ultralytics import YOLO


detection_model = YOLO('./models/traffic_signs_detection_model.pt')
segmentation_model = YOLO('./models/zebra_segmentation_model.pt')


detection_model.export(format="coreml")
segmentation_model.export(format="coreml")
