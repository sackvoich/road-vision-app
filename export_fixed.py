# export_fixed.py
from ultralytics import YOLO

print("--- Экспортируем модель дорожных знаков с NMS ---")
model_signs = YOLO("models/zebra_segmentation_model.pt")

# Экспортируем с явным указанием nms=True и размера изображения
model_signs.export(
    format="coreml", 
    nms=True,
    imgsz=640
)

print("--- Экспорт завершен. ---")
# Пока не трогаем модель сегментации