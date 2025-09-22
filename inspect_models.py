import coremltools as ct

# Загрузка моделей
traffic_signs_model = ct.models.MLModel('./models/traffic_signs_detection_model.mlpackage')
zebra_model = ct.models.MLModel('./models/zebra_segmentation_model.mlpackage')

# Получение информации о моделях
print("=== Traffic Signs Detection Model ===")
traffic_signs_spec = traffic_signs_model.get_spec()
print("Input description:")
print(traffic_signs_spec.description.input)
print("\nOutput description:")
print(traffic_signs_spec.description.output)

print("\n=== Zebra Segmentation Model ===")
zebra_spec = zebra_model.get_spec()
print("Input description:")
print(zebra_spec.description.input)
print("\nOutput description:")
print(zebra_spec.description.output)