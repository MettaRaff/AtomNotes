import sounddevice as sd

# Показать все доступные устройства
print(sd.query_devices())

# Укажите ID нужного устройства вручную
device_id = 3  # Измените на правильное значение