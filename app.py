
    import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. เรียกใช้โมเดล MobileNetV2
MODEL_PATH = 'lung_disease_mobilenetv2.h5'
model = load_model(MODEL_PATH)

# ลำดับคลาส
labels_map = {0: 'Normal (ปอดปกติ)', 1: 'PNEUMONIA (ปอดบวม)', 2: 'TB (วัณโรค)'}
IMG_SIZE = (224, 224) # MobileNetV2 ใช้ขนาด 224x224

def predict_xray(img_path):
    # โหลดและแปลงรูปภาพ
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array) # Preprocess ตรงของ MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)

    # ทำนายผล
    preds = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(preds)
    confidence = preds[predicted_idx] * 100

    return f"ผลวิเคราะห์: {labels_map[predicted_idx]}\nความมั่นใจ: {confidence:.1f}%"

    
