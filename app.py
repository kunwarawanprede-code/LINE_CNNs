import os
import io
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, abort
import tensorflow as tf
from tensorflow.keras.layers import InputLayer

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

app = Flask(__name__)

# ดึง Token และ Secret จาก Environment Variables ใน Render
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ---------------------------------------------------------
# แก้ไขปัญหา Keras Version mismatch (FixedInputLayer)
# ---------------------------------------------------------
class FixedInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        kwargs.pop('optional', None)
        if batch_shape is not None and 'shape' not in kwargs:
            kwargs['shape'] = batch_shape[1:]
        super().__init__(**kwargs)

# ชื่อไฟล์โมเดล
MODEL_PATH = 'lung_disease_mobilenetv2.h5'

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'FixedInputLayer': FixedInputLayer, 'InputLayer': FixedInputLayer},
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_NAMES = ['Normal', 'Pneumonia'] 

# ---------------------------------------------------------
# Route หลักสำหรับ Webhook
# ---------------------------------------------------------
@app.route("/", methods=['GET'])
def index():
    return "Your service is live 🚀"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# ---------------------------------------------------------
# ฟังก์ชันรับและประมวลผลรูปภาพจาก LINE
# ---------------------------------------------------------
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        # 1. ดึงไฟล์รูปภาพจาก LINE
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = io.BytesIO(message_content.content)

        # 2. ปรับทิศทางภาพด้วย EXIF + แปลงเป็น RGB
        img = Image.open(image_bytes)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        # 3. Preprocess รูปภาพ (224x224)
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. ให้โมเดลทำนายผล
        if model is None:
            raise Exception("Model is not loaded properly.")

        predictions = model.predict(img_array)
        
        if predictions.shape[-1] == 1:
            score = float(predictions[0][0])
            if score > 0.5:
                result_text = f"ผลการวิเคราะห์: {CLASS_NAMES[1]} ({score*100:.2f}%)"
            else:
                result_text = f"ผลการวิเคราะห์: {CLASS_NAMES[0]} ({(1-score)*100:.2f}%)"
        else:
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0])) * 100
            result_text = f"ผลการวิเคราะห์: {predicted_class}\nความมั่นใจ: {confidence:.2f}%"

        # 5. ส่งข้อความตอบกลับไปยัง LINE
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result_text)
        )

    except Exception as e:
        print(f"Error handling image: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="เกิดข้อผิดพลาดในการประมวลผลรูปภาพ กรุณาลองใหม่อีกครั้ง")
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
