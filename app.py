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

# ---------------------------------------------------------
# พจนานุกรมแปลชื่อคลาสเป็นภาษาไทยแบบทางการ
# ---------------------------------------------------------
CLASS_NAMES_TH = {
    'Normal': 'ปกติ (Normal)',
    'Pneumonia': 'ภาวะปอดอักเสบ/ปอดบวม (Pneumonia)',
    'Tuberculosis': 'วัณโรคปอด (Tuberculosis)'
}

# ⚠️ เช็กเรียงลำดับให้ตรงกับตอน Train โมเดล
CLASS_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']

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
        
        # คำนวณผลลัพธ์แบบ Multi-class
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx]) * 100

        # ---------------------------------------------------------
        # 5. เช็กค่าความเชื่อมั่น (Threshold Check)
        # ---------------------------------------------------------
        CONFIDENCE_THRESHOLD = 50.0  # สามารถปรับเปลี่ยนระดับ % ขั้นต่ำตรงนี้ได้ครับ

        if confidence < CONFIDENCE_THRESHOLD:
            result_text = (
                "⚠️ **ไม่สามารถวิเคราะห์ภาพถ่ายนี้ได้**\n\n"
                "ระบบไม่แน่ใจว่าภาพนี้เป็นภาพถ่ายเอ็กซเรย์ทรวงอก หรือลักษณะภาพไม่ตรงกับเงื่อนไขการตรวจวิเคราะห์\n\n"
                "📌 **คำแนะนำ:**\n"
                "• กรุณาส่งภาพถ่ายเอ็กซเรย์ทรวงอก (Chest X-ray) ที่มีความคมชัดและเห็นปอดชัดเจนใหม่อีกครั้ง"
            )
        else:
            if predicted_idx < len(CLASS_NAMES):
                raw_class = CLASS_NAMES[predicted_idx]
                predicted_label = CLASS_NAMES_TH.get(raw_class, raw_class)
            else:
                predicted_label = f"คลาสที่ {predicted_idx}"

            result_text = (
                "📋 **ผลการวิเคราะห์ภาพถ่ายเอ็กซเรย์ทรวงอก**\n\n"
                f"• **ผลการประมวลผล:** {predicted_label}\n"
                f"• **ระดับความเชื่อมั่น:** {confidence:.2f}%\n\n"
                "⚠️ *หมายเหตุ: ผลการวิเคราะห์นี้จัดทำโดยระบบปัญญาประดิษฐ์เพื่อการคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้ทดแทนการวินิจฉัยโดยแพทย์ผู้เชี่ยวชาญได้*"
            )

        # 6. ส่งข้อความตอบกลับไปยัง LINE
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result_text)
        )

    except Exception as e:
        print(f"Error handling image: {e}")
        error_text = (
            "ระบบไม่สามารถประมวลผลรูปภาพนี้ได้ในขณะนี้\n\n"
            "ข้อแนะนำ:\n"
            "1. ตรวจสอบไฟล์รูปภาพว่าเป็นภาพถ่ายเอ็กซเรย์ทรวงอกที่ชัดเจน\n"
            "2. ลองส่งไฟล์รูปภาพใหม่อีกครั้ง"
        )
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=error_text)
            )
        except Exception as line_err:
            print(f"Failed to send error reply to LINE: {line_err}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
