from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# ====== โหลด TOKEN / SECRET จาก Environment บน Render ======
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ====== โหลดโมเดล (ไฟล์อยู่ใน repo เดียวกับ app.py) ======
model = load_model("best_cnn_xray_E40.keras")


# ====== ฟังก์ชัน Preprocess รูป X-ray ======
def preprocess_image(path):
    # แปลงเป็น RGB 3 channel (ให้ตรงกับตอนเทรน)
    img = Image.open(path).convert("RGB")

    # ปรับขนาดให้ตรงกับตอนเทรน (ถ้าเทรนไว้ 224x224 ก็ใช้ 224)
    img = img.resize((224, 224))

    # แปลงเป็น array และ normalize
    img = np.array(img) / 255.0     # shape (224, 224, 3)

    # เพิ่ม batch dimension -> (1, 224, 224, 3)
    img = img.reshape(1, 224, 224, 3)

    return img


@app.route("/", methods=["GET"])
def home():
    return "OK", 200


# ====== Webhook จาก LINE ======
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print("webhook error:", e)
        abort(400)

    return "OK", 200


# ====== รับข้อความตัวอักษร (echo) ======
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # ไม่ตอบ event ทดสอบจาก LINE (ตอนกด Verify)
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return "OK"

    received_text = event.message.text
    reply = f"คุณพิมพ์ว่า: {received_text}"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ====== รับรูป X-ray แล้วทำนาย ======
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        # 1) ดาวน์โหลดรูปจาก LINE
        message_id = event.message.id
        message_content = line_bot_api.get_message_content(message_id)

        # 2) เซฟเป็นไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            for chunk in message_content.iter_content():
                temp.write(chunk)
            temp_path = temp.name

        print("Downloaded image:", temp_path)

        # 3) Preprocess รูป
        img = preprocess_image(temp_path)

        # 4) ทำนาย
        pred = model.predict(img)
        print("Raw prediction:", pred)

        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100.0

        classes = ["Normal", "Pneumonia", "Tuberculosis"]
        result = classes[class_id]

        # 5) ส่งผลกลับ LINE (พร้อม % ความมั่นใจ)
        reply_text = (
            f"ผลวินิจฉัยจากภาพ X-ray\n"
            f"- โรค: {result}\n"
            f"- ความมั่นใจ: {confidence:.2f}%"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

    except Exception as e:
        # ถ้า error ให้ log รายละเอียดไว้ดูใน Render logs
        print("ERROR in handle_image:", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="ขอโทษค่ะ ระบบวิเคราะห์ภาพมีปัญหา ลองส่งใหม่อีกครั้งนะคะ"
            )
        )
