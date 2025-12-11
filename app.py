from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# ====== โหลด TOKEN / SECRET จาก Environment (Render) ======
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ====== โหลดโมเดล ======
model = load_model("best_cnn_xray_E40.keras")

# ====== ฟังก์ชัน Preprocess รูป X-ray ======
def preprocess_image(path):
    # แปลงเป็นขาวดำ
    img = Image.open(path).convert("L")
    # ปรับขนาดให้ตรงกับตอนเทรน
    img = img.resize((224, 224))
    # แปลงเป็น array และ normalize
    img = np.array(img) / 255.0
    # เพิ่มมิติให้เป็น (1, 224, 224, 1)
    img = img.reshape(1, 224, 224, 1)
    return img


@app.route("/", methods=["GET"])
def home():
    return "OK", 200


# ====== Webhook จาก LINE ======
@app.route("/callback", methods=["POST"])
def callback():
    # 1) อ่าน signature จาก header
    signature = request.headers.get("X-Line-Signature", "")
    # 2) อ่าน body (ข้อมูล event จาก LINE)
    body = request.get_data(as_text=True)

    # debug ดู event ที่วิ่งเข้ามา
    print("===== Webhook Body =====")
    print(body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Check CHANNEL_SECRET หรือ Webhook URL ให้ถูก")
        abort(400)
    except Exception as e:
        print("webhook error:", e)
        abort(400)

    return "OK", 200


# ====== รับข้อความ (Echo) ======
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # ไม่ตอบ event ทดสอบจาก LINE (ตอนกด Verify)
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return

    user_id = event.source.user_id
    received_text = event.message.text
    reply = f"คุณพิมพ์ว่า: {received_text}"

    try:
        # ใช้ push_message แทน reply_message เพื่อตัดปัญหา reply token
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=reply)
        )
    except LineBotApiError as e:
        print("Reply error (text):", e.status_code, e.error.message, e.error.details)


# ====== รับรูป X-ray ======
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # กันไม่ให้ error ตอน Verify / ทดสอบ
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return

    user_id = event.source.user_id

    # ดาวน์โหลดรูปจาก LINE
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    # เซฟเป็นไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        for chunk in message_content.iter_content():
            temp.write(chunk)
        temp_path = temp.name

    # Preprocess รูปและทำนาย
    try:
        img = preprocess_image(temp_path)
        pred = model.predict(img)
        class_id = int(np.argmax(pred))

        classes = ["Normal", "Pneumonia", "Tuberculosis"]
        result = classes[class_id]
        reply_text = f"ผลวินิจฉัยจากภาพ X-ray: {result}"
    except Exception as e:
        print("Model/predict error:", e)
        reply_text = "ขอโทษค่ะ ระบบวิเคราะห์ภาพมีปัญหา ลองใหม่อีกครั้งนะ"

    # ส่งผลกลับ LINE (ใช้ push)
    try:
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=reply_text)
        )
    except LineBotApiError as e:
        print("Reply error (image):", e.status_code, e.error.message, e.error.details)
