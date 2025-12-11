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

# ====== โหลด TOKEN / SECRET จาก Environment ======
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ====== โหลดโมเดล ======
model = load_model("best_cnn_xray_E40.keras")

# ====== ฟังก์ชัน Preprocess รูป X-ray ======
def preprocess_image(path):
    img = Image.open(path).convert("L")       # แปลงเป็นขาวดำ
    img = img.resize((224, 224))              # ปรับขนาด
    img = np.array(img) / 255.0               # Normalize
    img = img.reshape(1, 224, 224, 1)         # เพิ่มมิติให้ตรงกับตอนเทรน
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

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        # กรณี SECRET ไม่ตรง
        print("❌ Invalid signature — ตรวจ CHANNEL_SECRET อีกครั้ง")
        abort(400)
    except Exception as e:
        # กัน error อื่น ๆ เช่น Invalid reply token
        print("❌ webhook error:", e)
        # ตอบ 200 กลับไปให้ LINE เพื่อกันระบบยิงซ้ำ
        return "OK", 200

    return "OK", 200


# ====== รับข้อความ (Echo) ======
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


# ====== รับรูป X-ray ======
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # ดาวน์โหลดรูปจาก LINE
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    # เซฟเป็นไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        for chunk in message_content.iter_content():
            temp.write(chunk)
        temp_path = temp.name

    # Preprocess รูป
    img = preprocess_image(temp_path)
    pred = model.predict(img)
    class_id = np.argmax(pred)

    classes = ["Normal", "Pneumonia", "Tuberculosis"]
    result = classes[class_id]

    # ส่งผลกลับ LINE
    reply_text = f"ผลวินิจฉัยจากภาพ X-ray: {result}"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )
