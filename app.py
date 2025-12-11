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

# ====== โหลด TOKEN / SECRET ======
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ====== โหลดโมเดล ======
model = load_model("best_cnn_xray_E40.keras")

# ====== ฟังก์ชัน Preprocess รูป X-ray ======
def preprocess_image(path):
    img = Image.open(path).convert("RGB")   # ⭐ แปลงเป็น RGB (3 channels)
    img = img.resize((224, 224))           # resize ให้ตรงกับตอน train
    img = np.array(img) / 255.0            # normalize
    img = img.reshape(1, 224, 224, 3)      # ⭐ โมเดลต้องรับ 3 ช่อง
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


# ====== ตอบข้อความธรรมดา ======
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # ไม่ตอบ event ที่มาจากปุ่ม Verify ของ LINE Developer
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return "OK"

    text = event.message.text
    reply = f"คุณพิมพ์ว่า: {text}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ====== รับรูป X-ray ======
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        # โหลดไฟล์รูปจาก LINE
        message_id = event.message.id
        message_content = line_bot_api.get_message_content(message_id)

        # เซฟไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            for chunk in message_content.iter_content():
                temp.write(chunk)
            temp_path = temp.name

        # Preprocess
        img = preprocess_image(temp_path)

        # Predict
        pred = model.predict(img)[0]
        class_id = np.argmax(pred)

        classes = ["Normal", "Pneumonia", "Tuberculosis"]
        result = classes[class_id]

        # แสดงความน่าจะเป็น (%)
        prob = float(pred[class_id]) * 100
        prob_text = f"{prob:.2f}%"

        # ส่งกลับ
        reply = f"ผลวินิจฉัย: {result}\nความมั่นใจ: {prob_text}"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

    except Exception as e:
        print("Predict ERROR:", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="ขออภัยค่ะ ระบบวิเคราะห์ภาพมีปัญหา ลองใหม่อีกครั้งนะคะ 🙏")
        )


if __name__ == "__main__":
    app.run()
