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

# Preprocess ฟังก์ชัน
def preprocess_image(path):
    img = Image.open(path).convert("L")  
    img = img.resize((224, 224))      
    img = np.array(img) / 255.0      
    img = img.reshape(1, 224, 224, 1) 
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


# ====== รับข้อความ (Echo) ======
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    received_text = event.message.text
    reply = f"คุณพิมพ์ว่า: {received_text}"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))


# ====== รับรูป X-ray ======
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # ดาวน์โหลดรูปจาก LINE
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        for chunk in message_content.iter_content():
            temp.write(chunk)
        temp_path = temp.name

    # Preprocess
    img = preprocess_image(temp_path)
    pred = model.predict(img)
    class_id = np.argmax(pred)

    classes = ["Normal", "Pneumonia", "Tuberculosis"]
    result = classes[class_id]

    # ส่งผลกลับ LINE
    reply_text = f"ผลวินิจฉัยจากภาพ X-ray: {result}"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
