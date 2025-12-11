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

# ================== โหลด TOKEN / SECRET จาก Environment ==================
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

print("CHANNEL_ACCESS_TOKEN is None? ->", CHANNEL_ACCESS_TOKEN is None)
print("CHANNEL_SECRET is None? ->", CHANNEL_SECRET is None)

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ================== โหลดโมเดล ==================
# ต้องมีไฟล์ชื่อ best_cnn_xray_E40.keras อยู่ใน repo เดียวกับ app.py
print("Loading model ...")
model = load_model("best_cnn_xray_E40.keras")
print("Model loaded OK")

# ================== ฟังก์ชันเตรียมรูป (Preprocess) ==================
def preprocess_image(path):
    # เปิดรูป
    img = Image.open(path)
    print("Original image mode:", img.mode)

    # บังคับให้เป็น RGB ก่อน แล้วค่อยแปลงเป็นขาวดำ (L)
    img = img.convert("RGB")
    img = img.convert("L")
    img = img.resize((224, 224))  # ให้ตรงกับตอนเทรน

    arr = np.array(img).astype("float32") / 255.0
    print("Image array shape before reshape:", arr.shape)

    # เพิ่มมิติให้เป็น (1, 224, 224, 1)
    arr = arr.reshape(1, 224, 224, 1)
    print("Image array shape after reshape:", arr.shape)

    return arr


# ================== หน้าเช็คว่า server ยังอยู่ ==================
@app.route("/", methods=["GET"])
def home():
    return "OK", 200


# ================== Webhook จาก LINE ==================
@app.route("/callback", methods=["POST"])
def callback():
    # 1) อ่าน signature จาก header
    signature = request.headers.get("X-Line-Signature", "")
    # 2) อ่าน body (ข้อมูล event จาก LINE)
    body = request.get_data(as_text=True)

    print("Request body:", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError as e:
        print("InvalidSignatureError:", e)
        abort(400)
    except Exception as e:
        print("Callback error:", e)
        abort(500)

    return "OK", 200


# ================== รับข้อความตัวอักษร ==================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # ไม่ตอบ event ทดสอบจากหน้า Verify ของ LINE
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return

    text = event.message.text.strip()

    if text.lower() in ["hi", "hello", "สวัสดี"]:
        reply = "สวัสดีค่ะ ✨ ส่งภาพ X-ray มาแล้วหนูจะช่วยทำนายให้นะคะ"
    else:
        reply = f"คุณพิมพ์ว่า: {text}\nหากต้องการให้ช่วยวิเคราะห์ภาพ X-ray ให้ส่งรูปมาได้เลยค่ะ 😊"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ================== รับรูปภาพ X-ray ==================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("===== IMAGE EVENT RECEIVED =====")

    try:
        # 1) ดาวน์โหลดรูปจาก LINE
        message_id = event.message.id
        print("message_id:", message_id)

        message_content = line_bot_api.get_message_content(message_id)
        print("Got message content from LINE")

        # 2) เซฟเป็นไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            for chunk in message_content.iter_content():
                temp.write(chunk)
            temp_path = temp.name

        print("Saved temp image at:", temp_path)

        # 3) เตรียมรูปเข้าโมเดล
        img_arr = preprocess_image(temp_path)

        # 4) ทำนายด้วยโมเดล
        pred = model.predict(img_arr)
        print("Raw prediction:", pred)

        class_id = int(np.argmax(pred, axis=1)[0])
        probability = float(np.max(pred))

        classes = ["Normal", "Pneumonia", "Tuberculosis"]
        result = classes[class_id]

        reply_text = (
            f"ผลวินิจฉัยจากภาพ X-ray: {result}\n"
            f"ความมั่นใจของโมเดล: {probability * 100:.2f}%"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

    except Exception as e:
        # ถ้ามีปัญหา ให้ log ลง console และส่งข้อความกลับไปใน LINE
        print("IMAGE ERROR:", e)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"ขอโทษค่ะ ระบบวิเคราะห์ภาพมีปัญหา: {e}")
        )
