from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageMessage,
)
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# สร้าง Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# โหลด TOKEN / SECRET จาก Environment Variables (ตั้งใน Render แล้ว)
# -----------------------------
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

if CHANNEL_ACCESS_TOKEN is None or CHANNEL_SECRET is None:
    print("ERROR: CHANNEL_ACCESS_TOKEN หรือ CHANNEL_SECRET ไม่มีค่าใน Environment !!")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# -----------------------------
# โหลดโมเดลที่เซฟไว้ใน GitHub (Render จะดึงมาด้วยเวลาดึง repo)
# -----------------------------
# ชื่อไฟล์ต้องตรงกับไฟล์ใน repo เช่น best_cnn_xray_E40.keras
model = load_model("best_cnn_xray_E40.keras")

# -----------------------------
# ฟังก์ชัน Preprocess รูป X-ray ให้ตรงกับที่โมเดลต้องการ
# ตอนนี้โมเดลต้องการ input ขนาด (224, 224, 3) = RGB
# -----------------------------
def preprocess_image(path):
    # เปิดภาพเป็น RGB (3 ช่องสี) เพราะโมเดลต้องการ 3-channel
    img = Image.open(path).convert("RGB")

    # ปรับขนาดภาพให้ตรงกับตอนเทรน
    img = img.resize((224, 224))

    # แปลงเป็น array และ normalize ให้อยู่ในช่วง 0-1
    img = np.array(img) / 255.0

    # เพิ่มมิติ batch ให้เป็น (1, 224, 224, 3)
    img = img.reshape(1, 224, 224, 3)

    return img


# -----------------------------
# route สำหรับเช็คว่า server ยังอยู่ดี (Render health check)
# เปิดใน browser: https://line-cnns-1.onrender.com/
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "OK", 200


# -----------------------------
# Webhook ที่ LINE จะยิงมาหาเรา
# -----------------------------
@app.route("/callback", methods=["POST"])
def callback():
    # 1) อ่าน signature จาก header
    signature = request.headers.get("X-Line-Signature", "")

    # 2) อ่าน body (ข้อมูล event จาก LINE)
    body = request.get_data(as_text=True)

    # 3) ให้ handler ตรวจและกระจาย event
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Check CHANNEL_SECRET/LINE header.")
        abort(400)
    except Exception as e:
        print("webhook error:", e)
        abort(400)

    return "OK", 200


# =============================
# Handler: ข้อความตัวอักษร (echo กลับไป)
# =============================
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    # ไม่ตอบ event ทดสอบจาก LINE (ตอนกด Verify)
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return

    user_text = event.message.text
    reply = f"คุณพิมพ์ว่า: {user_text}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# =============================
# Handler: รับรูป X-ray แล้วให้โมเดลทำนาย
# =============================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # ไม่ตอบ event ทดสอบจาก LINE เช่นตอน Verify
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return

    try:
        # 1) ดาวน์โหลดรูปจาก LINE
        message_id = event.message.id
        message_content = line_bot_api.get_message_content(message_id)

        # 2) เซฟเป็นไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            for chunk in message_content.iter_content():
                temp.write(chunk)
            temp_path = temp.name

        # 3) Preprocess รูปให้ตรงกับโมเดล
        img = preprocess_image(temp_path)

        # 4) ให้โมเดลทำนาย
        preds = model.predict(img)[0]  # ได้ array ยาว 3 ค่า
        classes = ["Normal", "Pneumonia", "Tuberculosis"]

        # แปลงเป็นเปอร์เซ็นต์
        probs = preds * 100.0
        top_idx = int(np.argmax(preds))
        top_class = classes[top_idx]

        # 5) สร้างข้อความตอบกลับ
        lines = []
        for cls, p in zip(classes, probs):
            lines.append(f"- {cls}: {p:.1f}%")

        reply_text = (
            "ผลวิเคราะห์เบื้องต้นจากภาพ X-ray 🩻\n\n"
            + "\n".join(lines)
            + f"\n\nสรุป: โมเดลให้ความน่าจะเป็นสูงสุดว่าเป็นกลุ่ม **{top_class}**"
        )

        # 6) ส่งผลกลับไปที่ LINE
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

    except Exception as e:
        print("IMAGE PREDICT ERROR:", e)
        # ถ้า error ให้ส่งข้อความขอโทษกลับไป
        error_text = "ขอโทษค่ะ ระบบวิเคราะห์ภาพมีปัญหา โปรดลองใหม่อีกครั้งนะคะ"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=error_text)
        )


# -----------------------------
# จุดเริ่มรันถ้ารันบนเครื่องตัวเอง (ไม่จำเป็นสำหรับ Render แต่มีไว้ไม่เสียหาย)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
