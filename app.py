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

# ====== โหลด TOKEN / SECRET จาก Environment (Render) ======
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ====== โหลดโมเดล ======
model = load_model("best_cnn_xray_E40.keras")

# ====== ฟังก์ชัน preprocess รูป X-ray ======
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
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("webhook error:", e)
        abort(400)

    return "OK", 200


# ====== รับข้อความ (echo ธรรมดา) ======
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    # ไม่ตอบตอน LINE ส่ง event ทดสอบ (เวลา Verify)
    if event.reply_token in (
        "00000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffff",
    ):
        return "OK"

    reply = f"คุณพิมพ์ว่า: {event.message.text}"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ====== รับรูป X-ray แล้วทำนาย + แสดงเปอร์เซ็นต์ ======
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        # 1) ดาวน์โหลดรูปจาก LINE
        message_id = event.message.id
        content = line_bot_api.get_message_content(message_id)

        # 2) เซฟเป็นไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            for chunk in content.iter_content():
                tf.write(chunk)
            temp_path = tf.name

        # 3) เตรียมภาพ
        img = preprocess_image(temp_path)

        # 4) ทำนายผลจากโมเดล
        pred = model.predict(img)[0]          # ได้ array เช่น [0.1, 0.7, 0.2]
        classes = ["Normal", "Pneumonia", "Tuberculosis"]

        # แปลงเป็นเปอร์เซ็นต์ (เผื่อ model ไม่ softmax ก็ยังดูเป็นสัดส่วนได้)
        probs = pred / np.sum(pred)
        probs = probs * 100.0

        # index ของ class ที่โอกาสมากที่สุด
        best_idx = int(np.argmax(probs))
        best_class = classes[best_idx]
        best_prob = float(probs[best_idx])

        # 5) สร้างข้อความตอบกลับ
        lines = ["ผลวินิจฉัยจากภาพ X-ray (เปอร์เซ็นต์ความน่าจะเป็น):"]
        for cls, p in zip(classes, probs):
            lines.append(f"- {cls}: {p:.2f}%")

        lines.append("")
        lines.append(f"สรุป: มีโอกาสเป็น \"{best_class}\" สูงที่สุด ({best_prob:.2f}%)")
        lines.append("")
        lines.append("※ เป็นการวิเคราะห์เบื้องต้นจากโมเดล AI เท่านั้น ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้")

        reply_text = "\n".join(lines)

    except Exception as e:
        print("IMAGE ERROR:", e)
        reply_text = "ขอโทษค่ะ ระบบวิเคราะห์ภาพมีปัญหา ลองส่งใหม่อีกครั้งนะคะ"

    # 6) ส่งข้อความกลับไป
