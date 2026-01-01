import os
import io
import numpy as np
from PIL import Image

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage

from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
app = Flask(__name__)

CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("Missing CHANNEL_ACCESS_TOKEN or CHANNEL_SECRET in environment variables.")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# -----------------------------
# MODEL (โหลดครั้งเดียว)
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")  # ถ้าไฟล์ชื่ออื่น ให้เปลี่ยน
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))      # ขนาดภาพที่โมเดลรับเข้า

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

# ชื่อคลาส (แก้ตามโมเดลของคุณ)
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TB"]


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """แปลง PIL image -> numpy tensor ตามรูปแบบโมเดล"""
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(pil_img).astype("float32") / 255.0  # normalize 0-1
    arr = np.expand_dims(arr, axis=0)                  # (1, H, W, 3)
    return arr


def predict(pil_img: Image.Image):
    """คืน (label, confidence)"""
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]             # shape: (num_classes,)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    conf = float(probs[idx])
    return label, conf


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK", 200


# -----------------------------
# HANDLERS
# -----------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event: MessageEvent):
    text = (event.message.text or "").strip()

    # ตอบแชททั่วไป
    if text.lower() in ["hi", "hello", "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ"]:
        msg = "สวัสดีครับ ✅ ส่งรูป X-ray มาได้เลย เดี๋ยวผมช่วยทำนาย (NORMAL / PNEUMONIA / TB)"
    else:
        msg = "พิมพ์ได้ครับ ✅ แต่ถ้าจะให้ทำนาย ส่ง “รูปภาพ X-ray” มาได้เลย"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event: MessageEvent):
    try:
        # 1) ดึงไฟล์รูปจาก LINE
        message_id = event.message.id
        content = line_bot_api.get_message_content(message_id)

        image_bytes = b""
        for chunk in content.iter_content():
            image_bytes += chunk

        # 2) เปิดเป็น PIL
        pil_img = Image.open(io.BytesIO(image_bytes))

        # 3) ทำนาย
        label, conf = predict(pil_img)

        # 4) ตอบกลับ
        reply = f"ผลการทำนาย: {label}\nความมั่นใจ: {conf*100:.2f}%"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    except Exception as e:
        # สำคัญ: กันบอทเงียบ + ช่วยดู log
        print("ERROR in handle_image:", repr(e))
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="เกิดข้อผิดพลาดตอนประมวลผลรูปภาพ ❌ ลองส่งใหม่อีกครั้งได้ไหม")
        )


if __name__ == "__main__":
    # Render จะไม่ใช้บรรทัดนี้ตอนรันด้วย gunicorn แต่ไว้เทส local ได้
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
