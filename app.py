import os
import io
import numpy as np
from PIL import Image

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage

# ใช้ TFLite runtime (เบากว่า TensorFlow มาก เหมาะกับ Render Free)
import tflite_runtime.interpreter as tflite


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# LINE Config (ตั้งใน Render -> Environment)
# -----------------------------
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("Missing CHANNEL_ACCESS_TOKEN or CHANNEL_SECRET in environment variables.")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# -----------------------------
# Model Config
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))  # ต้องตรงกับที่เทรน
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TB"]   # แก้ชื่อให้ตรงกับคลาสของคุณ

# -----------------------------
# Load TFLite model (โหลดครั้งเดียว)
# -----------------------------
print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded!")


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """PIL -> np array (1, IMG_SIZE, IMG_SIZE, 3) float32 0-1"""
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_tflite(pil_img: Image.Image):
    """return (label, confidence)"""
    x = preprocess_image(pil_img)

    # set input
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    # get output
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    conf = float(probs[idx])
    return label, conf


# -----------------------------
# Routes
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
# Handlers
# -----------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event: MessageEvent):
    text = (event.message.text or "").strip()

    if text.lower() in ["hi", "hello", "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ"]:
        msg = "สวัสดีครับ ✅ ส่งรูป X-ray มาได้เลย เดี๋ยวทำนาย (NORMAL / PNEUMONIA / TB)"
    else:
        msg = "พร้อมใช้งานครับ ✅ ถ้าจะทำนาย ส่งรูป X-ray มาได้เลย"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event: MessageEvent):
    try:
        # 1) ดึง bytes รูปจาก LINE
        message_id = event.message.id
        content = line_bot_api.get_message_content(message_id)

        image_bytes = b""
        for chunk in content.iter_content():
            image_bytes += chunk

        # 2) เปิดภาพ
        pil_img = Image.open(io.BytesIO(image_bytes))

        # 3) ทำนาย
        label, conf = predict_tflite(pil_img)

        # 4) ตอบกลับ
        reply = f"ผลการทำนาย: {label}\nความมั่นใจ: {conf*100:.2f}%"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    except Exception as e:
        # กันบอทเงียบ + ให้ดู Logs ได้
        print("ERROR in handle_image:", repr(e))
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="เกิดข้อผิดพลาดตอนประมวลผลรูปภาพ ❌ ลองส่งใหม่อีกครั้งได้ไหม")
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
