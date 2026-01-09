import os
import io
import numpy as np
from PIL import Image

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage

from tflite_runtime.interpreter import Interpreter  # ✅ ชัด ไม่ชน tensorflow

# ---------------------------
# Paths / Config
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")  # ✅ ชี้แบบ absolute
CLASS_NAMES = ["Normal", "Pneumonia", "TB"]

CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("Missing env vars: CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# ---------------------------
# Load TFLite model once
# ---------------------------
if not os.path.exists(MODEL_PATH):
    # debug ช่วยดูว่าในเครื่องจริงมีไฟล์อะไรบ้าง
    files = os.listdir(BASE_DIR)
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        f"FILES IN BASE_DIR: {files}\n"
        f"(ต้องชื่อ 'model.tflite' และอยู่โฟลเดอร์เดียวกับ app.py)"
    )

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_shape = input_details[0]["shape"]
if len(in_shape) == 4:
    _, IN_H, IN_W, IN_C = in_shape
elif len(in_shape) == 3:
    _, IN_H, IN_W = in_shape
    IN_C = 1
else:
    raise RuntimeError(f"Unsupported input shape: {in_shape}")

IN_DTYPE = input_details[0]["dtype"]

# ---------------------------
# LINE + Flask
# ---------------------------
app = Flask(__name__)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ---------------------------
# ✅ Pretty Reply Helpers (เปอร์เซ็นต์ + อิโมจิน่ารัก)
# ---------------------------
def fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:.1f}%"
    except:
        return "N/A"

def build_pretty_reply(pred_label: str, probs_dict: dict) -> str:
    # confidence ของคลาสที่ทำนาย
    confidence = probs_dict.get(pred_label, max(probs_dict.values()) if probs_dict else 0.0)

    # เรียงจากมาก -> น้อย
    items = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)

    emoji = {"Normal": "✅", "Pneumonia": "🚨", "TB": "🚨"}
    label_th = {
        "Normal": "ปกติ (Normal)",
        "Pneumonia": "ปอดอักเสบ (Pneumonia)",
        "TB": "วัณโรค (TB)"
    }

    lines = []
    lines.append("🩺✨ ผลวิเคราะห์ภาพ X-ray ✨")
    lines.append(f"🎯 ผลทำนาย: {emoji.get(pred_label,'📍')} {label_th.get(pred_label, pred_label)}")
    lines.append(f"🔎 ความมั่นใจ: {fmt_pct(confidence)}")
    lines.append("")
    lines.append("📊💖 ความน่าจะเป็นของแต่ละกลุ่ม")

    for k, v in items:
        lines.append(f"• {emoji.get(k,'•')} {label_th.get(k,k)} → {fmt_pct(float(v))}")

    if confidence < 0.65:
        lines.append("")
        lines.append("⚠️😿 ความมั่นใจยังไม่สูงมาก นำข้อมูลไปใช้อย่างระมัดระวังนะคะ🙇‍♀️")

    lines.append("")
    lines.append("📌 หมายเหตุ: หากนำไปใช้ทางการแพทย์ %ความมั่นใจควรมากกว่า65% ")
    return "\n".join(lines)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((int(IN_W), int(IN_H)))

    x = np.array(img, dtype=np.float32)

    if int(IN_C) == 1:
        x = np.mean(x, axis=2, keepdims=True)

    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    if IN_DTYPE == np.float32:
        return x.astype(np.float32)
    elif IN_DTYPE == np.uint8:
        return (x * 255.0).clip(0, 255).astype(np.uint8)
    else:
        return x.astype(IN_DTYPE)

def predict(image_bytes: bytes):
    x = preprocess_image(image_bytes)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])
    y = np.array(y).squeeze()

    # dequantize ถ้าเป็น uint8
    if y.dtype == np.uint8:
        scale, zero_point = output_details[0].get("quantization", (1.0, 0))
        if scale and scale != 0:
            y = (y.astype(np.float32) - zero_point) * scale
        else:
            y = y.astype(np.float32)

    # softmax ถ้าไม่รวม ~1
    if not np.isclose(np.sum(y), 1.0, atol=1e-2):
        e = np.exp(y - np.max(y))
        y = e / np.sum(e)

    idx = int(np.argmax(y))
    conf = float(y[idx])
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
    return label, conf, y

@app.route("/", methods=["GET"])
def home():
    return "OK - LINE CNNs (TFLite) is running", 200

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    msg = event.message.text.strip().lower()
    if msg in ["help", "วิธีใช้", "ใช้ยังไง", "ช่วยด้วย"]:
        reply = (
            "ส่งรูป X-ray มาได้เลย แล้วฉันจะทำนายว่าเป็น Normal / Pneumonia / TB 🩻✨\n"
            "หมายเหตุ: ผลลัพธ์เป็นการทดลอง ไม่ใช่การวินิจฉัยแพทย์นะคะ 💖"
        )
    else:
        reply = "ส่งรูป X-ray มาได้เลยน้า 🙂🩻"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    content = line_bot_api.get_message_content(event.message.id)
    image_bytes = b"".join(content.iter_content())

    try:
        label, conf, probs = predict(image_bytes)

        # probs เป็น array -> แปลงเป็น dict ตาม CLASS_NAMES
        probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(CLASS_NAMES), len(probs)))}

        # ✅ ตอบแบบสวย + เปอร์เซ็นต์ + อิโมจิ
        reply = build_pretty_reply(label, probs_dict)

    except Exception as e:
        reply = f"😿 ทำนายไม่สำเร็จ: {type(e).__name__}: {e}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
