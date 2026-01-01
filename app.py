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
            "ส่งรูป X-ray มาได้เลย แล้วฉันจะทำนายว่าเป็น Normal / Pneumonia / TB\n"
            "หมายเหตุ: ผลลัพธ์เป็นการทดลอง ไม่ใช่การวินิจฉัยแพทย์"
        )
    else:
        reply = "ส่งรูป X-ray มาได้เลย 🙂"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    content = line_bot_api.get_message_content(event.message.id)
    image_bytes = b"".join(content.iter_content())

    try:
        label, conf, probs = predict(image_bytes)
        probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)

        prob_lines = []
        for i, p in enumerate(probs):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            prob_lines.append(f"- {name}: {float(p):.3f}")

        reply = (
            f"ผลทำนาย: {label}\n"
            f"ความมั่นใจ: {conf:.3f}\n\n"
            f"รายละเอียดความน่าจะเป็น:\n" + "\n".join(prob_lines) +
            "\n\nหมายเหตุ: เพื่อการทดลอง/การเรียนรู้ ไม่ใช่การวินิจฉัย"
        )
    except Exception as e:
        reply = f"ทำนายไม่สำเร็จ: {type(e).__name__}: {e}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
