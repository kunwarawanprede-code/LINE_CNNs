import os
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, request, abort

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

MODEL_PATH = 'model.tflite'

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite Model loaded successfully!")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

CLASS_NAMES_TH = {
    'Normal': 'ปกติ (Normal)',
    'Pneumonia': 'ภาวะปอดอักเสบ/ปอดบวม (Pneumonia)'
}
CLASS_NAMES = ['Normal', 'Pneumonia']

# ---------------------------------------------------------
# ฟังก์ชันปรับแสงและคอนทราสต์ภาพเอ็กซเรย์ (CLAHE Enhancement)
# ---------------------------------------------------------
def enhance_xray_image(pil_img):
    # 1. แปลง PIL Image เป็น numpy array (BGR สำหรับ OpenCV)
    img_np = np.array(pil_img)
    
    # 2. แปลงเป็น Grayscale เพื่อปรับ Equalization
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 3. ใช้ CLAHE เพิ่ม Contrast เฉพาะจุด ไม่ให้ภาพขาวจม
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # 4. แปลงกลับเป็น RGB 3 channels
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(enhanced_rgb)

@app.route("/", methods=['GET'])
def index():
    return "Your service is live 🚀"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = io.BytesIO(message_content.content)

        # 1. โหลดภาพและแก้ EXIF Orientation
        img = Image.open(image_bytes)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        # 2. ปรับแต่งโทนสี/คอนทราสต์ภาพก่อนส่งให้โมเดล
        img = enhance_xray_image(img)

        # 3. Resize และ Normalize
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. ทำนายผล
        if interpreter is None:
            raise Exception("TFLite Model is not loaded properly.")

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        if predictions.shape[-1] == 1:
            score = float(predictions[0][0])
            if score > 0.5:
                predicted_label = CLASS_NAMES_TH[CLASS_NAMES[1]]
                confidence = score * 100
            else:
                predicted_label = CLASS_NAMES_TH[CLASS_NAMES[0]]
                confidence = (1 - score) * 100
        else:
            raw_class = CLASS_NAMES[np.argmax(predictions[0])]
            predicted_label = CLASS_NAMES_TH.get(raw_class, raw_class)
            confidence = float(np.max(predictions[0])) * 100

        result_text = (
            "📋 **ผลการวิเคราะห์ภาพถ่ายเอ็กซเรย์ทรวงอก**\n\n"
            f"• **ผลการประมวลผล:** {predicted_label}\n"
            f"• **ระดับความเชื่อมั่น:** {confidence:.2f}%\n\n"
            "⚠️ *หมายเหตุ: ผลการวิเคราะห์นี้จัดทำโดยระบบปัญญาประดิษฐ์เพื่อการคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้ทดแทนการวินิจฉัยโดยแพทย์ผู้เชี่ยวชาญได้*"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result_text)
        )

    except Exception as e:
        print(f"Error handling image: {e}")
        error_text = (
            "ระบบไม่สามารถประมวลผลรูปภาพนี้ได้ในขณะนี้\n\n"
            "ข้อแนะนำ:\n"
            "1. ตรวจสอบไฟล์รูปภาพว่าเป็นภาพถ่ายเอ็กซเรย์ทรวงอกที่ชัดเจน\n"
            "2. ลองส่งไฟล์รูปภาพใหม่อีกครั้ง"
        )
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=error_text)
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
