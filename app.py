import os
import io
import numpy as np
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

        # 1. อ่านภาพ + แก้ปัญหา EXIF Orientation + แปลงเป็น RGB
        img = Image.open(image_bytes)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        # 2. Resize เป็น 224x224 (ตามที่โมเดลถูก Train มา)
        img = img.resize((224, 224))
        
        # 3. Normalize pixel (0 - 1) มาตรฐาน
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. ส่งประมวลผลในโมเดล TFLite
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
