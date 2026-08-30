import os
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', '')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET', '')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

MODEL_PATH = 'lung_disease_mobilenetv2.h5'
model = load_model(MODEL_PATH)

labels_map = {0: 'Normal (ปอดปกติ)', 1: 'PNEUMONIA (ปอดบวม)', 2: 'TB (วัณโรค)'}
IMG_SIZE = (224, 224)

@app.route("/", methods=['GET'])
def index():
    return "LINE Bot Model Service is Running!"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    temp_img_path = f"/tmp/{event.message.id}.jpg"
    
    with open(temp_img_path, 'wb') as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    img = image.load_img(temp_img_path, target_size=IMG_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(preds)
    confidence = preds[predicted_idx] * 100

    reply_text = f"🩺 ผลวิเคราะห์: {labels_map[predicted_idx]}\n📈 ความมั่นใจ: {confidence:.1f}%"
    
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
