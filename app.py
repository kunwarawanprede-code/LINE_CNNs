from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os

app = Flask(__name__)

# ดึงค่า TOKEN / SECRET จากตัวแปรสภาพแวดล้อม (เดี๋ยวไปใส่ใน Render)
CHANNEL_ACCESS_TOKEN = os.getenv("iREsvpzCRyS7EldzM5P4JhEShiubs6OMPyJpI+B25+twZGFzdZJsmMesUY8xNAbhDCxf/6SEp7QaLc32POCol+YGd1AM5HVoskCCQggWKLI5xa9jYnvj9sop2XKl5XXz8fYHzWnJ5O/EL6TyKF/uHQdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("159e482291bafbd19f4e42b0c0f0e1b5")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


@app.route("/", methods=["GET"])
def home():
    return "OK"   # ให้ Render เช็คสุขภาพได้


@app.route("/callback", methods=['POST'])
def callback():
    # รับลายเซ็นจาก header
    signature = request.headers.get('X-Line-Signature', '')

    # รับ body (ข้อมูล event จาก LINE)
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


# เมื่อมีข้อความเข้ามา
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_text = event.message.text.strip().lower()

    if user_text in ["hi", "hello", "สวัสดี"]:
        reply = "สวัสดีค่าา 👋 ส่งรูป X-ray มาในอนาคตได้ เดี๋ยวหนูจะช่วยวิเคราะห์ให้ 🩻"
    else:
        reply = "ตอนนี้เป็นบอทตัวทดลองอยู่ค่ะ พิมพ์ว่า “สวัสดี” ดูได้เลย 😊"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


if __name__ == "__main__":
    # เวลา run บนเครื่องตัวเอง
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
