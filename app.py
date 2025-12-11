from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os

# สร้าง Flask app
app = Flask(__name__)

# -----------------------------
# อ่าน TOKEN / SECRET จาก Environment ของ Render
# -----------------------------
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

# ถ้าอ่านค่าไม่ได้ ให้ขึ้น error ชัด ๆ เลย
if CHANNEL_ACCESS_TOKEN is None:
    raise ValueError(
        "CHANNEL_ACCESS_TOKEN is not set. Check Environment Variables on Render."
    )

if CHANNEL_SECRET is None:
    raise ValueError(
        "CHANNEL_SECRET is not set. Check Environment Variables on Render."
    )

# สร้างตัวเชื่อมกับ LINE
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# -----------------------------
# route สำหรับเช็คว่า server ยังอยู่ดี (Render health check)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "OK"


# -----------------------------
# route ที่ LINE ยิง webhook มาหาเรา
# -----------------------------
@app.route("/callback", methods=["POST"])
def callback():
    # 1) อ่านลายเซ็นจาก header
    signature = request.headers.get("X-Line-Signature", "")

    # 2) อ่าน body (ข้อมูล event จาก LINE)
    body = request.get_data(as_text=True)

    # 3) ให้ handler ตรวจและกระจาย event
    try:
        handler.handle(body, signature)
    except Exception as e:
        # ถ้า token/secret ผิด หรือเซ็นไม่ตรง → 400
        print("webhook error:", e)

    return "OK" , 200


# -----------------------------
# ฟังก์ชันตอบข้อความแบบง่าย ๆ (echo text)
# ถ้ามีข้อความอะไรเข้ามา จะตอบกลับข้อความเดิม
# -----------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # ตัวอย่าง: ตอบกลับข้อความเดิมไปก่อน
    incoming_text = event.message.text

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=incoming_text)
    )


# ส่วนนี้ไว้รันบนเครื่องเราเอง (ไม่กระทบ Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
