import cv2
import numpy as np
import json
import os
import time
import threading
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont  # 新增：PIL 库

# 初始化模型
app = FaceAnalysis(name='antelopev2', root='./insightface_models')
app.prepare(ctx_id=0)

DB_PATH = 'user_db.json'
THRESHOLD = 0.6

# 全局变量：子线程计算结果，主线程显示
last_faces = []
frame_buffer = None
lock = threading.Lock()
# ========== 关键：中文显示函数 ==========
def cv2_put_chinese(img, text, position, color=(0, 255, 0), font_size=20):
    # 解决 OpenCV 不支持中文的问题
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 请根据你的系统修改字体路径
    # Windows 常用字体路径：
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
    # Linux 常用："NotoSansCJK-Regular.ttc"
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r', encoding="utf-8") as f:
            data = json.load(f)
        for uid in data:
            data[uid]['embedding'] = np.array(data[uid]['embedding'])
        return data
    return {}

def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ====================== 子线程：后台识别人脸（不卡画面） ======================
def face_worker(db):
    global last_faces, frame_buffer
    while True:
        time.sleep(0.03)
        if frame_buffer is None:
            continue

        # 拿一帧去检测
        with lock:
            frame = frame_buffer.copy()

        # 后台计算（这里再怎么卡都不影响画面）
        faces = app.get(frame)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.embedding
            best_match_user = None
            max_sim = -1

            for user_id, user_data in db.items():
                sim = calculate_similarity(embedding, user_data['embedding'])
                if sim > max_sim:
                    max_sim = sim
                    best_match_user = user_id

            if max_sim > THRESHOLD:
                label = f"Welcome {best_match_user}"
                color = (0, 255, 0)
                print(f"[通过] {best_match_user} | {max_sim:.2f}")
            else:
                label = "Unknown"
                color = (0, 0, 255)
                print(f"[拒绝] {max_sim:.2f}")

            results.append((x1, y1, x2, y2, label, color))

        # 更新结果
        with lock:
            last_faces = results

# ====================== 主线程：只负责显示（永远丝滑） ======================
def check_access(camera_index=0):
    global frame_buffer, last_faces
    print("启动门禁系统... 按 'q' 退出")

    # 摄像头高速模式
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    db = load_db()
    if not db:
        print("数据库为空！")
        cap.release()
        return

    # 启动子线程
    t = threading.Thread(target=face_worker, args=(db,), daemon=True)
    t.start()

    # 主线程：只显示画面，绝对不卡顿
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 把帧传给子线程
        with lock:
            frame_buffer = frame.copy()

        # 绘制最新结果
        with lock:
            for (x1, y1, x2, y2, label, color) in last_faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                frame = cv2_put_chinese(frame, label, (x1, y1 - 30), color=color, font_size=22)

        # 画面永远流畅
        cv2.imshow("Face Gateway", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_access()