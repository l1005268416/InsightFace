import cv2
import numpy as np
import json
import os
import time
import insightface
from insightface.app import FaceAnalysis

# 初始化 —— 强制GPU优先，不自动回退CPU
app = FaceAnalysis(
    name='antelopev2', 
    root='./insightface_models',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0)

DB_PATH = 'user_db.json'
THRESHOLD = 0.6

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            data = json.load(f)
            for uid in data:
                data[uid]['embedding'] = np.array(data[uid]['embedding'])
            return data
    return {}

def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def check_access(camera_index=0):
    print("启动门禁系统... 按 'q' 退出")
    
    # ==================== 关键优化：高速摄像头模式 ====================
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Windows 高速模式
    
    # 强制低分辨率 = 巨幅提升流畅度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 摄像头缓存只保留1帧 = 无延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    db = load_db()
    if not db:
        print("数据库为空，请先注册人脸！")
        cap.release()
        return

    # 清理缓存
    for _ in range(10):
        cap.read()
    
    frame_count = 0
    last_faces = []  # 保存上一次结果，让画面不跳不卡

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 每8帧检测一次（比15更流畅，又不卡）
        if frame_count % 15 == 0:
            faces = app.get(frame)
            last_faces = []  # 清空上一次结果
            
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
                
                label = "Unknown"
                color = (0, 0, 255)
                
                if max_sim > THRESHOLD:
                    label = f"Welcome {best_match_user}"
                    color = (0, 255, 0)
                    print(f"[通过] {best_match_user} | 相似度：{max_sim:.2f}")
                else:
                    print(f"[拒绝] 相似度：{max_sim:.2f}")
                
                last_faces.append((x1, y1, x2, y2, label, color))

        # ==================== 核心：不检测时也持续绘制画面 ====================
        for (x1, y1, x2, y2, label, color) in last_faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示画面（永远不卡）
        cv2.imshow("Face Gateway", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_access()