import cv2
import numpy as np
import json
import os
import time
import insightface
from insightface.app import FaceAnalysis

# 初始化
app = FaceAnalysis(name='antelopev2', root='./insightface_models')
app.prepare(ctx_id=0)

DB_PATH = 'user_db.json'
THRESHOLD = 0.6  # 相似度阈值，0.6-0.8 之间较为安全

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            data = json.load(f)
            # 将列表转回 numpy 数组
            for uid in data:
                data[uid]['embedding'] = np.array(data[uid]['embedding'])
            return data
    return {}

def calculate_similarity(vec1, vec2):
    # 计算余弦相似度
    # 公式: (A·B) / (||A|| * ||B||)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def check_access(camera_index=0):
    print("启动门禁系统... 按 'q' 退出")
    
    # 加载摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    db = load_db()
    if not db:
        print("数据库为空，请先注册人脸！")
        cap.release()
        return

    # 丢弃前20帧缓存
    for _ in range(20):
        cap.read()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 每15帧处理一次人脸检测，降低CPU占用
        if frame_count % 15 == 0:
            faces = app.get(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
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
                    print(f"[ACCESS GRANTED] User: {best_match_user}, Similarity: {max_sim:.2f}")
                else:
                    print(f"[ACCESS DENIED] Similarity: {max_sim:.2f} (Threshold: {THRESHOLD})")
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 实时显示画面
        cv2.imshow("Face Gateway", frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_access()