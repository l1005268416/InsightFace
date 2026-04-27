import cv2
import numpy as np
import json
import os
import insightface
from insightface.app import FaceAnalysis

# 初始化全局应用 (单例模式，避免重复加载模型)
app = FaceAnalysis(name="antelopev2", root="./insightface_models")
app.prepare(ctx_id=0)

DB_PATH = "user_db.json"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def register_face(image_path, user_id):
    print(f"正在注册用户: {user_id} ...")
    
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return False

    # 2. 检测人脸
    # get() 返回所有检测到的人脸对象
    faces = app.get(img)
    
    if len(faces) == 0:
        print("未检测到人脸，请确保图片清晰且包含正面人脸。")
        return False
    
    # 3. 选择最佳人脸
    # 我们选择检测分数最高（最清晰/最正面）的那张脸
    # 也可以根据框的大小选择最大的脸
    best_face = max(faces, key=lambda x: x.det_score)
    
    # 4. 提取特征向量 (Embedding)
    # embedding 是一个 float32 数组，长度为 512
    embedding = best_face.embedding
    
    # 5. 存入数据库
    db = load_db()
    
    # 检查是否已存在该用户
    if user_id in db:
        print(f"用户 {user_id} 已存在，更新其特征...")
    
    # 将 numpy 数组转换为列表以便 JSON 存储
    db[user_id] = {
        "embedding": embedding.tolist(),
        "image_path": image_path
    }
    
    save_db(db)
    print(f"用户 {user_id} 注册成功。")
    return True

if __name__ == "__main__":
    # 测试注册
    # register_face("images/user1.png", "Alice")
    # register_face("images/user2.png", "Bob")
    register_face("images/user3.jpg", "liubeibei")
