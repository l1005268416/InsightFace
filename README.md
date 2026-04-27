# 人脸识别门禁系统

基于 InsightFace 的人脸识别门禁系统（demo程序），支持人脸注册和实时门禁识别。

## 环境要求

- Python 3.10+
- Windows 系统（需要 GUI 显示实时画面）

## 依赖安装

```bash
pip install insightface opencv-python onnxruntime Pillow numpy
```

## 项目结构

```
InsightFace/
├── gateway.py              # 门禁识别主程序（实时摄像头）
├── register_face.py        # 人脸注册程序
├── user_db.json            # 用户人脸特征数据库
├── images/                 # 用户注册图片目录
│   ├── user1.png
│   ├── user2.png
│   └── user3.jpg
└── insightface_models/     # AI 模型目录
    └── models/
        └── antelopev2/
            ├── 1k3d68.onnx
            ├── 2d106det.onnx
            ├── genderage.onnx
            ├── glintr100.onnx
            └── scrfd_10g_bnkps.onnx
```

## 快速开始

### 1. 下载模型

模型会自动下载，或手动下载后放入 `insightface_models/models/` 目录：

[antelopev2.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip)

解压后确保文件结构为 `insightface_models/models/antelopev2/*.onnx`。

### 2. 注册人脸

```bash
python register_face.py
```

在 `register_face.py` 的 `__main__` 中修改要注册的图片路径和用户 ID：

```python
if __name__ == "__main__":
    register_face("images/user3.jpg", "liubeibei")
```

### 3. 启动门禁系统

```bash
python gateway.py
```

按 `q` 键退出。

## 核心功能

| 功能 | 说明 |
|------|------|
| 人脸检测 | 使用 `scrfd_10g_bnkps.onnx` 模型检测画面中的人脸 |
| 特征提取 | 使用 `glintr100.onnx` 模型提取 512 维人脸特征向量 |
| 1:N 识别 | 与数据库中所有人进行余弦相似度比对 |
| 实时显示 | 线程分离设计，子线程计算 + 主线程渲染，画面流畅不卡顿 |
| 中文标签 | 使用 PIL 渲染中文字体，解决 OpenCV 中文显示问题 |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| THRESHOLD | 0.6 | 识别相似度阈值（0.6~0.8） |
| det-size | (640, 640) | 人脸检测输入尺寸 |
| embedding dim | 512 | 特征向量维度 |

## 常见问题

### Q: 运行时报 `AssertionError`
A: 模型文件解压后多了一层目录，需要将 `.onnx` 文件从 `antelopev2/antelopev2/` 移到 `antelopev2/` 下。


### Q: 识别率低
A: 注册和识别时的光线、角度应尽量一致；可适当调低 `THRESHOLD` 值。
