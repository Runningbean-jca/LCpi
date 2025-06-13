# CV Module Deployment Instructions (Face Recognition System Based on OpenCV)

This module implements the face detection and recognition functions based on OpenCV and is deployed on edge devices such as Raspberry PI. The system consists of three core scripts and completes the entire process of image acquisition, model training and face recognition.
---

## 🛠 Environment Preparation

```bash
python -m venv venv
source venv/bin/activate
pip install wheel
pip install numpy
pip install opencv-contrib-python
```

## 📸 test_camera.py

### Use the camera to collect facial images and save them to the dataset/ directory for subsequent training.
    • Detect human faces in the video stream;
    • Grayscale the face area, crop it and save it the.jpg;
    • By default, 50 images are collected and can be adjusted as needed.
    • Press q to stop the collection.

## 🧠 face_training.py

### Train the face recognition model using the LBPH algorithm.
    • Load the images in dataset/ and label the ids;
    • Train the model and generate:
    • mytrainer.xml (Model File)
    • Label.pickle (label Mapping)

## 👀 face_recognition.py

### Load the model and recognize faces in real time through the camera.
    • Detection using Haar Cascade;
    • Recognition using the LBPH model;
    • Display the recognition box and the name of the person;
    • The default confidence threshold is 60;
    • Press q to exit the program.

## 🚀 Fast Start
```bash
python test_camera.py         # 步骤 1：采集图像
python face_training.py       # 步骤 2：训练模型
python face_recognition.py    # 步骤 3：运行识别
```