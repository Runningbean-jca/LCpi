import cv2
import numpy as np
import pickle
import os
import time
import subprocess
import RPi.GPIO as GPIO
import requests
import pyaudio
import wave
from openai import OpenAI  # 如果你用本地 LLM API（如 LLaMA_CPP），可以保留

# 初始化 DeepSeek 客户端
deepseek_client = OpenAI(
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 地址
    api_key="your_deepseek_api_key"  # 替换为你的 DeepSeek API 密钥
)

# 图像识别部分
def image_recognition():
    # 加载 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    # 加载 LBPH 人脸识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./mytrainer.xml')
    # 加载标签
    with open('label.pickle', 'rb') as f:
        origin_labels = pickle.load(f)
        labels = {v: k for k, v in origin_labels.items()}
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            gray_roi = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(gray_roi)
            if conf >= 60:
                name = labels[id_]
                print(f"识别到的人脸是: {name}")
                cap.release()
                return name
    cap.release()
    return None

# 语音识别部分
def speech_recognition():
    # 语音识别参数
    asr_program = './whisper.cpp/main'
    asr_model = './whisper.cpp/models/ggml-tiny.bin'
    recording_file = './recording/input.wav'
    # 录制语音
    record_wav(recording_file)
    # 调用语音识别程序
    subprocess.run([asr_program, "-m", asr_model, "-otxt", recording_file])
    # 读取识别结果
    with open(f"{recording_file}.txt", "r") as file:
        request = file.read().strip()
    print(f"识别到的语音是: {request}")
    return request

# 录制语音
def record_wav(output_file):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("开始录音...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("录音结束。")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# 语音回答
def speak_back(text):
    tts_program = './piper/piper'
    tts_model = './piper/en_US-amy-medium.onnx'
    subprocess.run([tts_program, "--model", tts_model, "--output_file", "output.wav", "--text", text])
    # 播放生成的语音
    subprocess.run(["aplay", "output.wav"])

# 查询明星简介
def get_celebrity_info(name):
    # 使用 Wikipedia API 查询明星简介
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "未找到该明星的简介。")
    return "未找到该明星的简介。"

# 调用 DeepSeek 模型生成回答
def deepseek_response(prompt):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",  # DeepSeek 模型名称
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content

# 主逻辑
def main():
    # 初始化 GPIO
    BUTTON = 8
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print("系统已启动，等待按钮按下...")
    while True:
        if GPIO.input(BUTTON) == GPIO.LOW:
            print("按钮按下，开始识别...")
            # 识别语音
            request = speech_recognition()
            if "你看到的是谁" in request:
                # 识别图像
                name = image_recognition()
                if name:
                    # 查询明星简介
                    info = get_celebrity_info(name)
                    # 生成 DeepSeek 回答
                    prompt = f"你看到的是 {name}。{info}"
                    response = deepseek_response(prompt)
                    print(f"生成的回答是: {response}")
                    speak_back(response)
                else:
                    print("未识别到人脸。")
                    speak_back("未识别到人脸。")
            else:
                # 处理日常对话
                response = deepseek_response(request)
                print(f"生成的回答是: {response}")
                speak_back(response)
            time.sleep(1)  # 防止按钮抖动

if __name__ == "__main__":
    main()