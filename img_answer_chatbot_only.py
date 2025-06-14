#
# Local LLM-based Voice Assistant running on Raspberry Pi
# 
# Ping Zhou
# https://github.com/zhoupingjay/llm_voice_chatbot_rpi
#
# It is inspired by Nick Bild's "Local LLM Assistant", with many enhancements:
# https://github.com/nickbild/local_llm_assistant

import os
from openai import OpenAI
import pyaudio
import wave
import whisper
import time
import argparse
import numpy as np
import subprocess
import RPi.GPIO as GPIO
import cv2
import pickle
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--llm_port', default=8080, help='Local port of the LLM server.', type=int)
parser.add_argument('--tts', default='./piper/piper', help='TTS program to use for speaking back.', type=str)
parser.add_argument('--tts_model', default='./piper/en_US-amy-medium.onnx', help='TTS model to use for speaking back.', type=str)
parser.add_argument('--asr', default='./whisper.cpp/main', help='Path to ASR program.', type=str)
parser.add_argument('--asr_model', default='./whisper.cpp/models/ggml-tiny.bin', help='Path to ASR model.', type=str)
parser.add_argument('--recording', default='./recording/input.wav', help='Temporary recording file for ASR.', type=str)

args = parser.parse_args()

BUTTON = 8
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def record_wav():
    form_1 = pyaudio.paInt16
    chans = 1
    samp_rate = 16000
    chunk = 4096
    wav_output_filename = args.recording

    audio = pyaudio.PyAudio()

    # Create pyaudio stream.
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        # input_device_index = 1,\
                        input = True, \
                        frames_per_buffer=chunk)
    print("recording")
    frames = []

    # Push-to-talk style: record until the button is released.
    while GPIO.input(BUTTON) == GPIO.LOW:
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording")

    # Stop the stream, close it, and terminate the pyaudio instantiation.
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio frames as .wav file.
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    return frames

def speak_back(sentence: str):
    os.system('echo "{0}" | {1} --model {2} --output-raw | aplay -r 22050 -f S16_LE -t raw -'.format(sentence, args.tts, args.tts_model))

# 人脸识别函数
def image_recognition():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./mytrainer.xml')
    with open('label.pickle', 'rb') as f:
        origin_labels = pickle.load(f)
        labels = {v: k for k, v in origin_labels.items()}
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    name = None
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            gray_roi = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(gray_roi)
            if conf >= 60:
                name = labels[id_]
                print(f"识别到的人脸是: {name}")
                break
    cap.release()
    return name

# 明星简介查询
def get_celebrity_info(name):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "未找到该明星的简介。")
    return "未找到该明星的简介。"

def main():
    client = OpenAI(
        base_url=f"http://127.0.0.1:{args.llm_port}/v1",
        api_key = "sk-no-key-required"
    )
    history = []
    system_msg = {"role": "system", "content": "Your name is Amy. You are an AI Assistant. Your priority is helping users with their requests. Limit your response within 150 words."}
    print("Ready...")
    while(True):
        if GPIO.input(BUTTON) == GPIO.LOW:
            record_wav()
            subprocess.run([args.asr, "-m", args.asr_model, "-otxt", args.recording])
            with open(f"{args.recording}.txt", "r") as file:
                request = file.read().strip()
            print("Transcription: ", request)
            print("========== HISTORY ==========")
            for h in history:
                print(h)
            print("========== END OF HISTORY ==========")
            if "你看到的是谁" in request:
                name = image_recognition()
                if name:
                    info = get_celebrity_info(name)
                    prompt = f"你看到的是 {name}。{info}"
                    messages = [ system_msg ]
                    messages.extend(history)
                    messages.append({"role": "user", "content": prompt})
                    completion = client.chat.completions.create(
                        model="LLaMA_CPP",
                        messages = messages,
                        stream=True,
                    )
                    print("LLM response: ")
                    print("=========================")
                    acc = ""
                    resp = ""
                    for chunk in completion:
                        txt = chunk.choices[0].delta.content
                        resp += txt or ""
                        if txt is None:
                            time.sleep(0.01)
                        else:
                            acc += txt
                            if "." in acc:
                                sentences = acc.split(".")
                                utterance = ".".join(sentences[:-1])
                                utterance = utterance.replace('"', '')
                                if len(utterance) > 10:
                                    print(utterance + ".")
                                    acc = sentences[-1].lstrip()
                                    speak_back(utterance)
                    if acc:
                        print(acc)
                        utterance = acc.replace('"', '')
                        speak_back(utterance)
                    history.append({"role": "user", "content": prompt})
                    history.append({"role": "Assistant", "content": resp})
                    if len(history) > 4:
                        history = history[2:]
                else:
                    print("未识别到人脸。")
                    speak_back("未识别到人脸。")
            else:
                messages = [ system_msg ]
                messages.extend(history)
                messages.append({"role": "user", "content": request})
                completion = client.chat.completions.create(
                    model="LLaMA_CPP",
                    messages = messages,
                    stream=True,
                )
                print("LLM response: ")
                print("=========================")
                acc = ""
                resp = ""
                for chunk in completion:
                    txt = chunk.choices[0].delta.content
                    resp += txt or ""
                    if txt is None:
                        time.sleep(0.01)
                    else:
                        acc += txt
                        if "." in acc:
                            sentences = acc.split(".")
                            utterance = ".".join(sentences[:-1])
                            utterance = utterance.replace('"', '')
                            if len(utterance) > 10:
                                print(utterance + ".")
                                acc = sentences[-1].lstrip()
                                speak_back(utterance)
                if acc:
                    print(acc)
                    utterance = acc.replace('"', '')
                    speak_back(utterance)
                history.append({"role": "user", "content": request})
                history.append({"role": "Assistant", "content": resp})
                if len(history) > 4:
                    history = history[2:]
            print("========== END OF RESPONSE ==========")
            input()
        else:
            time.sleep(0.2)

if __name__ == "__main__":
    main()