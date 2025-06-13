# 🤖 LCpi: Multimodal LLM & CV Assistant on Raspberry Pi 4B

This project implements a fully local **multimodal dialogue system** that integrates:

- 🧠 A quantized **LLM module** (via `llama.cpp`) for natural language understanding and response generation
- 👁️ A lightweight **CV module** (via OpenCV + Haar + LBPH) for real-time face recognition
- 🎤 End-to-end pipeline on a **Raspberry Pi 4B** (4GB RAM, CPU-only)

All components run locally without relying on cloud services, making this system ideal for edge AI, privacy-sensitive use cases, and educational applications.

---

## 📁 Submodules

### [`chat_box/`](chat_box/README.md)

> 💬 Deploy and run the **LLM-based voice chatbot**  
> See [`chat_box/README.md`](chat_box/README.md) for installation and usage instructions.

This module implements:
- Local speech-to-text using `whisper.cpp`
- Text-based conversation using quantized LLaMA model (TinyLLaMA-1.1B-Chat)
- Streaming token generation + `piper` text-to-speech synthesis
- Dialogue context tracking and system prompts

---

### [`cv_box/`](cv_box/README.md)

> 🧍 Deploy and run the **CV-based face recognition system**  
> See [`cv_box/README.md`](cv_box/README.md) for full setup and training workflow.

This module includes:
- Face image collection and dataset generation
- Training with OpenCV’s `LBPHFaceRecognizer`
- Real-time webcam-based recognition using Haar cascade + LBPH
- Outputs structured identity/confidence info per frame

---

## 🔄 Multimodal Fusion: Connecting LLM & CV

The fusion of language and vision is achieved through a lightweight coordination pipeline:

1. When the user asks a **vision-related query** (e.g., “What do you see?” or “Who is in front of you?”), the LLM detects trigger keywords.
2. The system **calls the CV module**, which performs face detection and recognition in real time.
3. The recognition results (e.g., `{"name": "Alice", "confidence": 0.92}`) are **converted into natural-language descriptions**.
4. This vision-derived text is **prepended to the LLM input prompt** as additional context.
5. The LLM responds with a context-aware answer based on both the query and the CV output.

> ✅ All fusion logic is implemented **on-device**, enabling interactive vision-grounded dialogue entirely offline.

---
## 🛠️ Project Structure

<pre>
LCpi/
├── chat_box/          # Voice-based LLM chatbot (Whisper + LLaMA + Piper)
│   └── README.md
├── cv_box/            # Face recognition (OpenCV + LBPH)
│   └── README.md
├── fusion/            # (optional) scripts to connect CV outputs to LLM prompts
├── assets/            # example images, audio, or demo scripts
└── README.md          # ← You're here
</pre>


---

## 📦 Requirements

- Raspberry Pi 4B (4GB+ recommended)
- Python 3.8+
- GCC for building `llama.cpp` / `whisper.cpp`
- Installed dependencies in each submodule (`requirements.txt` inside `chat_box/` and `cv_box/`)

---

## 🌐 Repository

All code, models, and deployment tutorials are available at:

👉 [https://github.com/Runningbean-jca/LCpi](https://github.com/Runningbean-jca/LCpi)

---

## 📌 License

This project is released under the MIT License.

---

Feel free to open an issue or PR if you'd like to contribute or report a bug.