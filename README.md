# CodeAlpha — Artificial Intelligence Internship Tasks

**Intern:** Jawad Mehfooz
**Student ID:** CA/DF1/316401
**Internship:** Artificial Intelligence Internship, CodeAlpha
**Duration:** 10 October 2026 – 10 November 2026

This repository contains the three tasks completed for the CodeAlpha AI Internship.
Each task is in its own folder with its own README, source code and requirements.

## Tasks

| # | Task | Folder | What it does |
|---|------|--------|--------------|
| 1 | Language Translation Tool | `CodeAlpha_LanguageTranslator/` | Streamlit web app that translates text between 20 languages (Google Translate, no API key), with copy button and text-to-speech |
| 2 | Chatbot for FAQs | `CodeAlpha_FAQChatbot/` | FAQ chatbot using TF-IDF + cosine similarity, with a Streamlit chat UI and CLI mode |
| 3 | Object Detection and Tracking | `CodeAlpha_ObjectDetection/` | Real-time object detection with YOLOv8 + ByteTrack tracking on video, with bounding boxes, labels and tracking IDs |

## How to run

Each folder has its own README with install and run instructions. In short,
for every project:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then follow the README in that folder (`streamlit run app.py` for Tasks 1 and 2,
`python detect.py --input data/vtest.avi --output output.mp4` for Task 3).

## Technologies

Python, Streamlit, deep-translator, gTTS, scikit-learn, OpenCV, Ultralytics YOLOv8.
