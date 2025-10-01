# Fire & Smoke Monitoring System using Raspberry Pi 4

## Overview
This project implements a **real-time fire and smoke detection and monitoring system** using a **Raspberry Pi 4**.  
The detection model is **YOLOv8s**, trained on a dataset from [Kaggle Smoke & Fire Detection YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo), obtaining a precision score of > 80%, recall of ~73% and MAP of > 80%.  
The system integrates:
- **Albumentations** for data augmentation
- **Quantization** for optimized inference
- **Streamlit** for the monitoring web app
- **Raspberry Pi Camera (PiCam)** with servo-based rotation
- **Audio alarms** for fire/smoke alerts

---

## Features
- 🔥 Real-time fire & smoke detection with bounding boxes  
- 🎛 Optimized quantized ONNX model for Raspberry Pi
- 📡Remote access to Raspberry Pi through SSH  
- 📷 PiCamera integration with rotation control
- 🎥Streaming possible using RTSP protocol 
- 🍓Using a version of Raspbery Pi OS headless because of hardware limitations of the PI 
- 🌐 Web interface via Streamlit for live monitoring  
- 🔊 Fire/Smoke alarms using `.mp3`/`.wav`  
- 🛠 Test scripts and modular code structure  

---

## Repository Structure
```
.
├── Training.ipynb              # YOLO training notebook
├── albumentations.ipynb        # Data augmentation notebook
├── quantization.ipynb          # Quantization experiments
├── quantize_model_onnx.py      # Script for ONNX quantization
├── best_80map.pt / .onnx       # Trained YOLOv8s models
├── best_albumentations.pt      # Augmentation-trained model
├── quantized_model.onnx        # Optimized model for deployment
├── pi_monitoring.py            # Attempt monitoring script on Pi(this file has no effect)
├── web_app.py                  # Streamlit-based monitoring dashboard
├── servo_code.py.txt           # Servo rotation control
├── start_stream.sh             # Stream start script
├── test*.py                    # Test scripts (Pi, Picam, Quantization)
├── alarm_fire.* / alarm_smoke.*# Audio alarms
├── camera_3D_model/            # 3D camera assets
├── services/                   # Supporting services
└── yolov8s.pt                  # YOLO base weights
```

---

## Getting Started

### Prerequisites
- Raspberry Pi 4 (with PiCamera module)  
- Python 3.8+  
- Streamlit, OpenCV, Torch, Ultralytics, Albumentations  
- Servo motor (optional, for rotation)

Create a virtual enviroment:
```bash
python -m venv venv
```

Ensure your are in the virtual enviroment:
```bash
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running on Raspberry Pi
Start monitoring locally:
```bash
./run start_stream.sh
```

Run web monitoring dashboard:
```bash
streamlit run web_app.py
```

Control servo rotation:
```bash
python servo_code.py.txt
```

Quantize model for faster inference:
```bash
python quantize_model_onnx.py
```

---

## Usage Workflow
1. Train or use pre-trained YOLOv8s model (`best_80map.pt`)  
2. Apply quantization → `quantized_model.onnx`  
3. Deploy on Raspberry Pi 4  
4. Start `./run start_stream.sh` for detection  
5. Launch `web_app.py` to view detections & alerts  
6. Servo rotates camera for extended field coverage  

---

## Photos and Videos
- Functionality of the entire project:
[![Watch to see how the project works]](project.mp4)

- Photos of the e-mail alerts:
[![Smoke detection 1]](smoke_detection1.jpeg)
[![Smoke detection 2]](smoke_detection2.jpeg)
[![Fire detection]](fire_detection.jpeg)

- Photo of the 3D camera module:
[![3D Camera]](3Dcamera.jpeg)

---

## Limitations & Future Work
- Model accuracy depends on dataset variety  
- Streamlit UI may require optimization for real-time streaming  
- Potential extension: MQTT/IoT integration for remote alerts  

---

## Contributions
This project was developed by a team of 4 members:

- [Alexandru Sauca](https://github.com/AlexandruSauca) – Project Lead, Raspberry Pi Integration, Model Training, Data Augmentation, Quantization, Streaming Method
- [Cristian Ion](https://github.com/Cristian86Ion) – Web App, Monitoring Dashboard, E-mail Receive Method, Screenshots of Danger Event Occurring, 3D Camera Design  
- [Gruia Rojisteanu](https://github.com/MiguCitric) – Raspberry Pi Integration, Streaming Method, Servos Solution  
- [Alexandru Barbu](https://github.com/AlexBrb278) – Model Training, Quantiaztion, 3D Camera Design

---

## License
This project is released under the MIT License.

---

## Acknowledgments
- [Kaggle Smoke & Fire Detection Dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- Streamlit framework for monitoring UI
