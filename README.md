# 🛣️ Road Pothole Analysis Dashboard

A professional, real-time infrastructure assessment dashboard using AI (YOLO11 + MiDaS) to detect potholes, estimate their depth, and classify road conditions automatically.

---

## 🚀 Features

- **Pothole Detection**: Uses a custom-trained **YOLO11 Segmentation** model for precise pothole identification.
- **Depth Estimation**: Integrates **MiDaS** depth models to estimate the relative depth of each detected anomaly.
- **Severity Scoring**: Automatically labels potholes (Low, Medium, High) based on area and depth ratio to help prioritize repairs.
- **Interactive Dashboard**: A modern, high-performance **React + Vite** frontend with canvas-based visual highlighting.
- **Road Health Index**: Summarizes overall road condition (Good, Moderate, Poor) based on total pothole coverage and severity.

---

## 🛠️ Technology Stack

- **Frontend**: React, Vite, Vanilla CSS (Industrial Dark Mode theme)
- **Backend**: FastAPI, Uvicorn
- **AI Models**: YOLOv11 (Ultralytics), MiDaS (Intel ISL)
- **Image Processing**: OpenCV, NumPy, Torch

---

## 📦 Project Structure

- `frontend/`: The React source code.
- `backend.py`: FastAPI server that serves the AI model endpoints.
- `pothole_detection_pipeline.py`: The core computation logic for features, depth, and severity.
- `best.pt`: Trained YOLOv11 weights.

---

## ⚡ Quick Start

### 1. Run the Backend
Ensure you have the required dependencies (torch, torchvision, opencv, fastapi, uvicorn, ultralytics):
```bash
python -m uvicorn backend:app --reload
```
The server will start at `http://localhost:8000`.

### 2. Run the Frontend
Navigate to the `frontend` folder and start the dev server:
```bash
cd frontend
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

---

## 🌍 Deployment

### Frontend (Vercel)
The React frontend is built for easy deployment to Vercel. Connect your repository, set the root directory to `frontend`, and configure your `VITE_API_BASE_URL` to point to your live backend.

### Backend (Render / Railway / GCP)
The FastAPI backend requires Python with GPU/CPU Torch. It is recommended to deploy this using a Docker container for a smooth environment setup.

---

## 👤 Author
Developed by **Parshva Dongare**