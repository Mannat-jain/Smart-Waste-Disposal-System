# Smart Waste Disposal System

A production-ready pipeline combining **Computer Vision**, **IoT Sensor Simulation**, **Route Optimization (Dijkstra + Genetic Algorithm)**, and **Adaptive ML Fine-Tuning** in a unified FastAPI service.

## 🎨 Interactive Live Dashboard
The system hosts a beautiful, real-time monitoring and control dashboard at the root URL:
👉 **[http://localhost:8000/](http://localhost:8000/)**

### Dashboard Features
- **Overview Stats**: Live aggregated throughput (items/hr), system latency, active bins, and feedback queue metrics.
- **IoT Sensors**: 12 live telemetry streams (ultrasonic fill levels, weights, temperatures, humidity, and gas VOC concentrations) polling in real time.
- **CV Classifier**: Drag-and-drop or select any waste image to classify it on the fly and view confidence levels and latency.
- **Route Optimizer**: Run Dijkstra + Genetic Algorithm optimization to calculate the most efficient collection path for eligible bins ($\ge 60\%$ fill) and view the path rendered dynamically on the SVG map.
- **Feedback Loop**: Interactive retraining queue where you can review flagged low-confidence classifications, correct their labels, and trigger fine-tuning.

---

## 🛠️ Components

| Module | File | Description |
|--------|------|-------------|
| **CV Classifier** | `models/classifier.py` | YOLOv8 detection crop → ResNet50 classification |
| **IoT Sensors** | `sensors/sensors.py` | Simulates 12 MQTT streams in the background |
| **Route Optimizer** | `routes/route_optimizer.py` | Dijkstra shortest path + GA routing |
| **Feedback Loop** | `feedback/feedback.py` | Automated dataset accumulation & nightly fine-tuning |
| **API Server** | `api/main.py` | FastAPI application serving endpoints & the dashboard |

---

## 🚀 Quick Start (Running the System)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the FastAPI Server**:
   Run uvicorn directly. The background IoT sensor simulator and retraining loops will automatically start up within the application lifecycle:
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Open the Dashboard**:
   Go to **[http://localhost:8000/](http://localhost:8000/)** in your web browser. You can also view the interactive API documentation at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🔌 API Reference

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/` | Serves the interactive dashboard UI |
| **POST** | `/classify` | Uploads an image for classification & bounding box |
| **GET** | `/sensors/live` | Retrieves live snapshots of all 12 simulated sensors |
| **POST** | `/feedback/flag` | Submits corrective labels for retraining |
| **GET** | `/feedback/queue` | Returns all pending feedback queue items |
| **GET** | `/route/optimize` | Runs Dijkstra + GA and returns optimized route JSON |
| **GET** | `/metrics` | Uptime, throughput, latency, and queue sizes |
| **POST** | `/train/trigger` | Manually triggers PyTorch fine-tuning |
| **GET** | `/health` | Core system health status |

---

## 🧠 Model Architecture & Pipeline
```
                    [ Waste Image ]
                           │
                           ▼
                 [ YOLOv8 Detector ] (Crop Object)
                           │
                           ▼
               [ ResNet50 Classifier ] (Softmax 5-class)
                           │
                           ▼
                    [ API Server ] ◄─── (Dashboard UI)
                           │
    ┌──────────────────────┴──────────────────────┐
    ▼                                             ▼
[ /sensors/live ]                         [ Feedback Loop ]
(12 MQTT telemetry streams)               (Flag low confidence < 70%)
    │                                             │
    ▼                                             ▼
[ /route/optimize ]                       [ Nightly Fine-Tune ]
(Dijkstra + Genetic Algorithm)            (Automatic PyTorch training)
```

---

## 🐳 Deployment Guide

### Option 1: Containerized (Docker & Docker Compose)
The easiest way to run the system in a production-ready containerized environment is using Docker.

1. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build -d
   ```
2. **Persistence**:
   - The `./data` directory is mounted to `/app/data` to persist flagged images and feedback datasets.
   - The `./models` directory is mounted to `/app/models` to persist model weights.

### Option 2: Deploying to Cloud Services (Render, Railway, AWS)
To deploy this system to a cloud provider:

1. **Render / Railway (Docker Deploy)**:
   - Push this repository to GitHub.
   - Create a new service on Render or Railway, selecting **Web Service** or **Docker Deployment**.
   - The platforms will automatically detect the `Dockerfile` and build the container.
   - Add a persistent disk mount for `/app/data` and `/app/models` to retain training metadata and weights.

2. **AWS EC2 / Virtual Server**:
   - Provision a virtual machine (Linux recommended).
   - Install Docker and run Docker Compose, or clone and run:
     ```bash
     pip install -r requirements.txt
     uvicorn api.main:app --host 0.0.0.0 --port 8000
     ```

