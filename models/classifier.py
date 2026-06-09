"""
Smart Waste Classifier — YOLOv8 + ResNet50 pipeline.
Classifies: Plastic, Paper, Glass, Metal, Organic.
"""
import cv2, io, time, logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic"]
CONFIDENCE_THRESHOLD = 0.70

@dataclass
class ClassificationResult:
    label: str
    class_id: int
    confidence: float
    bbox: Optional[list]
    latency_ms: float
    flagged: bool

class ResNet18WasteClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.backbone(x)


class WasteClassifier:
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    def __init__(self, resnet_weights, yolo_weights=None, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device=="auto" else "cpu")
        self.resnet = ResNet18WasteClassifier(num_classes=len(CLASSES))
        state = torch.load(resnet_weights, map_location=self.device)
        self.resnet.load_state_dict(state)
        self.resnet.to(self.device).eval()
        self.yolo = None
        if yolo_weights:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_weights)
            except ImportError:
                logger.warning("ultralytics not installed; YOLO stage skipped.")

    def _detect_bbox(self, img_bgr):
        if self.yolo is None: return None
        results = self.yolo(img_bgr, verbose=False)
        if results and len(results[0].boxes):
            x1,y1,x2,y2 = results[0].boxes[0].xyxy[0].tolist()
            return [int(x1),int(y1),int(x2),int(y2)]
        return None

    def _classify_crop(self, img_pil):
        tensor = self.TRANSFORM(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.resnet(tensor), dim=1).squeeze()
        conf, idx = probs.max(0)
        return CLASSES[idx.item()], conf.item()

    def predict(self, image_bytes):
        t0 = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None: raise ValueError("Could not decode image.")
        bbox = self._detect_bbox(img_bgr)
        crop = img_bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else img_bgr
        img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label, confidence = self._classify_crop(img_pil)
        latency = (time.perf_counter()-t0)*1000
        flagged = confidence < CONFIDENCE_THRESHOLD
        return ClassificationResult(
            label=label, class_id=CLASSES.index(label), confidence=confidence,
            bbox=bbox, latency_ms=round(latency,2), flagged=flagged)
