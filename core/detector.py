"""
Core Fire & Smoke Detection Module
Powered by Ultralytics YOLO (PyTorch & TensorRT Engine)
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

class FireSmokeDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_fire: float = 0.25,
        conf_smoke: float = 0.20,
        device: str = "cuda:0"
    ):
        self.conf_fire = conf_fire
        self.conf_smoke = conf_smoke
        
        if model_path is None:
            root = Path(__file__).resolve().parents[1]
            engine_p = root / "models" / "fire_smoke_yolo11m_finetuned.engine"
            pt_p = root / "models" / "fire_smoke_yolo_m.pt"
            model_path = str(engine_p if engine_p.exists() else pt_p)
            
        self.model_path = model_path
        print(f"[FireSmokeDetector] Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")
        self.names = self.model.names  # e.g., {0: 'fire', 1: 'smoke'}
        print(f"[FireSmokeDetector] Classes recognized: {self.names}")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on an input frame and returns detected fire and smoke boxes.
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=min(self.conf_fire, self.conf_smoke),
            verbose=False
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        fire_count = 0
        smoke_count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0]]
            cls_name = self.names.get(cls_id, str(cls_id)).lower()

            if "fire" in cls_name and conf >= self.conf_fire:
                fire_count += 1
                detections.append({
                    "class": "fire",
                    "confidence": conf,
                    "box": xyxy
                })
            elif "smoke" in cls_name and conf >= self.conf_smoke:
                smoke_count += 1
                detections.append({
                    "class": "smoke",
                    "confidence": conf,
                    "box": xyxy
                })

        return {
            "detections": detections,
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "is_hazard": (fire_count > 0 or smoke_count > 0),
            "inference_ms": round(infer_ms, 2)
        }

    def annotate(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draws visual bounding boxes, confidence badges, and status labels on the frame.
        """
        annotated = frame.copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            is_fire = det["class"] == "fire"
            color = (0, 69, 255) if is_fire else (180, 180, 180)  # Red for fire, Gray for smoke
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Badge text
            label = f"{det['class'].upper()}: {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Header status banner
        banner_color = (0, 0, 200) if result["is_hazard"] else (40, 140, 40)
        status_txt = f"HAZARD DETECTED! (Fire: {result['fire_count']}, Smoke: {result['smoke_count']})" if result["is_hazard"] else "SYSTEM SECURE: Normal"
        cv2.putText(annotated, f"{status_txt} | {result['inference_ms']}ms", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

        return annotated
