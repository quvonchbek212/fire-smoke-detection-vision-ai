"""
FastAPI Live Testing Studio for Fire & Smoke Detection
"""

import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from core.detector import FireSmokeDetector
import numpy as np

app = FastAPI(title="Fire & Smoke Detection AI")
detector = FireSmokeDetector()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fire & Smoke Detection Vision AI</title>
        <meta charset="utf-8">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
            .card { background: #1e293b; padding: 2rem; border-radius: 12px; max-width: 800px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
            h1 { color: #f97316; margin-top: 0; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-right: 8px; }
            .badge-fire { background: #ef4444; color: white; }
            .badge-smoke { background: #64748b; color: white; }
            button { background: #f97316; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 1rem; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🔥 Fire & Smoke Detection Vision AI</h1>
            <p>High-speed industrial safety monitoring system utilizing YOLO11 and TensorRT FP16 acceleration.</p>
            <div>
                <span class="badge badge-fire">Fire Hazard Detection</span>
                <span class="badge badge-smoke">Early Smoke Detection</span>
                <span class="badge" style="background:#0284c7; color:white;">4ms Latency</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 1.5rem 0;">
            <h3>API Endpoints:</h3>
            <ul>
                <li><code>POST /detect</code> - Analyze single image multipart upload</li>
                <li><code>GET /health</code> - Service and model readiness check</li>
                <li><code>GET /docs</code> - Interactive Swagger API Documentation</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok", "model": detector.model_path, "classes": detector.names}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})
    
    result = detector.detect(frame)
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
