#!/usr/bin/env python3
"""
Taqqoslovchi Test Skripti: Original Model vs Fine-tuned Model
Video4.MOV va Video5.MOV dagi sinovlar (False Positives, Detection Stability, Speed)
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "traffic_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORIG_PATH = ROOT_DIR / "models" / "fire_smoke_yolo_m.engine"
MODEL_FINE_PATH = ROOT_DIR / "models" / "fire_smoke_yolo11m_finetuned.engine"

def run_evaluation(video_path: str, max_frames: int = 0, step: int = 1, conf_thresh: float = 0.20):
    video_name = Path(video_path).stem
    print(f"\n=======================================================")
    print(f"🎬 Video: {video_name} ({video_path})")
    print(f"⚙️ Sozlamalar: Step={step}, Conf={conf_thresh}, MaxFrames={'Hammasi' if max_frames==0 else max_frames}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Video ochilmadi: {video_path}")
        return None

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Hajmi: {width}x{height} | FPS: {fps:.1f} | Jami kadrlar: {total_video_frames} ({total_video_frames/fps:.1f} soniya)")

    m_orig = YOLO(str(MODEL_ORIG_PATH), task="detect")
    m_fine = YOLO(str(MODEL_FINE_PATH), task="detect")

    stats = {
        "video": video_name,
        "processed_frames": 0,
        "orig_total_dets": 0,
        "orig_fire_dets": 0,
        "orig_smoke_dets": 0,
        "orig_danger_frames": 0,
        "orig_latencies": [],
        "fine_total_dets": 0,
        "fine_fire_dets": 0,
        "fine_smoke_dets": 0,
        "fine_danger_frames": 0,
        "fine_latencies": [],
        "detections_log": []
    }

    frame_idx = 0
    saved_samples = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        if step > 1 and (frame_idx % step != 0):
            continue

        stats["processed_frames"] += 1

        # Model 1: Original
        t0 = time.perf_counter()
        res_orig = m_orig.predict(frame, conf=conf_thresh, imgsz=640, verbose=False)[0]
        dt_orig = (time.perf_counter() - t0) * 1000
        stats["orig_latencies"].append(dt_orig)

        # Model 2: Fine-tuned
        t1 = time.perf_counter()
        res_fine = m_fine.predict(frame, conf=conf_thresh, imgsz=640, verbose=False)[0]
        dt_fine = (time.perf_counter() - t1) * 1000
        stats["fine_latencies"].append(dt_fine)

        boxes_orig = res_orig.boxes
        boxes_fine = res_fine.boxes

        n_orig = len(boxes_orig) if boxes_orig is not None else 0
        n_fine = len(boxes_fine) if boxes_fine is not None else 0

        has_orig_fire = False
        has_orig_smoke = False
        if n_orig > 0:
            stats["orig_danger_frames"] += 1
            for b in boxes_orig:
                c_id = int(b.cls[0])
                c_name = m_orig.names.get(c_id, str(c_id))
                conf = float(b.conf[0])
                stats["orig_total_dets"] += 1
                if c_name == "fire":
                    stats["orig_fire_dets"] += 1
                    has_orig_fire = True
                elif c_name == "smoke":
                    stats["orig_smoke_dets"] += 1
                    has_orig_smoke = True

        has_fine_fire = False
        has_fine_smoke = False
        if n_fine > 0:
            stats["fine_danger_frames"] += 1
            for b in boxes_fine:
                c_id = int(b.cls[0])
                c_name = m_fine.names.get(c_id, str(c_id))
                conf = float(b.conf[0])
                stats["fine_total_dets"] += 1
                if c_name == "fire":
                    stats["fine_fire_dets"] += 1
                    has_fine_fire = True
                elif c_name == "smoke":
                    stats["fine_smoke_dets"] += 1
                    has_fine_smoke = True

        if n_orig > 0 or n_fine > 0:
            log_item = {
                "frame": frame_idx,
                "sec": round(frame_idx / fps, 2),
                "orig": [(m_orig.names[int(b.cls[0])], round(float(b.conf[0]), 3), [int(x) for x in b.xyxy[0]]) for b in boxes_orig] if n_orig else [],
                "fine": [(m_fine.names[int(b.cls[0])], round(float(b.conf[0]), 3), [int(x) for x in b.xyxy[0]]) for b in boxes_fine] if n_fine else [],
            }
            stats["detections_log"].append(log_item)

            if saved_samples < 20:
                saved_samples += 1
                # Plot side-by-side
                img_o = res_orig.plot()
                img_f = res_fine.plot()
                cv2.putText(img_o, f"ORIGINAL (Boxes: {n_orig})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(img_f, f"FINE-TUNED (Boxes: {n_fine})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                h_view = 540
                w_view = int(width * (h_view / height))
                comb = np.hstack([cv2.resize(img_o, (w_view, h_view)), cv2.resize(img_f, (w_view, h_view))])
                sample_path = OUTPUT_DIR / f"{video_name}_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(sample_path), comb)

        if stats["processed_frames"] % 250 == 0:
            elapsed = time.time() - t_start
            cur_fps = stats["processed_frames"] / elapsed if elapsed > 0 else 0
            print(f"  [{stats['processed_frames']}/{total_video_frames // step}] Kadrlar ishlandi | Tezlik: {cur_fps:.1f} FPS | Orig xavf: {stats['orig_danger_frames']} | Fine xavf: {stats['fine_danger_frames']}")

    cap.release()
    total_time = time.time() - t_start
    avg_fps = stats["processed_frames"] / total_time if total_time > 0 else 0
    print(f"\n✅ {video_name} yakunlandi! Jami vaqt: {total_time:.2f}s | O'rtacha tezlik: {avg_fps:.1f} FPS")
    return stats

if __name__ == "__main__":
    v4 = "/home/tenzorsoft/Videos/video4.MOV"
    v5 = "/home/tenzorsoft/Videos/video5.MOV"

    print("🚀 Video4.MOV ni to'liq sinovdan o'tkazish boshlanmoqda (Sequential, Step=1)...")
    res4 = run_evaluation(v4, step=1, conf_thresh=0.20)

    print("\n🚀 Video5.MOV ni sinovdan o'tkazish boshlanmoqda (Step=5, representative sampling)...")
    res5 = run_evaluation(v5, step=5, conf_thresh=0.20)

    print("\n" + "=" * 65)
    print("📊 YAKUNIY TAQQOSLASH HISOBOTI")
    print("=" * 65)
    for res in [res4, res5]:
        if not res: continue
        print(f"\n📁 Video: {res['video']}")
        print(f"   Jami ko'rilgan kadrlar : {res['processed_frames']}")
        print(f"   --- ESKI MODEL (fire_smoke_yolo_m) ---")
        print(f"   Xavf aniqlangan kadrlar : {res['orig_danger_frames']} ta ({(res['orig_danger_frames']/res['processed_frames'])*100:.2f}%)")
        print(f"   Jami obyektlar          : {res['orig_total_dets']} (Olov: {res['orig_fire_dets']}, Tutun: {res['orig_smoke_dets']})")
        print(f"   O'rtacha kechikish      : {np.mean(res['orig_latencies']):.2f} ms")
        print(f"   --- YANGI MODEL (YOLO11m Finetuned) ---")
        print(f"   Xavf aniqlangan kadrlar : {res['fine_danger_frames']} ta ({(res['fine_danger_frames']/res['processed_frames'])*100:.2f}%)")
        print(f"   Jami obyektlar          : {res['fine_total_dets']} (Olov: {res['fine_fire_dets']}, Tutun: {res['fine_smoke_dets']})")
        print(f"   O'rtacha kechikish      : {np.mean(res['fine_latencies']):.2f} ms")
        if res['detections_log']:
            print(f"   --- Aniqlangan holatlar tafsiloti (dastlabki 5 ta) ---")
            for d in res['detections_log'][:5]:
                print(f"     Kadr {d['frame']} ({d['sec']}s): Orig={d['orig']} | Fine={d['fine']}")
