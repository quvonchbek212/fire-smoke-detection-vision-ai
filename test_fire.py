#!/usr/bin/env python3
"""
🔥 SafeVision — Fire & Smoke Detection (Jonli Sinov va Monitoring)
Model: YOLO11m Fine-tuned (TensorRT FP16 / PyTorch)

Qo'llab-quvvatlanadigan manbalar:
  1. RTSP IP Kamera oqimi (masalan: rtsp://admin:pass@192.168.1.10:554/stream)
  2. Web-kamera (0, 1, 2...)
  3. Video fayl (.mp4, .avi, .mkv, .mov)
  4. Rasm yoki rasmlar papkasi (.jpg, .png)

Ishlatish namunalari:
  # Interaktiv rejimda (menyudan tanlash):
  python3 scripts/test_fire.py

  # Web-kamerada sinash (0-kamera):
  python3 scripts/test_fire.py --source 0

  # RTSP kamera oqimida sinash:
  python3 scripts/test_fire.py --source rtsp://admin:pass@192.168.1.50:554/h264Preview_01_main

  # Video faylda sinash:
  python3 scripts/test_fire.py --source ./snapshots/6.avi

  # Natijaviy videoni yozib olish:
  python3 scripts/test_fire.py --source 0 --save-video
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# TCP orqali RTSP ni barqaror qabul qilish (UDP paket yo'qolishining oldini oladi)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT_DIR / "models" / "fire_smoke_yolo_m.engine"
DEFAULT_PT = ROOT_DIR / "models" / "fire_smoke_yolo_m.pt"
OUTPUT_DIR = ROOT_DIR / "outputs"


def get_default_model() -> Path:
    if DEFAULT_ENGINE.exists():
        return DEFAULT_ENGINE
    if DEFAULT_PT.exists():
        return DEFAULT_PT
    return DEFAULT_ENGINE


def is_video_file(path_str: str) -> bool:
    ext = Path(path_str).suffix.lower()
    return ext in [".avi", ".mp4", ".mkv", ".mov", ".flv", ".webm", ".ts", ".m4v"]


def is_image_file(path_str: str) -> bool:
    ext = Path(path_str).suffix.lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args():
    parser = argparse.ArgumentParser(description="SafeVision Fire & Smoke Live Test")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Manba: 0 (web-kamera), rtsp://... (RTSP oqim), video.mp4 yoki rasm yo'li. Ko'rsatilmasa menyu chiqadi.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(get_default_model()),
        help=f"Model fayli (.engine yoki .pt). Standart: {get_default_model().name}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ishonch chegarasi (confidence threshold: 0.1 - 0.9). Standart: 0.25",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Eski model (fire_smoke_yolo_m) va Yangi modelni yonma-yon solishtirish",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kadr o'lchami. Standart: 640",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Jonli video oynasini ko'rsatish (cv2.imshow)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Video oynasini ko'rsatmaslik (faqat terminal va faylga yozish)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Xavf aniqlangan kadrlarni outputs/ papkasiga saqlash",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help="Butun oqimni outputs/annotated_output.mp4 fayliga yozib olish",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maksimal ishlov beriladigan kadrlar soni (0 bo'lsa cheksiz)",
    )
    return parser.parse_args()


def interactive_select_source() -> str:
    print("\n" + "=" * 62)
    print("🔥 SafeVision — Fire & Smoke Detection Test Menyusi")
    print("=" * 62)
    print("  [1] Web-kamera (USB yoki noutbuk kamerasi - 0)")
    print("  [2] RTSP Oqim (IP kamera URL kiritish)")
    print("  [3] Video fayl (mavjud videoni ko'rish)")
    print("  [4] Rasm yoki Zavod papkasi (snapshots)")
    print("  [0] Chiqish")
    print("=" * 62)

    choice = input("Tanlovingizni kiriting [1/2/3/4]: ").strip()
    if choice == "1":
        cam_idx = input("Kamera raqamini kiriting (standart 0): ").strip() or "0"
        return cam_idx
    elif choice == "2":
        rtsp_url = input("RTSP URL ni kiriting (masalan: rtsp://admin:pass@ip:554/stream): ").strip()
        return rtsp_url
    elif choice == "3":
        def_vid = "./snapshots/6.avi"
        vid_path = input(f"Video yo'lini kiriting (Enter bossangiz: {def_vid}): ").strip()
        return vid_path if vid_path else def_vid
    elif choice == "4":
        def_img = "./snapshots/maxam/"
        img_path = input(f"Rasm yoki papka yo'li (Enter: {def_img}): ").strip()
        return img_path if img_path else def_img
    elif choice == "0":
        sys.exit("Dastur to'xtatildi.")
    else:
        return choice


def draw_hud(frame, status_text, fps, latency_ms, is_danger):
    """
    Kadr ustiga chiroyli professional axborot paneli (HUD) chizish
    """
    h, w = frame.shape[:2]
    # Yuqori panel foni
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 20), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Status chizig'i
    status_color = (0, 0, 255) if is_danger else (0, 220, 0)
    cv2.circle(frame, (25, 27), 10, status_color, -1)

    cv2.putText(
        frame,
        f"SafeVision RTX 5090 | {status_text}",
        (45, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    right_info = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f}ms"
    (rw, _), _ = cv2.getTextSize(right_info, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(
        frame,
        right_info,
        (w - rw - 20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )

    # Pastki qismda yordam ko'rsatgichi
    hint = "Chiqish: [Q] | Rasm saqlash: [S] | To'xtatish: [SPACE]"
    cv2.putText(
        frame,
        hint,
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def process_stream(source, model, args, names):
    is_webcam = source.isdigit()
    cap_src = int(source) if is_webcam else source

    print(f"\n[2/3] Oqim ochilmoqda: {source}...")
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"\n❌ XATO: Oqim ochilmadi! Manzilni tekshiring: {source}")
        if not is_webcam and source.startswith("rtsp://"):
            print("💡 Maslahat: RTSP URL yoki tarmoq (IP) aloqasini tekshiring.")
        elif is_webcam:
            print("💡 Maslahat: Kamera kompyuterga ulanganligini yoki /dev/video ruxsatini tekshiring.")
        return

    # Kadr parametrlarini olish
    fps_stream = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"      Oqim o'lchami: {width}x{height} | {fps_stream:.1f} FPS | Jami kadrlar: {total_frames}")

    video_writer = None
    if args.save_video:
        out_vid_path = OUTPUT_DIR / f"test_recording_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, 25.0, (width, height))
        print(f"      Video yozib olinmoqda: {out_vid_path}")

    window_name = "🔥 SafeVision — Fire & Smoke Real-Time Detection (RTX 5090)"
    if args.show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception as e:
            print(f"Grafik oyna ochilmadi ({e}), matnli rejimda davom etiladi.")
            args.show = False

    old_model = None
    if getattr(args, "compare", False):
        old_engine = ROOT_DIR / "models" / "fire_smoke_yolo_m.engine"
        old_pt = ROOT_DIR / "models" / "fire_smoke_yolo_m.pt"
        old_path = old_engine if old_engine.exists() else old_pt
        print(f"      [Taqqoslash] Eski model yuklanmoqda: {old_path.name}...")
        old_model = YOLO(str(old_path), task="detect")

    print("\n[3/3] Monitoring boshlandi! (To'xtatish uchun terminalda Ctrl+C yoki oynada 'Q' bosing)\n")

    frame_count = 0
    t_start = time.time()
    fps_smooth = 0.0
    paused = False

    # Ranglar: Olov -> Qizil (0, 0, 255), Tutun -> Zangori (255, 180, 0)
    COLORS = {
        0: (0, 0, 255),    # fire
        1: (255, 180, 0),  # smoke
    }

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if not is_webcam and not str(source).startswith("rtsp://"):
                        print("\n🏁 Video fayl yakunlandi.")
                        break
                    else:
                        print("⚠️ Kadr olinmadi, qayta ulanishga harakat qilinmoqda...")
                        time.sleep(0.5)
                        continue

                frame_count += 1
                if args.limit > 0 and frame_count > args.limit:
                    print(f"\nLimit ({args.limit} kadr) bajarildi.")
                    break

                t0 = time.perf_counter()
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
                latency_ms = (time.perf_counter() - t0) * 1000

                # FPS hisoblash
                instant_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
                fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

                res = results[0]
                boxes = res.boxes
                num_boxes = len(boxes) if boxes is not None else 0
                is_danger = num_boxes > 0

                dets_info = []
                frame_annotated = frame.copy()
                if is_danger:
                    for b in boxes:
                        cls_id = int(b.cls[0])
                        cls_name = names.get(cls_id, str(cls_id)).upper()
                        conf_val = float(b.conf[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0])

                        color = COLORS.get(cls_id, (0, 255, 255))
                        cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 3)

                        label = f"{cls_name} {conf_val*100:.1f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(frame_annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                        cv2.putText(
                            frame_annotated,
                            label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        dets_info.append(f"{cls_name}: {conf_val:.2f}")

                status_str = f"🚨 XAVF ANIQLANDI: {', '.join(dets_info)}" if is_danger else "Status: XAVFSIZ (Toza)"
                draw_hud(frame_annotated, status_str, fps_smooth, latency_ms, is_danger)

                if is_danger:
                    print(f"[{frame_count:05d}] 🚨 {', '.join(dets_info)} | Tezlik: {latency_ms:.1f}ms ({fps_smooth:.1f} FPS)")
                    if args.save:
                        snap_name = f"detected_live_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(str(OUTPUT_DIR / snap_name), frame_annotated)

                if old_model is not None:
                    # Eski model bilan taqqoslash (Side-by-Side)
                    res_old = old_model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
                    left_img = res_old.plot()
                    cv2.rectangle(left_img, (0, 0), (left_img.shape[1], 45), (30, 30, 30), -1)
                    cv2.putText(left_img, "ESKI MODEL (fire_smoke_yolo_m)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

                    right_img = frame_annotated
                    cv2.rectangle(right_img, (0, 0), (right_img.shape[1], 45), (20, 50, 20), -1)
                    cv2.putText(right_img, "YANGI MODEL (YOLO11m Finetuned)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    # Ekranga qulay sig'ishi uchun har ikkala tomonni 960x540 ga masshtablash
                    left_scaled = cv2.resize(left_img, (960, 540))
                    right_scaled = cv2.resize(right_img, (960, 540))
                    display_frame = np.hstack([left_scaled, right_scaled])

                    # Pastki qismda vaqt ko'rsatkichi
                    if total_frames > 0:
                        cur_sec = frame_count / fps_stream
                        tot_sec = total_frames / fps_stream
                        time_str = f"Vaqt: {int(cur_sec//60):02d}:{int(cur_sec%60):02d} / {int(tot_sec//60):02d}:{int(tot_sec%60):02d} | Kadr: {frame_count}/{total_frames}"
                        cv2.putText(display_frame, time_str, (display_frame.shape[1]//2 - 180, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                else:
                    display_frame = frame_annotated

                if video_writer is not None:
                    video_writer.write(display_frame)

            if args.show:
                cv2.imshow(window_name, display_frame)

                # Haqiqiy video tezligini ta'minlash (sun'iy tezlashmaslik uchun):
                target_frame_time = 1.0 / max(1.0, fps_stream)
                elapsed = time.perf_counter() - t0
                wait_sec = max(0.001, target_frame_time - elapsed)
                wait_ms = int(wait_sec * 1000) if not paused else 40

                key = cv2.waitKey(wait_ms) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                    print("\nFoydalanuvchi tomonidan to'xtatildi.")
                    break
                elif key in [ord("s"), ord("S")]:
                    manual_snap = OUTPUT_DIR / f"manual_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(manual_snap), frame)
                    print(f"📷 Snapshot saqlandi: {manual_snap}")
                elif key in [ord(" "), ord("p"), ord("P")]:
                    paused = not paused
                    print("To'xtatildi (Pause)" if paused else "Davom ettirildi")
                elif key in [ord("d"), ord("D")]:  # 5 soniya oldinga
                    cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames, cur_pos + int(fps_stream * 5)))
                    print("⏩ +5 soniya oldinga")
                elif key in [ord("a"), ord("A")]:  # 5 soniya orqaga
                    cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur_pos - int(fps_stream * 5)))
                    print("⏪ -5 soniya orqaga")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()


def process_images(source_path, model, args, names):
    path = Path(source_path)
    if path.is_file():
        image_files = [str(path)]
    else:
        patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.webp"]
        image_files = []
        for p in patterns:
            image_files.extend(glob.glob(str(path / p), recursive=True))
        image_files = sorted(image_files)

    if args.limit > 0:
        image_files = image_files[: args.limit]

    if not image_files:
        print(f"❌ '{source_path}' ichida rasm fayllari topilmadi.")
        return

    print(f"\n[2/3] {len(image_files)} ta rasm topildi. Sinov boshlanmoqda...")
    if args.show:
        print("💡 Oyna boshqaruvi: [SPACE/ENTER] - keyingi rasm | [Q] - chiqish")

    window_name = "🔥 SafeVision — Fire & Smoke Image/Folder Test"
    if args.show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception:
            args.show = False

    total_time = 0.0
    detected_count = 0

    try:
        for i, img_path in enumerate(image_files, 1):
            t0 = time.perf_counter()
            results = model.predict(source=img_path, conf=args.conf, imgsz=args.imgsz, verbose=False)
            dt = (time.perf_counter() - t0) * 1000
            total_time += dt

            res = results[0]
            boxes = res.boxes
            num_boxes = len(boxes) if boxes is not None else 0
            is_danger = num_boxes > 0

            status = "Toza (olov/tutun yo'q)"
            if is_danger:
                detected_count += 1
                dets_str = []
                for b in boxes:
                    c_id = int(b.cls[0])
                    c_name = names.get(c_id, str(c_id)).upper()
                    conf = float(b.conf[0])
                    dets_str.append(f"{c_name}: {conf:.2f}")
                status = f"🚨 TOPILDI: {', '.join(dets_str)}"

                if args.save:
                    out_name = f"detected_{Path(img_path).name}"
                    annotated = res.plot()
                    cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)

            print(f"  [{i:03d}/{len(image_files):03d}] {Path(img_path).name[:35]:35s} | {dt:5.1f} ms | {status}")

            if args.show:
                annotated_img = res.plot()
                # HUD qo'shish
                fps_val = 1000.0 / dt if dt > 0 else 0
                hud_text = f"[{i}/{len(image_files)}] {Path(img_path).name[:25]} | {status}"
                draw_hud(annotated_img, hud_text, fps_val, dt, is_danger)

                cv2.imshow(window_name, annotated_img)
                # Agar xavf topilsa 1.2 soniya ko'rsatib turadi, toza bo'lsa 300 ms da o'tib ketadi
                wait_time = 1200 if is_danger else 350
                key = cv2.waitKey(wait_time) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    print("\nFoydalanuvchi tomonidan to'xtatildi.")
                    break
                elif key in [ord(" "), 13]:  # Space yoki Enter
                    continue
    finally:
        if args.show:
            cv2.destroyAllWindows()

    avg_time = total_time / len(image_files) if image_files else 0
    print("=" * 60)
    print("Natijalar:")
    print(f"  Jami tekshirilgan rasmlar: {len(image_files)}")
    print(f"  Olov/tutun topilgan kadrlar : {detected_count}")
    print(f"  O'rtacha tezlik           : {avg_time:.2f} ms/kadr ({1000/avg_time if avg_time else 0:.1f} FPS)")
    if detected_count > 0:
        print(f"  Belgilangan rasmlar saqlandi: {OUTPUT_DIR}")
    print("=" * 60)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = args.source
    if not source:
        source = interactive_select_source()

    model_path = Path(args.model)
    if not model_path.exists():
        fallback = get_default_model()
        if fallback.exists():
            print(f"Ogohlantirish: {model_path} topilmadi, standart {fallback} ishlatiladi.")
            model_path = fallback
        else:
            sys.exit(f"❌ XATO: Model fayli topilmadi: {model_path}")

    print("=" * 60)
    print("🔥 SafeVision — Fire & Smoke Real-Time Test")
    print(f"Model  : {model_path.name}")
    print(f"Manba  : {source}")
    print(f"Conf   : {args.conf}")
    print("=" * 60)

    print(f"[1/3] Model yuklanmoqda ({model_path.name})...")
    model = YOLO(str(model_path), task="detect")
    names = model.names
    print(f"      Aniqlanadigan klasslar: {names}")

    # Agar manba web-kamera, RTSP yoki video bo'lsa
    is_stream = (
        source.isdigit()
        or source.startswith("rtsp://")
        or source.startswith("http://")
        or source.startswith("https://")
        or (Path(source).is_file() and is_video_file(source))
    )

    if is_stream:
        process_stream(source, model, args, names)
    else:
        process_images(source, model, args, names)


if __name__ == "__main__":
    main()
