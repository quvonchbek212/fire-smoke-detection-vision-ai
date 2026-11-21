#!/usr/bin/env python3
"""
Fire & Smoke Detection — Fine-tuning (Qayta o'qitish) skripti
RTX 5090 uchun maxsus sozlangan. O'qitish tugagach avtomatik TensorRT engine quradi.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import time
from pathlib import Path

import cv2
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT_DIR / "models" / "yolo11m_base.pt"
PROJECT_DIR = ROOT_DIR / "runs"
DATA_YAML = ROOT_DIR / "data" / "dataset.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Fire & Smoke Detection Model")
    parser.add_argument(
        "--base",
        type=str,
        default=str(DEFAULT_BASE),
        help=f"Boshlang'ich model vazni (.pt). Standart: {DEFAULT_BASE.name}",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML),
        help="Dataset YAML fayli yo'li",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="O'qitish davrlari (epochs). Standart: 30",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Rasm o'lchami. Standart: 640",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch o'lchami. RTX 5090 uchun 32",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU qurilma indeksi. Standart: 0",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fire_yolo11m_v1",
        help="Eksperiment nomi",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs). Standart: 10",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="To'xtatilgan o'qitishni davom ettirish uchun checkpoint (masalan: runs/fire_yolo11m_v1/weights/last.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"XATO: Davom ettirish uchun checkpoint topilmadi: {resume_path}")
        print("=" * 65)
        print(f"🔄 SafeVision — O'qitish davom ettirilmoqda (Resume): {resume_path}")
        print("=" * 65)
        model = YOLO(str(resume_path))
        t0 = time.perf_counter()
        results = model.train(resume=True, workers=4)
    else:
        base_path = Path(args.base)
        if not base_path.exists():
            sys.exit(f"XATO: Boshlang'ich model topilmadi: {base_path}")

        data_path = Path(args.data)
        if not data_path.exists():
            sys.exit(f"XATO: Dataset YAML fayli topilmadi: {data_path}")

        print("=" * 65)
        print("🚀 SafeVision — Fire & Smoke Fine-tuning (RTX 5090)")
        print(f"Base Model : {base_path.name}")
        print(f"Dataset    : {data_path}")
        print(f"Epochs     : {args.epochs}")
        print(f"Batch size : {args.batch}")
        print(f"Imgsz      : {args.imgsz}")
        print(f"Device     : {args.device} (NVIDIA GeForce RTX 5090)")
        print(f"Patience   : {args.patience}")
        print("=" * 65)

        # GPU xotirasi tekshiruvi
        import torch
        if torch.cuda.is_available():
            free_mem_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            print(f"  GPU Erkin Xotira: {free_mem_gb:.1f} GB")
            if free_mem_gb < 10:
                print("  ⚠️ DIQQAT: GPU xotirasi 10 GB dan kam! Agar SafeVision ishlab turgan bo'lsa:")
                print("     'sudo systemctl stop safevision' buyrug'i bilan uni vaqtincha to'xtating.")

        # Modelni yuklash
        print(f"\n[1/2] Boshlang'ich model yuklanmoqda ({base_path.name})...")
        model = YOLO(str(base_path))

        t0 = time.perf_counter()

        # O'qitishni boshlash (RTX 5090 ga moslangan optimal parametrlar)
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(PROJECT_DIR),
            name=args.name,
            workers=4,
            save=True,
            save_period=5,
            patience=args.patience,
            pretrained=True,
            plots=True,
            verbose=True,
            # Fine-tuning giperparametrlari (VRAM va aniqlik balansi)
            amp=True,              # FP16 aralash aniqlik (VRAM ni 50% tejaydi, tezlikni 2 baravar oshiradi)
            cache=False,           # NVMe SSD dan o'qiladi, RAM/VRAM to'lib ketmaydi
            lr0=0.003,             # Fine-tuning uchun xavfsiz boshlang'ich tezlik
            lrf=0.01,              # Yakuniy tezlik koeffitsiyenti
            warmup_epochs=1.5,      # Yumshoq qizdirish
            mosaic=0.5,            # Sanoat kadrlarida kontekst buzilmasligi uchun 0.5
            mixup=0.1,             # Yengil aralashtirish
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,             # Zavod yoritilishi va qorong'ilikka chidamlilik
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,            # Oynali burish
        )

    t_train = time.perf_counter() - t0
    best_pt = PROJECT_DIR / args.name / "weights" / "best.pt"

    print("\n" + "=" * 65)
    print("🎉 O'QITISH YAKUNLANDI!")
    print(f"  Umumiy vaqt     : {t_train / 60:.1f} daqiqa")
    print(f"  Eng yaxshi vazn : {best_pt}")

    # Avtomatik TensorRT Engine qurish
    if best_pt.exists():
        print("\n⚡ Yangi model RTX 5090 uchun TensorRT Engine (FP16) ga o'tkazilmoqda...")
        fine_model = YOLO(str(best_pt))
        engine_path = fine_model.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,
            device=args.device,
        )
        print(f"✅ Yangi TensorRT Engine tayyor: {engine_path}")

        # models/ papkasiga ham nusxalash
        target_engine = ROOT_DIR / "models" / "fire_smoke_yolo11m_finetuned.engine"
        import shutil
        shutil.copy(engine_path, target_engine)
        print(f"✅ Asosiy models/ papkasiga saqlandi: {target_engine}")

    print("=" * 65)


if __name__ == "__main__":
    main()
