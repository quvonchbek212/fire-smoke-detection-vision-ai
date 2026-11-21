#!/usr/bin/env python3
"""
Fire & Smoke Model Exporter — ONNX va TensorRT (.engine)
RTX 5090 (Blackwell) uchun maxsus optimallashtirilgan.

Ishlatish:
    # 1. Barcha modellarni ONNX va TensorRT ga eksport qilish:
    python fire_detection/scripts/export_models.py

    # 2. Yangi o'qitilgan best.pt faylini TensorRT engine qilish:
    python fire_detection/scripts/export_models.py --model fire_detection/runs/fire_factory_v1/weights/best.pt
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Export Fire Model to ONNX and TensorRT Engine")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Alohida .pt model yo'li (agar berilmasa, models/ ichidagilar eksport qilinadi)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kirish o'lchami. Standart: 640",
    )
    parser.add_argument(
        "--format",
        choices=["all", "engine", "onnx"],
        default="all",
        help="Qaysi formatga eksport qilinsin. Standart: all",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU indeksi. Standart: 0",
    )
    return parser.parse_args()


def export_single(model_path: Path, imgsz: int, fmt: str, device: str):
    print(f"\n--- Model eksporti: {model_path.name} ---")
    model = YOLO(str(model_path))

    if fmt in ["onnx", "all"]:
        print(f"  [1] ONNX formatiga eksport qilinmoqda...")
        onnx_file = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,
            device=device,
        )
        print(f"      ✅ ONNX tayyor: {onnx_file}")

    if fmt in ["engine", "all"]:
        print(f"  [2] TensorRT Engine (FP16) qurilmoqda...")
        engine_file = model.export(
            format="engine",
            imgsz=imgsz,
            half=True,
            device=device,
        )
        print(f"      ✅ TensorRT Engine tayyor: {engine_file}")


def main():
    args = parse_args()

    if args.model:
        m_path = Path(args.model)
        if not m_path.exists():
            sys.exit(f"XATO: Fayl topilmadi: {m_path}")
        targets = [m_path]
    else:
        targets = list(MODELS_DIR.glob("*.pt"))
        if not targets:
            sys.exit(f"MODELS papkasida .pt fayllar topilmadi: {MODELS_DIR}")

    print("=" * 60)
    print("⚡ SafeVision — Fire Model Export (ONNX & TensorRT)")
    print(f"Modellar soni : {len(targets)}")
    print(f"Format        : {args.format}")
    print(f"Imgsz         : {args.imgsz}")
    print(f"Device        : {args.device}")
    print("=" * 60)

    for target in targets:
        export_single(target, args.imgsz, args.format, args.device)

    print("\n" + "=" * 60)
    print("🎉 Barcha eksportlar muvaffaqiyatli yakunlandi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
