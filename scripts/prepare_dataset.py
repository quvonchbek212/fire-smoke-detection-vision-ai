#!/usr/bin/env python3
"""
Fire & Smoke Dataset Tayyorlash Skripti
Yuklangan zip arxivni ochadi, data.yaml ni to'g'rilaydi va kadrlar/label'lar sonini tekshiradi.
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ZIP_PATH = DATA_DIR / "fire_smoke_hardnegatives_complete.zip"
EXTRACT_DIR = DATA_DIR / "combined_dataset"
YAML_PATH = DATA_DIR / "dataset.yaml"


def main():
    print("=" * 60)
    print("📦 Fire & Smoke Dataset Tayyorlash")
    print(f"Zip fayl       : {ZIP_PATH}")
    print(f"Chiqarish joyi : {EXTRACT_DIR}")
    print("=" * 60)

    if not ZIP_PATH.exists():
        sys.exit(f"XATO: Zip fayl topilmadi: {ZIP_PATH}. Yuklanish yakunlanishini kuting.")

    print(f"[1/3] Arxiv ochilmoqda ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB)...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("      ✅ Arxiv to'liq ochildi!")

    # data.yaml ni sozlash
    print("[2/3] dataset.yaml sozlanmoqda...")
    # Agar arxiv ichida train/val to'g'ridan-to'g'ri bo'lsa yoki pastki papkada bo'lsa
    base_path = EXTRACT_DIR
    if (EXTRACT_DIR / "combined_dataset").exists():
        base_path = EXTRACT_DIR / "combined_dataset"

    yaml_content = f"""# Fire & Smoke Hard-Negatives Dataset
path: {base_path}
train: train/images
val: val/images
test: test/images

# Klasslar
names:
  0: smoke
  1: fire

nc: 2
"""
    with open(YAML_PATH, "w") as f:
        f.write(yaml_content)
    print(f"      ✅ YAML yangilandi: {YAML_PATH}")

    # Fayllarni sanash
    print("[3/3] Kadrlar va Label'lar statistikasi tekshirilmoqda...")
    for split in ["train", "val", "test"]:
        img_dir = base_path / split / "images"
        lbl_dir = base_path / split / "labels"
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"  👉 {split.upper():5s} | Rasmlar: {n_img:6d} ta | Yorliqlar (labels): {n_lbl:6d} ta")

    print("=" * 60)
    print("🎉 Dataset o'qitishga (Fine-tuning) 100% tayyor!")
    print("\nO'qitishni boshlash uchun quyidagi buyruqni bering:")
    print("    python scripts/finetune_fire.py --data data/dataset.yaml --epochs 50 --batch 16")
    print("=" * 60)


if __name__ == "__main__":
    main()
