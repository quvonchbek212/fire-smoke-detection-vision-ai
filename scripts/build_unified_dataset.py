#!/usr/bin/env python3
"""
Build Unified Fire & Smoke Dataset
Combines:
  1. FASDD_CV (95k images) -> classes: 0: fire, 1: smoke
  2. D-Fire (21.5k images) -> remapped: 0 (smoke) -> 1, 1 (fire) -> 0
  3. ISFire-4 (7.3k images) -> remapped: 0,2 -> 1 (smoke); 1,3 -> 0 (fire)
  4. SafeVision Factory (3k images) -> Hard Negatives (empty labels)
Standardized to:
  0: fire
  1: smoke
"""

import os
import sys
import glob
import shutil
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UNIFIED_DIR = DATA_DIR / "unified_dataset"
YAML_PATH = DATA_DIR / "dataset.yaml"

# Sources
FASDD_DIR = DATA_DIR / "FASDD_CV"
DFIRE_DIR = DATA_DIR / "fire_smoke_hardnegatives_complete" / "combined_dataset"
ISFIRE_DIR = DATA_DIR / "ISFire-4" / "Dataset"
SAFEVISION_RAW = Path("/home/tenzorsoft/Desktop/safevision_rtx5090/backend/dataset_raw")
SAFEVISION_MAXAM = Path("/home/tenzorsoft/Desktop/safevision_rtx5090/snapshots/maxam")


def safe_link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def process_dataset(src_dir: Path, prefix: str, split_map: dict, class_map: dict, stats: dict):
    print(f"\n--- Qayta ishlanmoqda: {prefix.upper()} ({src_dir.name}) ---")
    processed_count = 0
    
    for src_split, dst_split in split_map.items():
        img_dir = src_dir / src_split / "images"
        lbl_dir = src_dir / src_split / "labels"
        
        if not img_dir.exists():
            continue
            
        dst_img_dir = UNIFIED_DIR / dst_split / "images"
        dst_lbl_dir = UNIFIED_DIR / dst_split / "labels"
        
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        print(f"  [{src_split} -> {dst_split}]: {len(images):6d} ta rasm")
        
        for img_path in images:
            stem = img_path.stem
            ext = img_path.suffix
            dst_stem = f"{prefix}_{stem}"
            
            dst_img = dst_img_dir / f"{dst_stem}{ext}"
            safe_link_or_copy(img_path, dst_img)
            
            # Label
            lbl_path = lbl_dir / f"{stem}.txt"
            dst_lbl = dst_lbl_dir / f"{dst_stem}.txt"
            
            new_lines = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if content:
                    for line in content.split("\n"):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                src_cls = int(parts[0])
                                if src_cls in class_map:
                                    target_cls = class_map[src_cls]
                                    new_line = f"{target_cls} {' '.join(parts[1:5])}"
                                    new_lines.append(new_line)
                                    stats[f"class_{target_cls}"] += 1
                            except ValueError:
                                pass
                                
            # Write new label
            with open(dst_lbl, "w", encoding="utf-8") as f:
                if new_lines:
                    f.write("\n".join(new_lines) + "\n")
                    stats[f"{dst_split}_annotated"] += 1
                else:
                    stats[f"{dst_split}_hard_negative"] += 1
                    
            stats[dst_split] += 1
            processed_count += 1
            
    print(f"  ✅ {prefix.upper()} jami: {processed_count} ta kadr qo'shildi.")


def process_safevision_hard_negatives(stats: dict, target_count=3000):
    print(f"\n--- SafeVision Zavod Hard Negatives qo'shilmoqda (Maks: {target_count} ta) ---")
    
    # 1. dataset_raw dan yig'ish (ishxona, maxam, ammafos)
    raw_images = list(SAFEVISION_RAW.glob("**/*.jpg"))
    # 2. snapshots/maxam dan yig'ish
    maxam_images = list(SAFEVISION_MAXAM.glob("**/*.jpg"))
    
    print(f"  Mavjud zavod kadrlari: dataset_raw={len(raw_images)}, maxam={len(maxam_images)}")
    
    random.seed(42)
    selected = []
    if raw_images:
        selected.extend(random.sample(raw_images, min(2000, len(raw_images))))
    if maxam_images:
        selected.extend(random.sample(maxam_images, min(1000, len(maxam_images))))
        
    random.shuffle(selected)
    selected = selected[:target_count]
    print(f"  Tanlab olingan zavod kadrlari: {len(selected)} ta")
    
    # 80% train, 10% val, 10% test
    n_total = len(selected)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    splits_data = {
        "train": selected[:n_train],
        "val": selected[n_train:n_train+n_val],
        "test": selected[n_train+n_val:]
    }
    
    for split_name, img_list in splits_data.items():
        dst_img_dir = UNIFIED_DIR / split_name / "images"
        dst_lbl_dir = UNIFIED_DIR / split_name / "labels"
        
        for i, img_path in enumerate(img_list, 1):
            dst_name = f"safevision_factory_{split_name}_{i:05d}{img_path.suffix}"
            dst_img = dst_img_dir / dst_name
            safe_link_or_copy(img_path, dst_img)
            
            # Bo'sh label (Hard Negative)
            dst_lbl = dst_lbl_dir / f"safevision_factory_{split_name}_{i:05d}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as f:
                pass  # Bo'sh fayl
                
            stats[split_name] += 1
            stats[f"{split_name}_hard_negative"] += 1
            stats["safevision_total"] += 1
            
    print(f"  ✅ SafeVision zavod fonlari (train: {n_train}, val: {n_val}, test: {n_total - n_train - n_val}) muvaffaqiyatli qo'shildi!")


def main():
    print("=" * 65)
    print("🚀 YAKUNIY GIBRID DATASETNI YARATISH (FIRE + SMOKE + FACTORY)")
    print(f"Chiqish joyi : {UNIFIED_DIR}")
    print("=" * 65)
    
    for split in ["train", "val", "test"]:
        (UNIFIED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        
    stats = {
        "train": 0, "val": 0, "test": 0,
        "train_annotated": 0, "val_annotated": 0, "test_annotated": 0,
        "train_hard_negative": 0, "val_hard_negative": 0, "test_hard_negative": 0,
        "class_0": 0, "class_1": 0,
        "safevision_total": 0
    }
    
    # 1. FASDD_CV (0: fire, 1: smoke)
    fasdd_splits = {"train": "train", "valid": "val", "test": "test"}
    fasdd_map = {0: 0, 1: 1}
    process_dataset(FASDD_DIR, "fasdd", fasdd_splits, fasdd_map, stats)
    
    # 2. D-Fire (0: smoke -> 1, 1: fire -> 0)
    dfire_splits = {"train": "train", "val": "val", "test": "test"}
    dfire_map = {0: 1, 1: 0}
    process_dataset(DFIRE_DIR, "dfire", dfire_splits, dfire_map, stats)
    
    # 3. ISFire-4 (0,2: smoke -> 1; 1,3: fire -> 0)
    isfire_splits = {"train": "train", "valid": "val", "test": "test"}
    isfire_map = {0: 1, 1: 0, 2: 1, 3: 0}
    process_dataset(ISFIRE_DIR, "isfire", isfire_splits, isfire_map, stats)
    
    # 4. SafeVision Factory Hard Negatives
    process_safevision_hard_negatives(stats, target_count=3000)
    
    # 5. dataset.yaml yozish
    yaml_content = f"""# SafeVision Sanoat Fire & Smoke Birlashtirilgan Dataset
path: {UNIFIED_DIR}
train: train/images
val: val/images
test: test/images

# Klasslar
names:
  0: fire
  1: smoke

nc: 2
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    with open(UNIFIED_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print("\n" + "=" * 65)
    print("🎉 YAKUNIY STATISTIKA:")
    print(f"  Train kadrlar : {stats['train']:6d} ta (Annotatsiyalangan: {stats['train_annotated']}, Hard Negative: {stats['train_hard_negative']})")
    print(f"  Val kadrlar   : {stats['val']:6d} ta (Annotatsiyalangan: {stats['val_annotated']}, Hard Negative: {stats['val_hard_negative']})")
    print(f"  Test kadrlar  : {stats['test']:6d} ta (Annotatsiyalangan: {stats['test_annotated']}, Hard Negative: {stats['test_hard_negative']})")
    print(f"  Jami kadrlar  : {stats['train'] + stats['val'] + stats['test']:6d} ta")
    print(f"  Olov qutilari (Class 0: fire)  : {stats['class_0']:6d} ta")
    print(f"  Tutun qutilari (Class 1: smoke): {stats['class_1']:6d} ta")
    print(f"  SafeVision zavod foni qo'shildi: {stats['safevision_total']:6d} ta")
    print(f"  dataset.yaml tayyorlandi       : {YAML_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
