# src/data_preprocessing.py
import os
import cv2
import numpy as np
import shutil
from ultralytics.data.utils import autosplit_dataset

def preprocess_data():
    # 경로 설정
    raw_path = "data/raw/"
    processed_path = "data/processed/"
    classes = ["rock", "paper", "scissors", "none"]
    
    # 처리된 데이터 저장 폴더 생성
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(processed_path, split, cls), exist_ok=True)
    
    # 각 클래스별 데이터 처리
    for cls in classes:
        class_path = os.path.join(raw_path, cls)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"클래스 {cls}의 이미지 {len(images)}개 처리 중...")
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"이미지를 읽을 수 없습니다: {img_path}")
                continue
            
            # 이미지 전처리 (크기 조정, 정규화 등)
            img = cv2.resize(img, (224, 224))
            
            # 데이터 증강
            augmented_images = augment_image(img)
            
            # 원본 이미지 저장
            cv2.imwrite(os.path.join(processed_path, "train", cls, img_name), img)
            
            # 증강된 이미지 저장
            for i, aug_img in enumerate(augmented_images):
                aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}{os.path.splitext(img_name)[1]}"
                cv2.imwrite(os.path.join(processed_path, "train", cls, aug_name), aug_img)
    
    # 데이터셋 분할 (train/val/test)
    autosplit_dataset(processed_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # YAML 파일 생성
    create_dataset_yaml()
    
    print("데이터 전처리 완료")

def augment_image(img):
    """이미지 증강 함수"""
    augmented = []
    
    # 회전
    for angle in [5, -5, 10, -10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)
    
    # 밝기 조정
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
    augmented.extend([bright, dark])
    
    # 노이즈 추가
    noise = img.copy()
    noise_factor = 0.05
    noise = np.clip(img + noise_factor * np.random.randn(*img.shape), 0, 255).astype(np.uint8)
    augmented.append(noise)
    
    # 좌우 반전
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    return augmented

def create_dataset_yaml():
    """데이터셋 YAML 파일 생성"""
    yaml_content = """
# YOLOv11 dataset configuration
path: data/processed  # dataset root dir
train: train  # train images (relative to 'path')
val: val  # val images (relative to 'path')
test: test  # test images (optional)

# Classes
names:
  0: rock
  1: paper
  2: scissors
  3: none
"""
    
    with open("data/dataset.yaml", "w") as f:
        f.write(yaml_content)

if __name__ == "__main__":
    preprocess_data()
