import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

def test_model_with_image(model_path, image_path):
    """
    이미지로 YOLO 모델 테스트
    
    Args:
        model_path: YOLO 모델 경로
        image_path: 테스트 이미지 경로
    """
    # 모델 로드
    try:
        model = YOLO(model_path)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 이미지 로드
    if not os.path.exists(image_path):
        print(f"이미지를 찾을 수 없음: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 읽을 수 없음: {image_path}")
        return
    
    # 이미지 크기 출력
    print(f"이미지 크기: {img.shape}")
    
    # 다양한 신뢰도 임계값으로 예측 테스트
    for conf_threshold in [0.5, 0.3, 0.2, 0.1]:
        print(f"\n신뢰도 임계값: {conf_threshold}")
        results = model.predict(img, conf=conf_threshold)
        
        # 결과 분석
        if len(results[0].boxes) > 0:
            print(f"감지된 객체 수: {len(results[0].boxes)}")
            
            for i, box in enumerate(results[0].boxes):
                label_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names.get(label_id, "unknown")
                
                # 바운딩 박스 좌표 (xyxy 형식)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"객체 {i+1}: 클래스 {label_id} ({class_name}), 신뢰도: {conf:.4f}")
                print(f"  바운딩 박스: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
        else:
            print("감지된 객체 없음")
        
        # 결과 시각화 및 저장
        result_img = results[0].plot(line_width=2, font_size=1)
        output_path = f"test_result_{conf_threshold:.1f}.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"결과 이미지 저장됨: {output_path}")

def test_model_with_folder(model_path, folder_path):
    """
    폴더 내 모든 이미지로 YOLO 모델 테스트
    
    Args:
        model_path: YOLO 모델 경로
        folder_path: 테스트 이미지 폴더 경로
    """
    # 모델 로드
    try:
        model = YOLO(model_path)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                 glob.glob(os.path.join(folder_path, "*.png"))
    
    if not image_files:
        print(f"이미지를 찾을 수 없음: {folder_path}")
        return
    
    print(f"총 {len(image_files)}개 이미지 테스트")
    
    # 결과 저장 폴더 생성
    results_folder = "test_results"
    os.makedirs(results_folder, exist_ok=True)
    
    # 각 이미지 테스트
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 이미지 테스트: {os.path.basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지를 읽을 수 없음: {img_path}")
            continue
        
        # 예측
        results = model.predict(img, conf=0.2)
        
        # 결과 분석
        if len(results[0].boxes) > 0:
            print(f"감지된 객체 수: {len(results[0].boxes)}")
            
            for j, box in enumerate(results[0].boxes):
                label_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names.get(label_id, "unknown")
                print(f"객체 {j+1}: 클래스 {label_id} ({class_name}), 신뢰도: {conf:.4f}")
        else:
            print("감지된 객체 없음")
        
        # 결과 시각화 및 저장
        result_img = results[0].plot(line_width=2, font_size=1)
        output_path = os.path.join(results_folder, f"result_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, result_img)
        print(f"결과 이미지 저장됨: {output_path}")

def test_model_classes(model_path):
    """
    모델의 클래스 정보 확인
    
    Args:
        model_path: YOLO 모델 경로
    """
    try:
        model = YOLO(model_path)
        print(f"모델 로드 성공: {model_path}")
        
        # 클래스 정보 출력
        print("\n클래스 정보:")
        for class_id, class_name in model.names.items():
            print(f"클래스 ID {class_id}: {class_name}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")

if __name__ == "__main__":
    # 모델 경로
    model_path = "models/best.pt"
    
    # 1. 모델 클래스 정보 확인
    print("===== 모델 클래스 정보 확인 =====")
    test_model_classes(model_path)
    
    # 2. 단일 이미지 테스트
    print("\n===== 단일 이미지 테스트 =====")
    test_image = input("\n테스트할 이미지 경로를 입력하세요 (기본: test_hand.jpg): ").strip() or "test_hand.jpg"
    test_model_with_image(model_path, test_image)
    
    # 3. 폴더 내 이미지 테스트 (선택 사항)
    test_folder = input("\n테스트할 이미지 폴더 경로를 입력하세요 (건너뛰려면 Enter): ").strip()
    if test_folder:
        print(f"\n===== 폴더 내 이미지 테스트: {test_folder} =====")
        test_model_with_folder(model_path, test_folder)
    
    print("\n테스트 완료!")
