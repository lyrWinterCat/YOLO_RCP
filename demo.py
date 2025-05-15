# -*- coding: utf-8 -*-
import cv2
import random
import os
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# 한글 텍스트 출력 함수
def put_korean_text(img, text, position, font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 파일 경로 (프로젝트 폴더에 폰트 파일 추가 필요)
    font_path = "./assets/font/NanumGothic.ttf"  # 또는 다른 한글 폰트 파일
    
    if not os.path.exists(font_path):
        print(f"경고: 폰트 파일을 찾을 수 없습니다: {font_path}")
        print("영문으로 표시합니다.")
        # 영문으로 대체
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, color, 2)
        return img
    
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 카메라 장치 확인 함수
def list_available_cameras(max_devices=10):
    available_cameras = []
    for device_idx in range(max_devices):
        try:
            cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(device_idx)
                    print(f"카메라 {device_idx}: 사용 가능")
                else:
                    print(f"카메라 {device_idx}: 프레임을 읽을 수 없음")
            else:
                print(f"카메라 {device_idx}: 열 수 없음")
            cap.release()
        except Exception as e:
            print(f"카메라 {device_idx} 확인 중 오류: {e}")
    return available_cameras

# YOLO 모델 로드 - ONNX 모델 사용
# model = YOLO("models/best.onnx")
model = YOLO("models/best2.pt")
print("모델 로드 완료")

# AI 판단 함수
def get_ai_move(user_move):
    counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return counter.get(user_move.lower(), "none")

# 클래스 이름 매핑 (테스트 결과 기반으로 수정)
label_map = {
    "Rock": "rock",
    "Paper": "paper",
    "Scissors": "scissors",
    "rock": "rock",
    "paper": "paper",
    "scissors": "scissors",
    "0": "rock",
    "1": "paper",
    "2": "scissors"
}

# 메인 함수
def main():
    print("사용 가능한 카메라 장치를 확인하는 중...")
    cameras = list_available_cameras()

    if not cameras:
        print("사용 가능한 카메라가 없습니다.")
        return

    print(f"사용 가능한 카메라 장치 인덱스: {cameras}")

    if len(cameras) == 1:
        camera_index = cameras[0]
        print(f"카메라 {camera_index}를 사용합니다.")
    else:
        print("\n여러 카메라가 감지되었습니다. 사용할 카메라를 선택하세요:")
        for i, idx in enumerate(cameras):
            print(f"{i+1}. 카메라 {idx}")

        while True:
            try:
                choice = int(input("번호를 입력하세요: ")) - 1
                if 0 <= choice < len(cameras):
                    camera_index = cameras[choice]
                    break
                else:
                    print("유효하지 않은 선택입니다. 다시 시도하세요.")
            except ValueError:
                print("숫자를 입력하세요.")

    # 카메라 설정
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"카메라 {camera_index}를 열 수 없습니다.")
        return

    print("\n[실시간 가위바위보 데모 시작]")
    print("웹캠을 켜고 손을 화면 중앙에 위치시켜 주세요. (종료: Q 키)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다. 다시 시도합니다.")
            continue

        # 프레임 좌우 반전 (거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 모델 예측 - 간단한 방식으로 변경
        results = model(frame)
        result = results[0]
        
        # 결과 처리
        boxes = result.boxes
        num_objects = len(boxes)
        
        # 화면에 결과 표시할 프레임 준비
        annotated_frame = result.plot()
        
        if num_objects == 0:
            # 손 객체가 없는 경우
            annotated_frame = put_korean_text(annotated_frame, "손을 인식하지 못했어요.", (30, 40), font_size=30, color=(100, 100, 100))
        elif num_objects > 1:
            # 2개 이상의 손 객체가 인식된 경우
            annotated_frame = put_korean_text(annotated_frame, "손이 2개 이상 인식됨! 화면 또는 자세를 조정해 주세요.", (30, 40), font_size=30, color=(0, 0, 255))
            print("손이 2개 이상 인식됨! 화면 또는 자세를 조정해 주세요.")
        else:
            # 정상적으로 하나의 손 객체만 인식된 경우
            best_idx = boxes.conf.argmax().item()
            label_id = int(boxes.cls[best_idx])
            conf = float(boxes.conf[best_idx])
            
            # 클래스 이름 가져오기
            class_name = result.names.get(label_id, "unknown")
            print(f"감지된 클래스: {class_name}, 신뢰도: {conf:.2f}")
            
            # 클래스 이름 매핑
            user_move = label_map.get(class_name, class_name.lower())
            ai_move = get_ai_move(user_move)
            
            # 화면에 텍스트 출력
            annotated_frame = put_korean_text(annotated_frame, f"사용자: {user_move} ({conf:.2f})", (30, 40), font_size=30, color=(0, 255, 0))
            annotated_frame = put_korean_text(annotated_frame, f"컴퓨터: {ai_move}", (30, 80), font_size=30, color=(0, 0, 255))
            print(f"사용자: {user_move} ({conf:.2f})  →  컴퓨터: {ai_move}")
        
        # 화면 출력
        cv2.imshow("YOLO RSP Demo", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[End Demo]")

if __name__ == "__main__":
    main()
