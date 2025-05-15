# -*- coding: utf-8 -*-
import cv2
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

# YOLO v11 모델 로드 - 향상된 설정
model = YOLO("models/best3.pt")
print("YOLO v11 모델 로드 완료")

# AI 판단 함수
def get_ai_move(user_move):
    counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return counter.get(user_move.lower(), "none")

# 클래스 이름 매핑
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
    camera_index = 0
    print(f"{camera_index}번 카메라를 사용합니다.")

    # 카메라 설정 - 향상된 설정
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # 카메라 해상도 설정 (선택사항)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
        
        # YOLO v11 모델 예측 - 향상된 설정
        results = model.predict(frame, conf=0.5, iou=0.45)
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
            
            # 승패 결정 추가
            result_text = determine_winner(user_move, ai_move)
            annotated_frame = put_korean_text(annotated_frame, result_text, (30, 120), font_size=30, color=(255, 165, 0))
            print(f"사용자: {user_move} ({conf:.2f})  →  컴퓨터: {ai_move}  →  {result_text}")
        
        # 화면 출력
        cv2.imshow("YOLO v11 RSP Demo", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[End Demo]")

# 승패 결정 함수 추가
def determine_winner(user_move, ai_move):
    if user_move == ai_move:
        return "무승부!"
    elif (user_move == "rock" and ai_move == "scissors") or \
         (user_move == "scissors" and ai_move == "paper") or \
         (user_move == "paper" and ai_move == "rock"):
        return "사용자 승리!"
    else:
        return "컴퓨터 승리!"

if __name__ == "__main__":
    main()
