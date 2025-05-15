# -*- coding: utf-8 -*-
import cv2
import numpy as np
import gradio as gr
import os
import time
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# 전역 변수로 모델과 이미지 미리 로드
model = None
hand_images = None

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

# 컴퓨터 손 이미지 로드 함수 - 한 번만 실행되도록 수정
def load_hand_images():
    global hand_images
    if hand_images is not None:
        return hand_images
        
    hands = {}
    try:
        # PIL을 사용하여 이미지 로드 (RGB 형식으로 올바르게 로드)
        from PIL import Image
        
        hands["rock"] = Image.open("assets/images/rock.png").convert("RGBA")
        hands["paper"] = Image.open("assets/images/paper.png").convert("RGBA")
        hands["scissors"] = Image.open("assets/images/scissors.png").convert("RGBA")
        hands["default"] = Image.open("assets/images/yolo_c.png").convert("RGBA")
        hands["none"] = Image.open("assets/images/none.png").convert("RGBA")
        
        # 이미지 크기 조정
        for key in hands:
            if hands[key] is not None:
                hands[key] = hands[key].resize((400, 400))
                # PIL 이미지를 numpy 배열로 변환
                hands[key] = np.array(hands[key])
        
        hand_images = hands
    except Exception as e:
        print(f"손 이미지 로드 중 오류: {e}")
    
    return hands

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

# 게임 결과 판정 함수
def determine_winner(user_move, ai_move):
    if user_move == ai_move:
        return "무승부!"
    elif (user_move == "rock" and ai_move == "scissors") or \
         (user_move == "scissors" and ai_move == "paper") or \
         (user_move == "paper" and ai_move == "rock"):
        return "사용자 승리!"
    else:
        return "컴퓨터 승리!"

# 모델 로드 함수 - 한 번만 실행되도록 수정
def load_model():
    global model
    if model is None:
        model = YOLO("models/best3.pt")
        # 추론 설정 최적화
        model.conf = 0.25  # 신뢰도 임계값 낮춤
        model.iou = 0.45   # IOU 임계값 조정
        print("모델 로드 완료")
    return model

# 웹캠 처리 함수 - 최적화
def process_webcam(webcam_image):
    if webcam_image is None:
        return None, None, "웹캠을 연결해주세요."
    
    # 모델 로드 (처음 한 번만)
    model = load_model()
    
    # 이미지 로드 (처음 한 번만)
    hands = load_hand_images()
    
    # 웹캠 이미지 좌우 반전 (거울 효과)
    frame = cv2.flip(webcam_image.copy(), 1)
    
    # 모델 예측 - 작은 이미지로 예측하여 속도 향상
    results = model(frame, verbose=False)  # verbose=False로 로그 출력 제거
    result = results[0]
    
    # 결과 처리
    boxes = result.boxes
    num_objects = len(boxes)
    
    # 손 객체 인식 결과 처리
    if num_objects == 0:
        # 손 객체가 없는 경우
        frame = put_korean_text(frame, "손을 인식하지 못했어요.", (30, 40), font_size=30, color=(100, 100, 100))
        result_text = "손을 인식하지 못했어요. 손 모양을 카메라에 보여주세요."
        
        # 기본 이미지 표시
        computer_hand_img = hands["default"]
    elif num_objects > 1:
        # 2개 이상의 손 객체가 인식된 경우
        # 바운딩 박스 그리기
        annotated_frame = result.plot()
        frame = annotated_frame
        
        frame = put_korean_text(frame, "손이 2개 이상 인식됨!", (30, 40), font_size=30, color=(0, 0, 255))
        result_text = "손이 2개 이상 인식됨! 화면 또는 자세를 조정해 주세요."
        
        # none.png 이미지 표시
        computer_hand_img = hands["none"]
    else:
        # 정상적으로 하나의 손 객체만 인식된 경우
        # 바운딩 박스 그리기
        annotated_frame = result.plot()
        frame = annotated_frame
        
        best_idx = boxes.conf.argmax().item()
        label_id = int(boxes.cls[best_idx])
        conf = float(boxes.conf[best_idx])
        
        # 클래스 이름 가져오기
        class_name = result.names.get(label_id, "unknown")
        
        # 클래스 이름 매핑
        user_move = label_map.get(class_name, class_name.lower())
        
        # 사용자 움직임에 대응하는 AI 움직임 선택
        ai_move = get_ai_move(user_move)
        
        # 화면에 텍스트 출력
        frame = put_korean_text(frame, f"사용자: {user_move} ({conf:.2f})", (30, 40), font_size=30, color=(0, 255, 0))
        frame = put_korean_text(frame, f"컴퓨터: {ai_move}", (30, 80), font_size=30, color=(0, 0, 255))
        
        # 승패 결정
        result_text = determine_winner(user_move, ai_move)
        frame = put_korean_text(frame, result_text, (30, 120), font_size=30, color=(255, 165, 0))
        result_text = f"사용자: {user_move} ({conf:.2f}) vs 컴퓨터: {ai_move} - {result_text}"
        
        # 컴퓨터 손 이미지 선택
        computer_hand_img = hands[ai_move] if ai_move in hands else hands["default"]
    
    return frame, computer_hand_img, result_text

# CSS 스타일 정의
css = """
.container {max-width: 1400px !important; margin: auto !important;}
.webcam-container {display: flex !important; justify-content: center !important; align-items: center !important;}
.result-container {display: flex !important; justify-content: center !important; align-items: center !important;}
.hand-image {width: 400px !important; height: 400px !important; object-fit: contain !important;}
.webcam-feed {width: 640px !important; height: 480px !important; object-fit: contain !important;}
"""

# 그라디오 인터페이스 구성
with gr.Blocks(title="절대 이길 수 없는 가위바위보 게임", css=css) as demo:
    # 앱 시작 시 모델과 이미지 미리 로드
    load_model()
    load_hand_images()
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 10px;">
        <h1>🎮 절대 이길 수 없는 가위바위보 게임</h1>
        <p>웹캠에 손 모양을 보여주세요! AI가 자동으로 인식하고 대응합니다.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3, elem_classes="webcam-container"):
            # 웹캠 입력 - 크기 증가
            webcam = gr.Image(source="webcam", streaming=True, label="게임 화면", elem_classes="webcam-feed")
        
        with gr.Column(scale=2, elem_classes="result-container"):
            # 컴퓨터 손 이미지 - 크기 증가
            computer_hand = gr.Image(label="컴퓨터의 선택", elem_classes="hand-image")
    
    # 결과 출력 영역
    result_text = gr.Textbox(label="게임 결과", value="손 모양을 카메라에 보여주세요.")
    
    # 이벤트 연결 - concurrency_limit 매개변수 제거
    webcam.stream(
        process_webcam,
        inputs=[webcam],
        outputs=[webcam, computer_hand, result_text],
        show_progress=False,
        max_batch_size=1  # 배치 크기 제한
    )
    
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px;">
        <h2>🎲 게임 방법</h2>
        <ol style="text-align: left; max-width: 600px; margin: auto;">
            <li>카메라에 손 모양(가위, 바위, 보)을 보여주세요.</li>
            <li>AI가 당신의 손 모양을 인식하고 컴퓨터의 선택과 함께 결과를 보여줍니다.</li>
            <li>컴퓨터는 항상 이기는 선택을 합니다!</li>
            <li>아직 공부중이라 반응이 조금 느릴 수 있어요!!</li>
        </ol>
    </div>
    """)

# 그라디오 앱 실행
if __name__ == "__main__":
    # 그라디오 3.50.2 버전에 맞게 queue 설정 조정
    demo.queue(max_size=1).launch()
