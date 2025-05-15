# -*- coding: utf-8 -*-
import cv2
import numpy as np
import gradio as gr
import os
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

# 모델 로드 - YOLO v11 사용
model = YOLO("models/best4.pt")
print("모델 로드 완료")

# 컴퓨터 손 이미지 미리 로드 (성능 향상)
hands = {}

def load_hand_images():
    global hands
    try:
        # PIL을 사용하여 이미지 로드 (RGB 형식으로 올바르게 로드)
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
        return True
    except Exception as e:
        print(f"손 이미지 로드 중 오류: {e}")
        return False

# 시작 시 이미지 로드
load_success = load_hand_images()

# AI 판단 함수 - YOLO v11의 높은 정확도를 활용
def get_ai_move(user_move):
    if user_move == "justhand":
        return "win"  # justhand는 무조건 컴퓨터 승리
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
    "2": "scissors",
    "3": "justhand"
}

# 게임 결과 판정 함수
def determine_winner(user_move, ai_move):
    if user_move == "justhand":
        return "컴퓨터 승리! (유효하지 않은 손 모양)"
    if user_move == ai_move:
        return "무승부!"
    elif (user_move == "rock" and ai_move == "scissors") or \
         (user_move == "scissors" and ai_move == "paper") or \
         (user_move == "paper" and ai_move == "rock"):
        return "사용자 승리!"
    else:
        return "컴퓨터 승리!"

# 웹캠 처리 함수 - YOLO v11 모델 활용
def process_webcam(webcam_image):
    if webcam_image is None:
        return None, None, "웹캠을 연결해주세요."
    
    # 웹캠 이미지 좌우 반전 (거울 효과)
    frame = cv2.flip(webcam_image.copy(), 1)
    
    # YOLO v11 모델 예측 - 향상된 신뢰도 설정
    results = model.predict(frame, conf=0.5, iou=0.45)
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
        computer_hand_img = hands["default"] if "default" in hands else None
    elif num_objects > 1:
        # 2개 이상의 손 객체가 인식된 경우
        # 바운딩 박스 그리기
        annotated_frame = result.plot()
        frame = annotated_frame
        
        frame = put_korean_text(frame, "손이 2개 이상 인식됨! 화면 또는 자세를 조정해 주세요.", (30, 40), font_size=30, color=(0, 0, 255))
        result_text = "손이 2개 이상 인식됨! 화면 또는 자세를 조정해 주세요."
        
        # 기본 이미지 표시
        computer_hand_img = hands["none"] if "none" in hands else None
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
        if user_move == "justhand":
            # 화면에 텍스트 출력
            frame = put_korean_text(frame, f"사용자: {user_move} ({conf:.2f})", (30, 40), font_size=30, color=(0, 255, 0))
            frame = put_korean_text(frame, "컴퓨터: 승리!", (30, 80), font_size=30, color=(0, 0, 255))
            
            # 판정패 메시지
            result_text = "판정패! (허용되지 않는 손 모양)"
            frame = put_korean_text(frame, result_text, (30, 120), font_size=30, color=(255, 0, 0))
            result_text = f"사용자: {user_move} ({conf:.2f}) vs 컴퓨터: 승리 - {result_text}"
            
            # 컴퓨터 손 이미지는 yolo_c.png 사용
            computer_hand_img = hands["default"]
        else:
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

# CSS 스타일 정의 - 그라디오 3.50.2 호환
css = """
.container {max-width: 1400px !important; margin: auto !important;}
.webcam-container {display: flex !important; justify-content: center !important; align-items: center !important;}
.result-container {display: flex !important; justify-content: center !important; align-items: center !important;}
.hand-image {width: 400px !important; height: 400px !important; object-fit: contain !important;}
.webcam-feed {width: 640px !important; height: 480px !important; object-fit: contain !important;}
"""

# 그라디오 인터페이스 구성 - 그라디오 3.50.2 문법
with gr.Blocks(title="절대 이길 수 없는 가위바위보 게임", css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 10px;">
        <h1>🎮 절대 이길 수 없는 가위바위보 게임</h1>
        <p>웹캠에 손 모양을 보여주세요! AI가 자동으로 인식하고 대응합니다.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3, elem_classes="webcam-container"):
            # 웹캠 입력 - 그라디오 3.50.2 문법
            webcam = gr.Image(source="webcam", streaming=True, label="게임 화면", elem_classes="webcam-feed")
        
        with gr.Column(scale=2, elem_classes="result-container"):
            # 컴퓨터 손 이미지
            computer_hand = gr.Image(label="컴퓨터의 선택", elem_classes="hand-image")
    
    # 결과 출력 영역
    result_text = gr.Textbox(label="게임 결과", value="손 모양을 카메라에 보여주세요.")
    
    # 이벤트 연결 - 그라디오 3.50.2 스트리밍 문법
    webcam.stream(
        fn=process_webcam,
        inputs=[webcam],
        outputs=[webcam, computer_hand, result_text],
        show_progress=False,
        preprocess=True,
        postprocess=True
    )
    
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px;">
        <h2>🎲 게임 방법</h2>
        <ol style="text-align: left; max-width: 600px; margin: auto;">
            <li>카메라에 손 모양(가위, 바위, 보)을 보여주세요.</li>
            <li>AI가 당신의 손 모양을 인식하고 컴퓨터의 선택과 함께 결과를 보여줍니다.</li>
            <li>컴퓨터는 항상 이기는 선택을 합니다!</li>
        </ol>
    </div>
    """)

# 그라디오 앱 실행 - 추가 옵션 설정
if __name__ == "__main__":
    # 이미지 미리 로드 확인
    if not load_success:
        print("경고: 일부 이미지를 로드하지 못했습니다. 기본 이미지를 사용합니다.")
    
    # 그라디오 3.50.2 실행 옵션
    demo.launch(
        share=False,  # 공유 링크 생성 여부
        server_name="0.0.0.0",  # 모든 IP에서 접근 가능
        server_port=7860,  # 기본 포트
        show_api=False,  # API 문서 표시 여부
        favicon_path="assets/images/yolo_c.png"  # 파비콘 설정
    )
