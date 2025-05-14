import gradio as gr
import cv2
import numpy as np
import random
import os
from ultralytics import YOLO

# ----------------------------
# 1. YOLO 모델 로드
# ----------------------------
# 사용자의 커스텀 학습된 YOLOv8 모델 경로를 지정합니다.
model = YOLO("model/best.pt")  # 필요 시 경로 수정

# ----------------------------
# 2. AI 판단 함수 (난이도 반영)
# ----------------------------
def get_ai_move(user_move, difficulty="hard"):
    counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    if difficulty == "hard":
        return counter.get(user_move, "foul")
    else:  # 쉬움 모드: 75% 확률로 AI가 지기도 함
        options = [counter[user_move]] * 3 + [user_move]
        return random.choice(options) if user_move in counter else "foul"

# ----------------------------
# 3. 손 이미지 불러오기 (assets 폴더)
# ----------------------------
def load_hand_image(hand_type):
    path = f"assets/{hand_type}.png"
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return np.zeros((224, 224, 3), dtype=np.uint8)  # 빈 이미지 리턴

# ----------------------------
# 4. 상태 변수 관리용 딕셔너리
# ----------------------------
state = {
    "mode": "normal",
    "difficulty": "hard",
    "rounds_left": 5,
    "wins": 0,
    "losses": 0,
    "streak": 0
}

# ----------------------------
# 5. 게임 플레이 함수
# ----------------------------
def play_game(frame, mode, difficulty):
    results = model.predict(frame, conf=0.5)
    user_move = "none"
    ai_move = "none"
    outcome = ""

    # YOLO 결과 해석
    if results and results[0].boxes:
        label_id = int(results[0].boxes.cls[0])
        user_move = results[0].names[label_id]
        ai_move = get_ai_move(user_move, difficulty)
    else:
        user_move = "foul"
        outcome = "❌ 손이 인식되지 않았어요. 다시 시도해주세요."

    # 승패 판단
    if user_move in ['rock', 'paper', 'scissors']:
        if ai_move == {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}[user_move]:
            outcome = "😈 AI가 이겼어요!"
            state["losses"] += 1
            state["streak"] = state["streak"] - 1 if state["streak"] > 0 else state["streak"] - 1
        elif user_move == ai_move:
            outcome = "😐 비겼어요!"
        else:
            outcome = "🎉 당신이 이겼어요!"
            state["wins"] += 1
            state["streak"] = state["streak"] + 1 if state["streak"] >= 0 else 1
    elif user_move == "foul":
        outcome = "❌ 반칙입니다! 손을 정확히 보여주세요."

    # 챌린지 모드라면 남은 라운드 감소
    if mode == "challenge":
        state["rounds_left"] -= 1

    # 결과 이미지 로드 및 상태 출력
    hand_img = load_hand_image(ai_move)
    score = f"현재 전적: {state['wins']}승 / {state['losses']}패"
    streak_msg = "🔥 10연승!" if state["streak"] >= 10 else "💀 10연패..." if state["streak"] <= -10 else ""

    return hand_img, f"유저: {user_move} / AI: {ai_move}\n{outcome}", score, state["rounds_left"], streak_msg

# ----------------------------
# 6. 게임 초기화 함수
# ----------------------------
def reset_game(mode_val, diff_val):
    state["mode"] = mode_val
    state["difficulty"] = diff_val
    state["wins"] = 0
    state["losses"] = 0
    state["rounds_left"] = 5 if mode_val == "challenge" else 99
    state["streak"] = 0
    return "카메라 앞에 손을 보여주세요!", "", state["rounds_left"], ""

# ----------------------------
# 7. Gradio UI 구성
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 YOLO 가위바위보 챌린지")
    gr.Markdown("게임 모드와 난이도를 선택한 뒤 게임을 시작하세요.")

    mode = gr.Radio(["normal", "challenge"], label="게임 모드", value="normal")
    difficulty = gr.Radio(["easy", "hard"], label="난이도", value="hard")
    start_btn = gr.Button("게임 시작")

    webcam = gr.Image(source="webcam", streaming=True)
    output_img = gr.Image(label="AI의 손")
    output_text = gr.Textbox(label="결과 메시지")
    score_text = gr.Textbox(label="전적")
    rounds_left = gr.Number(label="남은 라운드", value=5)
    streak_text = gr.Textbox(label="이벤트")

    # 게임 초기화
    start_btn.click(fn=reset_game, inputs=[mode, difficulty],
                    outputs=[output_text, score_text, rounds_left, streak_text])

    # 웹캠 입력 시 게임 진행
    webcam.change(fn=lambda frame: play_game(frame, state["mode"], state["difficulty"]),
                  inputs=webcam,
                  outputs=[output_img, output_text, score_text, rounds_left, streak_text])

# ----------------------------
# 8. 실행
# ----------------------------
demo.launch()
