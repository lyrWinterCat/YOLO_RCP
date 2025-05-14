import gradio as gr
import cv2
import numpy as np
import random
import os
from ultralytics import YOLO

# ----------------------------
# 1. YOLO 모델 로드
# ----------------------------
model = YOLO("models/best.pt")

# ----------------------------
# 2. AI 판단 함수 (난이도 반영)
# ----------------------------
def get_ai_move(user_move, difficulty="hard"):
    counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    if difficulty == "hard":
        return counter.get(user_move, "foul")
    else:
        options = [counter[user_move]] * 3 + [user_move]
        return random.choice(options) if user_move in counter else "foul"

# ----------------------------
# 3. 손 이미지 불러오기
# ----------------------------
def load_hand_image(hand_type):
    path = f"assets/images/{hand_type}.png"
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return np.zeros((224, 224, 3), dtype=np.uint8)

# ----------------------------
# 4. 상태 변수
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
# 5. 게임 로직
# ----------------------------
def play_game(frame, mode, difficulty):
    if frame is None:
        default_img = load_hand_image("yolo_c")
        return default_img, "카메라를 켜주세요!", "", state["rounds_left"], ""

    results = model.predict(frame, conf=0.5)
    user_move = "none"
    ai_move = "none"
    outcome = ""

    if results and len(results[0].boxes) > 0:
        label_id = int(results[0].boxes.cls[0])
        user_move = results[0].names[label_id]
        ai_move = get_ai_move(user_move, difficulty)
    else:
        user_move = "foul"
        outcome = "❌ 손이 인식되지 않았어요. 다시 시도해주세요."

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

    if mode == "challenge":
        state["rounds_left"] -= 1

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
    default_img = load_hand_image("yolo_c")
    return default_img, "카메라 앞에 손을 보여주세요!", "", state["rounds_left"], ""

# ----------------------------
# 7. Gradio 인터페이스 구성
# ----------------------------
default_ai_img = load_hand_image("yolo_c")

with gr.Blocks(title="절대 이길 수 없는 가위바위보 게임") as demo:
    gr.Markdown("# 💻 절대 이길 수 없는 가위바위보 게임")

    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=True, label="게임 화면", width=400, height=400)
        output_img = gr.Image(value=default_ai_img, label="AI의 손", width=400, height=400)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## 🎮 게임 방법
            0. 웹캠을 켜고 카메라를 연결해주세요.
            1. 게임 모드와 난이도를 선택하세요.  
            2. '게임 시작' 버튼을 클릭하세요.  
            3. 웹캠에 손을 보여주세요.  
            4. 카운트다운이 끝나면 가위, 바위, 보 중 하나를 내세요.  
            5. AI는 항상 이기는 손 모양을 냅니다 (어려움 모드).
            """)
        with gr.Column():
            gr.Markdown("""
            ## 🕹️ 모드 설명
            - **Normal**: 제한 없이 계속 플레이  
            - **Challenge**: 10라운드 후 결과 요약

            ## 🎯 난이도 설명
            - **Easy**: AI가 가끔 실수를 합니다   
            - **Hard**: AI가 항상 이깁니다
            """)

    mode = gr.Radio(["normal", "challenge"], label="게임 모드", value="normal")
    difficulty = gr.Radio(["easy", "hard"], label="난이도", value="hard")
    start_btn = gr.Button("게임 시작")

    output_text = gr.Textbox(label="결과 메시지")
    score_text = gr.Textbox(label="전적")
    rounds_left = gr.Number(label="남은 라운드", value=5)
    streak_text = gr.Textbox(label="이벤트")

    start_btn.click(
        fn=reset_game,
        inputs=[mode, difficulty],
        outputs=[output_img, output_text, score_text, rounds_left, streak_text]
    )

    webcam.change(
        fn=lambda frame: play_game(frame, state["mode"], state["difficulty"]),
        inputs=webcam,
        outputs=[output_img, output_text, score_text, rounds_left, streak_text]
    )

# ----------------------------
# 8. 실행
# ----------------------------
demo.launch()