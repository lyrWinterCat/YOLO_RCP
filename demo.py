import cv2
import random
from ultralytics import YOLO

# ----------------------------
# 1. YOLO 모델 로드
# ----------------------------
model = YOLO("model/best.pt")  # YOLOv8 사용자 모델 경로

# ----------------------------
# 2. AI 판단 함수
# ----------------------------
def get_ai_move(user_move):
    counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return counter.get(user_move, "none")

# ----------------------------
# 3. 웹캠 실행 및 실시간 감지
# ----------------------------
cap = cv2.VideoCapture(0)  # 기본 웹캠 장치

print("\n[실시간 가위바위보 데모 시작]")
print("웹캠을 켜고 손을 화면 중앙에 위치시켜 주세요. (종료: Q 키)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # YOLO로 손 예측
    results = model.predict(frame, conf=0.5)
    user_move = "none"
    ai_move = "none"

    if results and results[0].boxes:
        label_id = int(results[0].boxes.cls[0])
        user_move = results[0].names[label_id]
        ai_move = get_ai_move(user_move)

        # 화면에 텍스트 출력
        cv2.putText(frame, f"User: {user_move}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Computer: {ai_move}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        print(f"User: {user_move}  →  Computer: {ai_move}")
    else:
        cv2.putText(frame, f"손을 인식 중...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)

    # 화면 출력
    cv2.imshow("YOLO RPS Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[데모 종료]")
# ----------------------------