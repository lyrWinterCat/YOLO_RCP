# src/data_collection.py
import cv2
import os
import time
import numpy as np

def collect_data():
    # 데이터 저장 경로 설정
    base_path = "data/raw/"
    classes = ["rock", "paper", "scissors", "none"]
    
    # 각 클래스별 폴더 생성
    for cls in classes:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    current_class = None
    frame_count = 0
    
    print("데이터 수집을 시작합니다.")
    print("키 안내: 'r'=바위, 'p'=보, 's'=가위, 'n'=없음, 'q'=종료")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 좌우 반전 (거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 중앙에 ROI(관심 영역) 표시
        h, w = frame.shape[:2]
        roi_size = min(h, w) // 2
        roi_x = w // 2 - roi_size // 2
        roi_y = h // 2 - roi_size // 2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
        
        # 현재 클래스 표시
        if current_class:
            cv2.putText(frame, f"Collecting: {current_class}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {frame_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(1)
        
        # 키 입력에 따라 클래스 설정
        if key == ord('r'):
            current_class = "rock"
            frame_count = 0
        elif key == ord('p'):
            current_class = "paper"
            frame_count = 0
        elif key == ord('s'):
            current_class = "scissors"
            frame_count = 0
        elif key == ord('n'):
            current_class = "none"
            frame_count = 0
        elif key == ord('q'):
            break
        
        # 스페이스바를 누르면 현재 프레임 저장
        if key == 32 and current_class:  # 스페이스바
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            timestamp = int(time.time() * 1000)
            filename = f"{base_path}{current_class}/{current_class}_{timestamp}.jpg"
            cv2.imwrite(filename, roi)
            frame_count += 1
            print(f"저장됨: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("데이터 수집 완료")

if __name__ == "__main__":
    collect_data()
