import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import glob
import sys
import re

# YOLO 모델 로드
model = YOLO("models/best.onnx", task='detect')
print("모델 로드 완료")

# 클래스 이름 매핑 - justhand 추가
label_map = {
    "Rock": "rock",
    "Paper": "paper",
    "Scissors": "scissors",
    "rock": "rock",
    "paper": "paper",
    "scissors": "scissors",
    "justhand": "justhand",
    "0": "rock",
    "1": "paper",
    "2": "scissors",
    "3": "justhand"
}

# 클래스 ID 매핑 - justhand 추가
class_map = {'paper': 0, 'rock': 1, 'scissors': 2, 'justhand': 3}

# 저장할 폴더 경로
img_dir = 'collected_data/images'
txt_dir = 'collected_data/labels'
bbox_dir = 'collected_data/bbox_i'

# 폴더가 없으면 생성
os.makedirs(img_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)
os.makedirs(bbox_dir, exist_ok=True)

# 파일명에서 숫자 추출 함수
def get_last_file_index(directory, prefix='img_', ext='.jpg'):
    if not os.path.exists(directory):
        return -1
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
    if not files:
        return -1
    # 파일명에서 숫자 추출
    indices = []
    for f in files:
        match = re.search(r'img_(\d{5})', f)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        return -1
    return max(indices)

# 바운딩 박스 좌표를 YOLO 포맷으로 변환
def convert_bbox_to_yolo(bbox, img_width, img_height):
    # bbox는 (xmin, ymin, xmax, ymax) 형식
    xmin, ymin, xmax, ymax = bbox
    
    # YOLO 포맷: (x_center, y_center, width, height) - 정규화된 좌표
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height

# 라벨 저장 함수
def save_yolo_label(filename, class_id, bbox):
    # bbox = (x_center, y_center, width, height) 정규화된 좌표
    x_center, y_center, width, height = bbox
    with open(filename, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 삭제된 bbox 이미지에 해당하는 원본 이미지와 라벨 파일도 함께 삭제하는 함수
def sync_deleted_files():
    print("\n[파일 동기화 시작]")
    
    # bbox_img 폴더의 모든 파일 목록 가져오기
    bbox_files = set([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(bbox_dir, "*.jpg"))])
    
    # 원본 이미지 폴더의 모든 파일 목록 가져오기
    img_files = set([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(img_dir, "*.jpg"))])
    
    # 라벨 폴더의 모든 파일 목록 가져오기
    txt_files = set([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(txt_dir, "*.txt"))])
    
    # bbox_img 폴더에 없는 파일 찾기
    deleted_files = (img_files.union(txt_files)) - bbox_files
    
    if not deleted_files:
        print("삭제할 파일이 없습니다. 모든 파일이 동기화되어 있습니다.")
        return
    
    print(f"총 {len(deleted_files)}개의 파일을 동기화합니다.")
    
    for filename in deleted_files:
        # 원본 이미지 파일 삭제
        img_path = os.path.join(img_dir, filename + ".jpg")
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"삭제됨: {img_path}")
        
        # 라벨 파일 삭제
        txt_path = os.path.join(txt_dir, filename + ".txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
            print(f"삭제됨: {txt_path}")
    
    print("[파일 동기화 완료]")

# 메인 데이터 수집 함수 - 클래스 지정 매개변수 추가
def collect_data(fixed_class=None):
    # 카메라 설정
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow 백엔드 사용
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 마지막 파일 인덱스 가져오기
    last_index = get_last_file_index(img_dir)
    count = last_index + 1
    print(f"마지막 파일 인덱스: {last_index}, 새 파일은 img_{count:05d}.jpg부터 시작합니다.")
    
    # 고정 클래스가 지정된 경우 표시
    if fixed_class:
        print(f"지정된 클래스: {fixed_class} (모든 이미지가 이 클래스로 저장됩니다)")
    
    start_time = time.time()
    
    print("\n[데이터 수집 시작]")
    print("웹캠을 켜고 손을 화면 중앙에 위치시켜 주세요. (종료: Q 키)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다. 다시 시도합니다.")
            continue

        # 프레임 좌우 반전 (거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 현재 화면 표시용 프레임
        display_frame = frame.copy()
        
        # 1초에 1장 저장
        current_time = time.time()
        if current_time - start_time >= 1.0:
            start_time = current_time
            
            # 모델 예측
            results = model(frame)
            result = results[0]
            
            # 결과 처리
            boxes = result.boxes
            num_objects = len(boxes)
            
            # 화면에 결과 표시할 프레임 준비
            annotated_frame = result.plot()
            
            if num_objects == 0:
                # 손 객체가 없는 경우
                cv2.putText(display_frame, "No hand detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            elif num_objects > 1:
                # 2개 이상의 손 객체가 인식된 경우
                cv2.putText(display_frame, "Multiple hands detected!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # 정상적으로 하나의 손 객체만 인식된 경우
                best_idx = boxes.conf.argmax().item()
                label_id = int(boxes.cls[best_idx])
                conf = float(boxes.conf[best_idx])
                
                # 클래스 이름 가져오기
                class_name = result.names.get(label_id, "unknown")
                
                # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
                box = boxes.xyxy[best_idx].tolist()
                
                # 이미지 크기
                h, w = frame.shape[:2]
                
                # YOLO 포맷으로 변환
                yolo_bbox = convert_bbox_to_yolo(box, w, h)
                
                # 파일명 생성
                filename = f"img_{count:05d}"
                
                # 이미지 저장
                img_path = os.path.join(img_dir, filename + ".jpg")
                cv2.imwrite(img_path, frame)
                
                # 클래스 이름 및 ID 결정 (고정 클래스가 있으면 그것을 사용)
                if fixed_class:
                    user_move = fixed_class
                    class_id = class_map.get(fixed_class, 3)  # 기본값은 justhand(3)
                else:
                    # 클래스 이름 매핑
                    user_move = label_map.get(class_name, class_name.lower())
                    class_id = class_map.get(user_move, 0)  # 기본값은 paper(0)
                
                # 라벨 저장
                label_path = os.path.join(txt_dir, filename + ".txt")
                save_yolo_label(label_path, class_id, yolo_bbox)
                
                # 바운딩 박스 그린 이미지 저장
                bbox_img = frame.copy()
                xmin, ymin, xmax, ymax = map(int, box)
                cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label_text = f"{user_move} ({conf:.2f})"
                cv2.putText(bbox_img, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                bbox_img_path = os.path.join(bbox_dir, filename + ".jpg")
                cv2.imwrite(bbox_img_path, bbox_img)
                
                print(f"저장 완료: {filename}.jpg - 클래스: {user_move}, 신뢰도: {conf:.2f}")
                
                # 화면에 텍스트 출력
                cv2.putText(display_frame, f"Class: {user_move} ({conf:.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Saved: {filename}.jpg", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                count += 1
        
        # 화면 출력
        cv2.imshow("YOLO Data Collection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[데이터 수집 종료] 총 {count - (last_index + 1)}개의 이미지와 라벨이 저장되었습니다.")

# 메인 함수
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sync":
            # 파일 동기화 모드
            sync_deleted_files()
        elif sys.argv[1] in class_map:
            # 클래스 지정 모드
            collect_data(fixed_class=sys.argv[1])
        else:
            print(f"알 수 없는 인자: {sys.argv[1]}")
            print(f"사용 가능한 클래스: {', '.join(class_map.keys())}")
            print("사용법: python script.py [--sync|rock|paper|scissors|justhand]")
    else:
        # 기본 데이터 수집 모드
        collect_data()
