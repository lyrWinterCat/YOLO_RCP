# src/train_model.py
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def train_model():
    # 모델 초기화
    model = YOLO('yolov11n.pt')  # 또는 다른 YOLO11 모델 크기 선택
    
    # 학습 설정
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,
        imgsz=224,
        batch=16,
        patience=20,
        device='0',  # GPU 사용 (CPU만 사용 시 'cpu')
        project='models',
        name='rps_model',
        save=True,
        exist_ok=True
    )
    
    # 학습 결과 시각화
    plot_results(results)
    
    # 모델 검증
    val_results = model.val()
    print(f"검증 결과: {val_results}")
    
    # 최종 모델 저장
    best_model_path = os.path.join('models', 'rps_model', 'weights', 'best.pt')
    final_model_path = os.path.join('models', 'best.pt')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_model_path)
        print(f"최종 모델 저장 완료: {final_model_path}")
    else:
        print(f"경고: 최종 모델을 찾을 수 없습니다: {best_model_path}")

def plot_results(results):
    """학습 결과 시각화"""
    # 결과 그래프 저장 경로
    os.makedirs('models/plots', exist_ok=True)
    
    # 손실 그래프
    plt.figure(figsize=(12, 8))
    plt.plot(results.results_dict['train/box_loss'], label='train box loss')
    plt.plot(results.results_dict['val/box_loss'], label='val box loss')
    plt.title('Training and Validation Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('models/plots/box_loss.png')
    
    # 정확도 그래프
    plt.figure(figsize=(12, 8))
    plt.plot(results.results_dict['metrics/precision'], label='precision')
    plt.plot(results.results_dict['metrics/recall'], label='recall')
    plt.plot(results.results_dict['metrics/mAP50'], label='mAP50')
    plt.title('Precision, Recall and mAP50')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('models/plots/metrics.png')
    
    print("학습 결과 그래프가 models/plots 폴더에 저장되었습니다.")

if __name__ == "__main__":
    train_model()
