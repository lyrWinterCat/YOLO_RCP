# 🎮 절대 이길 수 없는 가위바위보 게임 (YOLO 기반)

YOLOv11를 활용한 실시간 가위바위보 인식 게임입니다. 웹캠을 통해 손 모양(가위/바위/보)을 인식하고, AI가 항상 이기는 선택을 하여 사용자와 대결합니다.

본 프로젝트는 OpenCV 기반 콘솔 버전과 Gradio 웹 인터페이스 두 가지를 제공합니다.

## 🔗 데모 링크

- 🤗 [Hugging Face Space에서 체험하기](https://huggingface.co/spaces/WinterCatS2/YOLO_RSP)
- 🎥 [YouTube 시연 영상 보기](https://youtu.be/5S1YVSPta5w)

---

## 📁 프로젝트 구조
```
YOLO_RSP/
├── assets/
│   ├── font/                 # 한글 폰트 (예: NanumGothic.ttf)
│   └── images/               # 손 모양 이미지 (rock.png, paper.png, scissors.png 등)
├── models/                   # 학습된 YOLOv11 모델 파일 (best2.pt, best3.pt 등)
├── test_img/                 # 테스트용 이미지 저장 폴더
├── app.py                    # ✅ Gradio 웹앱 메인 실행 파일
├── app_test2.py              # Gradio 테스트용 보조 파일
├── demo.py                   # ✅ OpenCV 기반 콘솔 인터페이스 실행 파일
├── dataset.py                # 데이터셋 처리용 유틸 (선택적 사용)
├── test.py                   # 테스트/디버깅용 스크립트
├── requirements.txt          # 필요한 패키지 목록
└── README.md                 # 프로젝트 설명 문서
```

---

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt

# 필요 시 수동 설치:
pip install gradio opencv-python ultralytics pillow
```

### 2. 모델 파일 위치 확인
models/best3.pt 또는 best2.pt 등 학습된 YOLOv8 모델을 models/ 폴더에 넣습니다.

### 3. Gradio 웹앱 실행
```bash
python app.py
```
→ 웹브라우저에서 Gradio UI가 자동 실행됩니다. <br>
→ 손 모양을 웹캠에 보여주면 AI가 인식하여 대응합니다.

### 4. 콘솔 기반 OpenCV 데모 실행 (선택)
```bash
python demo.py
```
→ 콘솔에서 카메라 장치 선택 후 실시간으로 손 모양을 감지합니다.

### 🎨 주요 기능 
✅ YOLOv11 모델 기반 손 모양 실시간 감지

✅ 한글 메시지 지원 (NanumGothic.ttf 사용)

✅ Gradio UI 또는 OpenCV 콘솔 중 선택 실행 가능

✅ AI는 항상 이기는 손 모양 선택

✅ 손이 없거나 2개 이상일 경우 사용자에게 피드백 표시

✅ 결과에 따라 컴퓨터의 손 이미지 출력

### 📦 전제 조건
Python 3.8 이상
gradio version = 3.50.2

YOLOv11으로 학습된 .pt 모델 (best2.pt, best3.pt 등)

#### 웹캠 사용 가능 환경

아래 파일이 assets 디렉토리에 존재해야 합니다:
```
assets/images/rock.png
assets/images/paper.png
assets/images/scissors.png
assets/images/yolo_c.png
assets/images/none.png
assets/font/NanumGothic.ttf
```

🧠 개발자 참고
app.py: Gradio 기반 인터랙티브 UI로 AI 반응형 웹 게임 구현

demo.py: 실시간 프레임 기반 콘솔 데모. 빠른 성능 확인 가능

dataset.py, test.py: 모델 학습 및 평가에 사용할 수 있는 보조 코드



