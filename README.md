# YOLO_RCP

rock-paper-scissors/
│
├── data/                      # 데이터셋 폴더
│   ├── raw/                   # 원본 이미지
│   ├── processed/             # 전처리된 이미지
│   └── dataset.yaml           # 데이터셋 설정 파일
│
├── models/                    # 학습된 모델 저장
│   └── best.pt                # 최종 학습 모델
│
├── src/                       # 소스 코드
│   ├── data_collection.py     # 데이터 수집 스크립트
│   ├── data_preprocessing.py  # 데이터 전처리 스크립트
│   ├── train_model.py         # 모델 학습 스크립트
│   ├── game_logic.py          # 게임 로직 구현
│   └── ui_components.py       # UI 컴포넌트
│
├── assets/                    # 게임 자산
│   ├── sounds/                # 효과음
│   ├── images/                # 이미지
│   └── animations/            # 애니메이션
│
├── main.py                    # 메인 게임 실행 파일
├── requirements.txt           # 필요 패키지 목록
└── README.md                  # 프로젝트 설명

# 환경설정
pip install -r requirements.txt

# 데이터 수집
python src/data_collection.py

# 데이터 전처리
python src/data_preprocessing.py

# 모델 학습
python src/train_model.py

# 게임 실행
python main.py
