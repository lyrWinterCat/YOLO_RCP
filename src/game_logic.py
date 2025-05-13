# src/game_logic.py
import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

class RockPaperScissorsGame:
    def __init__(self, model_path='models/best.pt', difficulty='hard'):
        # 모델 로드
        self.model = YOLO(model_path)
        
        # 게임 상태
        self.states = {
            'INTRO': 0,
            'SETUP': 1,
            'COUNTDOWN': 2,
            'CAPTURE': 3,
            'RESULT': 4,
            'SUMMARY': 5
        }
        self.current_state = self.states['INTRO']
        
        # 게임 설정
        self.difficulty = difficulty  # 'easy' or 'hard'
        self.game_mode = 'normal'  # 'normal' or 'challenge'
        self.max_rounds = 10 if self.game_mode == 'challenge' else float('inf')
        
        # 게임 통계
        self.stats = {
            'player_wins': 0,
            'ai_wins': 0,
            'draws': 0,
            'fouls': 0,
            'rounds_played': 0,
            'player_choices': {'rock': 0, 'paper': 0, 'scissors': 0, 'none': 0}
        }
        
        # 카운트다운 변수
        self.countdown_start = 0
        self.countdown_duration = 3  # 초
        
        # 결과 변수
        self.player_choice = None
        self.ai_choice = None
        self.result = None
        self.confidence = 0.0
        
        # 웹캠 설정
        self.cap = None
        self.frame = None
    
    def start_game(self):
        """게임 시작"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return False
        return True
    
    def stop_game(self):
        """게임 종료"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def process_frame(self):
        """현재 프레임 처리"""
        ret, self.frame = self.cap.read()
        if not ret:
            return False
        
        # 프레임 좌우 반전 (거울 효과)
        self.frame = cv2.flip(self.frame, 1)
        return True
    
    def detect_hand(self, frame=None):
        """손 제스처 감지"""
        if frame is None:
            frame = self.frame
        
        # 모델로 예측
        results = self.model(frame, verbose=False)[0]
        
        # 결과 처리
        if len(results.boxes) == 0:
            return 'none', 0.0
        
        # 가장 높은 신뢰도의 예측 가져오기
        best_idx = results.boxes.conf.argmax().item()
        confidence = results.boxes.conf[best_idx].item()
        class_id = int(results.boxes.cls[best_idx].item())
        
        # 클래스 ID를 이름으로 변환
        class_names = ['rock', 'paper', 'scissors', 'none']
        predicted_class = class_names[class_id]
        
        return predicted_class, confidence
    
    def determine_ai_choice(self, player_choice):
        """AI의 선택 결정 (항상 이기는 전략)"""
        if self.difficulty == 'easy' and random.random() < 0.2:
            # 20% 확률로 AI가 랜덤하게 선택 (쉬운 난이도)
            return random.choice(['rock', 'paper', 'scissors'])
        
        # 어려운 난이도 또는 쉬운 난이도의 80% 확률
        if player_choice == 'rock':
            return 'paper'  # 바위를 이기는 보
        elif player_choice == 'paper':
            return 'scissors'  # 보를 이기는 가위
        elif player_choice == 'scissors':
            return 'rock'  # 가위를 이기는 바위
        else:
            return random.choice(['rock', 'paper', 'scissors'])
    
    def determine_winner(self, player_choice, ai_choice):
        """승자 결정"""
        if player_choice == 'none':
            return 'foul'
        
        if player_choice == ai_choice:
            return 'draw'
        
        winning_combinations = {
            'rock': 'scissors',
            'paper': 'rock',
            'scissors': 'paper'
        }
        
        if winning_combinations[player_choice] == ai_choice:
            return 'player'
        else:
            return 'ai'
    
    def update_stats(self, player_choice, result):
        """게임 통계 업데이트"""
        self.stats['rounds_played'] += 1
        self.stats['player_choices'][player_choice] += 1
        
        if result == 'player':
            self.stats['player_wins'] += 1
        elif result == 'ai':
            self.stats['ai_wins'] += 1
        elif result == 'draw':
            self.stats['draws'] += 1
        elif result == 'foul':
            self.stats['fouls'] += 1
    
    def update_game_state(self):
        """게임 상태 업데이트"""
        if self.current_state == self.states['INTRO']:
            # 인트로 화면에서 설정 화면으로 전환
            self.current_state = self.states['SETUP']
        
        elif self.current_state == self.states['SETUP']:
            # 설정 완료 후 카운트다운 시작
            self.current_state = self.states['COUNTDOWN']
            self.countdown_start = time.time()
        
        elif self.current_state == self.states['COUNTDOWN']:
            # 카운트다운이 끝나면 캡처 상태로 전환
            elapsed = time.time() - self.countdown_start
            if elapsed >= self.countdown_duration:
                self.current_state = self.states['CAPTURE']
        
        elif self.current_state == self.states['CAPTURE']:
            # 손 제스처 캡처 및 결과 처리
            self.player_choice, self.confidence = self.detect_hand()
            self.ai_choice = self.determine_ai_choice(self.player_choice)
            self.result = self.determine_winner(self.player_choice, self.ai_choice)
            self.update_stats(self.player_choice, self.result)
            self.current_state = self.states['RESULT']
        
        elif self.current_state == self.states['RESULT']:
            # 결과 표시 후 다음 라운드 준비 또는 게임 종료
            if self.stats['rounds_played'] >= self.max_rounds and self.game_mode == 'challenge':
                self.current_state = self.states['SUMMARY']
            else:
                # 잠시 결과를 표시한 후 다시 카운트다운으로
                time.sleep(2)  # 실제 구현에서는 타이머 사용
                self.current_state = self.states['COUNTDOWN']
                self.countdown_start = time.time()
        
        elif self.current_state == self.states['SUMMARY']:
            # 게임 종료 및 통계 표시
            pass
    
    def get_countdown_text(self):
        """카운트다운 텍스트 가져오기"""
        if self.current_state != self.states['COUNTDOWN']:
            return ""
        
        elapsed = time.time() - self.countdown_start
        remaining = self.countdown_duration - elapsed
        
        if remaining <= 0:
            return "가위바위보!"
        else:
            return str(max(1, int(remaining)))
    
    def get_result_text(self):
        """결과 텍스트 가져오기"""
        if self.current_state != self.states['RESULT']:
            return ""
        
        if self.result == 'player':
            return "플레이어 승리!"
        elif self.result == 'ai':
            return "AI 승리!"
        elif self.result == 'draw':
            return "무승부!"
        elif self.result == 'foul':
            return "반칙! 올바른 손 모양을 보여주세요."
