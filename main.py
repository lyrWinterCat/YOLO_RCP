# main.py
import pygame
import sys
import cv2
import time
from src.game_logic import RockPaperScissorsGame
from src.ui_components import GameUI

def main():
    # 게임 초기화
    game = RockPaperScissorsGame()
    ui = GameUI(width=800, height=600)
    
    # 게임 시작
    if not game.start_game():
        print("게임을 시작할 수 없습니다.")
        return
    
    # 배경 음악 재생
    ui.sounds['background'].play(-1)  # 무한 반복
    
    # 게임 루프
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 좌클릭
                    # 인트로 화면에서 버튼 클릭 처리
                    if game.current_state == game.states['INTRO']:
                        if ui.is_button_hovered('start', event.pos):
                            ui.play_sound('button')
                            game.current_state = game.states['SETUP']
                        
                        elif ui.is_button_hovered('mode_normal', event.pos):
                            ui.play_sound('button')
                            game.game_mode = 'normal'
                            game.max_rounds = float('inf')
                        
                        elif ui.is_button_hovered('mode_challenge', event.pos):
                            ui.play_sound('button')
                            game.game_mode = 'challenge'
                            game.max_rounds = 10
                        
                        elif ui.is_button_hovered('difficulty_easy', event.pos):
                            ui.play_sound('button')
                            game.difficulty = 'easy'
                        
                        elif ui.is_button_hovered('difficulty_hard', event.pos):
                            ui.play_sound('button')
                            game.difficulty = 'hard'
                    
                    # 결과 화면에서 다시 시작 버튼 클릭 처리
                    elif game.current_state == game.states['RESULT'] or game.current_state == game.states['SUMMARY']:
                        if ui.is_button_hovered('restart', event.pos):
                            ui.play_sound('button')
                            if game.current_state == game.states['SUMMARY']:
                                # 게임 통계 초기화
                                game.stats = {
                                    'player_wins': 0,
                                    'ai_wins': 0,
                                    'draws': 0,
                                    'fouls': 0,
                                    'rounds_played': 0,
                                    'player_choices': {'rock': 0, 'paper': 0, 'scissors': 0, 'none': 0}
                                }
                            game.current_state = game.states['COUNTDOWN']
                            game.countdown_start = time.time()
        
        # 웹캠 프레임 처리
        if game.current_state != game.states['INTRO']:
            if not game.process_frame():
                running = False
                continue
        
        # 화면 그리기
        if game.current_state == game.states['INTRO']:
            ui.draw_intro_screen()
        
        elif game.current_state == game.states['SETUP']:
            ui.draw_setup_screen()
            ui.draw_webcam(game.frame, 240, 200)
            
            # 손이 인식되면 카운트다운 시작
            player_choice, confidence = game.detect_hand()
            if player_choice != 'none' and confidence > 0.5:
                game.current_state = game.states['COUNTDOWN']
                game.countdown_start = time.time()
                ui.play_sound('countdown')
        
        elif game.current_state == game.states['COUNTDOWN']:
            countdown_text = game.get_countdown_text()
            ui.draw_countdown_screen(countdown_text)
            ui.draw_webcam(game.frame, 240, 300)
            
            # 카운트다운 종료 확인
            elapsed = time.time() - game.countdown_start
            if elapsed >= game.countdown_duration:
                game.current_state = game.states['CAPTURE']
        
        elif game.current_state == game.states['CAPTURE']:
            # 손 제스처 캡처 및 결과 처리
            game.player_choice, game.confidence = game.detect_hand()
            game.ai_choice = game.determine_ai_choice(game.player_choice)
            game.result = game.determine_winner(game.player_choice, game.ai_choice)
            game.update_stats(game.player_choice, game.result)
            
            # 결과에 따른 사운드 재생
            if game.result == 'player':
                ui.play_sound('win')
            elif game.result == 'ai':
                ui.play_sound('lose')
            else:
                ui.play_sound('draw')
            
            game.current_state = game.states['RESULT']
        
        elif game.current_state == game.states['RESULT']:
            ui.draw_result_screen(
                game.player_choice,
                game.ai_choice,
                game.get_result_text(),
                game.confidence
            )
            
            # 일정 시간 후 다음 라운드로 (또는 요약 화면으로)
            if time.time() - game.countdown_start > game.countdown_duration + 3:
                if game.stats['rounds_played'] >= game.max_rounds and game.game_mode == 'challenge':
                    game.current_state = game.states['SUMMARY']
                else:
                    game.current_state = game.states['COUNTDOWN']
                    game.countdown_start = time.time()
                    ui.play_sound('countdown')
        
        elif game.current_state == game.states['SUMMARY']:
            ui.draw_summary_screen(game.stats)
        
        # 화면 업데이트
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    # 게임 종료
    game.stop_game()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
