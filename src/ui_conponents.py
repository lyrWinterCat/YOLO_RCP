# src/ui_components.py
import pygame
import cv2
import numpy as np

class GameUI:
    def __init__(self, width=800, height=600):
        # Pygame 초기화
        pygame.init()
        pygame.mixer.init()
        
        # 화면 설정
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("절대 이길 수 없는 가위바위보 게임")
        
        # 폰트 설정
        self.title_font = pygame.font.Font(None, 60)
        self.large_font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # 색상 설정
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # 이미지 로드
        self.load_images()
        
        # 사운드 로드
        self.load_sounds()
        
        # 버튼 정의
        self.buttons = {}
        self.create_buttons()
    
    def load_images(self):
        """게임에 필요한 이미지 로드"""
        self.images = {
            'background': pygame.image.load('assets/images/background.png'),
            'rock': pygame.image.load('assets/images/rock.png'),
            'paper': pygame.image.load('assets/images/paper.png'),
            'scissors': pygame.image.load('assets/images/scissors.png'),
            'none': pygame.image.load('assets/images/none.png'),
            'logo': pygame.image.load('assets/images/logo.png')
        }
        
        # 이미지 크기 조정
        for key in self.images:
            if key != 'background' and key != 'logo':
                self.images[key] = pygame.transform.scale(self.images[key], (150, 150))
        
        self.images['background'] = pygame.transform.scale(self.images['background'], (self.width, self.height))
        self.images['logo'] = pygame.transform.scale(self.images['logo'], (300, 100))
    
    def load_sounds(self):
        """게임에 필요한 사운드 로드"""
        self.sounds = {
            'background': pygame.mixer.Sound('assets/sounds/background.mp3'),
            'countdown': pygame.mixer.Sound('assets/sounds/countdown.mp3'),
            'win': pygame.mixer.Sound('assets/sounds/win.mp3'),
            'lose': pygame.mixer.Sound('assets/sounds/lose.mp3'),
            'draw': pygame.mixer.Sound('assets/sounds/draw.mp3'),
            'button': pygame.mixer.Sound('assets/sounds/button.mp3')
        }
        
        # 배경 음악 설정
        self.sounds['background'].set_volume(0.3)
    
    def create_buttons(self):
        """게임에 필요한 버튼 생성"""
        # 시작 화면 버튼
        self.buttons['start'] = {
            'rect': pygame.Rect(self.width//2 - 100, 300, 200, 50),
            'text': '게임 시작',
            'color': self.GREEN,
            'hover_color': (0, 200, 0)
        }
        
        self.buttons['mode_normal'] = {
            'rect': pygame.Rect(self.width//2 - 220, 380, 200, 50),
            'text': '일반 모드',
            'color': self.BLUE,
            'hover_color': (0, 0, 200)
        }
        
        self.buttons['mode_challenge'] = {
            'rect': pygame.Rect(self.width//2 + 20, 380, 200, 50),
            'text': '챌린지 모드',
            'color': self.RED,
            'hover_color': (200, 0, 0)
        }
        
        # 난이도 버튼
        self.buttons['difficulty_easy'] = {
            'rect': pygame.Rect(self.width//2 - 220, 460, 200, 50),
            'text': '쉬움',
            'color': self.GREEN,
            'hover_color': (0, 200, 0)
        }
        
        self.buttons['difficulty_hard'] = {
            'rect': pygame.Rect(self.width//2 + 20, 460, 200, 50),
            'text': '어려움',
            'color': self.RED,
            'hover_color': (200, 0, 0)
        }
        
        # 결과 화면 버튼
        self.buttons['restart'] = {
            'rect': pygame.Rect(self.width//2 - 100, 500, 200, 50),
            'text': '다시 시작',
            'color': self.GREEN,
            'hover_color': (0, 200, 0)
        }
    
    def draw_button(self, key, hover=False):
        """버튼 그리기"""
        button = self.buttons[key]
        color = button['hover_color'] if hover else button['color']
        
        pygame.draw.rect(self.screen, color, button['rect'], border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK, button['rect'], 2, border_radius=10)
        
        text = self.medium_font.render(button['text'], True, self.BLACK)
        text_rect = text.get_rect(center=button['rect'].center)
        self.screen.blit(text, text_rect)
    
    def is_button_hovered(self, key, pos):
        """버튼 위에 마우스가 있는지 확인"""
        return self.buttons[key]['rect'].collidepoint(pos)
    
    def draw_intro_screen(self):
        """인트로 화면 그리기"""
        # 배경
        self.screen.blit(self.images['background'], (0, 0))
        
        # 로고
        logo_rect = self.images['logo'].get_rect(center=(self.width//2, 150))
        self.screen.blit(self.images['logo'], logo_rect)
        
        # 제목
        title = self.title_font.render("절대 이길 수 없는 가위바위보", True, self.WHITE)
        title_rect = title.get_rect(center=(self.width//2, 230))
        self.screen.blit(title, title_rect)
        
        # 버튼
        mouse_pos = pygame.mouse.get_pos()
        
        for key in ['start', 'mode_normal', 'mode_challenge', 'difficulty_easy', 'difficulty_hard']:
            hover = self.is_button_hovered(key, mouse_pos)
            self.draw_button(key, hover)
    
    def draw_setup_screen(self):
        """설정 화면 그리기"""
        # 배경
        self.screen.blit(self.images['background'], (0, 0))
        
        # 안내 메시지
        text = self.large_font.render("손을 중앙에 위치시켜 주세요", True, self.WHITE)
        text_rect = text.get_rect(center=(self.width//2, 100))
        self.screen.blit(text, text_rect)
    
    def draw_countdown_screen(self, countdown_text):
        """카운트다운 화면 그리기"""
        # 배경
        self.screen.blit(self.images['background'], (0, 0))
        
        # 카운트다운 텍스트
        text = self.title_font.render(countdown_text, True, self.RED)
        text_rect = text.get_rect(center=(self.width//2, self.height//2))
        self.screen.blit(text, text_rect)
    
    def draw_result_screen(self, player_choice, ai_choice, result_text, confidence):
        """결과 화면 그리기"""
        # 배경
        self.screen.blit(self.images['background'], (0, 0))
        
        # 플레이어 선택
        player_text = self.medium_font.render("당신의 선택", True, self.WHITE)
        player_text_rect = player_text.get_rect(center=(self.width//4, 150))
        self.screen.blit(player_text, player_text_rect)
        
        player_img_rect = self.images[player_choice].get_rect(center=(self.width//4, 250))
        self.screen.blit(self.images[player_choice], player_img_rect)
        
        # AI 선택
        ai_text = self.medium_font.render("AI의 선택", True, self.WHITE)
        ai_text_rect = ai_text.get_rect(center=(3*self.width//4, 150))
        self.screen.blit(ai_text, ai_text_rect)
        
        ai_img_rect = self.images[ai_choice].get_rect(center=(3*self.width//4, 250))
        self.screen.blit(self.images[ai_choice], ai_img_rect)
        
        # 결과 텍스트
        result = self.large_font.render(result_text, True, self.RED)
        result_rect = result.get_rect(center=(self.width//2, 350))
        self.screen.blit(result, result_rect)
        
        # 신뢰도 표시
        conf_text = self.small_font.render(f"인식 신뢰도: {confidence:.2f}", True, self.WHITE)
        conf_rect = conf_text.get_rect(center=(self.width//2, 400))
        self.screen.blit(conf_text, conf_rect)
        
        # 다시 시작 버튼
        mouse_pos = pygame.mouse.get_pos()
        hover = self.is_button_hovered('restart', mouse_pos)
        self.draw_button('restart', hover)
    
    def draw_summary_screen(self, stats):
        """게임 요약 화면 그리기"""
        # 배경
        self.screen.blit(self.images['background'], (0, 0))
        
        # 제목
        title = self.large_font.render("게임 결과 요약", True, self.WHITE)
        title_rect = title.get_rect(center=(self.width//2, 100))
        self.screen.blit(title, title_rect)
        
        # 통계 정보
        stats_texts = [
            f"총 라운드: {stats['rounds_played']}",
            f"플레이어 승리: {stats['player_wins']}",
            f"AI 승리: {stats['ai_wins']}",
            f"무승부: {stats['draws']}",
            f"반칙: {stats['fouls']}"
        ]
        
        for i, text in enumerate(stats_texts):
            rendered = self.medium_font.render(text, True, self.WHITE)
            rect = rendered.get_rect(center=(self.width//2, 180 + i*50))
            self.screen.blit(rendered, rect)
        
        # 선택 분포 그래프 (간단한 텍스트로 표시)
        choices = stats['player_choices']
        choice_text = self.small_font.render(
            f"선택 분포: 바위({choices['rock']}), 보({choices['paper']}), 가위({choices['scissors']}), 기타({choices['none']})",
            True, self.WHITE
        )
        choice_rect = choice_text.get_rect(center=(self.width//2, 430))
        self.screen.blit(choice_text, choice_rect)
        
        # 다시 시작 버튼
        mouse_pos = pygame.mouse.get_pos()
        hover = self.is_button_hovered('restart', mouse_pos)
        self.draw_button('restart', hover)
    
    def convert_cv_to_pygame(self, cv_image):
        """OpenCV 이미지를 Pygame 이미지로 변환"""
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image = cv2.resize(cv_image, (320, 240))
        pygame_image = pygame.surfarray.make_surface(cv_image.swapaxes(0, 1))
        return pygame_image
    
    def draw_webcam(self, frame, x, y):
        """웹캠 화면 그리기"""
        if frame is not None:
            pygame_frame = self.convert_cv_to_pygame(frame)
            self.screen.blit(pygame_frame, (x, y))
            
            # 손 위치 가이드 (중앙에 사각형)
            pygame.draw.rect(self.screen, self.GREEN, (x + 110, y + 70, 100, 100), 2)
    
    def play_sound(self, sound_key):
        """사운드 재생"""
        self.sounds[sound_key].play()
