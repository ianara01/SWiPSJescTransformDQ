"""
Docstring for core.search.rl_agent
State: (awg, par, turns, rpm)

Action: ±1 이동

Reward: margin - penalty
"""
# core/search/rl_agent.py

# -*- coding: utf-8 -*-
"""
Core RL Agent for ESC Motor Design Optimization
Advanced DQN (Deep Q-Network) with Experience Replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

import math
from utils.utils import awg_area_mm2
import configs.config as cfg

def calculate_fill_factor(awg, par, turns):
    """
    물리적 점적률 실시간 계산
    """
    # 1. 소선 단면적 (mm2)
    bare_area = awg_area_mm2(int(awg))
    
    # 2. 전체 구리 면적 (Total Copper Area)
    # turns는 슬롯 내 한쪽 면 기준이므로 전체 도체 수는 turns * 2 (이층권 기준)
    total_copper_area = bare_area * par * (turns * 2)
    
    # 3. 슬롯 면적 (Config에서 참조, 없으면 기본값 70mm2 가정)
    slot_area = getattr(cfg, "SLOT_AREA_MM2", 70.0)
    
    return total_copper_area / slot_area

def get_refined_reward(self, margin_pct, fail_prob, state):
    """
    점적률이 연동된 정밀 보상 함수
    state: (awg, par, turns, rpm)
    """
    awg, par, turns, _ = state
    fill_factor = calculate_fill_factor(awg, par, turns)
    
    # 기본 보상: 성능 마진 (토크/전압 여유)
    reward = margin_pct * 2.0
    
    # --- 점적률 패널티 (Soft & Hard Constraint) ---
    # 1. Hard Limit: 80% 초과 시 즉시 탈락 (강력한 음수 보상)
    if fill_factor > 0.80:
        return -50.0 
    
    # 2. Warning Zone: 70% ~ 80% 구간 (점진적 감점)
    # 제작 난이도 상승을 반영하여 지수적으로 보상을 깎음
    if fill_factor > 0.70:
        penalty = math.exp((fill_factor - 0.70) * 20) # 70% 넘으면 급격히 상승
        reward -= penalty

    # 3. Efficiency Zone: 점적률이 너무 낮으면(30% 미만) 동손 증가 및 공간 낭비
    if fill_factor < 0.30:
        reward -= 5.0

    # 4. 물리적 실패(전압 제한 초과 등) 시 추가 패널티
    if fail_prob > 0.05:
        reward -= 20.0

    return reward

# =============================================================================
# 1. Q-Network: 설계 상태를 분석하여 보상을 예측하는 심층 신경망
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 3층 Dense 레이어로 구성 (ESC 설계의 비선형성 학습)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# 2. Advanced DQNAgent: 학습 및 결정 엔진
# =============================================================================
class DQNAgent:
    def __init__(self, state_size=4, action_size=27): # AWG, Par, Turns, RPM (4) / 3x3x3 Actions (27)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # 경험 재생 버퍼
        
        # 하이퍼파라미터
        self.gamma = 0.95    # 미래 보상 할인율
        self.epsilon = 1.0   # 탐험(Exploration) 확률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 메인 모델 및 타겟 모델 (학습 안정성 확보)
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """설계 경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """현재 상태에서 최적의 설계 변경 행동 선택"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # 탐험
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item() # 학습된 최적 행동

    def replay(self, batch_size):
        """경험 재생을 통한 신경망 학습"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            
            # Target Q-Value 계산
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.target_model(next_state)).item())
            
            target_f = self.model(state)
            target_f[action] = target
            
            # 오차 역전파 및 가중치 업데이트
            loss = nn.MSELoss()(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_reward(self, margin_pct, fail_prob, fill_factor=None):
        """
        고도화된 다중 목표 보상 체계
        1. 마진이 높을수록 보상 가중
        2. 물리적 실패(전압 초과 등) 시 강력한 패널티
        3. 점적률(Slot Fill Factor) 초과 시 패널티 (추가 가능)
        """
        if fail_prob > 0.05: return -20.0 # 물리적 불능 설계
        
        reward = margin_pct * 2.0
        
        # 점적률 제약 조건 고도화 (예: 75% 초과 시 패널티)
        if fill_factor and fill_factor > 0.75:
            reward -= 5.0
            
        return reward
# ================================End of File=============================================