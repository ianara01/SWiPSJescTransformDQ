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
from types import SimpleNamespace

import math
from utils.utils import awg_area_mm2
import configs.config as cofg

def calculate_fill_factor(awg, par, turns):
    """
    물리적 점적률 실시간 계산
    """
    # 1. 소선 단면적 (mm2)
    bare_area = awg_area_mm2(awg)
    
    # 2. 전체 구리 면적 (Total Copper Area)
    # turns는 슬롯 내 한쪽 면 기준이므로 전체 도체 수는 turns * 2 (이층권 기준)
    total_copper_area = bare_area * par * (turns * 2)
    
    # 3. 슬롯 면적 (Config에서 참조, 없으면 기본값 70mm2 가정)
    slot_area = getattr(cofg, "slot_area_mm2_list", [130.0])[0]
    
    return total_copper_area / slot_area


# =====================================================================================
#             Mode 5. rl research : 강화학습 기반 설계 탐색 (RL Search 모드)
# =====================================================================================
def evaluate_design_physically(state):
    from core.physics import apply_envelope_for_case, kw_rpm_to_torque_nm, estimate_mlt_mm, compute_lengths_side_basis
    import configs.config as cofg

    """
    RL 에이전트가 제안한 단일 설계안(state)의 물리적 타당성 검토
    엔진 인터페이스(dict)에 맞춰 수정된 버전
    """

    # [클래스 정의] 점(.)과 get()을 동시에 지원하는 하이브리드 객체
    class EnvelopeObject(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self # 점(.) 접근 가능하게 함
        def get(self, key, default=None):
            return super().get(key, default)
        
    awg, par, turns, rpm = state
    
    # 1. 목표 출력으로부터 필요 토크 계산
    target_kw = getattr(cofg, "Target_Power_kW", 1.5)
    target_torque = kw_rpm_to_torque_nm(target_kw, rpm)

    # 2. 물리 엔진용 입력 데이터
    case_dict = {
        "awg": int(awg),
        "parallels": int(par),
        "turns": int(turns),
        "rpm": int(rpm),
        "target_torque_Nm": target_torque
    }
    # 3. RPM_ENV 구성 및 데이터 보정
    # RL 모드 단독 실행 시 RPM_ENV가 비어있을 수 있으므로 config의 후보군으로 강제 생성
    # [수정] RPM_ENV 로드 및 기본값 강제 할당
    # cofg.RPM_ENV가 비어있어서 발생하는 [WARN]을 해결하기 위해 여기서 직접 생성합니다.
    raw_env = getattr(cofg, "RPM_ENV", {})
    if not raw_env or rpm not in raw_env:
        # 엔진이 rpm_env[rpm] 에 접근한 뒤 그 내부 속성을 보므로 구조를 맞춰줌
        inner_data = {
            "awg_list": list(cofg.awg_candidates),
            "par_lo": min(cofg.par_candidates),
            "par_hi": max(cofg.par_candidates),
            "turn_lo": min(cofg.turn_candidates_base),
            "turn_hi": max(cofg.turn_candidates_base),
            "nslot_lo": cofg.NSLOT_USER_RANGE[0],   # 수정된 부분
            "nslot_hi": cofg.NSLOT_USER_RANGE[1],   # 수정된 부분
        }
        # rpm_env[600] = EnvelopeObject({...})
        rpm_env = {rpm: EnvelopeObject(inner_data)}
    else:
        # 기존 데이터가 있다면 래핑만 수행
        rpm_env = {k: EnvelopeObject(v) if isinstance(v, dict) else v for k, v in raw_env.items()}
    # 4. MLT 및 권선 길이 계산 (config.py 변수명 정확히 참조)
    # [추가] MLT 및 길이 계산을 위한 파라미터 로드 (config에서 가져옴)
    # 실제 환경에 맞는 변수명으로 조정 필요
    try:
        # estimate_mlt_mm에 필요한 변수들을 config에서 매칭
        mlt_base = estimate_mlt_mm(
            slot_pitch_mm=cofg.slot_pitch_mm_nom, # config: slot_pitch_mm_nom
            stack_mm=cofg.Stack_rotor,           # config: Stack_rotor (Rotor stack 기준)
            coil_span_slots=cofg.coil_span_slots_list[0], # config: list의 첫번째 값
            N_slots=cofg.N_slots,                # config: N_slots
            D_use=cofg.D_use                     # config: D_use
        )
        
        # MLT 스케일 적용 (config: MLT_scale_list)
        mlt_scale = cofg.MLT_scale_list[1] if len(cofg.MLT_scale_list) > 1 else 1.0
        mlt_mm = mlt_base * mlt_scale
        
        # 3상 총 길이 계산
        l_phase_m, l_total_m = compute_lengths_side_basis(
            turns_per_slot_side=int(turns),
            MLT_mm=mlt_mm,
            m=3, # 3상 고정
            coils_per_phase=int(cofg.N_slots / (3 * 2)) # 단순화된 계산 (예시)
        )
    except Exception as e:
        # print(f"[DEBUG] Length calculation failed: {e}")
        l_phase_m, l_total_m = 0.0, 0.0

    # 5. 물리 엔진 호출
    try:
        # SimpleNamespace를 쓰지 말고 그냥 딕셔너리(rpm_env)를 보냅니다.
        # 만약 apply_envelope_for_case 내부에서 .awg_list(점)를 쓴다면 SimpleNamespace가 맞지만,
        # 에러 메시지를 보면 내부에서 .get()을 쓰고 있으므로 딕셔너리여야 합니다.
        # 엔진에서 해당 케이스의 유효 범위(Envelope)를 가져옴
        valid_awgs, valid_pars, _ = apply_envelope_for_case(case_dict, rpm_env)

        # 에이전트가 제안한 값이 유효 범위 내에 있는지 확인
        is_ok = (int(awg) in valid_awgs) and (int(par) in valid_pars)

        # [성공] 엔진 가이드라인을 만족함
        # 실제 정밀 해석(V_margin 등) 함수가 있다면 여기서 추가 호출 가능
        # 현재는 성공 여부에 따른 기본 마진 점수 부여
        margin_pct = 10.0 if is_ok else -20.0
        fail_prob = 0.0 if is_ok else 1.0
            
    except Exception as e:
        print(f"[PHYS_ERR] {e}") # 여기서 .get 에러가 났던 것임
        return {"margin": -100.0, "fail": 1.0, "L_phase_m": 0, "L_total_m": 0}

    # 상세 결과를 딕셔너리로 묶어서 반환
    return {
        "margin": margin_pct,
        "fail": fail_prob,
        "L_phase_m": l_phase_m,
        "L_total_m": l_total_m
    }

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
    def __init__(self, state_size=4, action_space=27): # AWG, Par, Turns, RPM (4) / 3x3x3 Actions (27)
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.memory = deque(maxlen=2000) # 경험 재생 버퍼
        
        # 하이퍼파라미터
        self.gamma = 0.95    # 미래 보상 할인율
        self.epsilon = 1.0   # 탐험(Exploration) 확률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 메인 모델 및 타겟 모델 (학습 안정성 확보)
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
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
            if len(self.memory) < batch_size:
                return

            minibatch = random.sample(self.memory, batch_size)
            
            # 데이터를 텐서 덩어리(Batch)로 변환
            states = torch.FloatTensor([x[0] for x in minibatch]).to(self.device)
            actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
            rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
            next_states = torch.FloatTensor([x[3] for x in minibatch]).to(self.device)
            dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)

            # 현재 Q-값과 다음 상태의 최대 Q-값 계산 (타겟 모델 활용)
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_model(next_states).max(1)[0].detach()
            
            # 벨만 방정식 적용
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            # 손실 계산 및 최적화 (한 번에 수행)
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def get_refined_reward(self, margin_pct, fail_prob, state):
        """
        점적률이 연동된 정밀 보상 함수
        state: (awg, par, turns, rpm)
        """
        import math
        awg, par, turns, _ = state
        fill_factor = calculate_fill_factor(awg, par, turns)
        
        # 기본 보상: 성능 마진 (토크/전압 여유)
        reward = margin_pct * 2.0
        
        # --- 점적률 패널티 (Soft & Hard Constraint) ---
        # 1. Hard Limit: 80% 초과 시 즉시 탈락 (강력한 음수 보상)
        if fill_factor > 0.85:
            return -50.0 
        
        # 2. Warning Zone: 70% ~ 85% 구간 (점진적 감점)
        # 제작 난이도 상승을 반영하여 지수적으로 보상을 깎음
        if fill_factor > 0.70:
            penalty = math.exp((fill_factor - 0.70) * 20) # 70% 넘으면 급격히 상승
            reward -= penalty

        # 3. Efficiency Zone: 점적률이 너무 낮으면(40% 미만) 동손 증가 및 공간 낭비
        if fill_factor < 0.40:
            reward -= 5.0

        # 4. 물리적 실패(전압 제한 초과 등) 시 추가 패널티
        if fail_prob > 0.05:
            reward -= 20.0

        return reward

    def get_reward(self, margin_pct, fail_prob, fill_factor, kt_val=None):
            # 1. 치명적 실패 (전압 부족 등)
            if fail_prob > 0.05: return -50.0 
            
            # 2. 마진 보상 (10~15% 사이가 최적임을 가정)
            reward = margin_pct * 1.5 
            
            # 3. 점적률 패널티 (지수 함수 적용하여 75% 근처에서 급격히 증가)
            if fill_factor > 0.70:
                reward -= math.exp((fill_factor - 0.70) * 20) 
                
            # 4. 성능 가중치 (동일 조건에서 높은 Kt 선호)
            if kt_val:
                reward += kt_val * 10.0
                
            return reward
# ================================End of File=============================================

"""
def evaluate_design_physically(state):
    from core.physics import apply_envelope_for_case, kw_rpm_to_torque_nm
    
    RL 에이전트가 제안한 단일 설계안(state)의 물리적 타당성 검토
    engine.py의 핵심 물리 로직을 단일 케이스에 대해 수행
    (예시값 반환, 실제 구현 시 physics.apply_envelope_for_case 호출)
    state: (awg, parallels, turns, rpm)
    
    awg, par, turns, rpm = state
    
    # 1. 목표 출력(kW)으로부터 필요 토크 계산
    # config에 정의된 Target_Power_kW를 기준으로 계산 (예: 1.5kW)
    # 목표 부하 계산
    target_kw = getattr(cofg, "Target_Power_kW", 1.5)
    target_torque = kw_rpm_to_torque_nm(target_kw, rpm)

    # 2. 단일 케이스 해석을 위한 텐서 생성 (Batch Size = 1)
    # core.physics 로직은 텐서 기반이므로 단일 값도 텐서로 변환 필요
    device = cofg.DEVICE if cofg.DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # [수정 포인트] awg_area_mm2 함수 대신 AWG_TABLE 참조
    try:
        # config.py의 AWG_TABLE에서 해당 AWG의 면적(area)을 가져옴
        area_val = cofg.AWG_TABLE[int(awg)]["area"]
    except KeyError:
        # AWG 번호가 테이블에 없을 경우 예외 처리 (기본값 또는 에러 리턴)
        print(f"[WARN] AWG {awg} not found in AWG_TABLE. Using fallback.")
        area_val = 0.5177  # 예: AWG20 기본값
    
    t_awg = torch.tensor([float(awg)], device=device)
    t_area = torch.tensor([float(area_val)], device=device) # 변환 완료
    t_par = torch.tensor([float(par)], device=device)
    t_turns = torch.tensor([float(turns)], device=device)

    # 3. 물리 엔진 호출
    try:
        # apply_envelope_for_case가 텐서를 인자로 받는지 확인 필요
        results = apply_envelope_for_case(
            awg_tensor=t_awg,
            awg_area_tensor=t_area,
            par_tensor=t_par,
            turns_tensor=t_turns,
            target_rpm=rpm,
            target_torque_nm=target_torque,
            motor_type=getattr(cofg, "MOTOR_TYPE", "IPM")
        )
        
        # [수정] 결과 딕셔너리에 margin_pct가 없다면 CSV 컬럼명과 일치하는 키를 찾습니다.
        # 파일에 정의된 V_LL_margin_pct를 사용하도록 매핑
        margin_pct = results.get('V_LL_margin_pct', results.get('V_margin_pct', 0.0)).item()
        
        fail_prob = 1.0 - results.get('success_mask', torch.tensor(0.0)).item()
        
    except Exception as e:
        print(f"[PHYS_ERR] Evaluation failed for state {state}: {e}")
        return -100.0, 1.0

    return margin_pct, fail_prob

"""