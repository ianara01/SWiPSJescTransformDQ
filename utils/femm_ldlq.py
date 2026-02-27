# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 23:05:52 2026

@author: user, SANG JIN PARK
"""

import femm
import math

import numpy as np

# 1. FEMM 실행 (환경 변수의 bin 폴더 내 실행파일 참조 - 환경 변수에 등록된 FEMM 실행 파일을 자동으로 찾아)
femm.openfemm()

# 27슬롯 모터의 자속 밀도 해석을 위한 새 문서 생성
femm.newdocument(0) # 0: Magnetics
femm.main_maximize()

# 2. 만약 특정 경로의 파일을 열고 싶다면 (mfiles 폴더 내 예제 등)
# femm.opendocument(r'C:\femm42\mfiles\example.fem')

def get_phase_flux_linkage(phase_name):
    """
    FEMM에서 회로(PhaseA/B/C)의 flux linkage [Wb-turn] 반환
    """
    props = femm.mo_getcircuitproperties(phase_name)
    # props = (current, voltage, flux_linkage)
    return props[2]

def abc_to_dq(psi_a, psi_b, psi_c):
    """
    Park transform (θe = 0 기준)
    """
    psi_d = (2/3) * (psi_a - 0.5*psi_b - 0.5*psi_c)
    psi_q = (2/3) * (math.sqrt(3)/2)*(psi_b - psi_c)
    return psi_d, psi_q

def inject_d_axis_current(Ipk):
    """
    d-axis current injection
    Assumes PhaseA aligned with d-axis (θe=0)
    """
    femm.mi_setcurrent("PhaseA",  Ipk)
    femm.mi_setcurrent("PhaseB", -0.5 * Ipk)
    femm.mi_setcurrent("PhaseC", -0.5 * Ipk)

def inject_q_axis_current(Ipk):
    """
    q-axis current injection
    """
    femm.mi_setcurrent("PhaseA", 0.0)
    femm.mi_setcurrent("PhaseB",  math.sqrt(3)/2 * Ipk)
    femm.mi_setcurrent("PhaseC", -math.sqrt(3)/2 * Ipk)
def compute_psi_pm():
    femm.mi_setcurrent("PhaseA", 0.0)
    femm.mi_setcurrent("PhaseB", 0.0)
    femm.mi_setcurrent("PhaseC", 0.0)

    femm.mi_analyze()
    femm.mi_loadsolution()

    psi_a = get_phase_flux_linkage("PhaseA")
    psi_b = get_phase_flux_linkage("PhaseB")
    psi_c = get_phase_flux_linkage("PhaseC")

    psi_d, _ = abc_to_dq(psi_a, psi_b, psi_c)
    return psi_d

def extract_ld_lq_from_femm(fem_file, I_test, compute_psi_pm=True):

    try:
        # --- PM flux ---
        psi_pm = compute_psi_pm()

        # --- d-axis ---
        Id = I_test
        Ipk = math.sqrt(2) * Id
        inject_d_axis_current(Ipk)

        femm.mi_analyze()
        femm.mi_loadsolution()

        psi_a = get_phase_flux_linkage("PhaseA")
        psi_b = get_phase_flux_linkage("PhaseB")
        psi_c = get_phase_flux_linkage("PhaseC")

        psi_d, _ = abc_to_dq(psi_a, psi_b, psi_c)
        Ld = (psi_d - psi_pm) / (math.sqrt(2) * Id)

        # --- q-axis ---
        Iq = I_test
        Ipk = math.sqrt(2) * Iq
        inject_q_axis_current(Ipk)

        femm.mi_analyze()
        femm.mi_loadsolution()

        psi_a = get_phase_flux_linkage("PhaseA")
        psi_b = get_phase_flux_linkage("PhaseB")
        psi_c = get_phase_flux_linkage("PhaseC")

        _, psi_q = abc_to_dq(psi_a, psi_b, psi_c)
        Lq = psi_q / (math.sqrt(2) * Iq)

        return Ld, Lq

    finally:
        femm.closefemm()


def calculate_ld_lq_from_flux(flux_abc, current_amp, theta_deg=0):
    """
    3상 자속과 전류를 받아 d-q축 자속 및 인덕턴스를 계산합니다.
    theta_deg: 현재 해석된 모델의 회전자 정렬 각도 (보통 0도 정렬 기준)
    """
    theta_rad = np.radians(theta_deg)
    
    # Park Transformation Matrix (2/3 정규화 방식)
    # d-axis가 A상과 일치한다고 가정할 때의 행렬
    T = (2/3) * np.array([
        [np.cos(theta_rad), np.cos(theta_rad - 2*np.pi/3), np.cos(theta_rad - 4*np.pi/3)],
        [-np.sin(theta_rad), -np.sin(theta_rad - 2*np.pi/3), -np.sin(theta_rad - 4*np.pi/3)]
    ])
    
    flux_abc_vec = np.array(flux_abc)
    flux_dq = np.dot(T, flux_abc_vec)
    
    psi_d = flux_dq[0]
    psi_q = flux_dq[1]
    
    # L = Psi / I (단, 영구자석 자속이 포함된 경우 별도 보정 필요하나 여기서는 단순 인덕턴스 위주)
    Ld = psi_d / current_amp if current_amp != 0 else 0
    Lq = psi_q / current_amp if current_amp != 0 else 0
    
    return Ld, Lq, psi_d, psi_q

#LdLq_DB = {
#    (17, 3, 20): {"Ld_mH": 0.42, "Lq_mH": 0.68},
#    (18, 2, 22): {"Ld_mH": 0.39, "Lq_mH": 0.61},
#}

if __name__ == "__main__":
    fem_file = r"...경로..."
    I_test = 10.0  # A rms

    Ld_H, Lq_H = extract_ld_lq_from_femm(fem_file, I_test)
    print(f"Ld = {Ld_H:.6e} H, Lq = {Lq_H:.6e} H")
    Ld_mH = Ld_H * 1e3
    Lq_mH = Lq_H * 1e3 
    print(f"Ld = {Ld_mH:.3f} mH, Lq = {Lq_mH:.3f} mH")
    #LdLq_DB[(AWG, Parallels, Turns)] = {"Ld_mH": Ld_mH, "Lq_mH": Lq_mH}
# utils/femm_ldlq.py