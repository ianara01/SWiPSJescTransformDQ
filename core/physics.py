# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:24:31 2026

@author: user, SANG JIN PARK
"""

from typing import TYPE_CHECKING
import math
import numpy as np
import pandas as pd
from configs.config import (
    AWG_TABLE,
    Br_20,
    Hcj_20,
    J_max_list,
    Ke_LL_rms_per_krpm_nom,
    Kt_rms_list,
    MARGIN_MIN_PCT,
    MARGIN_MIN_V,
    NSLOT_USER_RANGE,
    SAFETY_RELAX,
    alpha_Br,
    alpha_Cu,
    alpha_Hcj,
    awg_candidates,
    par_candidates,
    rho_Cu_20C,
    slot_area_mm2_list,
    slot_fill_limit_list,
    slot_pitch_mm_nom,
    turn_candidates_base,
    Nref_turn,
    D_use,
    N_slots,
    m,
    m_max_list,
    Vdc_list,
    Ke_scale_list,
)
from utils.utils import T, _row_get, awg_area_mm2

#from math import floor
#from __future__ import annotations
from itertools import product
from datetime import datetime
from collections import defaultdict
from bisect import bisect_left
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import copy  # [ADD] auto_adjust_by_pass 에서 사용
import matplotlib.pyplot as plt

import torch

# ======================== 기본 환경 =========================================
# 1. 환경 변수 먼저 선언
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
ITYPE  = torch.int32

# ============================================================
# [UTIL] β-sweep 사인/코사인 가져오기 (전역 재사용)
#   β-sweep 전역 상수 (coarse만 기본 반영; 정밀화는 후보에만 선택 적용 가능)
#   - SIN_FINE/COS_FINE 있으면 그걸 우선 사용
#   - 없으면 beta_samples로 즉석 생성 (torch)
# ============================================================
def _get_sin_cos_fine(beta_samples: int):
    SIN_FINE = globals().get("SIN_FINE", None)
    COS_FINE = globals().get("COS_FINE", None)
    if SIN_FINE is not None and COS_FINE is not None:
        # 길이가 다르면 beta_samples에 맞춰 재생성(호환)
        try:
            if int(SIN_FINE.numel()) == int(beta_samples) and int(COS_FINE.numel()) == int(beta_samples):
                return SIN_FINE, COS_FINE
        except Exception:
            pass
    # fallback: torch로 생성
    DTYPE  = globals().get("DTYPE", torch.float32)
    DEVICE = globals().get("DEVICE", torch.device("cpu"))
    BETA_FINE = torch.linspace(0.0, math.pi/2, steps=int(beta_samples), device=DEVICE, dtype=DTYPE)
    return torch.sin(BETA_FINE), torch.cos(BETA_FINE)

# 2. Config 정의 및 생성
@dataclass
class RunConfig:
    device: torch.device
    dtype: torch.dtype
    itype: torch.int32
    # 추가적인 파라미터들도 이곳에 통합 가능
# 실행 시점에 생성
cfg = RunConfig(device=DEVICE, dtype=DTYPE, itype=ITYPE)

# ========================= 물리적 (Physics) =================================
def resistivity_at_T(Tamb: float) -> float:
    return rho_Cu_20C * (1 + alpha_Cu * (Tamb - 20.0))

def estimate_mlt_mm(slot_pitch_mm, stack_mm, coil_span_slots, N_slots, D_use,
                    alpha_end=1.0, C_end_mm=10.0,
                    span_is_inclusive=True):
    """
    Old estimate_mlt_mm function
    def estimate_mlt_mm(slot_pitch_mm, end_factor, stack_mm, coil_span_slots):
        # 직선(2*stack) + 엔드턴(코일 스팬*slot_pitch*보정계수)
        return 2 * stack_mm + 2 * end_factor * coil_span_slots * slot_pitch_mm
    - coil_span_slots: 현장 정의가 '포함 개수'일 가능성이 높아 y=coil_span_slots-1로 변환
    - end-turn: arc(=y*slot_pitch) 대신 chord로 계산 (과대평가 방지)
    """
    y_raw = int(coil_span_slots)
    y = max(1, y_raw - 1) if span_is_inclusive else max(1, y_raw)   # ✅ 핵심
    R = 0.5 * float(D_use)
    theta = math.pi * y / float(N_slots)            # [rad]
    chord_mm = 2.0 * R * math.sin(theta)

    end_one_mm = float(alpha_end) * chord_mm + float(C_end_mm)
    return 2.0 * float(stack_mm) + 2.0 * end_one_mm

def estimate_mlt_from_resistance(R_phase_ohm: float,
                                 A_wire_mm2: float,
                                 Tamb: float,
                                 turns_per_slot_side: int,
                                 coils_per_phase: int) -> float:
    """
    상 저항으로부터 평균 한 턴 길이(MLT)를 역산

    Parameters
    ----------
    R_phase_ohm : 측정된 상 저항 [Ohm] (온도 T_C에서)
    A_wire_mm2  : 도체 단면적 [mm^2] (병렬 1가닥 기준)
    Tamb         : 저항 측정 온도 [°C]
    turns_per_slot_side : 슬롯 한쪽 면 턴수
    coils_per_phase     : 상당 코일 개수 (더블레이어면 slots_per_phase/2)

    Returns
    -------
    MLT_mm : 평균 한 턴 길이 [mm]
    example to execute:
    MLT_meas_mm = estimate_mlt_from_resistance(
    R_phase_ohm=0.82,      # 예시 측정 상저항
    A_wire_mm2=1.04,       # AWG18
    Tamb=25,                # 측정 온도
    turns_per_slot_side=30,
    coils_per_phase=4
    )

    print(f"Measured MLT ≈ {MLT_meas_mm:.1f} mm")
    """
    # 온도 보정 저항률
    rho_T = resistivity_at_T(Tamb)          # [Ohm·m]
    
    # 단면적 변환 (mm² → m²)
    A_m2 = A_wire_mm2 * 1e-6               # [m²]
    
    # 상 전체 도체 길이
    L_total_phase_m = R_phase_ohm * A_m2 / rho_T   # [m]
    
    # 상 직렬 턴수
    N_turns_phase = turns_per_slot_side * coils_per_phase

    if N_turns_phase <= 0:
        raise ValueError("Invalid turn count")    
    
    # 평균 한 턴 길이
    MLT_m = L_total_phase_m / N_turns_phase  # [m]

    return MLT_m * 1000.0  # → [mm]    

def Br_T(T):
    return Br_20*(1+alpha_Br*(T-20))

def Hcj_T(T):
    return Hcj_20*(1+alpha_Hcj*(T-20))

def get_dynamic_constraints(N_slots, m):
    # 슬롯/상/레이어 구조에 따른 최소 단위는 유지하되, 
    # 실제 제작 가능한 최대치(예: 24 또는 48)까지는 허용하도록 변경
    logical_limit = N_slots // m  # 예: 24슬롯 3상이면 8개
    manufacture_limit = 60        # 공정상 한계치 직접 설정    
    return min(logical_limit * 6, manufacture_limit) # 논리적 한계의 배수까지 허용

def debug_emf_cap(
    rpm, Vdc, m_max,
    Ke_use, gate_pct,
    Nref_turn, coils_per_phase,
    nphase_cap, nslot_cap,
    user_range,
):
    print(
        f"[EMF_CAP] rpm={rpm:.0f} "
        f"Vdc={Vdc:.1f} m_max={m_max:.3f} "
        f"Vavail_LL={(m_max*Vdc/math.sqrt(3)):.2f} "
        f"Ke_use={Ke_use:.4f} "
        f"gate_pct={gate_pct:.4f} "
        f"Nref={Nref_turn} coils_per_phase={coils_per_phase} "
        f"nphase_cap={nphase_cap} "
        f"nslot_cap={nslot_cap} "
        f"user={user_range}"
    )

# === EMF Cap 기반 Nslot 상한 계산 물리 (Physics) ==================================
def compute_nslot_emf_cap(
    Ke_use: float,
    rpm: int,
    Vavail_LL_rms: float,
    gate_pct_scalar: float,
    Nref_turn: float,
    coils_per_phase: int,
    relax_emf: float,
) -> int:
    """
    EMF 제약 기반 nslot 상한 계산.
    relax_emf 가 클수록 '빡빡한' 캡 (실질적으로 Ke_scale*relax_emf 같은 효과).
    """
    if Ke_use <= 0 or rpm <= 0 or coils_per_phase <= 0:
        return 10**9  # 사실상 무제한
    
    rpm_f = float(rpm)  # 계산이 필요하면 여기서만 float으로
    rpm_per_krpm = rpm_f / 1000.0

    numer = (1.0 - gate_pct_scalar) * Vavail_LL_rms
    denom = Ke_use * rpm_per_krpm
    if denom <= 0:
        return 10**9

    # relax_emf 로 EMF 모델을 보수적으로 만들기
    # relax_emf → 1.0 이면 원래 식, >1 이면 더 빡빡한 상한
    nphase_cap = math.floor((numer / denom) * (Nref_turn / max(1e-9, relax_emf)))
    if nphase_cap < 0:
        return 0
    return max(0, nphase_cap // coils_per_phase)

# ======== [DROP-IN] P(kW) → T(N·m) 자동 생성 블록 ========
# 붙여넣기 위치 예: 하이퍼파라미터(YAML/딕셔너리) 로딩 직후, 그리드 확장 전에.

def calculate_reverse_power(
    cfg,
    awg_area_mm2: float,
    parallels: int,
    turns_per_slot_side: int,
    T_oper_C: float = 120.0,
    Vdc: float | None = None,
    m_max: float | None = None,
    Ke_LL_rms_per_krpm: float | None = None,
    L_phase_mH: float | None = None,
    limit_by_voltage: bool = False,
    *,
    # --- 스윕/row에서 주입 권장 ---
    J_A_per_mm2: float | None = None,
    Kt_rms: float | None = None,
    rpm_list_in: list[int] | None = None,
    MLT_mm: float | None = None,
    # --- 필요 시(geometry로 MLT 추정) ---
    slot_pitch_mm: float | None = None,
    stack_mm: float | None = None,
    coil_span_slots: int | None = None,
    alpha_end: float = 1.0,
    C_end_mm: float = 10.0,
):
    """
    스윕 파이프라인과 상호작용하도록 설계된 역산 유틸.
    - R_phase@T, I_rms(요구/제한), Torque, P(kW)를 rpm별로 산출
    - limit_by_voltage=True이면 (Vdc,m_max,Ke, L)로 전압제한 기반 I_limit 적용

    주의(정의):
    - Ke_LL_rms_per_krpm: line-line RMS 기준 [V_LL_rms / krpm]
    - Vavail_LL_rms ≈ Vdc * m_max / sqrt(3)  (SVPWM 등에서 흔히 쓰는 근사)
    - 모든 전압 제한은 "상(phase) RMS" 기준으로 통일해 비교
    """

    # ---------------------------
    # 내부: cfg/전역에서 값 가져오기
    # ---------------------------
    def _pick(key: str, default: Any):
        if cfg is not None:
            # dict 스타일 cfg/hp
            if isinstance(cfg, dict) and key in cfg:
                return cfg[key]
            # 객체 스타일 cfg
            if hasattr(cfg, key):
                return getattr(cfg, key)
        return globals().get(key, default)

    # ---------- 0) 전역/기본값 ----------
    _N_slots = int(_pick("N_slots", 24))
    _m       = int(_pick("m", 3))
    _coils_per_phase = int(_pick("coils_per_phase", (_N_slots // _m) // 2))

    # pole_pairs 추정: cfg에 pole_pairs가 있으면 최우선, 없으면 poles/2, 그것도 없으면 2(=4극) 가정
    pole_pairs = _pick("pole_pairs", None)
    if pole_pairs is None:
        poles = _pick("poles", None)
        if poles is not None:
            pole_pairs = int(poles) // 2
        else:
            pole_pairs = 2
    pole_pairs = int(pole_pairs)

    rho_20 = float(_pick("rho_Cu_20C", 1.724e-8))
    alpha  = float(_pick("alpha_Cu",   0.00393))

    if rpm_list_in is None:
        rpm_list_in = list(map(int, _pick("rpm_list", [600, 1800, 3600])))

    # ---------- 1) MLT 결정 ----------
    if MLT_mm is not None:
        MLT_m = float(MLT_mm) * 1e-3
    else:
        if slot_pitch_mm is not None and stack_mm is not None and coil_span_slots is not None:
            # estimate_mlt_mm는 기존 스크립트에 존재한다는 전제
            MLT_calc_mm = estimate_mlt_mm(
                slot_pitch_mm=float(slot_pitch_mm),
                stack_mm=float(stack_mm),
                coil_span_slots=int(coil_span_slots),
                N_slots=_N_slots,
                D_use=float(_pick("D_use", 0.0)),
                alpha_end=float(alpha_end),
                C_end_mm=float(C_end_mm),
                span_is_inclusive=True,
            )
            MLT_m = float(MLT_calc_mm) * 1e-3
        else:
            MLT_m = float(_pick("MLT_FIXED_M", 0.224))

    # ---------- 2) 온도 저항률 & 상저항 ----------
    rho_T = rho_20 * (1.0 + alpha * (float(T_oper_C) - 20.0))

    # 상 직렬 턴수(더블레이어 관점의 일관식)
    N_turns_phase_series = int(turns_per_slot_side) * _coils_per_phase

    L_phase_m = MLT_m * N_turns_phase_series

    A_total_m2 = (float(awg_area_mm2) * int(parallels)) * 1e-6
    if A_total_m2 <= 0:
        return {"error": "invalid_area"}

    R_phase_ohm = rho_T * L_phase_m / A_total_m2

    # ---------- 3) 전류(요구) / Kt ----------
    if J_A_per_mm2 is None:
        J_A_per_mm2 = float(_pick("J_RATED_DEFAULT", 15.0))

    # 요구 전류(열/전류밀도 기준): "권선군 총 단면적(awg_area*parallels)"에 J 곱
    I_req_rms_A = float(J_A_per_mm2) * float(awg_area_mm2) * int(parallels)

    if Kt_rms is None:
        kt_list = _pick("Kt_rms_list", [0.5])
        Kt_rms = float(kt_list[0])
    Kt_rms = float(Kt_rms)

    # ---------- 4) 전압제한(옵션) 준비 ----------
    L_phase_H = 0.0
    if L_phase_mH is not None:
        L_phase_H = float(L_phase_mH) * 1e-3  # mH -> H

    # 전압제한 계산에 필요한 값이 모이면 True
    voltage_ready = (
        limit_by_voltage
        and (Vdc is not None)
        and (m_max is not None)
        and (Ke_LL_rms_per_krpm is not None)
    )

    # Vavail: LL RMS -> phase RMS
    Vph_avail_rms = None
    if voltage_ready:
        Vavail_LL_rms = float(Vdc) * float(m_max) / math.sqrt(3.0)
        Vph_avail_rms = Vavail_LL_rms / math.sqrt(3.0)  # = Vdc*m_max/3

    # ---------- 5) rpm별 계산 ----------
    out = {
        "T_oper_C": float(T_oper_C),
        "rho_T": float(rho_T),
        "R_phase_ohm": float(R_phase_ohm),
        "L_phase_m": float(L_phase_m),
        "L_phase_mH_used": float(L_phase_H * 1e3),
        "MLT_mm": float(MLT_m * 1e3),
        "N_turns_phase_series": int(N_turns_phase_series),
        "I_req_rms_A": float(I_req_rms_A),
        "J_A_per_mm2": float(J_A_per_mm2),
        "Kt_rms": float(Kt_rms),
        "pole_pairs": int(pole_pairs),
        "limit_by_voltage": bool(voltage_ready),
    }

    if voltage_ready:
        out.update({
            "Vdc": float(Vdc),
            "m_max": float(m_max),
            "Ke_LL_rms_per_krpm": float(Ke_LL_rms_per_krpm),
            "Vph_avail_rms": float(Vph_avail_rms),
        })

    for rpm in rpm_list_in:
        rpm_i = int(rpm)

        # 5-1) 기본은 요청 전류 그대로
        I_used = float(I_req_rms_A)
        I_limit = None
        V_margin = None

        # 5-2) 전압 제한 적용(옵션)
        if voltage_ready and Vph_avail_rms is not None:
            # back-emf: LL rms -> phase rms
            E_LL_rms = float(Ke_LL_rms_per_krpm) * (rpm_i / 1000.0)
            E_ph_rms = E_LL_rms / math.sqrt(3.0)

            # 전기각속도
            omega_mech = 2.0 * math.pi * rpm_i / 60.0
            omega_e    = pole_pairs * omega_mech

            # 전압 여유(phase rms)
            V_margin = float(Vph_avail_rms) - float(E_ph_rms)

            if V_margin <= 0.0:
                I_limit = 0.0
            else:
                # |Z| = sqrt(R^2 + (wL)^2)
                Zmag = math.sqrt((R_phase_ohm ** 2) + ((omega_e * L_phase_H) ** 2))
                if Zmag <= 0.0:
                    I_limit = float("inf")
                else:
                    I_limit = V_margin / Zmag

            I_used = min(I_used, float(I_limit))

            out[f"E_LL_rms_{rpm_i}rpm_V"] = float(E_LL_rms)
            out[f"V_margin_ph_rms_{rpm_i}rpm_V"] = float(V_margin)
            out[f"I_limit_rms_{rpm_i}rpm_A"] = float(I_limit)
            out[f"I_used_rms_{rpm_i}rpm_A"] = float(I_used)
        else:
            out[f"I_used_rms_{rpm_i}rpm_A"] = float(I_used)

        # 5-3) 토크/출력
        T_nm = Kt_rms * I_used
        P_kw = (T_nm * rpm_i) / 9550.0

        out[f"T_nm_{rpm_i}rpm"] = float(T_nm)
        out[f"P_{rpm_i}rpm_kW"] = float(P_kw)

    return out

import numpy as np
import pandas as pd

def process_reverse_power(
    df_batch: pd.DataFrame,
    rpm: float,
    Vdc: float,
    m_max: float,
    Ke_use=None,
    Kt_rms=None,
    T_Nm=None,
    Ld_mH=None,
    Lq_mH=None,
    cfg=None,
    *,
    Ke_scale=None,
    Ke_nom=None,
):
    """
    df_batch 각 row(AWG/Parallels/Turns...)에 대해 reverse power 관련 컬럼을 추가/갱신한다.
    반환: df_batch (copy)
    """

    # ---------- 기본 방탄 ----------
    if df_batch is None or not isinstance(df_batch, pd.DataFrame) or df_batch.empty:
        return df_batch

    df = df_batch.copy()
    # ------------------------------------------------------------------
    # Backward/Forward compatible Ke handling:
    # - legacy: caller passes Ke_use (already scaled)
    # - new:    caller passes Ke_scale (then we compute Ke_use = Ke_nom * Ke_scale)
    # ------------------------------------------------------------------
    # ---------- Ke 처리 (호환) ----------
    try:
        if Ke_use is None:
            if Ke_scale is None:
                raise TypeError("process_reverse_power requires Ke_use or Ke_scale.")

            if Ke_nom is None:
                if cfg is not None and hasattr(cfg, "Ke_LL_rms_per_krpm_nom"):
                    Ke_nom = float(getattr(cfg, "Ke_LL_rms_per_krpm_nom"))
                else:
                    Ke_nom = float(globals().get("Ke_LL_rms_per_krpm_nom", 20.0))

            Ke_use = float(Ke_nom) * float(Ke_scale)
        Ke_use = float(Ke_use)
    except Exception as e:
        # Ke 처리 실패면 전체 NaN 처리
        for c in ["P_rev_kW","I_used_rev_A","I_lim_rev_A","I_req_rev_A","T_rev_Nm"]:
            df[c] = np.nan
        df["reverse_power_err"] = f"Ke handling error: {e}"
        return df

    # ---------- Kt_rms / T_Nm 처리 ----------
    # (전역 스칼라로 들어오면 스칼라 사용, 아니면 df에서 컬럼으로 받기)
    if Kt_rms is None:
        if "Kt_rms" in df.columns:
            # row마다 다를 수 있으므로 벡터로 처리
            Kt_vec = df["Kt_rms"].astype(float).to_numpy()
        else:
            for c in ["P_rev_kW","I_used_rev_A","I_lim_rev_A","I_req_rev_A","T_rev_Nm"]:
                df[c] = np.nan
            df["reverse_power_err"] = "missing Kt_rms (arg None and no df['Kt_rms'])"
            return df
    else:
        Kt_vec = np.full(len(df), float(Kt_rms), dtype=np.float64)

    if T_Nm is None:
        if "T_Nm" in df.columns:
            T_vec = df["T_Nm"].astype(float).to_numpy()
        else:
            for c in ["P_rev_kW","I_used_rev_A","I_lim_rev_A","I_req_rev_A","T_rev_Nm"]:
                df[c] = np.nan
            df["reverse_power_err"] = "missing T_Nm (arg None and no df['T_Nm'])"
            return df
    else:
        T_vec = np.full(len(df), float(T_Nm), dtype=np.float64)

    # ---------- Ld/Lq 사용 정책 ----------
    # 1) 인자로 Ld/Lq가 들어오면 스칼라 max(Ld,Lq) 사용
    # 2) 아니면 df 컬럼에서 row별로 max(Ld,Lq) 사용(없으면 None)
    if (Ld_mH is not None) and (Lq_mH is not None):
        L_use_vec = np.full(len(df), float(max(float(Ld_mH), float(Lq_mH))), dtype=np.float64)
        has_L = True
    else:
        has_L = ("Ld_mH" in df.columns) and ("Lq_mH" in df.columns)
        if has_L:
            L_use_vec = np.maximum(df["Ld_mH"].astype(float).to_numpy(),
                                   df["Lq_mH"].astype(float).to_numpy())
        else:
            L_use_vec = None

    # ---------- 입력 벡터 ----------
    try:
        awg_vec   = df["AWG"].astype(int).to_numpy()
        par_vec   = df["Parallels"].astype(int).to_numpy()
        nslot_vec = df["Turns_per_slot_side"].astype(int).to_numpy()
        J_vec     = df["J_max_A_per_mm2"].astype(float).to_numpy()

        # AWG area: 프로젝트에서 awg_area_mm2()가 있으면 그걸 쓰는 편이 안전
        if "awg_area_mm2" in globals() and callable(globals()["awg_area_mm2"]):
            area_mm2_vec = np.array([float(globals()["awg_area_mm2"](int(a))) for a in awg_vec], dtype=np.float64)
        else:
            # fallback: AWG_TABLE이 있으면 사용
            area_mm2_vec = np.array([float(AWG_TABLE[int(a)]["area"]) for a in awg_vec], dtype=np.float64)

    except Exception as e:
        for c in ["P_rev_kW","I_used_rev_A","I_lim_rev_A","I_req_rev_A","T_rev_Nm"]:
            df[c] = np.nan
        df["reverse_power_err"] = f"input vector build error: {e}"
        return df

    # ---------- 계산 ----------
    I_req_list  = np.full(len(df), np.nan, dtype=np.float64)
    I_lim_list  = np.full(len(df), np.nan, dtype=np.float64)
    I_used_list = np.full(len(df), np.nan, dtype=np.float64)
    T_list      = np.full(len(df), np.nan, dtype=np.float64)
    P_list      = np.full(len(df), np.nan, dtype=np.float64)
    err_list    = np.array([""] * len(df), dtype=object)

    rpm_i = int(round(float(rpm)))

    for i in range(len(df)):
        try:
            L_phase_mH = float(L_use_vec[i]) if (L_use_vec is not None and np.isfinite(L_use_vec[i])) else None

            rev = calculate_reverse_power(
                cfg=cfg,
                awg_area_mm2=float(area_mm2_vec[i]),
                parallels=int(par_vec[i]),
                turns_per_slot_side=int(nslot_vec[i]),
                T_oper_C=120.0,
                Vdc=float(Vdc),
                m_max=float(m_max),
                Ke_LL_rms_per_krpm=float(Ke_use),
                L_phase_mH=L_phase_mH,
                limit_by_voltage=True,
                J_A_per_mm2=float(J_vec[i]),
                Kt_rms=float(Kt_vec[i]),
                rpm_list_in=[rpm_i],
                MLT_mm=float(df["MLT_mm"].iloc[i]) if "MLT_mm" in df.columns else None,
            )

            I_req_list[i]  = float(rev.get("I_req_rms_A", np.nan))
            I_lim_list[i]  = float(rev.get(f"I_limit_rms_{rpm_i}rpm_A", np.nan))
            I_used_list[i] = float(rev.get(f"I_used_rms_{rpm_i}rpm_A", np.nan))
            T_list[i]      = float(rev.get(f"T_nm_{rpm_i}rpm", np.nan))
            P_list[i]      = float(rev.get(f"P_{rpm_i}rpm_kW", np.nan))

        except Exception as e:
            err_list[i] = str(e)

    df["I_req_rev_A"]  = I_req_list
    df["I_lim_rev_A"]  = I_lim_list
    df["I_used_rev_A"] = I_used_list
    df["T_rev_Nm"]     = T_list
    df["P_rev_kW"]     = P_list
    df["I_ratio_used_over_req"] = df["I_used_rev_A"] / np.maximum(1e-9, df["I_req_rev_A"])

    # 에러 메시지 저장(있을 때만)
    df["reverse_power_err"] = np.where(err_list != "", err_list, np.nan)

    return df

def kw_rpm_to_torque_nm(p_kw: float, rpm: int) -> float:
    """P[kW], rpm → T[N·m]. 9550 상수식: T = 9550 * PkW / rpm"""
    if rpm <= 0:
        raise ValueError(f"rpm must be > 0 (got {rpm})")
    return 9550.0 * float(p_kw) / int(rpm)

def infer_nslot_feasible_range_for_rpm(
    rpm: int,
    Ke_scale_list: list[float],
    Ld_mH_list: list[float],
    Lq_mH_list: list[float],
    Vdc_list: list[float],
    m_max_list: list[float],
    stack_mm_list: list[float],
    slot_pitch_mm_scales: list[float],
    end_factor_list: list[float],
    coil_span_slots_list: list[int],
    MLT_scale_list: list[float],
    L_total_min_m: float,
    L_total_max_m: float,
    relax_emf: float = 1.0,
    verbose: bool = True,
    *,
    length_is_total_3ph: bool = True,   # ✅ 추가: L_total이 3상 합인가?
    alpha_end: float = 1.0,             # ✅ 하드코딩 제거(최소한 파라미터화)
    C_end_mm: float = 10.0,
):
    """
    주어진 rpm에서, geometry + EMF + 길이 제약을 모두 고려하여
    가능한 Nslot 범위를 추정한다.

    반환:
      (global_min_nslot, global_max_nslot, count_geom_with_feasible_range)
    """
    global_min = None
    global_max = None
    geom_hit   = 0
    
    # ✅ Lq 비어있으면 Ld로 대체
    if not Lq_mH_list:
        Lq_mH_list = Ld_mH_list

    # ✅ 전역 캐시(가독/안전)
#    Ke_nom = float(globals().get("Ke_LL_rms_per_krpm_nom", Ke_LL_rms_per_krpm_nom))
#    m_ph   = float(globals().get("m", m))
    
    for Ke_scale in Ke_scale_list:
        Ke_use = Ke_LL_rms_per_krpm_nom * Ke_scale
        for Ld_mH in Ld_mH_list:
            for Lq_mH in Lq_mH_list:
                for Vdc in Vdc_list:
                    for m_max in m_max_list:
                        Vavail_LL_rms = m_max * Vdc / math.sqrt(3)
                        gate_pct = max(
                            MARGIN_MIN_PCT,
                            MARGIN_MIN_V / max(1e-9, Vavail_LL_rms)
                        )

                        # EMF cap 이 geometry에 무관한 상한 제공
                        nslot_emf_cap = compute_nslot_emf_cap(
                            Ke_use, rpm, Vavail_LL_rms, gate_pct, Nref_turn,
                            coils_per_phase, relax_emf
                        )
                        if nslot_emf_cap <= 0:
                            continue

                        for stack_mm in stack_mm_list:
                            for slot_pitch_scale in slot_pitch_mm_scales:
                                slot_pitch_mm = slot_pitch_mm_nom * float(slot_pitch_scale)
                                for end_factor in end_factor_list:
                                    for coil_span_slots in coil_span_slots_list:
                                        MLT_base_mm = estimate_mlt_mm(
                                            slot_pitch_mm=slot_pitch_mm,
                                            stack_mm=float(stack_mm),
                                            coil_span_slots=int(coil_span_slots),
                                            N_slots=int(N_slots),
                                            D_use=float(D_use),
                                            alpha_end=float(alpha_end),
                                            C_end_mm=float(C_end_mm),
                                            span_is_inclusive=True,
                                        )

                                        # [가장 안쪽 루프]
                                        for mlt_scale in MLT_scale_list:
                                            MLT_mm = MLT_base_mm * mlt_scale
                                            if MLT_mm <= 0: continue
                                
                                            denom = (m * coils_per_phase * MLT_mm * 1e-3)
                                            if denom <= 0: continue
                                
                                            # 1) 권선 길이 제약 완화 (질문자님 수정 반영)
                                            Nslot_len_min = math.ceil((L_total_min_m * 0.5) / denom)
                                            Nslot_len_max = math.floor((L_total_max_m * 1.5) / denom)
                                            
                                            if Nslot_len_min > Nslot_len_max: continue
                                
                                            # 2) EMF와 길이 제약의 교집합 계산
                                            low = Nslot_len_min
                                            high = min(Nslot_len_max, nslot_emf_cap)
                                            
                                            if low > high: continue
                                
                                            # 3) 전역 범위(global_min/max) 업데이트 (주석 풀고 유지해야 함)
                                            geom_hit += 1
                                            if global_min is None or low < global_min:
                                                global_min = low
                                            if global_max is None or high > global_max:
                                                global_max = high

    # =============================================================
    # ✅ 중요: 모든 루프가 완전히 종료된 후 '이 위치'에서 Fallback 체크
    # =============================================================
    if global_min is None or global_max is None:
        if verbose:
            print("[WARN] No feasible range found after all loops. Falling back to default.")
        global_min = 5   # 프로젝트 최소 권선 수
        global_max = 40  # 프로젝트 최대 권선 수
        geom_hit = 1     # geo=0 방지를 위해 1로 세팅
    # =============================================================

    if verbose:
        print(f"[NSLOT RANGE] rpm={rpm:.0f}, relax_emf={relax_emf}")
        print(f"  feasible geometries : {geom_hit}")
        print(f"  global Nslot range  : {global_min} .. {global_max}")

    return global_min, global_max, geom_hit

# ============================================================
# 6) run_sweep 패치(핵심 부분만)
#    - 기존 run_sweep 시그니처에 RPM_ENV=None 추가
#    - case loop 시작하자마자 envelope 적용 + torch tensor 갱신
# ============================================================

def winding_ssot_3ph_double_layer(N_slots: int, m: int = 3) -> dict:
    """
    3상 더블레이어(정상적인 분포권선 전제)에서
    - slots_per_phase = N_slots / m
    - coils_per_phase = slots_per_phase / 2   (더블레이어 → slot-side가 2개 층)
    를 '절대 기준'으로 고정한다.

    반환 dict:
      slots_per_phase, coils_per_phase, check_ok
    """
    if N_slots % m != 0:
        raise ValueError(f"N_slots({N_slots}) must be divisible by m({m})")

    slots_per_phase = N_slots // m
    if slots_per_phase % 2 != 0:
        # 더블레이어 기준에서 보통 slots_per_phase는 짝수여야 coils_per_phase가 정수
        raise ValueError(
            f"slots_per_phase({slots_per_phase}) must be even for double-layer. "
            f"(N_slots={N_slots}, m={m})"
        )

    coils_per_phase = slots_per_phase // 2
    return {
        "slots_per_phase": int(slots_per_phase),
        "coils_per_phase": int(coils_per_phase),
        "check_ok": True
    }

def assert_and_lock_coils_per_phase(N_slots, m, coils_per_phase_user=None):
    """
    coils_per_phase를 '자동 산정'하고,
    사용자가 전역으로 이미 넣어둔 값과 불일치하면 즉시 오류를 내서
    side 기준이 절대 흔들리지 않게 만든다.
    """
    ss = winding_ssot_3ph_double_layer(N_slots=N_slots, m=m)
    cpp = ss["coils_per_phase"]

    if coils_per_phase_user is not None and int(coils_per_phase_user) != cpp:
        raise RuntimeError(
            f"[COILS_PER_PHASE MISMATCH] user={coils_per_phase_user} but SSOT={cpp}. "
            "Fix this immediately. (Side-basis would break!)"
        )
    return cpp, ss

# =========================
# ★ 여기서 전역 고정 ★
# =========================
# 기존 전역: N_slots, m 사용
coils_per_phase, _SSOT = assert_and_lock_coils_per_phase(
    N_slots=N_slots, m=m, coils_per_phase_user=globals().get("coils_per_phase", None)
)
slots_per_phase = _SSOT["slots_per_phase"]

print(f"[SSOT] N_slots={N_slots}, m={m} => slots_per_phase={slots_per_phase}, coils_per_phase={coils_per_phase} (double-layer, side-basis LOCKED)")

# ============================================================
# ============================================================
# 0) Envelope 데이터 구조
# ============================================================

@dataclass
class RpmEnvelope:
    rpm_key: int                # 그룹 키(정수 rpm)
    rpm_repr: float             # 표시/대표 rpm (median/mean 등)
    rpm_samples: List[float]    # 이 그룹에 들어온 실제 rpm들(디버그/추적용)
    # 추천 탐색 범위
    awg_list: List[int]
    par_lo: int
    par_hi: int
    nslot_lo: int
    nslot_hi: int
    # 디버깅/설명용
    awg_rep: int
    par_need_rep: int
    nslot_hi_est: int
    I_worst: float

def _ceil_int(x: float) -> int:
    return int(math.ceil(x - 1e-12))

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

# ============================================================
# 1) case 단위 envelope 계산 (rpm/PkW 기준)
#    - 질문에서 주신 prescan_case_envelope(case)와 결합하기 쉽게 구성
# ============================================================
def prescan_case_envelope(case) -> dict:
    """
    (사용자께서 올려주신 버전과 동일 컨셉)
    - PASS 씨앗 목적: 느슨한(유리한) 조건으로 envelope를 잡음
    """
    rpm = float(case["rpm"])
    raw = case.get("P_kW", 0.0)
    PkW = float(raw) if raw is not None else 0.0
    T   = 9550.0 * PkW / rpm if rpm > 1e-9 else 0.0

    # 탐색 초기: PASS 씨앗이면 "유리하게" 잡아야 함
    # - 전류를 크게 만드는 Kt_min을 쓰면 더 빡세짐(보수적)
    # - 반대로 PASS 씨앗 목적이면 Kt를 상단(유리)로 잡을 수도 있음
    Kt_min = min(Kt_rms_list)
    J_max  = max(J_max_list)       # J는 상단이 유리
    I_worst = T / max(1e-9, Kt_min)

    # awg별 필요 par(J 조건)
    par_need_by_awg = []
    for awg in awg_candidates:
        area_mm2 = awg_area_mm2(int(awg))
        par_need = _ceil_int(I_worst / (J_max * area_mm2))
        par_need_by_awg.append((int(awg), float(area_mm2), int(par_need)))

    # "par_need 최소" 주는 awg를 대표로(대개 굵은 선)
    awg_best, area_best, par_need_best = min(par_need_by_awg, key=lambda x: x[2])

    # 전압 envelope(EMF cap)을 느슨하게 잡기 위해:
    Vavail = max(m_max_list) * max(Vdc_list) / math.sqrt(3)
    gate_pct = max(MARGIN_MIN_PCT, (MARGIN_MIN_V / Vavail) if (MARGIN_MIN_V is not None) else 0.0)

    Ke_use = Ke_LL_rms_per_krpm_nom * min(Ke_scale_list)  # Ke_scale 작은 쪽이 전압은 유리(턴 상한 높아짐)
    if Ke_use <= 0 or rpm <= 0:
        nslot_emf_cap = 10**9
    else:
        nphase_cap = math.floor(((1.0 - gate_pct) * Vavail / (Ke_use * (rpm/1000.0))) * (Nref_turn / SAFETY_RELAX))
        nslot_emf_cap = max(0, nphase_cap // max(1, coils_per_phase))

    # fill 관점 nslot 상한(대표 slot_area/fill_limit로 근사)
    slot_area = max(slot_area_mm2_list)
    fill_lim  = max(slot_fill_limit_list)

    # ---- guard: avoid div-by-zero in prescan ----
    try:
        area_best = float(area_best)
    except Exception:
        area_best = 0.0
    if area_best <= 0.0:
        # AWG area lookup failed -> can't compute fill cap; treat as "no cap"
        nslot_fill_cap_est = 10**9
    else:
        try:
            par_need_best = int(par_need_best)
        except Exception:
            par_need_best = 1
        par_need_best = max(1, par_need_best)  # clamp (I_rms=0 etc.)
        nslot_fill_cap_est = math.floor((fill_lim * slot_area) / (2.0 * par_need_best * area_best))

    nslot_hi_est = min(NSLOT_USER_RANGE[1], int(nslot_emf_cap), int(nslot_fill_cap_est))

    return {
        "I_worst": float(I_worst),
        "awg_rep": int(awg_best),
        "par_need_rep": int(par_need_best),
        "nslot_hi_est": int(nslot_hi_est),
    }

# ============================================================
# 2) rpm-adaptive envelope builder
#    - 케이스들을 rpm별로 묶고, 각 rpm에서 살아남는 범위를 만든다
# ============================================================

def build_rpm_adaptive_envelopes(
    cases: List[dict],
    *,
    # PAR 쪽 안전 여유
    safety_extra_par: int = 2,
    par_hard_max: Optional[int] = None,     # PAR_HARD_MAX를 넘지 않게
    # AWG 쪽 범위 확장 폭
    awg_span_down: int = 1,                 # 대표 awg보다 더 굵게(번호↓) 몇 단계 포함
    awg_span_up: int = 1,                   # 대표 awg보다 더 얇게(번호↑) 몇 단계 포함
    # NSLOT 쪽 안전 여유
    nslot_expand_lo: int = 0,
    nslot_expand_hi: int = 0,
    # turn_candidates_base / user range 반영
    use_turn_candidates_base: bool = True,
    verbose: bool = True,
) -> Dict[int, RpmEnvelope]:
    """
    반환: rpm_key(int, 그룹핑 키) -> RpmEnvelope

    rpm 개념 정리
    - rpm_f    : case에 들어있는 실수 rpm (측정/계산 결과일 수 있음)
    - rpm_key  : rpm_f를 정수로 라운드한 그룹 키 (예: 1799.6 -> 1800)
    - Envelope : rpm_key 그룹에 대해 AWG/PAR/NSLOT 범위를 누적(UNION 또는 INTERSECTION 정책)
    """

    # awg 후보를 정렬해 인덱싱 가능하게
    awg_sorted = sorted(set(int(a) for a in awg_candidates))
    awg_to_idx = {a:i for i,a in enumerate(awg_sorted)}

    # turn_candidates_base 범위
    tbase_lo = int(turn_candidates_base[0]) if (use_turn_candidates_base and turn_candidates_base) else NSLOT_USER_RANGE[0]
    tbase_hi = int(turn_candidates_base[-1]) if (use_turn_candidates_base and turn_candidates_base) else NSLOT_USER_RANGE[1]
    
    # rpm_key(int) -> 누적 dict
    env_by_rpm: Dict[int, dict] = {}

    for case in cases:
        rpm_f: float = float(case["rpm"])
        rpm_key: int = int(round(rpm_f))
        info = prescan_case_envelope(case)

        awg_rep = int(info["awg_rep"])
        par_need = int(info["par_need_rep"])
        nslot_hi_est = int(info["nslot_hi_est"])
        I_worst = float(info["I_worst"])

        # ---- AWG 리스트 생성(대표 awg 중심으로 ±span) ----
        if awg_rep in awg_to_idx:
            i0 = awg_to_idx[awg_rep]
            i_lo = _clamp_int(i0 - awg_span_down, 0, len(awg_sorted)-1)
            i_hi = _clamp_int(i0 + awg_span_up,   0, len(awg_sorted)-1)
            awg_list = awg_sorted[i_lo:i_hi+1]
        else:
            # 혹시 테이블/후보 불일치 시: 전체 후보
            awg_list = awg_sorted[:]

        # ---- PAR 범위: 최소 par_need부터 + safety_extra_par 까지 ----
        par_lo = max(1, par_need)
        par_hi = par_need + int(safety_extra_par)
        if par_hard_max is not None:
            par_hi = min(par_hi, int(par_hard_max))
        if par_hi < par_lo:
            par_hi = par_lo

        # ---- NSLOT 범위: user range + base + emf/fill 근사 상한 ----
        user_lo, user_hi = NSLOT_USER_RANGE
        nslot_lo = max(user_lo, tbase_lo) - int(nslot_expand_lo)
        nslot_hi = min(user_hi, tbase_hi, nslot_hi_est) + int(nslot_expand_hi)
        nslot_lo = max(user_lo, nslot_lo)
        nslot_hi = min(user_hi, nslot_hi)
        if nslot_hi < nslot_lo:
            # 이 rpm/case에서는 근사상 불가능 → 아주 보수적으로 user range만 남기거나 빈 envelope 처리
            nslot_lo, nslot_hi = user_lo, min(user_hi, user_lo)

        # rpm별로 "가장 보수적/가장 공격적"을 어떻게 합칠지 정책 필요:
        # - PASS 씨앗 찾기 목적이면: (par_hi는 max로, nslot_hi는 max로) => UNION 성격
        # - 안전하게 줄이려면: INTERSECTION 성격
        # 여기서는 PASS 씨앗 목적에 맞춰 "UNION"으로 축적합니다.
        e = env_by_rpm.get(rpm_key)
        if e is None:
            env_by_rpm[rpm_key] = dict(
                rpm_key=rpm_key,
                rpm_samples=[rpm_f],
                awg_set=set(awg_list),
                par_lo=par_lo,
                par_hi=par_hi,
                nslot_lo=nslot_lo,
                nslot_hi=nslot_hi,
                awg_rep=awg_rep,
                par_need_rep=par_need,
                nslot_hi_est=nslot_hi_est,
                I_worst=I_worst,
            )
        else:
            e["rpm_samples"].append(rpm_f)
            e["awg_set"].update(awg_list)
            # PASS 씨앗 목적: UNION 누적
            # - par_lo: 더 작은 값도 허용(범위 확장)
            # - par_hi: 더 큰 값도 허용(범위 확장)
            # - nslot_lo: 더 작은 값도 허용하되 아래에서 user_lo로 clamp됨
            # - nslot_hi: 더 큰 값도 허용
            e["par_lo"]   = min(e["par_lo"], par_lo)
            e["par_hi"]   = max(e["par_hi"], par_hi)
            e["nslot_lo"] = min(e["nslot_lo"], nslot_lo)
            e["nslot_hi"] = max(e["nslot_hi"], nslot_hi)
            
            # 디버그용 worst 추적(큰 전류가 worst)
            if I_worst > e["I_worst"]:
                e["I_worst"] = I_worst
                e["awg_rep"] = awg_rep
                e["par_need_rep"] = par_need
                e["nslot_hi_est"] = nslot_hi_est

    out: Dict[int, RpmEnvelope] = {}
    for rpm_key, e in sorted(env_by_rpm.items(), key=lambda kv: kv[0]):
        awg_list = sorted(int(x) for x in e["awg_set"])
        # 대표 rpm 샘플(필요하면 평균/중앙값으로)
        rpm_repr = float(np.median(e["rpm_samples"])) if len(e["rpm_samples"]) else float(rpm_key)

        out[rpm_key] = RpmEnvelope(
            rpm_key=int(rpm_key),
            rpm_repr=float(rpm_repr),
            rpm_samples=list(e["rpm_samples"]),
            awg_list=awg_list,
            par_lo=int(e["par_lo"]),
            par_hi=int(e["par_hi"]),
            nslot_lo=int(e["nslot_lo"]),
            nslot_hi=int(e["nslot_hi"]),
            awg_rep=int(e["awg_rep"]),
            par_need_rep=int(e["par_need_rep"]),
            nslot_hi_est=int(e["nslot_hi_est"]),
            I_worst=float(e["I_worst"]),
        )

    if verbose:
        print("\n[RPM-ENV] ------------------------")
        for rpm_key, env in out.items():
            print(
                f"  rpm_key={rpm_key:d} (repr={env.rpm_repr:.2f}, n={len(env.rpm_samples)})"
                f" | AWG={env.awg_list}"
                f" | PAR=[{env.par_lo}..{env.par_hi}]"
                f" | NSLOT=[{env.nslot_lo}..{env.nslot_hi}]"
                f"  (rep_awg={env.awg_rep}, par_need={env.par_need_rep},"
                f" nslot_hi_est={env.nslot_hi_est}, Iworst={env.I_worst:.2f}A)"
            )
        print("---------------------------------\n")

    return out

# ============================================================
# 3) run_sweep()에 "케이스마다 envelope 적용"하는 훅
#    - 전역 awg_candidates/par_candidates/turn_candidates_base를 바꾸지 않고
#      case 루프 안에서 로컬 후보만 바꿔 쓰는 방식(안전)
# ============================================================

def apply_envelope_for_case(case: dict, RPM_ENV: Dict[int, RpmEnvelope]):
    """
    반환:
      awg_list_case, par_candidates_case, nslot_user_range_case
    """
    rpm_k = int(round(float(case.get("rpm", 0))))
    env = RPM_ENV.get(rpm_k)

    # ---- 1) 기본(전역) 후보 ----
    awg_base = list(awg_candidates)
    par_base = list(par_candidates)
#    nslot_base = tuple(NSLOT_USER_RANGE)

    # ---- 2) env가 있으면 env 우선 ----
    if env is not None:
        # fallback: 기존 전역
        # ============================================================
        # [TOP5-3] Tuple(...) 제거 (typing.Tuple은 런타임 변환이 아님)
        # ============================================================
        awg_list_case = list(env.awg_list)
        par_candidates_case = list(range(int(env.par_lo), int(env.par_hi) + 1))
        nslot_user_range_case = (int(env.nslot_lo), int(env.nslot_hi))

        # 방탄: 전역 후보와 교집합(없는 AWG/PAR 방지)
        awg_list_case = [a for a in awg_list_case if a in awg_base]
        par_candidates_case = [p for p in par_candidates_case if p in par_base]

        # 방탄: 비면 fallback
        if not awg_list_case:
            awg_list_case = awg_base
        if not par_candidates_case:
            par_candidates_case = par_base

        return awg_list_case, par_candidates_case, nslot_user_range_case

    # ---- 3) env가 없으면 “강한 룰 기반 컷” 적용 ----
    if rpm_k <= 800:
        awg_list_case = [a for a in awg_base if a in (16, 17, 18)]
        par_candidates_case = [p for p in par_base if 2 <= p <= 18]
        nslot_user_range_case = (12, 60)
    elif rpm_k <= 2200:
        awg_list_case = [a for a in awg_base if a in (16, 17, 18)]
        par_candidates_case = [p for p in par_base if 2 <= p <= 14]
        nslot_user_range_case = (10, 45)
    else:
        awg_list_case = [a for a in awg_base if a in (16, 17)]
        par_candidates_case = [p for p in par_base if 2 <= p <= 10]
        nslot_user_range_case = (8, 30)

    # 방탄: 비면 전역으로
    if not awg_list_case:
        awg_list_case = awg_base
    if not par_candidates_case:
        par_candidates_case = par_base

    return awg_list_case, par_candidates_case, nslot_user_range_case

def rebuild_awg_par_tensors(awg_list_case: List[int], par_candidates_case: List[int]):
    """
    ============================================================
    # [TOP5-2] 텐서 갱신을 단일 체계로 정리
    #   - run_sweep에서 실제 사용하는 이름(awg_vec, awg_area, par_vec)만 갱신
    #   - AWG_AREA_T/PAR_T 같은 "또 다른 전역"은 만들지 않음(혼선/버그 방지)
    ============================================================
    """
    global awg_vec, awg_area, par_vec
    awg_vec  = T(list(awg_list_case), config=cfg)
    awg_area = T([awg_area_mm2(int(a)) for a in awg_list_case], config=cfg)
    par_vec  = T(list(par_candidates_case), config=cfg)
    return awg_vec, awg_area, par_vec


def compute_lengths_side_basis(
    turns_per_slot_side,   # scalar or tensor
    MLT_mm,                # scalar or tensor
    m: int,
    coils_per_phase: int,
):
    """
    'side 기준' 길이 산정의 단일 진실.
      L_phase = coils_per_phase * turns_per_slot_side * MLT
      L_total(3상) = m * L_phase

    단위:
      MLT_mm [mm] -> [m]
    """
    MLT_m = MLT_mm * 1e-3
    L_phase_m = (coils_per_phase * turns_per_slot_side) * MLT_m
    L_total_m = (m * L_phase_m)
    return L_phase_m, L_total_m

def worstcase_margin_ok_fast(row, wc: dict, target_pct: float = 0.05) -> bool:
    """
    단순 스케일 기반 worst-case 마진 빠른 체크 (대략 필터용).
    wc 예: {"Vdc":325.0, "m_max":0.925, "Ke_scale":1.05, "R_scale":1.40, "L_scale":0.85}
    """
    try:
        Vavail_wc = wc["m_max"] * wc["Vdc"] / math.sqrt(3)
        # ✅ row 타입 방탄 + 별칭 허용(V_LL_req_V)
        Vreq_nom = _row_get(row, "Vreq_LL_rms", None)
        if Vreq_nom is None:
            Vreq_nom = _row_get(row, "V_LL_req_V", None)
        Vreq_nom = float(Vreq_nom)
        
        Vreq_wc   = Vreq_nom \
                    * float(wc.get("Ke_scale", 1.0)) \
                    * (float(wc.get("L_scale", 1.0)) ** 0.5)

        Vmargin_wc_pct = (Vavail_wc - Vreq_wc) / max(1e-9, Vavail_wc)
        return bool(Vmargin_wc_pct >= target_pct)
    except Exception:
        return False

# --- (옵션) 정확한 Worst-case 마진 체크(간이 스케일 X, β-스윕 재계산 O) ---
def worstcase_margin_ok(row: pd.Series, wc: dict, target_pct: float = 0.05) -> bool:
    """
    wc 예시: {"Vdc":325.0, "m_max":0.925, "Ke_scale":1.05, "R_scale":1.40, "L_scale":0.85}
    - 기존 row의 rpm, Nslot, Par, AWG, MLT 등은 그대로 사용
    - Ke/L/R/Vavail은 wc 보정하여 β-스윕으로 Vreq 재계산
    """
    try:
        # ✅ row 타입 방탄(itertuples/dict/Series 모두)
        rpm      = float(_row_get(row, "rpm"))
        Vavail   = wc["m_max"] * wc["Vdc"] / math.sqrt(3)

        Ke_scale = float(_row_get(row, "Ke_scale")) * wc.get("Ke_scale", 1.0)
        Ld_mH    = float(_row_get(row, "Ld_mH"))   * wc.get("L_scale", 1.0)
        Lq_mH    = float(_row_get(row, "Lq_mH", _row_get(row, "Ld_mH"))) * wc.get("L_scale", 1.0)
        Kt_rms   = float(_row_get(row, "Kt_rms"))
        T_Nm     = float(_row_get(row, "T_Nm"))
        
        I_rms    = T_Nm / Kt_rms

        Nslot    = int(_row_get(row, "Turns_per_slot_side"))
        Par      = int(_row_get(row, "Parallels"))
        # ✅ awg_area_mm2로 통일 (AWG_TABLE dict 혼용 제거)
        A_wire   = float(_row_get(row, "A_wire_mm2", np.nan))
        if np.isnan(A_wire):
            A_wire = float(awg_area_mm2(int(_row_get(row, "AWG"))))

        MLT_mm   = float(_row_get(row, "MLT_mm"))
        coil_per_phase = int(globals().get("coils_per_phase", 4))
        poles    = int(globals().get("p", 2) * 2)  # electrical pole count (for f_e)
        f_e      = (poles/2) * rpm / 60.0         # electrical frequency for p pole pairs
        omega_e  = 2*math.pi*f_e

        # 저항은 온도 상승/선경/병렬 영향 + wc 스케일
        rho20=1.724e-8; alpha=0.00393
        T_oper_C = 120.0
        rho_T = rho20 * (1 + alpha * (T_oper_C - 20.0)) * wc.get("R_scale", 1.0)

        # 역기전력 상수
        Ke_nom = float(globals().get("Ke_LL_rms_per_krpm_nom", 20.0))
        Nref   = float(globals().get("Nref_turn", 20))
        # 상 저항
        L_phase = coil_per_phase * Nslot * (MLT_mm * 1e-3)
        A_tot   = max(1e-9, A_wire * Par) * 1e-6
        Rph     = rho_T * L_phase / A_tot

        # ψf from E_LL_rms (LL,rms): E = (√3/√2) ωe ψf
        Nphase  = Nslot * coil_per_phase
        E_LL    = Ke_nom * Ke_scale * (rpm/1000.0) * (Nphase/Nref)
        psi_f   = E_LL / ((math.sqrt(3)/math.sqrt(2)) * omega_e + 1e-12)

        # ✅ β 스윕: 전역 SIN_FINE/COS_FINE(torch) 재사용 + torch scalar 캐스팅
        SINb, COSb = _get_sin_cos_fine(121)
        SINb = SINb.view(-1, 1); COSb = COSb.view(-1, 1)
#        DTYPE  = globals().get("DTYPE", torch.float32)
#        DEVICE = globals().get("DEVICE", torch.device("cpu"))
        Rph_t   = T(Rph, config=cfg)
        psi_t   = T(psi_f, config=cfg)
        omega_t = T(omega_e, config=cfg)
        Ipk_t   = T(math.sqrt(2.0) * I_rms, config=cfg)
        Ld_t    = T(float(Ld_mH)*1e-3, config=cfg)
        Lq_t    = T(float(Lq_mH)*1e-3, config=cfg)
        K_V_t   = T((math.sqrt(3)/math.sqrt(2)), config=cfg)
 
        Id_t = (-Ipk_t) * SINb
        Iq_t = ( Ipk_t) * COSb
        v_d  = (Rph_t * Id_t) - (omega_t * Lq_t) * Iq_t
        v_q  = (Rph_t * Iq_t) + (omega_t * Ld_t) * Id_t + (omega_t * psi_t)
        Vll  = K_V_t * torch.sqrt(v_d*v_d + v_q*v_q)
        vmin = float(torch.min(Vll).item())

        Vmargin_pct = (Vavail - vmin) / max(1e-9, Vavail)
        return bool(Vmargin_pct >= target_pct)
    except Exception:
        return False


def compute_required_par_bounds(
    *,
    cases_local: List[Dict[str, Any]],
    awg_list: List[int],
    jmax_list: List[float],
    kt_list: List[float],
    par_hard_min: int,
    par_hard_max: int,
    safety_extra: int = 0,
    # ✅ rpm-only 케이스용: autotune에서만 쓰는 "가정 부하"
    seed_power_kW: Optional[float] = None,
    # ✅ seed_power_kW도 없을 때 마지막 fallback
    seed_power_kW_fallback: float = 1.0,
    # ✅ 매우 낮은 rpm에서 토크가 과도하게 커지지 않게 하한
    rpm_min_for_torque: float = 1.0,
):
    """
    PAR 후보 범위 자동 튜닝용:
    - 최악조건 기준으로 필요한 par 상한(par_reco_max)을 계산한다.

    케이스 부하 해석 우선순위:
      1) case["T_Nm"] (not None)
      2) case["P_kW"] (not None) -> T = 9550*P/rpm
      3) 둘 다 None (rpm-only) -> seed_power_kW 가정으로 T 생성

    반환 dict:
      - worst_case: {'rpm', 'P_kW', 'T_Nm', 'src'}
      - par_need_worst / par_reco_max / hard_max_ok ...
    """
    # ----------------------------
    # 입력 정리/검증
    # ----------------------------
    awg_list = [int(a) for a in awg_list]
    jmax_list = [float(j) for j in jmax_list]
    kt_list = [float(k) for k in kt_list]

    if par_hard_min <= 0 or par_hard_max <= 0:
        raise ValueError("par_hard_min/max must be positive.")
    if par_hard_min > par_hard_max:
        raise ValueError("par_hard_min must be <= par_hard_max.")

    # worst는 보통: 얇은선(A_min), 낮은 J, 낮은 Kt, 높은 T
    awg_areas = [(a, awg_area_mm2(a)) for a in awg_list]
    awg_areas = [(a, A) for a, A in awg_areas if np.isfinite(A) and A > 0]
    if not awg_areas:
        raise ValueError("AWG_TABLE lookup failed for given awg_candidates.")

    awg_min, A_min = min(awg_areas, key=lambda x: x[1])
    awg_max, A_max = max(awg_areas, key=lambda x: x[1])

    J_min = min(jmax_list) if jmax_list else None
    Kt_min = min([k for k in kt_list if k > 0], default=None)

    if J_min is None or J_min <= 0:
        raise ValueError("J_max_list must contain positive values.")
    if Kt_min is None or Kt_min <= 0:
        raise ValueError("Kt_rms_list must contain positive values.")

    # seed power 결정
    if seed_power_kW is None:
        # 전역에 설정해두면 자동 사용
        seed_power_kW = globals().get("P_SEED_KW_FOR_AUTOTUNE", None)
    if seed_power_kW is None:
        seed_power_kW = float(seed_power_kW_fallback)
    seed_power_kW = float(seed_power_kW)

    # ----------------------------
    # 케이스에서 "유효 토크" 계산
    # ----------------------------
    def _case_to_torque(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # rpm
        try:
            rpm = float(case.get("rpm", 0.0))
        except Exception:
            return None
        if not np.isfinite(rpm) or rpm <= 0:
            return None

        # 1) T_Nm 직접
        T_raw = case.get("T_Nm", None)
        if T_raw is not None:
            try:
                T = float(T_raw)
                if np.isfinite(T) and T > 0:
                    return {"rpm": rpm, "P_kW": case.get("P_kW", None), "T_Nm": T, "src": "T_Nm"}
            except Exception:
                pass

        # 2) P_kW로 환산
        P_raw = case.get("P_kW", None)
        if P_raw is not None:
            try:
                PkW = float(P_raw)
                if np.isfinite(PkW) and PkW > 0:
                    rpm_eff = max(float(rpm_min_for_torque), rpm)
                    T = kw_rpm_to_torque_nm(PkW, rpm_eff)
                    if np.isfinite(T) and T > 0:
                        return {"rpm": rpm, "P_kW": PkW, "T_Nm": float(T), "src": "P_kW"}
            except Exception:
                pass

        # 3) rpm-only -> seed power 가정
        rpm_eff = max(float(rpm_min_for_torque), rpm)
        PkW = float(seed_power_kW)
        T = kw_rpm_to_torque_nm(PkW, rpm_eff)
        if np.isfinite(T) and T > 0:
            return {"rpm": rpm, "P_kW": PkW, "T_Nm": float(T), "src": "seed_power"}

        return None

    # ----------------------------
    # 최악 케이스 선택
    # ----------------------------
    T_max = -1.0
    worst_case = None

    for c in cases_local:
        wc = _case_to_torque(c)
        if wc is None:
            continue
        if wc["T_Nm"] > T_max:
            T_max = wc["T_Nm"]
            worst_case = wc

    if worst_case is None or not np.isfinite(T_max) or T_max <= 0:
        raise ValueError("No valid case torque could be inferred from cases_local.")

    # ----------------------------
    # 필요한 par 계산
    # ----------------------------
    I_worst = float(T_max) / float(Kt_min)   # Irms worst
    if not np.isfinite(I_worst) or I_worst <= 0:
        raise ValueError("Computed I_worst is invalid.")

    # 최악(A_min, J_min)에서 필요한 par
    par_need_worst = int(math.ceil(I_worst / (float(J_min) * float(A_min))))
    par_need_worst = max(int(par_hard_min), par_need_worst)

    # 권장 상한 = par_need_worst + safety_extra
    par_reco_max = int(par_need_worst) + int(safety_extra)

    hard_max_ok = (int(par_hard_max) >= int(par_reco_max))

    info = dict(
        worst_case=worst_case,
        awg_min=awg_min, A_min=float(A_min),
        awg_max=awg_max, A_max=float(A_max),
        J_min=float(J_min), Kt_min=float(Kt_min),
        T_max=float(T_max), I_worst=float(I_worst),
        par_need_worst=int(par_need_worst),
        par_reco_max=int(par_reco_max),
        hard_max_ok=bool(hard_max_ok),
        par_hard_min=int(par_hard_min),
        par_hard_max=int(par_hard_max),
        seed_power_kW=float(seed_power_kW),
        rpm_min_for_torque=float(rpm_min_for_torque),
    )
    return info

def _is_placeholder_df(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and (list(df.columns) == ["Note"])

def rerun_seed_if_no_feasible(df_pass1: pd.DataFrame) -> bool:
    if not isinstance(df_pass1, pd.DataFrame):
        return True
    if _is_placeholder_df(df_pass1):
        return True
    if df_pass1.empty:
        return True
    return False

# physics.py
# ===================================================
# Ld/Lq cache (FEMM feedback 연동의 핵심)
#   - key: (AWG, Parallels, Turns_per_slot_side)
#   - value: {"Ld_mH": float, "Lq_mH": float}
# ===================================================
_LDLQ_CACHE: dict[tuple[int, int, int], dict[str, float]] = {}

def register_ldlq_from_femm(
    key: tuple[int, int, int],
    Ld_mH: float,
    Lq_mH: float,
    *,
    overwrite: bool = True
) -> None:
    """Register FEMM-extracted Ld/Lq into the global cache."""
    k = (int(key[0]), int(key[1]), int(key[2]))
    if (not overwrite) and (k in _LDLQ_CACHE):
        return
    _LDLQ_CACHE[k] = {"Ld_mH": float(Ld_mH), "Lq_mH": float(Lq_mH)}

def register_ldlq_from_femm_db(ldlq_db: dict, *, overwrite: bool = True) -> int:
    """
    Bulk-register a dict like:
      {(awg,par,nslot): {"Ld_mH":..,"Lq_mH":..}, ...}
    Returns number registered.
    """
    n = 0
    if not isinstance(ldlq_db, dict):
        return 0
    for k, v in ldlq_db.items():
        if not (isinstance(k, tuple) and len(k) == 3 and isinstance(v, dict)):
            continue
        if ("Ld_mH" not in v) or ("Lq_mH" not in v):
            continue
        register_ldlq_from_femm(k, v["Ld_mH"], v["Lq_mH"], overwrite=overwrite)
        n += 1
    return n

# physics.py
def get_ld_lq(
    key: tuple[int, int, int],
    *,
    fallback_Ld_mH=None,
    fallback_Lq_mH=None
):
    """Fetch cached (Ld_mH,Lq_mH). If missing, return fallbacks."""
    k = (int(key[0]), int(key[1]), int(key[2]))
    v = _LDLQ_CACHE.get(k)
    if isinstance(v, dict):
        return v.get("Ld_mH", fallback_Ld_mH), v.get("Lq_mH", fallback_Lq_mH)
    return fallback_Ld_mH, fallback_Lq_mH

def apply_ldlq_feedback(df: pd.DataFrame, ldlq_db: dict):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    def _key(r: pd.Series):
        try:
            return (
                int(_row_get(r, "AWG")),
                int(_row_get(r, "Parallels")),
                int(_row_get(r, "Turns_per_slot_side")),
            )
        except Exception:
            return None

    def _ld(r: pd.Series):
        k = _key(r)
        v = ldlq_db.get(k) if k is not None else None
        if isinstance(v, dict) and "Ld_mH" in v:
            return float(v["Ld_mH"])
        return _row_get(r, "Ld_mH", np.nan)

    def _lq(r: pd.Series):
        k = _key(r)
        v = ldlq_db.get(k) if k is not None else None
        if isinstance(v, dict) and "Lq_mH" in v:
            return float(v["Lq_mH"])
        return _row_get(r, "Lq_mH", np.nan)

    out = df.copy()
    out["Ld_mH"] = out.apply(_ld, axis=1)
    out["Lq_mH"] = out.apply(_lq, axis=1)
    return out
