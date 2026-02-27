# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 09:46:17 2026

@author: USER, SANG JIN PARK
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import time
import os
import sys
import math
import pandas as pd
import numpy as np
import logging
import copy  # [ADD] auto_adjust_by_pass 에서 사용
from scipy.stats import norm

from bisect import bisect_left
from itertools import product
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from utils.utils import T, _row_get, _row_get_first, init_planned_combos_denominator, _ensure_iterable, topk_pretrim_df
from utils.utils import save_rank_and_fixes_workbook, cuda_sync, gpu_mem_info_gb, print_gpu_banner

from utils.femm_builder import build_fem_from_winding, build_femm_for_top_designs
from core.physics import apply_envelope_for_case, rebuild_awg_par_tensors, worstcase_margin_ok_fast, estimate_mlt_mm, calculate_reverse_power
from core.physics import kw_rpm_to_torque_nm, infer_nslot_feasible_range_for_rpm, compute_required_par_bounds, debug_emf_cap
from core.physics import compute_lengths_side_basis, resistivity_at_T, awg_area_mm2, process_reverse_power, Br_T, Hcj_T
from core.physics import build_rpm_adaptive_envelopes, worstcase_margin_ok, get_ld_lq, _get_sin_cos_fine
from core.search.narrowing import compute_dynamic_tile_size, estimate_winding_temp, failure_probability,save_pass_patterns
from core.search.narrowing import (
    compute_esc_optimal_window,
    apply_window_to_globals,
    compute_bayesian_window
    )

# --- ensure progress dict identity (engine <-> progress.py) ---
from core.progress import (
    progress_add_tiles, progress_tick_tile,
    progress_update, progress_newline,
    GEO, PROF, PROG2
    )                      # progress.py

from configs.config import (
    AWG_TABLE,
    D_use,
    ENABLE_PROFILING,
    ITYPE,
    Ke_LL_rms_per_krpm_nom,
    LIMIT_MIN,
    L_total_max_m,
    L_total_min_m,
    MARGIN_MIN_PCT,
    MARGIN_MIN_V,
    MARGIN_TARGET,
    N_slots,
    Nref_turn,
    PAIR_TILE,
    PAR_HARD_MAX,
    PAR_HARD_MIN,
    POWER_MODE,
    RANK_GROUP_COLS,
    RANK_GROUP_TOPK,
    RANK_PRESET,
    RANK_TOPK,
    SAFETY_RELAX,
    SAL_RATIO_LIST,
    TILE_NSLOTS,
    TOPK,
    T_max,
    USE_PRECOUNT_PLANNED,
    _RANK_PRESETS,
    coils_per_phase,
    g_eff,
    kw1,
    m,
    p,
    mu0,
    mu_r_mag,
    slot_pitch_mm_nom,
    t_m,
    rpm_list,
    Ld_mH_list,
    Vdc_list,
    m_max_list,
    Ke_scale_list,
    Kt_rms_list,
    J_max_list,
    awg_candidates,
    stack_mm_list,
    end_factor_list,
    slot_pitch_mm_list,
    coil_span_slots_list,
    MLT_scale_list,
    P_kW_list,
    T_Nm_list,
    NSLOT_USER_RANGE,
    turn_candidates_base,
    slot_area_mm2_list,
    slot_fill_limit_list,
    Lq_mH_list,
    RPM_QUOTA,
    LIMIT_ROWS,
    slots_per_phase,
    par_candidates,
)

import configs.config as C

if TYPE_CHECKING:
    from core.winding_spec import WindingConnSpec

# =============================================================================
L_total_max_m = float(getattr(C, "L_total_max_m", 130.0))
L_total_min_m = float(getattr(C, "L_total_min_m", 50.0))
DEVICE = getattr(C, "DEVICE", None) or __import__("torch").device("cpu")
DTYPE = getattr(C, "DTYPE", None) or __import__("torch").float32
ITYPE = getattr(C, "ITYPE", None) or __import__("torch").int32

# core/engine.py 상단
from configs.config  import * # 혹은 from config import *

# 전역 상태 변수들 초기화 (NameError 방지)
awg_vec = None
par_vec = None
awg_area = None
awg_candidates = []
par_candidates = []
results = []

# config에 정의된 기본값들을 engine의 모듈 레벨 전역 변수로 초기화
# config. 을 빼고 변수명만 사용
turn_candidates_base = globals().get('turn_candidates_base', [])
awg_candidates = globals().get('awg_candidates', [])
par_candidates = globals().get('par_candidates', [])

# =============================================================================
# Torch runtime defaults (import-safe)
# =============================================================================
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    ITYPE = torch.int32
else:  # pragma: no cover
    DEVICE = None
    DTYPE = None
    ITYPE = None

# ======================= 본 계산 =======================
results = []
funnel  = {"pass_all":0, "skip_empty_par":0, "skip_empty_turn":0}

# 전역 변수들을 모듈 수준에서 미리 선언
#turn_candidates_base = []
#awg_candidates = []
#par_candidates = []

# ---- [NEW] 통계 카운터(전역) ----
STATS = {
    "tiles_len_empty": 0,   # 길이 마스크만으로 비는 타일 수
    "pairs_parJ_empty": 0,  # 병렬/전류밀도 프리필터에서 전부 탈락한 배치 수
    "fill_fails": 0,         # 2D에서 '충전율'로 컷된 조합 개수
    "volt_fails": 0,         # 2D에서 '전압'으로 컷된 조합 개수
    "pass_count": 0         # 2D 최종 통과 조합 개수
}

mlt_cache = {}
ROWS_TOTAL = 0                  # 누적 행수(= append된 후보 총계)
LAST_SNAPSHOT = time.perf_counter()


def auto_tile_limit(search_space_size):
    if search_space_size > 5_000_000:
        return 1_000_000
    if search_space_size > 1_000_000:
        return 500_000
    return None


def _process_reverse_power_with_ldlq_cache(
    df_batch,
    *,
    rpm,
    Ke_scale,
    Ld_mH,
    Lq_mH,
    Vdc,
    m_max,
    Kt_rms,
    T_Nm,
    cfg,
):
    """
    Run process_reverse_power() but override (Ld/Lq) per (AWG,Parallels,Turns) key when FEMM cache exists.
    Cache key: (AWG, Parallels, Turns_per_slot_side)
    """
    if df_batch is None or len(df_batch) == 0:
        return df_batch

    # 캐시가 실제로 적용될 키가 하나도 없으면 원래처럼 한 번만 계산(빠름)
    try:
        keys = df_batch[["AWG", "Parallels", "Turns_per_slot_side"]].dropna()
        if keys.empty:
            raise Exception
        uniq = {(int(a), int(p), int(n)) for a, p, n in keys.to_numpy()}
        any_cached = False
        for k in uniq:
            ld, lq = get_ld_lq(k, fallback_Ld_mH=Ld_mH, fallback_Lq_mH=Lq_mH)
            if (ld is not None and float(ld) != float(Ld_mH)) or (lq is not None and float(lq) != float(Lq_mH)):
                any_cached = True
                break
        if not any_cached:
            return process_reverse_power(
                df_batch,
                rpm=rpm, Ke_scale=Ke_scale, Ld_mH=Ld_mH, Lq_mH=Lq_mH,
                Vdc=Vdc, m_max=m_max, Kt_rms=Kt_rms, T_Nm=T_Nm, cfg=cfg,
            )
    except Exception:
        return process_reverse_power(
            df_batch,
            rpm=rpm, Ke_scale=Ke_scale, Ld_mH=Ld_mH, Lq_mH=Lq_mH,
            Vdc=Vdc, m_max=m_max, Kt_rms=Kt_rms, T_Nm=T_Nm, cfg=cfg,
        )

    # 캐시 적용: 키별로 서브배치를 나눠서 계산
    outs = []
    for (awg, par, nslot), sub in df_batch.groupby(["AWG", "Parallels", "Turns_per_slot_side"], sort=False):
        key = (int(awg), int(par), int(nslot))
        ld_use, lq_use = get_ld_lq(key, fallback_Ld_mH=Ld_mH, fallback_Lq_mH=Lq_mH)
        outs.append(
            process_reverse_power(
                sub,
                rpm=rpm, Ke_scale=Ke_scale, Ld_mH=float(ld_use), Lq_mH=float(lq_use),
                Vdc=Vdc, m_max=m_max, Kt_rms=Kt_rms, T_Nm=T_Nm, cfg=cfg,
            )
        )
    return pd.concat(outs, ignore_index=True) if outs else df_batch

class EarlyStop(Exception):
    pass


@dataclass
class RunConfig:
    device: torch.device
    dtype: torch.dtype
    itype: torch.int32
    # 추가적인 파라미터들도 이곳에 통합 가능
# 실행 시점에 생성
cfg = RunConfig(device=DEVICE, dtype=DTYPE, itype=ITYPE)

# 텐서 변환 헬퍼
SQRT2_T               = T(math.sqrt(2.0), config=cfg)
SQRT3_OVER_SQRT2_T    = T(math.sqrt(3) / math.sqrt(2), config=cfg)
ONE_I                 = torch.tensor(1, dtype=ITYPE, device=DEVICE)
BETA_FINE = torch.linspace(0., math.pi/2, steps=121, device=DEVICE, dtype=DTYPE)
SIN_FINE, COS_FINE = torch.sin(BETA_FINE), torch.cos(BETA_FINE)
# ============================================================
def build_power_torque_cases(hp: dict,
                             pair_mode: str = "product",
                             torque_round: int | None = 4):
    """
    하이퍼파라미터 dict(hp)에서 rpm/P/T 그리드를 구성해 case 리스트를 돌려줌.
    - 우선순위: P_kW_list가 있으면 rpm×P로 T 자동생성 → T_Nm_list 무시
                (불일치/중복 방지)
    - P_kW_list 없고 T_Nm_list만 있으면 rpm×T로 조합
    - pair_mode:
        "product" : 데카르트 곱(권장, 탐색용)
        "zip"     : 동일 인덱스끼리 페어링(길이 불일치 시 오류)
    - torque_round: 소수 자릿수 반올림(None이면 반올림 안함)
    반환: cases = [{ "rpm": ..., "P_kW": ..., "T_Nm": ...}, ...]
    """
    rpm_list   = [int(round(x)) for x in hp.get("rpm_list", [])]
    p_list     = hp.get("P_kW_list", None)
    t_list     = hp.get("T_Nm_list", None)

    if not rpm_list:
        raise ValueError("rpm_list 가 비어있습니다. 최소 1개 이상의 rpm 이 필요합니다.")

    # 1) P_kW_list 우선
    if p_list and len(p_list) > 0:
        p_list = list(map(float, p_list))
        cases = []
        if pair_mode == "zip":
            if len(p_list) != len(rpm_list):
                raise ValueError(f"pair_mode='zip'인데 길이가 다릅니다: len(P_kW_list)={len(p_list)}, len(rpm_list)={len(rpm_list)}")
            for rpm, pkW in zip(rpm_list, p_list):
                T = kw_rpm_to_torque_nm(pkW, rpm)
                if torque_round is not None:
                    T = round(T, torque_round)
                cases.append({"rpm": int(rpm), "P_kW": float(pkW), "T_Nm": T})
        else:  # product
            for rpm, pkW in product(rpm_list, p_list):
                T = kw_rpm_to_torque_nm(pkW, rpm)
                if torque_round is not None:
                    T = round(T, torque_round)
                cases.append({"rpm": int(rpm), "P_kW": float(pkW), "T_Nm": T})
        print(f"[P→T] built {len(cases)} cases from rpm_list × P_kW_list (mode={pair_mode})")
        return cases

    # 2) T_Nm_list만 존재할 때
    if t_list and len(t_list) > 0:
        t_list = list(map(float, t_list))
        cases = []
        if pair_mode == "zip":
            if len(t_list) != len(rpm_list):
                raise ValueError(f"pair_mode='zip'인데 길이가 다릅니다: len(T_Nm_list)={len(t_list)}, len(rpm_list)={len(rpm_list)}")
            for rpm, T in zip(rpm_list, t_list):
                Tn = round(T, torque_round) if torque_round is not None else T
                cases.append({"rpm": int(rpm), "P_kW": None, "T_Nm": float(Tn)})
        else:  # product
            for rpm, T in product(rpm_list, t_list):
                Tn = round(T, torque_round) if torque_round is not None else T
                cases.append({"rpm": int(rpm), "P_kW": None, "T_Nm": float(Tn)})
        print(f"[T-only] built {len(cases)} cases from rpm_list × T_Nm_list (mode={pair_mode})")
        return cases

    # 3) 둘 다 없음 → "rpm-only" 케이스를 만든다 (max_power 모드용)
    #    - run_sweep에서 T/P를 계산해 사용
    cases = [{"rpm": int(rpm), "P_kW": None, "T_Nm": None} for rpm in rpm_list]
    print(f"[rpm-only] built {len(cases)} cases from rpm_list (no load)")
    return cases

# ============================================================
# B-FLOW STEP 4
#   1) PASS-1: auto_generate_inputs + sweep
#   2) pass_rows 선택 → auto_adjust_by_pass
#   3) PASS-2: 튜닝된 hp로 sweep
#   4) PASS-2 결과를 저장(최종본)
# ============================================================
# B-FLOW STEP 1
#   auto_generate_inputs() + pass1용 hp, cases 준비
# ============================================================

def auto_generate_inputs(
    rpm_list,
    P_kW_list=None,
    T_Nm_list=None,
    MOTOR_TYPE="IPM",
    Ld_mH_list=(1.5, 2.0),
    SAL_RATIO_LIST=(1.3, 1.5, 1.7),
    # 전압/전류/Ke 등 기본 sweep
    Vdc_list=(380, 540),
    m_max_list=(0.925, 0.95, 0.975),
    Ke_scale_list=(0.95, 1.00),
    Kt_rms_list=(0.35, 0.40, 0.45, 0.50),
    J_max_list=(8.0, 10.0, 12.0),
    awg_candidates=(17, 18, 19),
    # geometry sweep
    stack_mm_list=(55.0, 56.0, 57.0),
    end_factor_list=(1.15, 1.25, 1.45),
    slot_pitch_mm_scales=(0.85, 0.95, 1.05, 1.15),
    coil_span_slots_list=(5, 6),
    MLT_scale_list=(0.95, 1.00, 1.05),
    # 길이 윈도우
    L_total_min_m=50.0,
    L_total_max_m=130.0,
    # NSLOT 사용자 하드 클램프
    user_nslot_min=10,
    user_nslot_max=92,
    # EMF 캡 relax
    relax_emf_for_range=1.1,
    verbose=True,
):
    # ------------------------------------------------------------------
    # 1) rpm/P -> case 생성 (T 자동)
    # ------------------------------------------------------------------
    hp: Dict[str, Any] = {}
    rpm_list = list(map(float, rpm_list))
    hp["rpm_list"] = rpm_list

    P_kW_list = list(map(float, P_kW_list)) if P_kW_list else []
    T_Nm_list = list(map(float, T_Nm_list)) if T_Nm_list else []

    if P_kW_list and len(P_kW_list) > 0:
        hp["P_kW_list"] = P_kW_list
        hp["T_Nm_list"] = []  # 항상 P→T 우선
        cases = build_power_torque_cases(
            {"rpm_list": rpm_list, "P_kW_list": P_kW_list},
            pair_mode="product",
            torque_round=4,
        )
    elif T_Nm_list and len(T_Nm_list) > 0:
        hp["P_kW_list"] = []
        hp["T_Nm_list"] = T_Nm_list
        if not T_Nm_list:
            raise ValueError("P_kW_list 와 T_Nm_list 가 모두 비어 있습니다.")
        cases = build_power_torque_cases(
            {"rpm_list": rpm_list, "T_Nm_list": T_Nm_list},
            pair_mode="product",
            torque_round=4,
        )
    else:
        # max_power 모드(또는 rpm-only)
        hp["P_kW_list"] = []
        hp["T_Nm_list"] = []
        cases = build_power_torque_cases(
            {"rpm_list": rpm_list},
            pair_mode="product",
            torque_round=4,
        )
        
    # ------------------------------------------------------------------
    # 2) Lq 자동 생성 (IPM/SPM)
    # ------------------------------------------------------------------
    Ld_mH_list = list(map(float, Ld_mH_list))
    hp["Ld_mH_list"] = Ld_mH_list

    mt = str(MOTOR_TYPE or "IPM").strip().upper()
    if mt == "SPM":
        hp["Lq_mH_list"] = Ld_mH_list[:]
    elif mt == "IPM":
        hp["Lq_mH_list"] = sorted({
            round(Ld * s, 6)
            for Ld in Ld_mH_list
            for s in SAL_RATIO_LIST
        })
    else:
        raise ValueError("MOTOR_TYPE must be 'SPM' or 'IPM'")

    # ------------------------------------------------------------------
    # 3) 기본 sweep 이관
    # ------------------------------------------------------------------
    hp["Vdc_list"]       = list(map(float, Vdc_list))
    hp["m_max_list"]     = list(map(float, m_max_list))
    hp["Ke_scale_list"]  = list(map(float, Ke_scale_list))
    hp["Kt_rms_list"]    = list(map(float, Kt_rms_list))
    hp["J_max_list"]     = list(map(float, J_max_list))
    hp["awg_candidates"] = list(map(int, awg_candidates))

    hp["stack_mm_list"]       = list(map(float, stack_mm_list))
    hp["end_factor_list"]     = list(map(float, end_factor_list))
    hp["slot_pitch_mm_list"]  = list(map(float, slot_pitch_mm_scales))
    hp["coil_span_slots_list"]= list(map(int, coil_span_slots_list))
    hp["MLT_scale_list"]      = list(map(float, MLT_scale_list))

    hp["L_total_min_m"] = float(L_total_min_m)
    hp["L_total_max_m"] = float(L_total_max_m)

    # ------------------------------------------------------------------
    # 4) par_candidates 자동 생성 (+ 하드 클램프)
    # ------------------------------------------------------------------
    par_min_global = 1
    par_max_global = 1

    for c in cases:
        T_c = float(c["T_Nm"])
        for Kt in hp["Kt_rms_list"]:
            I_rms = T_c / Kt
            for Jm in hp["J_max_list"]:
                for awg in hp["awg_candidates"]:
                    area = AWG_TABLE[int(awg)]["area"]
                    if area <= 0:
                        continue
                    min_par = math.ceil(I_rms / (Jm * area))
                    par_min_global = max(par_min_global, min_par)
                    par_max_global = max(par_max_global, min_par + 6)  # 여유

    # 1턴 여유
    par_min_global = max(1, par_min_global - 1)
    par_max_global = max(par_min_global, par_max_global)

    # 전역 하드 클램프 적용
    par_min_global = max(PAR_HARD_MIN, par_min_global)
    par_max_global = min(PAR_HARD_MAX, par_max_global)

    if par_min_global > par_max_global:
        # 안전 fallback: 최소 범위라도 확보
        par_min_global = PAR_HARD_MIN
        par_max_global = min(PAR_HARD_MIN + 4, PAR_HARD_MAX)

    hp["par_candidates"] = list(range(par_min_global, par_max_global + 1))

    if verbose and hp["par_candidates"]:
        print(f"[AUTO] par_candidates (clamped) = "
              f"{hp['par_candidates'][0]}..{hp['par_candidates'][-1]}")

    # ------------------------------------------------------------------
    # 5) EMF+길이 기반 NSLOT_USER_RANGE 자동 추천
    # ------------------------------------------------------------------
    nmin_all = None
    nmax_all = None
    geom_hits = 0

    for r in hp["rpm_list"]:
        nmin_r, nmax_r, hit_r = infer_nslot_feasible_range_for_rpm(
            rpm=r,
            Ke_scale_list=hp["Ke_scale_list"],
            Ld_mH_list=hp["Ld_mH_list"],
            Lq_mH_list=hp["Lq_mH_list"],
            Vdc_list=hp["Vdc_list"],
            m_max_list=hp["m_max_list"],
            stack_mm_list=hp["stack_mm_list"],
            slot_pitch_mm_scales=hp["slot_pitch_mm_list"],
            end_factor_list=hp["end_factor_list"],
            coil_span_slots_list=hp["coil_span_slots_list"],
            MLT_scale_list=hp["MLT_scale_list"],
            L_total_min_m=hp["L_total_min_m"],
            L_total_max_m=hp["L_total_max_m"],
            relax_emf=relax_emf_for_range,
            verbose=False,
        )
        if nmin_r is None or nmax_r is None:
            continue
        geom_hits += hit_r
        nmin_all = nmin_r if nmin_all is None else min(nmin_all, nmin_r)
        nmax_all = nmax_r if nmax_all is None else max(nmax_all, nmax_r)

    if nmin_all is None or nmax_all is None:
        nmin_all, nmax_all = user_nslot_min, user_nslot_max

    hp["NSLOT_USER_RANGE"] = [
        max(int(nmin_all), int(user_nslot_min)),
        min(int(nmax_all), int(user_nslot_max)),
    ]

    hp["turn_candidates_base"] = list(range(
        hp["NSLOT_USER_RANGE"][0],
        hp["NSLOT_USER_RANGE"][1] + 1
    ))

    if verbose:
        lo, hi = hp["NSLOT_USER_RANGE"]
        print(f"[AUTO] NSLOT_USER_RANGE = {lo}..{hi}")
        print(f"[AUTO] turn_candidates_base = "
              f"{hp['turn_candidates_base'][0]}..{hp['turn_candidates_base'][-1]}")
        print(f"[AUTO] feasible geom hits (sum) ≈ {geom_hits:,}")

    return hp, cases

def bflow_pass1_build_hp_and_cases(
    rpm_list,
    P_kW_list=None,
    T_Nm_list=None,
    motor_type="IPM",
    verbose=True,
):
    """
    B안 PASS-1:
      1) auto_generate_inputs() 호출
      2) hp_auto, cases_auto 를 돌려준다.
      3) (원하면) globals 에도 반영 가능

    반환:
        hp_auto, cases_auto
    """
    hp_auto, cases_auto = auto_generate_inputs(
        rpm_list=rpm_list,
        P_kW_list=P_kW_list,
        T_Nm_list=T_Nm_list,
        MOTOR_TYPE=motor_type,
        Ld_mH_list=Ld_mH_list,
        SAL_RATIO_LIST=SAL_RATIO_LIST,
        Vdc_list=Vdc_list,
        m_max_list=m_max_list,
        Ke_scale_list=Ke_scale_list,
        Kt_rms_list=Kt_rms_list,
        J_max_list=[j*1.15 for j in J_max_list],
        awg_candidates=awg_candidates,
        stack_mm_list=stack_mm_list,
        end_factor_list=end_factor_list,
        slot_pitch_mm_scales=slot_pitch_mm_list,
        coil_span_slots_list=coil_span_slots_list,
        MLT_scale_list=MLT_scale_list,
        L_total_min_m=L_total_min_m,
        L_total_max_m=L_total_max_m,
        user_nslot_min=10,
        user_nslot_max=92,
        relax_emf_for_range=1.1,
        verbose=verbose,
    )

    if verbose:
        print("[B-FLOW] PASS-1 hp/cases 준비 완료.")
        print(f"  rpm_list    = {hp_auto['rpm_list']}")
        print(f"  P_kW_list   = {hp_auto.get('P_kW_list')}")
        print(f"  NSLOT_RANGE = {hp_auto['NSLOT_USER_RANGE']}")
        print(f"  par_range   = {hp_auto['par_candidates'][0]}..{hp_auto['par_candidates'][-1]}")

    return hp_auto, cases_auto

# ============================================================
# B-FLOW STEP 2
#   주어진 hp/cases 를 이용해 run_sweep() 한 번 돌리고
#   결과 DataFrame 을 반환하는 헬퍼
# ============================================================
def _apply_hp_to_globals(hp: dict, cases_local):
    """
    auto_generate_inputs() 또는 auto_adjust_by_pass() 결과 hp를
    run_sweep()에서 사용하는 전역 변수에 반영.
    """
    global cases, rpm_list, P_kW_list, T_Nm_list
    global awg_candidates, par_candidates
    global NSLOT_USER_RANGE, turn_candidates_base
    global stack_mm_list, end_factor_list
    global slot_pitch_mm_list, coil_span_slots_list
    global MLT_scale_list, slot_area_mm2_list, slot_fill_limit_list
    global Ke_scale_list, Ld_mH_list, Lq_mH_list
    global Vdc_list, m_max_list, Kt_rms_list, J_max_list
    global RPM_QUOTA
    # ⚠️ 텐서도 다시 만들어야 함
    global awg_vec, par_vec, awg_area, DEVICE, ITYPE, DTYPE, AWG_TABLE

    # 기본 리스트들
    rpm_list           = hp.get("rpm_list", rpm_list)
    P_kW_list          = hp.get("P_kW_list", P_kW_list)
    T_Nm_list          = hp.get("T_Nm_list", T_Nm_list)

    awg_candidates     = hp.get("awg_candidates", awg_candidates)
    par_candidates     = hp.get("par_candidates", par_candidates)

    NSLOT_USER_RANGE     = hp.get("NSLOT_USER_RANGE", NSLOT_USER_RANGE)
    turn_candidates_base = hp.get("turn_candidates_base", turn_candidates_base)

    stack_mm_list        = hp.get("stack_mm_list", stack_mm_list)
    end_factor_list      = hp.get("end_factor_list", end_factor_list)
    slot_pitch_mm_list   = hp.get("slot_pitch_mm_list", slot_pitch_mm_list)
    coil_span_slots_list = hp.get("coil_span_slots_list", coil_span_slots_list)
    MLT_scale_list       = hp.get("MLT_scale_list", MLT_scale_list)

    slot_area_mm2_list   = hp.get("slot_area_mm2_list", slot_area_mm2_list)
    slot_fill_limit_list = hp.get("slot_fill_limit_list", slot_fill_limit_list)

    Ke_scale_list      = hp.get("Ke_scale_list", Ke_scale_list)
    Ld_mH_list         = hp.get("Ld_mH_list", Ld_mH_list)
    Lq_mH_list         = hp.get("Lq_mH_list", Lq_mH_list)

    Vdc_list           = hp.get("Vdc_list", Vdc_list)
    m_max_list         = hp.get("m_max_list", m_max_list)
    Kt_rms_list        = hp.get("Kt_rms_list", Kt_rms_list)
    J_max_list         = hp.get("J_max_list", J_max_list)

    # RPM_QUOTA 도 반영 (없으면 기존 유지)
    # ------------------------------------------------------------
    # RPM_QUOTA는 "덮어쓰기"가 아니라 "병합" (hp에 3600이 빠져 있어도 기존 전역을 유지)
    #   - 전역 RPM_QUOTA를 기본으로 두고
    #   - hp에 RPM_QUOTA가 있으면 해당 키만 update
    # ------------------------------------------------------------
    _rpm_quota_hp = hp.get("RPM_QUOTA", None)
    if isinstance(_rpm_quota_hp, dict):
        _merged = dict(RPM_QUOTA) if isinstance(RPM_QUOTA, dict) else {}
        _merged.update({int(k): int(v) for k, v in _rpm_quota_hp.items()})
        RPM_QUOTA = _merged
        # (선택) 실제 적용값 1회 출력
        print(f"[RPM_QUOTA] effective = {RPM_QUOTA}")

    # case 리스트
    cases = list(cases_local)

    # ---- [NEW] AWG/Par 텐서 재생성 ----
    if awg_candidates:
        awg_vec  = T(awg_candidates, config=cfg)
        awg_area = T(
            [AWG_TABLE[a]["area"] for a in awg_candidates],
            config=cfg
        )
    if par_candidates:
        par_vec  = T(par_candidates, config=cfg)

def _reset_sweep_accumulators(case_total: int | None = None):
    global results, STATS, ROWS_TOTAL, funnel
    global PROG, GEO, PROF, PROG2
    """
    run_sweep()에 사용하는 누적용 전역(results, STATS, ROWS_TOTAL, funnel 등) 초기화.
    PASS-1 / PASS-2를 독립적으로 돌리기 위해 필요.
    """
    results = []
    funnel  = {"pass_all": 0, "skip_empty_par": 0, "skip_empty_turn": 0}
    ROWS_TOTAL = 0

    STATS = {
        "tiles_len_empty": 0,
        "pairs_parJ_empty": 0,
        "fill_fails": 0,
        "volt_fails": 0,
        "pass_count": 0,
    }

    t0 = time.perf_counter()
    # IMPORTANT: progress 모듈이 참조하는 dict 객체와 "같은 객체"를 유지해야 함.
    # dict 재할당(=새 dict로 교체) 금지. clear()/update()로만 초기화.
    # IMPORTANT: progress.py가 참조하는 dict 객체와 "같은 객체"를 유지해야 함.
    # dict 재할당 금지. clear()/update()로만 초기화.
    if isinstance(PROG, dict):
        PROG.clear()
        PROG.update({"t0": t0, "last": 0.0, "case_total": 0, "case_done": 0, "geo_done": 0})
    if isinstance(GEO, dict):
        GEO.clear()
        GEO.update({"case_total": int(case_total or 0), "case_idx": 0, "geo_steps": 0, "tile_hits": 0})
    if isinstance(PROF, dict):
        PROF.clear()
        # start_wall=None이면 wall='-'가 계속 뜨므로 t0로 시작
        PROF.update({"start_wall": t0, "combos_evaluated": 0, "combos_planned": 0,
                     "gpu_ms_mask": 0.0, "gpu_ms_collect": 0.0})
    if isinstance(PROG2, dict):
        PROG2.clear()
        PROG2.update({"tiles_done": 0, "tiles_total": 0})

    # case_total 반영(진행률 표시용) - PROG/GEO 둘 다에 넣음
    if case_total is not None:
        try:
            ct = int(case_total)
            if isinstance(PROG, dict): PROG["case_total"] = ct
            if isinstance(GEO, dict):  GEO["case_total"]  = ct
        except Exception:
            pass

# ============================================================
# B-FLOW STEP 3
#   PASS-1 결과에서 auto_adjust_by_pass() 입력용 pass_rows 추출
# ============================================================
def bflow_select_pass_rows_for_autotune(
    df_pass1: pd.DataFrame,
    min_margin_pct: float = 0.005,
    topk: int = 1000,
):
    """
    PASS-1 결과(df_pass1) 중에서
    - 기본 전압/충전율/전류밀도 조건을 만족하고
    - 어느 정도 전압 여유(min_margin_pct 이상)를 가진 행들을
      auto_adjust_by_pass()에 투입할 pass_rows로 선택.

    반환:
        pass_rows (DataFrame)
    """
    if df_pass1 is None or df_pass1.empty or "Note" in df_pass1.columns:
        print("[B-FLOW] PASS-1 결과에 유효 후보가 없습니다. auto_adjust 생략.")
        return df_pass1.iloc[0:0]  # 빈 DF

    df = df_pass1.copy()

    needed = {"rpm","AWG","Parallels","Turns_per_slot_side",
              "slot_area_mm2","J_A_per_mm2","Ke_scale","Kt_rms"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"[B-FLOW] auto_adjust용 필수 컬럼 누락: {missing}")

    # 전압 여유 관련 파생이 없으면 지금 생성
    if "V_margin_pct" not in df.columns:
        if {"Vavail_LL_rms","Vreq_LL_rms"} <= set(df.columns):
            vmax = df["Vavail_LL_rms"]
            vreq = df["Vreq_LL_rms"]
            df["V_margin"]     = vmax - vreq
            df["V_margin_pct"] = df["V_margin"] / vmax.replace(0, np.nan)
        else:
            df["V_margin_pct"] = 0.0  # 없는 경우엔 무조건 0으로 취급

    # 기본 PASS 조건(전압/충전율/전류밀도) + 최소 전압 여유
    mask = pd.Series(True, index=df.index)
    if {"Vreq_LL_rms","Vavail_LL_rms"} <= set(df.columns):
        mask &= df["Vreq_LL_rms"] <= df["Vavail_LL_rms"]
    if {"Slot_fill_ratio","slot_fill_limit"} <= set(df.columns):
        mask &= df["Slot_fill_ratio"] <= df["slot_fill_limit"]
    if {"J_A_per_mm2","J_max_A_per_mm2"} <= set(df.columns):
        mask &= df["J_A_per_mm2"] <= df["J_max_A_per_mm2"]

    mask &= df["V_margin_pct"] >= min_margin_pct

    df_passrows = df[mask].copy()

    # 너무 많으면 약간 정렬 후 상위 topk만 사용
    if len(df_passrows) > topk:
        df_passrows = df_passrows.sort_values(
            ["Pcu_W","J_A_per_mm2","Slot_fill_ratio"],
            ascending=[True, True, True]
        ).head(topk)

    print(f"[B-FLOW] auto_adjust용 pass_rows = {len(df_passrows)} 행 (min_margin_pct={min_margin_pct})")
    return df_passrows


def _need_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"필수 컬럼 누락: {miss}")

def _prepare_pass_for_rank(df: pd.DataFrame,
                           margin_target: float = MARGIN_TARGET,
                           margin_min_pct: float = MARGIN_MIN_PCT,
                           margin_min_v: float = MARGIN_MIN_V) -> pd.DataFrame:
    df = df.copy()
    _need_cols(df, ["Vavail_LL_rms","Vreq_LL_rms","Pcu_W",
                    "J_A_per_mm2","J_max_A_per_mm2","Slot_fill_ratio","slot_fill_limit"])

    df["V_margin"]     = df["Vavail_LL_rms"] - df["Vreq_LL_rms"]
    denom              = df["Vavail_LL_rms"].replace(0, np.nan)
    df["V_margin_pct"] = df["V_margin"] / denom

    BIG = 1e6
    df["V_margin_penalty"] = np.where(
        df["V_margin_pct"] < 0, BIG,
        (df["V_margin_pct"] - margin_target).abs()
    )

    # 하드 게이트(여유 >= max(MIN%, MIN_V/ Vavail)):
    gate_pct = np.maximum(margin_min_pct, margin_min_v / np.maximum(df["Vavail_LL_rms"], 1e-9))
    mask = (
        (df["J_A_per_mm2"]     <= df["J_max_A_per_mm2"]) &
        (df["Slot_fill_ratio"] <= df["slot_fill_limit"]) &
        (df["Vreq_LL_rms"]     <= df["Vavail_LL_rms"])   &
        (df["V_margin_pct"]    >= gate_pct)
    )
    return df[mask].copy()

def _select_top(df_sorted: pd.DataFrame,
                topk: int | None = RANK_TOPK,
                group_topk: int | None = RANK_GROUP_TOPK,
                group_cols: list[str] = RANK_GROUP_COLS) -> pd.DataFrame:
    out = df_sorted
    if group_topk:
        gcols = [c for c in group_cols if c in out.columns]
        if gcols:
            out = out.groupby(gcols, dropna=False, group_keys=False).head(group_topk)
    if topk:
        out = out.head(topk)
    return out

def _rank_df(df_pass: pd.DataFrame, preset: str = RANK_PRESET) -> pd.DataFrame:
    if preset not in _RANK_PRESETS:
        raise ValueError(f"알 수 없는 프리셋: {preset}  (가능값: {list(_RANK_PRESETS)})")
    sort_cols, asc = _RANK_PRESETS[preset]
    using = [c for c in sort_cols if c in df_pass.columns]
    return df_pass.sort_values(using, ascending=asc[:len(using)], kind="mergesort")

def make_rank(df_candidates: pd.DataFrame) -> pd.DataFrame:
    """후보 → 파생지표/필터 → 프리셋 정렬 → TOPK 추출"""
    dfp = _prepare_pass_for_rank(df_candidates)
    dfs = _rank_df(dfp, preset=RANK_PRESET)
    dft = _select_top(dfs, topk=RANK_TOPK, group_topk=RANK_GROUP_TOPK, group_cols=RANK_GROUP_COLS)
    return dft

def auto_adjust_by_pass(
    hp_raw,
    pass_rows,
    verbose=True
):
    """
    hp_raw : auto_generate_inputs() 결과 파라미터 dict
    pass_rows : DataFrame 또는 dict 리스트
                반드시 다음 key 포함:
                - AWG, Parallels, Turns_per_slot_side
                - slot_area_mm2
                - J_A_per_mm2, Ke_scale, Kt_rms

    반환:
        hp (튜닝된 파라미터 dict)
    """
    if len(pass_rows) < 5:
        print("[B-FLOW][GUARD] PASS rows too small → keep original hp")
        return hp_raw

    # 여기서 DataFrame -> dict list 변환
    if isinstance(pass_rows, pd.DataFrame):
        pass_rows = pass_rows.to_dict(orient="records")
        
    if not pass_rows:
        if verbose:
            print("[PASS→AUTO] pass_rows 비어 있음 → auto_adjust 생략, hp_raw 그대로 사용")
        return hp_raw

    hp = copy.deepcopy(hp_raw)

    # 전역 하드 클램프/테이블도 참조
    global PAR_HARD_MIN, PAR_HARD_MAX, AWG_TABLE

    # -----------------------------------------------------------
    # 1) PASS 기반: AWG 및 Par 범위 자동 수축 (+ 하드 클램프 + 교집합)
    # -----------------------------------------------------------
    awg_used = sorted({ int(r["AWG"]) for r in pass_rows })
    par_used = sorted({ int(r["Parallels"]) for r in pass_rows })

    # PASS에서 실제 쓴 범위 주변으로 1 step 여유
    awg_min = max(min(awg_used) - 1,  min(AWG_TABLE.keys()))
    awg_max = min(max(awg_used) + 1,  max(AWG_TABLE.keys()))
    
    #  하드 클램프 사용
    par_min = max(min(par_used) - 1,  PAR_HARD_MIN)
    par_max = min(max(par_used) + 2,  PAR_HARD_MAX)

    # 기존 hp_raw 범위와 교차 (혹시 hp_raw에서 더 좁게 잡았을 수도 있음)
    old_awg = sorted(set(hp_raw.get("awg_candidates", AWG_TABLE.keys())))
    old_par = sorted(set(hp_raw.get("par_candidates", range(PAR_HARD_MIN, PAR_HARD_MAX+1))))

    awg_range = [a for a in range(awg_min, awg_max+1) if a in old_awg and a in AWG_TABLE]
    par_range = [p for p in range(par_min, par_max+1) if (p in old_par)]

    # 혹시라도 비면, 원래 범위 유지 (안전 fallback)
    if not awg_range:
        awg_range = old_awg
    if not par_range:
        par_range = old_par

    hp["awg_candidates"] = awg_range
    hp["par_candidates"] = par_range

    if verbose:
        if awg_range and len(awg_range) > 0:
            print(f"[PASS→AUTO] AWG range → {awg_range[0]}..{awg_range[-1]}")
        else:
            print(f"[PASS→AUTO] AWG range → Empty (check constraints)")
            # 리스트가 비었다면 기본값으로 복구하는 안전장치
            awg_range = [18, 19, 20]
        print(f"[PASS→AUTO] PAR range → {par_range[0]}..{par_range[-1]}")

    # -----------------------------------------------------------
    # 2) PASS 기반: Nslot 범위 자동 정렬 (turns) + 기존 범위와 교집합
    # -----------------------------------------------------------
    turns_used = sorted({ int(r["Turns_per_slot_side"]) for r in pass_rows })

    turns_min = max(min(turns_used) - 3,  4)    # 최소 4턴 정도는 유지
    turns_max = min(max(turns_used) + 40, 200)  # 지나치게 길어지는 것 방지

    # 기존 NSLOT_USER_RANGE와 교차
    old_nslot = hp_raw.get("NSLOT_USER_RANGE", [turn_candidates_base[0], turn_candidates_base[-1]])
    old_lo, old_hi = int(old_nslot[0]), int(old_nslot[1])

    lo = max(turns_min, old_lo)
    hi = min(turns_max, old_hi)
    if lo > hi:  # 완전히 꼬이면, 그냥 기존 범위를 유지
        lo, hi = old_lo, old_hi

    hp["turn_candidates_base"] = list(range(lo, hi + 1))
    hp["NSLOT_USER_RANGE"]     = [lo, hi]

    if verbose:
        print(f"[PASS→AUTO] Nslot(Turn) range → {lo}..{hi}")

    # -----------------------------------------------------------
    # 3) slot_area_mm2 범위 자동 설정 (+ 기존과 교집합)
    # -----------------------------------------------------------
    sa_vals = sorted({ float(r["slot_area_mm2"]) for r in pass_rows })
    sa_min = max(int(min(sa_vals)) - 20,  60)
    sa_max = min(int(max(sa_vals)) + 40, 200)

    old_sa = sorted(set(hp_raw.get("slot_area_mm2_list", slot_area_mm2_list)))
    sa_list = [s for s in range(sa_min, sa_max + 1, 5) if s in old_sa]

    if not sa_list:  # 안전 fallback
        sa_list = old_sa

    hp["slot_area_mm2_list"] = sa_list

    if verbose:
        print(f"[PASS→AUTO] slot_area_mm2_list → {sa_list[0]}..{sa_list[-1]} (step=5)")

    # -----------------------------------------------------------
    # 4) Ke_scale / Kt_rms / J_max 기반 재설정 (PASS 주변만)
    # -----------------------------------------------------------
    ke_vals = sorted({ float(r["Ke_scale"])     for r in pass_rows })
    kt_vals = sorted({ float(r["Kt_rms"])       for r in pass_rows })
    j_vals  = sorted({ float(r["J_A_per_mm2"])  for r in pass_rows })

    def _expand_range(vals, pad_low, pad_high):
        vmin, vmax = min(vals), max(vals)
        return vmin - pad_low*(vmax-vmin+1e-9), vmax + pad_high*(vmax-vmin+1e-9)

    if ke_vals:
        lo, hi = _expand_range(ke_vals, 0.5, 0.5)  # ±50% 여유
        base = hp_raw.get("Ke_scale_list", Ke_scale_list)
        hp["Ke_scale_list"] = sorted({ v for v in base if lo <= v <= hi })
        if not hp["Ke_scale_list"]:  # fallback
            hp["Ke_scale_list"] = base[:]
    else:
        hp["Ke_scale_list"] = hp_raw.get("Ke_scale_list", Ke_scale_list)

    if kt_vals:
        lo, hi = _expand_range(kt_vals, 0.3, 0.3)
        base = hp_raw.get("Kt_rms_list", Kt_rms_list)
        hp["Kt_rms_list"] = sorted({ v for v in base if lo <= v <= hi })
        if not hp["Kt_rms_list"]:
            hp["Kt_rms_list"] = base[:]
    else:
        hp["Kt_rms_list"] = hp_raw.get("Kt_rms_list", Kt_rms_list)

    if j_vals:
        lo, hi = _expand_range(j_vals, 0.3, 0.3)
        base = hp_raw.get("J_max_list", J_max_list)
        hp["J_max_list"] = sorted({ v for v in base if lo <= v <= hi })
        if not hp["J_max_list"]:
            hp["J_max_list"] = base[:]
    else:
        hp["J_max_list"] = hp_raw.get("J_max_list", J_max_list)

    if verbose:
        print(f"[PASS→AUTO] Kt_rms_list → {hp['Kt_rms_list']}")
        print(f"[PASS→AUTO] Ke_scale_list → {hp['Ke_scale_list']}")
        print(f"[PASS→AUTO] J_max_list  → {hp['J_max_list']}")

    # -----------------------------------------------------------
    # 5) rpm별 PASS 기반 RPM_QUOTA 자동 조정 (기존과 병합)
    # -----------------------------------------------------------
    rpm_vals = sorted({ int(r["rpm"]) for r in pass_rows })
    old_quota = hp_raw.get("RPM_QUOTA", {})
    new_quota = dict(old_quota)

    # PASS에서 실제로 설계가 잡힌 rpm 들은 "최소 1개라도 유지"하도록 제한
    for r in rpm_vals:
        prev = old_quota.get(r, 9999)
        new_quota[r] = min(prev, 1)

    hp["RPM_QUOTA"] = new_quota

    if verbose:
        print(f"[PASS→AUTO] RPM_QUOTA → {hp['RPM_QUOTA']}")

    if min(hp["par_candidates"]) > 20:
        print("[B-FLOW][GUARD] par over-compressed → restoring")
        hp["par_candidates"] = list(range(PAR_HARD_MIN, PAR_HARD_MIN + 20))

    return hp

def init_planned_combos_denominator():
    """사전 분모 계산을 실행하고 PROF['combos_planned']에 세팅."""
    total = estimate_total_combos_planned()
    PROF["combos_planned"] = int(total)
    return total

def _par_list_from_globals(par0: int, max_steps: int) -> list[int]:
    """
    par_candidates가 있으면 그 후보를 따르고,
    없으면 [par0, par0+1, ...]로 fallback.
    max_steps는 '증가 스텝 수' 의미라서 총 길이는 max_steps+1.
    """
    try:
        cand = globals().get("par_candidates", None)
        if cand:
            par_list = sorted({int(x) for x in cand if int(x) >= int(par0)})
            if par_list:
                return par_list[: max_steps + 1]
    except Exception:
        pass
    # fallback: 연속 증가
    return [int(par0) + i for i in range(max_steps + 1)]

# ================= (옵션) 단일 후보 자동 수정 제안기 ==========================================
def suggest_margin_fix(row, target_margin_pct=0.10,
                       max_turn_steps=6, max_parallel_steps=4,
                       T_oper_C=120.0, coil_per_phase=4, poles=4,
                       beta_samples=121,
                       logger: logging.Logger = logging.getLogger("suggest_fix")):
    
    """
    턴↓(Ke↓, R↓) + 필요 시 병렬↑(R↓) + β-sweep(약계자)로 Vreq 최소화.
    목표 전압여유 달성 제안 반환(dict). 실패 시 no_fix / error 반환.

    변경점 정리:
      - try 블록 누락된 except 추가(문법 오류 제거)
      - 중간의 '예: ...' placeholder 블록 제거(미정의 함수/변수 참조 방지)
      - β 테이블은 _get_sin_cos_fine()만 사용(중복/충돌 제거)
      - 후보별 P_kW를 전압/전류 한계 하에서 직접 계산하여 반환(P_kW_calculated_new)
     """
     
         
    t0 = time.perf_counter()
    try:
        
        # ---- 전역 상수/상수화 ----
        m_phases = int(globals().get("m", 3))
        coil_per_phase_glob = int(globals().get("coils_per_phase", coil_per_phase))
        coil_per_phase = coil_per_phase_glob

        # Torch device/dtype (전역 일관)
        DTYPE  = globals().get("DTYPE", torch.float32)
        DEVICE = globals().get("DEVICE", torch.device("cpu"))

        # ---- row 입력 방탄 접근 ----
        rpm_raw = _row_get_first(row, ["rpm", "rpm_case", "rpm_check"], default=None)
        if rpm_raw is None:
            return dict(status="no_fix", reason="missing_rpm", runtime_s=0.0)
        rpm = float(rpm_raw)

        Vavail_raw = _row_get_first(row, ["Vavail_LL_rms", "V_LL_max_V"], default=None)
        if Vavail_raw is None:
            return dict(status="no_fix", reason="missing_Vavail", runtime_s=0.0)
        Vavail = float(Vavail_raw)
        
        Ke_scale = float(_row_get_first(row, ["Ke_scale"], default=None))
        Ld_mH    = float(_row_get_first(row, ["Ld_mH"], default=None))
        Lq_mH    = float(_row_get_first(row, ["Lq_mH"], default=Ld_mH))
        Kt_rms   = float(_row_get_first(row, ["Kt_rms"], default=None))
        T_Nm     = float(_row_get_first(row, ["T_Nm", "T_check_Nm"], default=None))
        if any(v is None for v in [Ke_scale, Ld_mH, Kt_rms, T_Nm]):
            return dict(status="no_fix", reason="missing_required_keys", runtime_s=0.0)

        # 요구토크 기준 전류(보수 체크)
        I_rms_req = float(T_Nm) / float(Kt_rms)

        Nslot0 = int(_row_get_first(row, ["Turns_per_slot_side"], default=None))
        Par0   = int(_row_get_first(row, ["Parallels"], default=None))
        if Nslot0 <= 0 or Par0 <= 0:
            return dict(status="no_fix", reason="invalid_Nslot_or_Par", runtime_s=0.0)

        # ---- 도체 단면적(AWG) ----
        A_wire = _row_get(row, "A_wire_mm2", np.nan)
        A_wire = float(A_wire) if A_wire is not None else float("nan")
        if np.isnan(A_wire):
            awg_raw = _row_get_first(row, ["AWG"], default=None)
            if awg_raw is None:
                return dict(status="no_fix", reason="missing_AWG", runtime_s=0.0)
            A_wire = float(awg_area_mm2(int(awg_raw)))
        if (not np.isfinite(A_wire)) or A_wire <= 0:
            raise ValueError("Invalid wire area (AWG lookup failed).")

        # ---- 기하 ----
        slot_area     = float(_row_get_first(row, ["slot_area_mm2"], default=None))
        slot_fill_lim = float(_row_get_first(row, ["slot_fill_limit"], default=None))
        MLT_mm        = float(_row_get_first(row, ["MLT_mm"], default=None))
        if any(v is None for v in [slot_area, slot_fill_lim, MLT_mm]):
            return dict(status="no_fix", reason="missing_geom_keys", runtime_s=0.0)
 
        rho20, alpha = 1.724e-8, 0.00393
        rho_T = rho20 * (1.0 + alpha * (float(T_oper_C) - 20.0))  # float
 
        Ke_nom = float(globals().get("Ke_LL_rms_per_krpm_nom", globals().get("Ke_LL_rms_per_krpm_nom", 20.0)))
        Nref   = float(globals().get("Nref_turn", globals().get("Nref_turn", 20.0)))
 
        pole_pairs = int(poles // 2)
        f_e     = float(pole_pairs) * rpm / 60.0
        omega_e = 2*math.pi * f_e
 
        # β–sweep: _get_sin_cos_fine()만 사용(중복/충돌 제거)
        SINb, COSb = _get_sin_cos_fine(beta_samples)
        SINb = SINb.to(device=DEVICE, dtype=DTYPE).view(-1, 1)
        COSb = COSb.to(device=DEVICE, dtype=DTYPE).view(-1, 1)
 
        K_V = (math.sqrt(3)/math.sqrt(2))  # float 상수
        Ld_H = float(Ld_mH) * 1e-3
        Lq_H = float(Lq_mH) * 1e-3

        def calc_vreq_ll_rms_batched(
            *,
            I_rms_vec: torch.Tensor,     # [N] rms
            Nslot_vec: torch.Tensor,     # [N] turns_per_slot_side
            Par_vec: torch.Tensor,       # [N] parallels
            rpm: int,
            Ke_scale: float,
            # --- design/const ---
            coil_per_phase: int,
            Ke_nom: float,
            Nref: float,
            omega_e: float,
            MLT_mm: float,
            A_wire_mm2: float,
            rho_T: float,
            slot_area_mm2: float,
            Ld_H: float,
            Lq_H: float,
            pole_pairs: int,
            # --- beta table ---
            SINb: torch.Tensor,          # [B,1]
            COSb: torch.Tensor,          # [B,1]
            SQRT2: float = math.sqrt(2.0),
            K_V: float = (math.sqrt(3)/math.sqrt(2)),
            mode: str = "mtpa",          # "mtpa" | "vmin" | "fw_max_torque_under_vlimit"
            Vavail_LL_rms_vec: torch.Tensor | None = None,  # [N]
        ):
            device = I_rms_vec.device
            dtype  = I_rms_vec.dtype

            I_rms = I_rms_vec.view(-1).to(device=device, dtype=dtype)  # [N]
            Nslot = Nslot_vec.view(-1).to(device=device, dtype=dtype)  # [N]
            Par   = Par_vec.view(-1).to(device=device, dtype=dtype)    # [N]

            N = int(I_rms.numel())
#            B = int(SINb.shape[0])

            # --- phase turns & back-emf -> psi_f ---
            Nphase   = Nslot * float(coil_per_phase)  # [N]
            E_LL_rms = float(Ke_nom) * float(Ke_scale) * (int(rpm)/1000.0) * (Nphase / float(Nref))  # [N]
            psi_f = E_LL_rms / (float(K_V) * float(omega_e) + 1e-12)  # [N]

            # --- resistance (per phase) ---
            L_phase_m = float(coil_per_phase) * Nslot * (float(MLT_mm) * 1e-3)  # [N] m
            A_tot_m2  = torch.clamp(Par * float(A_wire_mm2), min=1e-12) * 1e-6  # [N] m^2
            Rph = float(rho_T) * L_phase_m / A_tot_m2                           # [N] ohm

            # --- beta sweep currents (peak) ---
            Ipk = (float(SQRT2) * I_rms).view(1, N)  # [1,N]
            Id  = (-Ipk) * SINb.to(device=device, dtype=dtype)  # [B,N]
            Iq  = ( Ipk) * COSb.to(device=device, dtype=dtype)  # [B,N]

            # --- dq voltages ---
            Rph_b = Rph.view(1, N)
            w     = torch.tensor(float(omega_e), device=device, dtype=dtype).view(1, 1)
            Ld    = torch.tensor(float(Ld_H), device=device, dtype=dtype).view(1, 1)
            Lq    = torch.tensor(float(Lq_H), device=device, dtype=dtype).view(1, 1)
            psi   = psi_f.view(1, N)

            v_d = (Rph_b * Id) - (w * Lq) * Iq
            v_q = (Rph_b * Iq) + (w * Ld) * Id + (w * psi)

            Vpk = torch.sqrt(v_d*v_d + v_q*v_q)   # [B,N]
            Vll = (float(K_V) * Vpk)              # [B,N] LL_rms

            # --- torque sweep ---
            p = float(pole_pairs)
            Te = (1.5 * p) * (psi * Iq + (Ld - Lq) * Id * Iq)  # [B,N]

            # --- choose beta ---
            if mode == "vmin":
                idx = torch.argmin(Vll, dim=0)  # [N]
            elif mode == "mtpa":
                idx = torch.argmax(Te, dim=0)   # [N]
            elif mode == "fw_max_torque_under_vlimit":
                if Vavail_LL_rms_vec is None:
                    raise ValueError("fw_max_torque_under_vlimit requires Vavail_LL_rms_vec")
                Vavail_b = Vavail_LL_rms_vec.view(1, N).to(device=device, dtype=dtype)  # [1,N]
                feasible = (Vll <= Vavail_b)
                Te_masked = torch.where(feasible, Te, torch.full_like(Te, -1e30))
                idx = torch.argmax(Te_masked, dim=0)
            else:
                raise ValueError(f"Unknown mode={mode}")

            n_idx = torch.arange(N, device=device)
            Vreq = Vll[idx, n_idx]  # [N]
            Tuse = Te[idx, n_idx]   # [N]

            slot_fill = (2.0 * Nslot * Par * float(A_wire_mm2)) / float(slot_area_mm2)  # [N]
            J = I_rms / torch.clamp(Par * float(A_wire_mm2), min=1e-12)                 # [N] A/mm2

            extra = dict(
                Rph=Rph, slot_fill=slot_fill, J=J,
                psi_f=psi_f, Nphase=Nphase, E_LL_rms=E_LL_rms, L_phase_m=L_phase_m,
            )
            return Vreq, Tuse, extra

        def make_vreq_mtpa_of_I(
            *,
            Nslot_vec, Par_vec,
            rpm, Ke_scale,
            coil_per_phase, Ke_nom, Nref, omega_e,
            MLT_mm, A_wire_mm2, rho_T, slot_area_mm2,
            Ld_H, Lq_H, pole_pairs,
            SINb, COSb,
            device, dtype,
        ):
            Nslot_vec = torch.as_tensor(Nslot_vec, device=device, dtype=dtype)
            Par_vec   = torch.as_tensor(Par_vec,   device=device, dtype=dtype)

            def vreq_mtpa_of_I(I_rms_vec: torch.Tensor) -> torch.Tensor:
                I_rms_vec = I_rms_vec.to(device=device, dtype=dtype).view(-1)
                Vreq, _, _ = calc_vreq_ll_rms_batched(
                    I_rms_vec=I_rms_vec,
                    Nslot_vec=Nslot_vec,
                    Par_vec=Par_vec,
                    rpm=int(rpm),
                    Ke_scale=float(Ke_scale),
                    coil_per_phase=int(coil_per_phase),
                    Ke_nom=float(Ke_nom),
                    Nref=float(Nref),
                    omega_e=float(omega_e),
                    MLT_mm=float(MLT_mm),
                    A_wire_mm2=float(A_wire_mm2),
                    rho_T=float(rho_T),
                    slot_area_mm2=float(slot_area_mm2),
                    Ld_H=float(Ld_H),
                    Lq_H=float(Lq_H),
                    pole_pairs=int(pole_pairs),
                    SINb=SINb, COSb=COSb,
                    mode="mtpa",
                )
                return Vreq
            return vreq_mtpa_of_I

        @torch.no_grad()
        def solve_I_vlimit_bisect_batched(
            *,
            I_hi: torch.Tensor,                 # [N]
            Vavail_LL_rms: torch.Tensor,         # [N]
            vreq_of_I_fn,
            n_iter: int = 24,
        ):
            
            I_hi = torch.clamp(I_hi, min=0.0)
            I_lo = torch.zeros_like(I_hi)
            for _ in range(int(n_iter)):
                I_mid = 0.5 * (I_lo + I_hi)
                V_mid = vreq_of_I_fn(I_mid)
                ok = (V_mid <= Vavail_LL_rms)
                I_lo = torch.where(ok, I_mid, I_lo)
                I_hi = torch.where(ok, I_hi, I_mid)
            return I_lo
        
        # ================== [REPLACE] 기존 for(dN)/for(Par) 탐색 블록 (batched 평가) ==================
        
        # ---- (A) 후보 조합 텐서 생성 (파이썬으로 리스트 만든 뒤 torch로 1번 변환) ----
        par_list = _par_list_from_globals(Par0, max_parallel_steps)  # 예: [Par0, Par0+1, ...] or candidates
        dN_list  = list(range(0, max_turn_steps + 1))

        Nslot_list  = []
        Par_list2   = []
        dTurns_list = []
        dPar_list   = []
        for dN in dN_list:
            Nslot = max(1, int(Nslot0) - int(dN))
            for Par in par_list:
                Nslot_list.append(int(Nslot))
                Par_list2.append(int(Par))
                dTurns_list.append(int(Nslot - int(Nslot0)))  # 음수/0
                dPar_list.append(int(int(Par) - int(Par0)))

        Ncand = len(Nslot_list)

        if Ncand == 0:
            dt = time.perf_counter() - t0
            return dict(status="no_fix", reason="no_candidates", runtime_s=round(dt,3))
           
        Nslot_candidates = torch.tensor(Nslot_list, device=DEVICE, dtype=DTYPE)  # [N]
        Par_candidates   = torch.tensor(Par_list2,  device=DEVICE, dtype=DTYPE)  # [N]
        dTurns_t         = torch.tensor(dTurns_list, device=DEVICE, dtype=torch.int32)
        dPar_t           = torch.tensor(dPar_list,   device=DEVICE, dtype=torch.int32)

        Vavail_vec = torch.full((Ncand,), float(Vavail), device=DEVICE, dtype=DTYPE)

        # ---- (B) 후보별 허용 전류 I_lim ----
        Jmax_val = _row_get(row, "J_max_A_per_mm2", None)
        J_limit = float(Jmax_val) if (Jmax_val is not None and float(Jmax_val) > 0) else None

        if J_limit is None:
            # 부하(요구 토크) 기반: I_lim = I_rms_req
            I_lim = torch.full((Ncand,), float(I_rms_req), device=DEVICE, dtype=DTYPE)
        else:
            # 최대가능 출력 기반: 후보별 I_lim
            # A_wire[mm^2], Par -> 총 도체 단면적[mm^2]
            I_lim = torch.clamp(Par_candidates * float(A_wire) * float(J_limit), min=0.0)  # [N] A_rms

        vreq_mtpa_of_I = make_vreq_mtpa_of_I(
            Nslot_vec=Nslot_candidates,
            Par_vec=Par_candidates,
            rpm=int(rpm), Ke_scale=float(Ke_scale),
            coil_per_phase=int(coil_per_phase),
            Ke_nom=float(Ke_nom), Nref=float(Nref), omega_e=float(omega_e),
            MLT_mm=float(MLT_mm), A_wire_mm2=float(A_wire),
            rho_T=float(rho_T), slot_area_mm2=float(slot_area),
            Ld_H=float(Ld_H), Lq_H=float(Lq_H), pole_pairs=int(pole_pairs),
            SINb=SINb, COSb=COSb,
            device=DEVICE, dtype=DTYPE,
        )

        I_vlimit = solve_I_vlimit_bisect_batched(
            I_hi=I_lim,
            Vavail_LL_rms=Vavail_vec,
            vreq_of_I_fn=vreq_mtpa_of_I,
            n_iter=24,
        )

        I_use = torch.minimum(I_lim, I_vlimit)

        Vreq_fw, Tuse_fw, extra = calc_vreq_ll_rms_batched(
            I_rms_vec=I_use,
            Nslot_vec=Nslot_candidates,
            Par_vec=Par_candidates,
            rpm=int(rpm),
            Ke_scale=float(Ke_scale),
            coil_per_phase=int(coil_per_phase),
            Ke_nom=float(Ke_nom),
            Nref=float(Nref),
            omega_e=float(omega_e),
            MLT_mm=float(MLT_mm),
            A_wire_mm2=float(A_wire),
            rho_T=float(rho_T),
            slot_area_mm2=float(slot_area),
            Ld_H=float(Ld_H),
            Lq_H=float(Lq_H),
            pole_pairs=int(pole_pairs),
            SINb=SINb, COSb=COSb,
            mode="fw_max_torque_under_vlimit",
            Vavail_LL_rms_vec=Vavail_vec,
            K_V=K_V,
        )


        Vmargin     = Vavail_vec - Vreq_fw
        Vmargin_pct = Vmargin / torch.clamp(Vavail_vec, min=1e-9)

        # 직접 계산된 출력
        P_kW_calc = (Tuse_fw * float(rpm)) / 9550.0

        slot_fill = extra["slot_fill"]
        ok_fill = (slot_fill <= float(slot_fill_lim))
        ok_margin = (Vmargin_pct >= float(target_margin_pct))
        ok_T = (Tuse_fw >= float(T_Nm) - 1e-9)
        feasible = ok_fill & ok_margin & ok_T

#        turn_steps = (torch.tensor(int(Nslot0), device=DEVICE, dtype=torch.int32) - Nslot_candidates.to(torch.int32))
#        par_steps  = (Par_candidates.to(torch.int32) - torch.tensor(int(Par0), device=DEVICE, dtype=torch.int32))
        cost = (-dTurns_t) * 1000 + dPar_t
        BIG  = torch.tensor(2**30, device=DEVICE, dtype=torch.int32)
        cost2 = torch.where(feasible, cost, BIG)

        best = int(torch.argmin(cost2).item())
        if int(cost2[best].item()) >= int(BIG.item()):
            dt = time.perf_counter() - t0
            return dict(status="no_fix", reason="limits_exhausted", runtime_s=round(dt,3))

        # ---- best 추출 ----
        Nslot_new = int(Nslot_candidates[best].item())
        Par_new   = int(Par_candidates[best].item())

        Vreq_new  = float(Vreq_fw[best].item())
        Vav_new   = float(Vavail_vec[best].item())
        Vm_new    = float(Vmargin[best].item())
        Vmp_new   = float(Vmargin_pct[best].item())

        T_new     = float(Tuse_fw[best].item())
        P_kW_new  = float(P_kW_calc[best].item())

        Rph_new   = float(extra["Rph"][best].item())
        fill_new  = float(slot_fill[best].item())
        J_new     = float(extra["J"][best].item())

        I_use_best = float(I_use[best].item())
        P_cu_W_new_est = float(m_phases) * (I_use_best ** 2) * float(Rph_new)

        dt = time.perf_counter() - t0

        return dict(
            status="ok",
            reason="target_margin_met",
            
            Turns_per_slot_side_new=Nslot_new,
            Parallels_new=Par_new,

            V_LL_max_V_new=Vav_new,
            V_LL_req_V_new=Vreq_new,
            V_LL_margin_V_new=Vm_new,
            V_LL_margin_pct_new=Vmp_new,

            T_Nm_calculated_new=T_new,
            P_kW_calculated_new=P_kW_new,
            I_rms_used_new=I_use_best,

            P_cu_W_new_est=P_cu_W_new_est,
            R_phase_ohm_new=float(Rph_new),
            Slot_fill_ratio_new=fill_new,
            J_A_per_mm2_new=J_new,

            dTurns=int(Nslot_new - int(Nslot0)),
            dPar=int(Par_new - int(Par0)),
            
            runtime_s=round(dt, 3),
        )
        # ================== [REPLACE END] ==================
    except Exception as e:
        logger.exception(f"[suggest_fix][ERROR] {e}")
        dt = time.perf_counter() - t0
        return dict(status="error", reason=str(e), runtime_s=round(dt,3))
    
def estimate_total_combos_planned() -> int:
    """
    run_sweep() 내부의 분모 증가 규칙과 동일하게
    CPU에서 전체 planned 조합 수를 사전 계산한다.
    (이미 build_power_torque_cases()로 만든 cases를 사용)
    """
    total = 0

    MULT_SA_FILL = len(slot_area_mm2_list) * len(slot_fill_limit_list)
    par_sorted   = sorted(par_candidates)
    n_par        = len(par_sorted)
    
     # ============================================================
     # [TOP5-5] (치명) 깊은 루프 내부 print/set 생성 제거
     #   - cases 요약 출력은 딱 1번만
     # ============================================================
    try:
        rpm_set  = sorted({float(c.get("rpm", 0.0)) for c in cases})
        pk_set   = sorted({c.get("P_kW", None) for c in cases})
        print(f"[INIT] cases built: {len(cases)} points (rpm_list={rpm_set}, P_kW_list={pk_set})")
    except Exception:
        pass
    
    # 🔁 기존: for rpm in rpm_list: for P_kW in P_kW_list:
    #    → 이제는 사전에 만든 cases 리스트를 그대로 사용
    for case in cases:
        rpm  = float(case["rpm"])
        #P_kW = float(case["P_kW"])
        T_Nm = float(case["T_Nm"])

        for Ke_scale in Ke_scale_list:
            Ke_use = Ke_LL_rms_per_krpm_nom * Ke_scale

            for Ld_mH in Ld_mH_list:
                for Lq_mH in Lq_mH_list:
                    for Vdc in Vdc_list:
                        for m_max in m_max_list:
                            Vavail_LL_rms = m_max * Vdc / math.sqrt(3)
                            gate_pct_scalar = max(
                                MARGIN_MIN_PCT,
                                MARGIN_MIN_V / float(Vavail_LL_rms)
                            )

                            for Kt_rms in globals().get("Kt_rms_list", [1.0]):
                                if Kt_rms <= 0:
                                    continue
                                I_rms_scalar = T_Nm / Kt_rms

                                for J_max in J_max_list:
                                    # ========= 가능한 (AWG, Par) 쌍 개수 P 계산 =========
                                    P = 0
                                    for awg in awg_candidates:
                                        area = AWG_TABLE[awg]['area']
                                        if area <= 0:
                                            continue

                                        min_par = math.ceil(I_rms_scalar / (J_max * area))
                                        if min_par < 1:
                                            min_par = 1

                                        idx = bisect_left(par_sorted, min_par)
                                        if idx < n_par:
                                            P += (n_par - idx)

                                    if P <= 0:
                                        continue

                                    # ========= 길이(MLT), EMF 기반 Nslot window =========
                                    for stack_mm in stack_mm_list:
                                        for slot_pitch_scale in slot_pitch_mm_list:
                                            slot_pitch_mm = slot_pitch_mm_nom * float(slot_pitch_scale)
                                            for end_factor in end_factor_list:
                                                for coil_span_slots in coil_span_slots_list:

                                                    MLT_base_mm = estimate_mlt_mm(
                                                        slot_pitch_mm=slot_pitch_mm,
                                                        stack_mm=float(stack_mm),
                                                        coil_span_slots=int(coil_span_slots),
                                                        N_slots=int(N_slots),
                                                        D_use=float(D_use),
                                                        span_is_inclusive=True,
                                                    )

                                                    for mlt_scale in MLT_scale_list:
                                                        MLT_mm = MLT_base_mm * mlt_scale
                                                        if MLT_mm <= 0:
                                                            continue

                                                        denom = (m * coils_per_phase *
                                                                 MLT_mm * 1e-3)
                                                        if denom <= 0:
                                                            continue

                                                        Nslot_len_min = math.ceil(
                                                            L_total_min_m / denom)
                                                        Nslot_len_max = math.floor(
                                                            L_total_max_m / denom)
                                                        if (Nslot_len_min >
                                                                Nslot_len_max):
                                                            continue

                                                        # EMF 기반 상한
                                                        if Ke_use <= 0 or rpm <= 0:
                                                            nslot_emf_cap = 10**9
                                                        else:
                                                            relax_emf = SAFETY_RELAX
                                                            nphase_cap = math.floor(
                                                                ((1.0 - gate_pct_scalar)
                                                                 * Vavail_LL_rms
                                                                 / (Ke_use * (rpm/1000.0)))
                                                                * (Nref_turn / relax_emf)
                                                            )
                                                            nslot_emf_cap = max(
                                                                0,
                                                                nphase_cap //
                                                                max(1, coils_per_phase)
                                                            )

                                                        # 사용자 NSLOT window 반영
                                                        user_lo, user_hi = NSLOT_USER_RANGE
                                                        base_low = max(
                                                            turn_candidates_base[0],
                                                            Nslot_len_min,
                                                            user_lo
                                                        )
                                                        base_high = min(
                                                            turn_candidates_base[-1],
                                                            Nslot_len_max,
                                                            nslot_emf_cap,
                                                            user_hi
                                                        )
                                                        if base_low > base_high:
                                                            continue

                                                        Tslots = base_high - base_low + 1

                                                        total += P * Tslots * MULT_SA_FILL
    return int(total)


@torch.inference_mode()  # [추가] 함수 전체를 인퍼런스 모드로 실행 (들여쓰기 절약)
def run_sweep(RPM_ENV=None):
    """
    - 전역 백업/복구 정확
    - progress 정상
    - 항상 (df_pass1, df_pass2) 반환
    - bflow/adaptive/full 공통 사용 가능
    - 타일 루프 안에 GPU 이벤트 삽입(e_mask0/e_mask1) → gpu_ms_mask 누적
    - GPU→CPU 수집 시 e_col0/e_col1로 gpu_ms_collect 누적
    - EMF 기반 Nslot 상한(nslot_emf_cap) 계산 순서 정상화
    - NSLOT_USER_RANGE(턴수 범위) 반영
    - USE_PRECOUNT_PLANNED=False인 경우 tile마다 combos_planned 누적
    - [NEW] case / geometry 레벨 progress 훅 추가 → 초기 구간에서도 [PROG] 출력
    
    정리된 구조:
    1. 초기 백업 및 환경 설정
    2. 데이터 텐서 및 리스트 정규화
    3. RPM/Case별 연산 루프 (Inference Mode)
    4. 강제 복구 (Finally)
    """
    global ROWS_TOTAL, p, funnel, results, STATS, PROG, PROG2, GEO, DEBUG_TILE_HIT
    global awg_vec, par_vec, awg_area, awg_candidates, par_candidates, LIMIT_ROWS

    from core.progress import set_progress_interval
    set_progress_interval(5.0)

    # --- [수정] awg_candidates가 비어있는지 검사하고 복구 ---
    if not awg_candidates or len(awg_candidates) == 0:
        # 이 로그가 찍힌다면 전역 변수 전달에 문제가 있는 상태입니다.
        print(f"[RECOVER] awg_candidates was empty in run_sweep. Forcing fallback [18, 19, 20].")
        awg_candidates = [17, 18, 19, 20] 

    # --- 기존 로직 계속 ---
    # 텐서 생성 (이제 awg_candidates가 확실히 존재하므로 에러가 나지 않음)
    awg_vec = T(awg_candidates, config=cfg)
    par_vec = T(par_candidates, config=cfg)
    awg_area = T([AWG_TABLE[a]["area"] for a in awg_candidates], config=cfg)
    # ============================================================
    # rpm-only(무부하) 케이스에서 0 붕괴 방지용 "가정 부하"
    #   - None이면 기존처럼 0 붕괴(권장X)
    #   - 예: 3kW 가정 -> rpm별 토크 자동 생성
    # ============================================================
    SEED_POWER_KW_RPM_ONLY = 3.0   # <-- 필요 시 1.0~5.0 등으로 조정
    # (선택) rpm당 너무 오래 걸리면 강제로 다음 rpm으로 넘어가는 예산
    #   - None이면 비활성
    MAX_WALL_S_PER_RPM = None      # 예: 900.0 (15분) 또는 1800.0 (30분)
    MAX_COMBOS_PER_RPM = None      # 예: 5_000_000
    
    # --- [1]  예외 클래스 정의 (초입부)
    class _NextCase(Exception):
        """현재 RPM의 쿼타를 채웠을 때 다음 RPM으로 넘어가기 위한 용도"""
        pass

    # --- [2] 내부 보조 함수 정의
    def _all_quotas_met():
        """모든 RPM의 목표 수량(Quota)을 달성했는지 확인"""
        if not RPM_QUOTA:
            return False
        return all(kept_per_rpm[r] >= q for r, q in RPM_QUOTA.items())
    
    # --- [3] 전역 상태 백업 함수 ---
    _G_BACKUP = {}
    def _backup_global(name: str):
        if name not in globals(): return
        v = globals()[name]
        if hasattr(v, "clone"): _G_BACKUP[name] = v.clone()
        elif isinstance(v, (list, dict, tuple)): _G_BACKUP[name] = type(v)(v)
        else: _G_BACKUP[name] = v

    # 백업 수행
    for _nm in ("awg_candidates", "par_candidates", "NSLOT_USER_RANGE", 
                "turn_candidates_base", "awg_vec", "par_vec", "awg_area"):
        _backup_global(_nm)

    try:
        # --- [4] 초기화 및 리스트 정규화 ---
        DEBUG_TILE_HIT = 0
        DEBUG_TILE_HIT_MAX = 1000
        ROWS_TOTAL = 0

        alpha_end = float(globals().get("alpha_end", 1.0))
        C_end_mm  = float(globals().get("C_end_mm", 10.0))

        kept_per_rpm = defaultdict(int)   # 반드시 새로고쳐야 하는 부분: rpm별로 quota 달성한 개수 누적 → 다음 케이스에서 스킵 여부 판단

        if results is None: results = []
        else: results.clear()
        
        if PROF.get("start_wall") is None:
            PROF["start_wall"] = time.perf_counter()

        # 리스트/이터러블 강제 변환 (튜플화)
        global slot_area_mm2_list, slot_fill_limit_list, MLT_scale_list, end_factor_list
        slot_area_mm2_list   = _ensure_iterable(slot_area_mm2_list)
        slot_fill_limit_list = _ensure_iterable(slot_fill_limit_list)
        MLT_scale_list       = _ensure_iterable(MLT_scale_list)
        end_factor_list      = _ensure_iterable(end_factor_list)

        # 케이스 및 쿼타 설정
        cases = globals().get("cases", [])
        if not cases:
            raise RuntimeError("[run_sweep] cases is empty.")
        
        if isinstance(RPM_QUOTA, dict) and RPM_QUOTA:
            need_rows = int(sum(RPM_QUOTA.values()))
            if LIMIT_ROWS is None or LIMIT_ROWS < need_rows:
                LIMIT_ROWS = need_rows

        # --- [5] GEO/PROG 초기화 (출력 정상화의 핵심) ---
        GEO["case_total"] = len(cases)
        GEO["case_idx"]   = 0
        GEO["geo_steps"]  = 0
        GEO["tile_hits"]  = 0
        PROG2["tiles_done"] = 0
        PROG2["tiles_total"] = 0
        
        # ---- progress guard: ct=0이면 progress가 0/0으로 고정됨 ----
        if isinstance(GEO, dict) and int(GEO.get("case_total", 0) or 0) == 0:
            _cases = globals().get("cases", None)
            if isinstance(_cases, list):
                GEO["case_total"] = len(_cases)
        # 텐서 무결성 체크
        if awg_vec is None or awg_vec.numel() != len(awg_candidates):
            awg_vec = T(awg_candidates, config=cfg)
        if par_vec is None or par_vec.numel() != len(par_candidates):
            par_vec = T(par_candidates, config=cfg)
        if awg_area is None:
            awg_area = T([AWG_TABLE[a]["area"] for a in awg_candidates], config=cfg)

        # --- [6] 핵심 연산 루프 ---
        for case_idx, case in enumerate(cases, start=1):
            # [LEVEL 1] 케이스(RPM)가 바뀔 때 한 번만 업데이트
            GEO["case_idx"] = case_idx - 1
            # ---- [NEW] rpm-adaptive envelope 적용 ----
            if RPM_ENV is not None:
                awg_list_case, par_candidates_case, nslot_user_range_case = apply_envelope_for_case(case, RPM_ENV)
                user_lo, user_hi = nslot_user_range_case
            
                # NOTE: 아래 3개 텐서는 사용하는 이름에 맞춰 교체하세요.
                #   awg_vec / awg_area / par_vec
                awg_vec, awg_area, par_vec = rebuild_awg_par_tensors(awg_list_case, par_candidates_case)
            
                if PROG.get("env_dbg_printed", False) is False:
                    print(f"[RPM-ENV/APPLY] rpm={float(case['rpm']):.0f}  AWG={awg_list_case}  PAR=[{par_candidates_case[0]}..{par_candidates_case[-1]}]"
                          f"  NSLOT={nslot_user_range_case}")
                    PROG["env_dbg_printed"] = True
            else:
                user_lo, user_hi = NSLOT_USER_RANGE
                # awg_vec, awg_area, par_vec는 기존 전역 사용
                
            GEO["case_idx"]  = case_idx
            GEO["geo_steps"] = 0
            
            rpm   = int(case["rpm"])
            quota_this_rpm = RPM_QUOTA.get(int(rpm)) if RPM_QUOTA else None
            # 이미 quota 달성한 rpm이면 이 case 자체를 스킵
            if quota_this_rpm is not None and kept_per_rpm[rpm] >= quota_this_rpm:
                print(f"[RPM-QUOTA] rpm={rpm:.0f} already met {kept_per_rpm[rpm]}/{quota_this_rpm} -> skip case")
                continue
            
            # case["T_Nm"]는 T-only면 존재, max_power 모드면 None일 수 있음
            T_case = case.get("T_Nm", None)
            P_kW_target = case.get("P_kW", None)

            #  rpm-only(부하 None)면 seed_power로 토크를 만들어 0 붕괴 방지
            if (T_case is None) and (P_kW_target is None) and (SEED_POWER_KW_RPM_ONLY is not None):
                P_kW_target = float(SEED_POWER_KW_RPM_ONLY)
                T_case = kw_rpm_to_torque_nm(P_kW_target, rpm)

            #  이제도 None일 수 있으니, None 유지 (0.0 강제 금지)
            T_Nm = (float(T_case) if (T_case is not None) else None)

            # =========================================
            #  (선택) rpm별 예산 초과 시 다음 rpm으로
            # (선택) rpm별 예산 타이머/콤보 카운터 시작
            case_wall_t0 = time.perf_counter()
            case_combos_t0 = int(PROF.get("combos_evaluated", 0))

            # -------- RPM budget guard ----------
            if MAX_WALL_S_PER_RPM is not None:
                if (time.perf_counter() - case_wall_t0) >= float(MAX_WALL_S_PER_RPM):
                    print(f"[BUDGET] rpm={rpm} wall_s>={MAX_WALL_S_PER_RPM} -> next rpm")
                    raise _NextCase
            if MAX_COMBOS_PER_RPM is not None:
                combos_now = int(PROF.get('combos_evaluated', 0)) - case_combos_t0
                if combos_now >= int(MAX_COMBOS_PER_RPM):
                    print(f"[BUDGET] rpm={rpm} combos>={MAX_COMBOS_PER_RPM} -> next rpm")
                    raise _NextCase

            omega_mech = 2.0 * math.pi * rpm / 60.0
            f_e     = p * rpm / 60.0
            omega_e = p * omega_mech
            
            try:
                
                for Ke_scale in Ke_scale_list:
                    Ke_use = Ke_LL_rms_per_krpm_nom * Ke_scale
    
                    for Ld_mH in Ld_mH_list:
                        Ld_H_scalar = Ld_mH * 1e-3
                        for Lq_mH in Lq_mH_list:
                            Lq_H_scalar = Lq_mH * 1e-3
    
                            for Vdc in Vdc_list:
                                for m_max in m_max_list:
                                    Vavail_LL_rms = m_max * Vdc / math.sqrt(3)
    
                                    for Kt_rms in Kt_rms_list:
                                        # ✅ T_Nm이 None이면 전류도 None로 두고, df에 NaN으로 기록
                                        I_rms_scalar = (float(T_Nm) / float(Kt_rms)) if (T_Nm is not None) else None
    
                                        # 텐서 캐스팅
                                        Vavail_LL_rms_t = T(Vavail_LL_rms, config=cfg)
                                        # ✅ None이면 0 텐서를 넣되, "필터/기록"은 NaN/skip로 처리
                                        I_rms_t         = T(float(I_rms_scalar) if I_rms_scalar is not None else 0.0, config=cfg)
                                        Ke_use_t        = T(Ke_use, config=cfg)
                                        omega_e_t       = T(omega_e, config=cfg)
                                        rpm_per_krpm_t  = T(rpm/1000.0, config=cfg)
                                        L_min           = float(L_total_min_m)
                                        L_max           = float(L_total_max_m)
                                        L_min_t         = T(L_total_min_m, config=cfg)
                                        L_max_t         = T(L_total_max_m, config=cfg)                                    
    
                                        for J_max in J_max_list:
                                            J_max_t = T(J_max, config=cfg)
    
                                            for stack_mm in stack_mm_list:
                                                for slot_pitch_scale in slot_pitch_mm_list:
                                                    slot_pitch_mm = slot_pitch_mm_nom * float(slot_pitch_scale)
                                                    for end_factor in end_factor_list:
                                                        for coil_span_slots in coil_span_slots_list:
    
                                                            geo_key = (stack_mm, slot_pitch_scale, end_factor, coil_span_slots)

                                                            GEO["geo_steps"] += 1  # ← geometry 조합 1 step 진입
                                                            
                                                            MLT_base_mm = mlt_cache.get(geo_key)
                                                            if MLT_base_mm is None:
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
    
                                                                mlt_cache[geo_key] = MLT_base_mm
    
                                                            for mlt_scale in MLT_scale_list:
                                                                MLT_mm = MLT_base_mm * mlt_scale
                                                                if MLT_mm <= 0:
                                                                    continue
                                                                MLT_mm_t = T(MLT_mm, config=cfg)
    
                                                                # [NEW] geometry 조합 하나 진입할 때마다 카운트 & 가끔 progress 출력
                                                                PROG["geo_done"] = PROG.get("geo_done", 0) + 1
                                                                # 너무 자주 찍히지 않도록, 일정 스텝마다만 태그 추가
                                                                if PROG["geo_done"] % 200 == 0:
                                                                    # case_idx가 1이고 geo_done이 처음 200일 때만 강제 출력
                                                                    force_first = (case_idx == 1 and PROG["geo_done"] == 200)
                                                                    progress_update(
                                                                        funnel=funnel,
                                                                        tag=(f"geo#{PROG['geo_done']} rpm={rpm:.0f}, "
                                                                             f"stack={stack_mm}, span={coil_span_slots}, "
                                                                             f"MLT_scale={mlt_scale}"),
                                                                             force=force_first
                                                                    )
    
                                                                # 길이 기반 Nslot 범위
                                                                denom = (m * coils_per_phase * MLT_mm * 1e-3)
                                                                if denom <= 0:
                                                                    continue
                                                                Nslot_len_min = math.ceil(L_min / denom)
                                                                Nslot_len_max = math.floor(L_max / denom)
    
                                                                # EMF 기반 Nslot 상한(여유 하한 반영)
                                                                gate_v = 0.0 if (MARGIN_MIN_V is None) else (MARGIN_MIN_V / float(Vavail_LL_rms))
                                                                gate_pct_scalar = max(MARGIN_MIN_PCT, gate_v)
                                                                
                                                                nphase_cap = None
                                                                if Ke_use <= 0 or rpm <= 0:
                                                                    nslot_emf_cap = 10**9
                                                                else:
                                                                    # [CHANGE] SAFETY_RELAX만 사용 (1.1 배수 제거)
                                                                    relax_emf = SAFETY_RELAX
                                                                    nphase_cap = math.floor(
                                                                        ((1.0 - gate_pct_scalar) * Vavail_LL_rms / (Ke_use * (rpm / 1000.0)))
                                                                        * (Nref_turn / relax_emf)
                                                                    )
                                                                    nslot_emf_cap = max(0, nphase_cap // max(1, coils_per_phase))
                                                                if not PROG.get("emf_dbg_printed", False):
                                                                    debug_emf_cap(
                                                                        rpm=rpm,
                                                                        Vdc=Vdc,
                                                                        m_max=m_max,
                                                                        Ke_use=Ke_use,
                                                                        gate_pct=gate_pct_scalar,
                                                                        Nref_turn=Nref_turn,
                                                                        coils_per_phase=coils_per_phase,
                                                                        nphase_cap=(nphase_cap if nphase_cap is not None else -1),
                                                                        nslot_cap=nslot_emf_cap,
                                                                        user_range=NSLOT_USER_RANGE,
                                                                    )
                                                                    PROG["emf_dbg_printed"] = True
    
                                                                base_low  = max(
                                                                    turn_candidates_base[0],
                                                                    Nslot_len_min,
                                                                    user_lo,
                                                                )
                                                                base_high = min(
                                                                    turn_candidates_base[-1],
                                                                    Nslot_len_max,
                                                                    nslot_emf_cap,
                                                                    user_hi,
                                                                )
                                                                if base_low > base_high:
                                                                    funnel["skip_empty_turn"] += 1
                                                                    continue
    
                                                                for slot_area_mm2 in slot_area_mm2_list:
                                                                    for slot_fill_limit in slot_fill_limit_list:
                                                                        # 병렬 사전조건
                                                                        par_min_per_awg = torch.ceil(I_rms_t / (J_max_t * awg_area)).to(ITYPE)
                                                                        par_min_per_awg = torch.clamp(par_min_per_awg, min=1)
                                                                        
                                                                        idx_awg = torch.arange(awg_vec.numel(), dtype=ITYPE, device=DEVICE)
                                                                        AWG_IDX_G, PAR_G = torch.meshgrid(idx_awg.long(), par_vec.long(), indexing="ij")
                                                                        PAR_OK = PAR_G >= par_min_per_awg[AWG_IDX_G]
                                                                        
                                                                        if not PAR_OK.any().item():
                                                                            if not PROG.get("par_dbg_printed", False):
                                                                                # CPU로 최소 정보만 출력
                                                                                i_val = float(I_rms_scalar)
                                                                                j_val = float(J_max)
                                                                                areas = [float(AWG_TABLE[int(a)]["area"]) for a in awg_candidates]
                                                                                par_min_list = [math.ceil(i_val/(j_val*ar)) for ar in areas]
                                                                                print("[PAR_DEAD] I_rms=%.2fA  J_max=%.2fA/mm2" % (i_val, j_val))
                                                                                print("          awg_candidates=", awg_candidates)
                                                                                print("          areas(mm2)     =", [round(x,4) for x in areas])
                                                                                print("          par_min        =", par_min_list)
                                                                                print("          par_candidates =", par_candidates)
                                                                                PROG["par_dbg_printed"] = True
                                                                        
                                                                            funnel["skip_empty_par"] += 1
                                                                            STATS["pairs_parJ_empty"] += 1
                                                                            continue
    
                                                                        sel          = torch.nonzero(PAR_OK, as_tuple=False)
                                                                        awg_idx_sel  = sel[:,0]
                                                                        par_sel      = par_vec[sel[:,1]]
                                                                        area_sel     = awg_area[awg_idx_sel]
                                                                        awg_val_sel  = awg_vec[awg_idx_sel]
    
                                                                        nslot_all = torch.arange(base_low, base_high+1, dtype=ITYPE, device=DEVICE)
                                                                        if nslot_all.numel() == 0:
                                                                            continue
                                                                        tile_size = compute_dynamic_tile_size(2048)
                                                                        tile_count = (nslot_all.numel() + TILE_NSLOTS - 1) // TILE_NSLOTS
                                                                        progress_add_tiles(int(tile_count))
    
                                                                        cpu_batches = []
    
                                                                        # pair 타일링 + nslot 타일링
                                                                        for p0 in range(0, awg_val_sel.numel(), PAIR_TILE):
                                                                            psel = slice(p0, p0+PAIR_TILE)
                                                                            pair_awg_t  = awg_val_sel[psel]
                                                                            pair_par_t  = par_sel[psel].to(ITYPE)
                                                                            pair_area_t = area_sel[psel].to(DTYPE)
    
                                                                            A_tot_pair  = pair_area_t * pair_par_t.to(DTYPE)
                                                                            mask_J_pair = (I_rms_t / A_tot_pair) <= J_max_t
                                                                            #nslot_fill_max_pair = torch.floor(
                                                                            #    (slot_fill_limit * slot_area_mm2) / (2.0 * pair_par_t.to(DTYPE) * pair_area_t)
                                                                            #).to(ITYPE).clamp(min=0)
                                                                            # [BFLOW] fill 완화(윈도잉): bflow일 때만 slot_fill_limit을 소폭 확대
                                                                            _fill_relax = float(globals().get("BFLOW_FILL_RELAX_PCT", 0.0)) if globals().get("BFLOW_ACTIVE", False) else 0.0
                                                                            _fill_limit_eff = float(slot_fill_limit) * (1.0 + _fill_relax)
                                                                            nslot_fill_max_pair = torch.floor(
                                                                                (_fill_limit_eff * slot_area_mm2) / (2.0 * pair_par_t.to(DTYPE) * pair_area_t)
                                                                            ).to(ITYPE).clamp(min=0)
    
                                                                            if not mask_J_pair.any().item():
                                                                                continue
    
                                                                            for tile_idx, t0 in enumerate(range(0, nslot_all.numel(), TILE_NSLOTS), start=1):
                                                                                # === 타일 루프 실제 진입 시점 ===
                                                                                GEO["tile_hits"] += 1
                                                                                
                                                                                # (선택) 처음 몇 개 타일은 디버그 메시지
                                                                                if DEBUG_TILE_HIT < DEBUG_TILE_HIT_MAX:
                                                                                    DEBUG_TILE_HIT += 1
                                                                                    print(
                                                                                        f"\n[DEBUG] TILE_HIT#{DEBUG_TILE_HIT} "
                                                                                        f"case={case_idx}/{len(cases)} "
                                                                                        f"rpm={rpm}, stack={stack_mm}, span={coil_span_slots}, "
                                                                                        f"MLT_mm={MLT_mm:.2f}, slotA={slot_area_mm2}, fillLim={slot_fill_limit}"
                                                                                    )
                                                                                    
                                                                                nslot_i = nslot_all[t0:t0+TILE_NSLOTS]
                                                                                nslot_f = nslot_i.to(DTYPE)
    
                                                                                # ---- [ADD] 분모 누적 (사전분모 미사용 시) ----
                                                                                if not USE_PRECOUNT_PLANNED:
                                                                                    PROF["combos_planned"] = int(
                                                                                        PROF.get("combos_planned", 0)
                                                                                        + A_tot_pair.numel() * nslot_i.numel()
                                                                                    )
                                                                                # ------------------------------------------------
    
                                                                                # --- GPU core 시작 (프로파일링) ---
                                                                                is_cuda = (DEVICE.type == "cuda")
                                                                                if ENABLE_PROFILING and is_cuda:
                                                                                    e_mask0 = torch.cuda.Event(enable_timing=True)
                                                                                    e_mask1 = torch.cuda.Event(enable_timing=True)
                                                                                    e_mask0.record()
                                                                                # ----------------------------------
    
                                                                                # === 1D rough ===
                                                                                L_phase_1d, L_total_1d = compute_lengths_side_basis(
                                                                                    turns_per_slot_side=nslot_f,
                                                                                    MLT_mm=MLT_mm_t,
                                                                                    m=m,
                                                                                    coils_per_phase=coils_per_phase
                                                                                )

                                                                                mask_len_1d = (L_total_1d >= L_min_t) & (L_total_1d <= L_max_t)
    
                                                                                Nphase_1d = nslot_f * coils_per_phase
                                                                                E_1d      = Ke_use_t * rpm_per_krpm_t * (Nphase_1d / Nref_turn)
                                                                                psi_f_1d  = E_1d / (SQRT3_OVER_SQRT2_T * omega_e_t + 1e-12)
                                                                                
                                                                                A_ref = A_tot_pair.max() # 가장 굵은 쪽으로 R을 최소 추정 → rough 통과 넓힘, mm^2
                                                                                Rph_approx = T(resistivity_at_T(120.0), config=cfg) * (coils_per_phase * nslot_f * (MLT_mm_t*1e-3)) / (A_ref*1e-6)
    
                                                                                Vd_peak    = T(0.0, config=cfg)
                                                                                Vq_peak    = (Rph_approx*SQRT2_T*I_rms_t) + (omega_e_t * psi_f_1d)
                                                                                V_phase_pk = torch.sqrt(Vd_peak**2 + Vq_peak**2)
                                                                                Vreq_1d    = SQRT3_OVER_SQRT2_T * V_phase_pk
    
                                                                                rough_ok = (Vreq_1d <= (1.0 - T(gate_pct_scalar, config=cfg)) * Vavail_LL_rms_t) & mask_len_1d
                                                                                if not rough_ok.any().item():
                                                                                    is_cuda = (DEVICE.type == "cuda")
                                                                                    if ENABLE_PROFILING and is_cuda:
                                                                                        e_mask1.record(); torch.cuda.synchronize()
                                                                                        PROF["gpu_ms_mask"] += float(e_mask0.elapsed_time(e_mask1))
                                                                                    progress_tick_tile(1)
                                                                                    progress_update(tag=f"tile {tile_idx}/{int(tile_count)} | rough-empty",
                                                                                                    force=(case_idx == 1 and tile_idx == 1))
                                                                                    STATS["tiles_len_empty"] += 1
                                                                                    elapsed_min = (time.perf_counter() - PROF["start_wall"]) / 60.0
                                                                                    if (LIMIT_MIN is not None) and elapsed_min >= LIMIT_MIN:
                                                                                        print(f"\n[EARLY STOP] rows={ROWS_TOTAL:,}, elapsed_time={elapsed_min:.1f} ≥ {LIMIT_MIN} min — saving current results...")
                                                                                        raise EarlyStop
                                                                                    continue
    
                                                                                # === 2D 정확계산 ===
                                                                                NS        = nslot_i[None, :]
                                                                                pair_ok   = mask_J_pair[:, None]
                                                                                fill_ok   = (NS <= nslot_fill_max_pair[:, None])
    
                                                                                A_tot_2d  = A_tot_pair[:, None].to(DTYPE)
                                                                                Nphase_2d = (NS * coils_per_phase).to(DTYPE)
    
                                                                                E_2d      = Ke_use_t * rpm_per_krpm_t * (Nphase_2d / Nref_turn)
                                                                                psi_f_2d  = E_2d / (SQRT3_OVER_SQRT2_T * omega_e_t + 1e-12)
    
                                                                                rho_T_t   = T(resistivity_at_T(120.0), config=cfg)
                                                                                Lp_2d     = (coils_per_phase * NS * (MLT_mm_t * 1e-3)).to(DTYPE)
                                                                                Rph_2d    = rho_T_t * Lp_2d / (A_tot_2d * 1e-6)
                                                                                
                                                                                emf_q = (omega_e_t * psi_f_2d).unsqueeze(0)
                                                                                Rph_3d = Rph_2d.unsqueeze(0)  # -> [1, P, T]
    
                                                                                # --- β-sweep: SIN_FINE/COS_FINE 재사용 (sin/cos 반복계산 제거) ---
                                                                                # I_rms_t가 python float일 수도 있으니 "torch scalar"로 캐스팅
                                                                                Ipk = (SQRT2_T * I_rms_t)
                                                                                # torch scalar로 캐스팅(필요할 때만 .to)
                                                                                if not torch.is_tensor(Ipk):
                                                                                    Ipk = torch.tensor(float(Ipk), device=DEVICE, dtype=DTYPE)
                                                                                else:
                                                                                    if (Ipk.device != DEVICE) or (Ipk.dtype != DTYPE):
                                                                                        Ipk = Ipk.to(device=DEVICE, dtype=DTYPE)
                                                                                # SIN_FINE/COS_FINE는 전역에서 이미 device/dtype 맞춰 생성된 캐시라고 가정
                                                                                # 결과: Id/Iq shape = [beta_steps]
                                                                                # [beta] -> [beta,1,1] 로 shape 고정 (Rph_2d[None,...] = [1,Nturn,Npar]와 브로드캐스트)
                                                                                SINf = SIN_FINE.view(-1, 1, 1)
                                                                                COSf = COS_FINE.view(-1, 1, 1)
                                                                                Id  = -Ipk * SINf
                                                                                Iq  =  Ipk * COSf
                                                                                
                                                                                # [PATCH] v_d/v_q 는 [beta, P, T]로 계산됨
                                                                                v_d = (Rph_3d * Id) + (-omega_e_t) * T(Lq_H_scalar, config=cfg) * Iq
                                                                                v_q = (Rph_3d * Iq) + ( omega_e_t) * T(Ld_H_scalar, config=cfg) * Id + emf_q
    
                                                                                v_phase_peak    = torch.sqrt(v_d**2 + v_q**2)
                                                                                vreq_candidates = SQRT3_OVER_SQRT2_T * v_phase_peak
                                                       
                                                                                if vreq_candidates.numel() == 0:
                                                                                    # 이 타일/조합은 유효 후보가 없음 → 스킵하지 말고
                                                                                    STATS["tiles_len_empty"] += 1
                                                                                    continue
                                                                                Vreq_2d, _ = vreq_candidates.min(dim=0)
    
                                                                                v_ok = (Vreq_2d <= (1.0 - T(gate_pct_scalar, config=cfg)) * Vavail_LL_rms_t * T(SAFETY_RELAX, config=cfg))
    
                                                                                STATS["fill_fails"] += (~fill_ok & pair_ok & rough_ok[None, :]).sum().item()
                                                                                STATS["volt_fails"] += (~v_ok & pair_ok & fill_ok & rough_ok[None, :]).sum().item()

                                                                                # [수정] Rph_k 계산을 위로 올림
                                                                                rho_T_t = T(resistivity_at_T(120.0), config=cfg)
                                                                                # Rph_k가 먼저 정의되어야 함
                                                                                #L_phase_1d: (N_turns,), A_tot_pair: (N_pairs,)
                                                                                # R_all: (N_pairs, N_turns) 형태의 2D 텐서가 생성됩니다 (Broadcasting)
                                                                                R_all = (rho_T_t * L_phase_1d[None, :] / (A_tot_pair[:, None] * 1e-6)).to(torch.float32)

                                                                                # Rph_k = (rho_T_t * Lp_k / (A_tot_k * 1e-6)).to(torch.float32)
                                                                                # -------------------------------------------------
                                                                                # [ESC RELIABILITY FILTER]
                                                                                # 고장 확률 기반 필터링 (선택 사항)
                                                                                # NOTE: df_batch already has Current_rms_A / R_phase_ohm
                                                                                # 권선 온도 추정 호출 (이제 안전함)
                                                                                # Rph_k(R_phase)와 I_rms_t(I_rms)가 확실히 정의되었으므로 NameError나 IndexError가 발생하지 않습니다.
                                                                                # -------------------------------------------------
                                                                                temp_estimated = estimate_winding_temp(I_rms_t, R_all)  # R_all은 (N_pairs, N_turns) 형태의 텐서입니다.

                                                                                # 4. 온도 기반 필터링 (선택 사항)
                                                                                # [수정 후] 텐서 연산으로 마스크 생성
                                                                                temp_ok = (temp_estimated <= 180.0)  # 180도 이하인 것들만 True인 불리언 텐서 생성: (N_pairs, N_turns) 크기의 Mask

                                                                                # rough_ok는 (N_turns,), temp_ok는 (N_pairs, N_turns)이므로 [None, :] 등을 맞춰줍니다.
                                                                                mask_all = pair_ok & fill_ok & rough_ok[None, :] & v_ok & temp_ok
                                                                                pass_idx = torch.nonzero(mask_all, as_tuple=False)
    
                                                                                # --- GPU core 종료/누적 ---
                                                                                is_cuda = (DEVICE.type == "cuda")
                                                                                if ENABLE_PROFILING and is_cuda:
                                                                                    e_mask1.record(); torch.cuda.synchronize()
                                                                                    PROF["gpu_ms_mask"] += float(e_mask0.elapsed_time(e_mask1))
                                                                                # --------------------------
    
                                                                                PROF["combos_evaluated"] += int(A_tot_pair.numel() * nslot_i.numel())
    
                                                                                added = 0
                                                                                if pass_idx.numel() > 0:
                                                                                    rows = pass_idx[:,0]; cols = pass_idx[:,1]

                                                                                    STATS["pass_count"] += pass_idx.size(0)

                                                                                    # 이제 수집 단계에서 필요한 k 변수들을 추출
                                                                                    Lp_k    = L_phase_1d.index_select(0, cols)
                                                                                    A_tot_k = A_tot_pair.index_select(0, rows)
                                                                                    Rph_k = (rho_T_t * Lp_k / (A_tot_k * 1e-6)).to(torch.float32)
    
                                                                                    nslot_k = nslot_f.index_select(0, cols)

                                                                                    Lt_k    = L_total_1d.index_select(0, cols)
                                                                                    Vreq_k  = Vreq_2d[rows, cols]
                                                                                        
                                                                                    par_k_i = pair_par_t.index_select(0, rows).to(torch.int32)   # int32 유지
                                                                                    par_k_f = par_k_i.to(torch.float32)
                                                                                    
                                                                                    area_k  = pair_area_t.index_select(0, rows)   # DTYPE (float64 가능)
                                                                                    area_k_f = area_k.to(torch.float32)
                                                                                    
                                                                                    awg_k   = pair_awg_t.index_select(0, rows)
                                                                                    # [수정] J_k 계산을 여기에 올림 (area_k_f 사용)
                                                                                    J_k     = (I_rms_t / A_tot_k).to(torch.float32)
                                                                                    
                                                                                    # 병렬 수는 그대로 사용 (sqrt 제거)
                                                                                    slotfill_k = (2.0 * nslot_k.to(torch.float32) * par_k_f * area_k_f) / float(slot_area_mm2)
                                                                                    
                                                                                    Pcu_k   = (m * (I_rms_t**2) * Rph_k).to(torch.float32)
                                                                                    
                                                                                    pack = torch.stack([
                                                                                        awg_k.to(torch.int32),
                                                                                        par_k_i,             # ← 여기도 par_k_i
                                                                                        nslot_k.to(torch.int32),
                                                                                        (nslot_k * coils_per_phase).to(torch.int32),
                                                                                        J_k, slotfill_k,
                                                                                        Lp_k.to(torch.float32), Lt_k.to(torch.float32),
                                                                                        Vreq_k.to(torch.float32), Rph_k, Pcu_k
                                                                                    ], dim=1)
    
                                                                                    # --- 수집/전송 프로파일 ---
                                                                                    is_cuda = (DEVICE.type == "cuda")
                                                                                    if ENABLE_PROFILING and is_cuda:
                                                                                        e_col0 = torch.cuda.Event(enable_timing=True)
                                                                                        e_col1 = torch.cuda.Event(enable_timing=True)
                                                                                        e_col0.record()
    
                                                                                    host = pack.detach().to("cpu", non_blocking=True).numpy()
    
                                                                                    if ENABLE_PROFILING and is_cuda:
                                                                                        e_col1.record(); torch.cuda.synchronize()
                                                                                        PROF["gpu_ms_collect"] += float(e_col0.elapsed_time(e_col1))
                                                                                    # --------------------------------
    
                                                                                    cpu_batches.append(host)
                                                                                    added = host.shape[0]
    
                                                                                flush_every = 64
                                                                                last_tile   = (t0 + TILE_NSLOTS) >= nslot_all.numel()
                                                                                if (len(cpu_batches) >= flush_every) or last_tile:
                                                                                    import numpy as np
                                                                                    if cpu_batches:
                                                                                        host_big = np.vstack(cpu_batches)
                                                                                        cpu_batches.clear()

                                                                                        df_batch = pd.DataFrame(host_big, columns=[
                                                                                            "AWG","Parallels","Turns_per_slot_side","N_turns_phase_series",
                                                                                            "J_A_per_mm2","Slot_fill_ratio","L_phase_m","L_total_m",
                                                                                            "Vreq_LL_rms","R_phase_ohm","Pcu_W"
                                                                                        ]).assign(
                                                                                            rpm=rpm, Ke_scale=Ke_scale, Ld_mH=Ld_mH, Vdc=Vdc, m_max=m_max,
                                                                                            T_Nm=(float(T_Nm) if T_Nm is not None else np.nan),
                                                                                            Kt_rms=Kt_rms, J_max_A_per_mm2=J_max,
                                                                                            stack_mm=stack_mm, end_factor=end_factor,
                                                                                            coil_span_slots=coil_span_slots,
                                                                                            slot_pitch_mm=round(slot_pitch_mm, 3),
                                                                                            MLT_mm=round(MLT_mm, 2),
                                                                                            slot_area_mm2=slot_area_mm2,
                                                                                            slot_fill_limit=slot_fill_limit,
                                                                                            Vavail_LL_rms=float(Vavail_LL_rms),
                                                                                            Current_rms_A=(round(float(I_rms_scalar), 3) if I_rms_scalar is not None else np.nan),
                                                                                            f_e_Hz=round(f_e, 3), omega_e_radps=round(omega_e, 3),
                                                                                            P_shaft_kW=(round((float(T_Nm) * (2.0*math.pi*rpm/60.0)) / 1000.0, 4) if T_Nm is not None else np.nan),
                                                                                            P_kW_case=(round(float(P_kW_target), 4) if P_kW_target is not None else None),
                                                                                            # 핵심: 항상 존재하는 power 축 (필터/heatmap 기준)
                                                                                            P_kW_eff=(round(float(P_kW_target), 4) if P_kW_target is not None
                                                                                                      else round((T_Nm * int(rpm)) / 9550.0, 4)),
                                                                                            Lq_mH=Lq_mH, Lq_over_Ld=(float(Lq_mH)/float(Ld_mH)) if Ld_mH else float("nan"),
                                                                                            kw1=kw1, g_eff=g_eff
                                                                                        )
                                                                                        # ===========================
                                                                                        # [NEW] Reverse power columns
                                                                                        # ===========================
                                                                                        # 목적:
                                                                                        # - J 기반 I_req(요구 전류)
                                                                                        # - 전압 제한 기반 I_limit (옵션)
                                                                                        # - 실제 사용 전류 I_used
                                                                                        # - 그에 따른 T, P (역산 power)
                                                                                        #
                                                                                        # 주의:
                                                                                        # - df_batch에는 AWG만 있고 area가 없으므로 AWG_TABLE로 area(mm2) 복원
                                                                                        # - Ke_use는 현재 case 루프의 스칼라(Ke_LL_rms_per_krpm_nom * Ke_scale)로 이미 존재
                                                                                        # - L_phase_mH는 Ld/Lq 중 무엇을 쓸지 정책이 필요:
                                                                                        #   보수적으로는 max(Ld_mH, Lq_mH), 또는 SPMSM이면 Lq 사용 등을 권장
                                                                                        
                                                                                        # run_sweep() 내부에서 df_batch가 준비된 시점에
                                                                                        df_batch = _process_reverse_power_with_ldlq_cache(
                                                                                            df_batch,
                                                                                            rpm=rpm,
                                                                                            Ke_scale=Ke_scale,
                                                                                            Ld_mH=Ld_mH,
                                                                                            Lq_mH=Lq_mH,
                                                                                            Vdc=Vdc,
                                                                                            m_max=m_max,
                                                                                            Kt_rms=Kt_rms,
                                                                                            T_Nm=T_Nm,
                                                                                            cfg=cfg,
                                                                                        )
                                                                                                
                                                                                        # 파생/별칭
                                                                                        df_batch["V_LL_max_V"]       = df_batch["Vavail_LL_rms"]
                                                                                        df_batch["V_LL_req_V"]       = df_batch["Vreq_LL_rms"]
                                                                                        df_batch["V_LL_margin_V"]    = df_batch["V_LL_max_V"] - df_batch["V_LL_req_V"]
                                                                                        df_batch["V_LL_margin_pct"]  = df_batch["V_LL_margin_V"] / df_batch["V_LL_max_V"]
                                                                                        df_batch["P_cu_W"]           = df_batch["Pcu_W"]
                                                                                        df_batch["J_margin"]         = df_batch["J_max_A_per_mm2"] - df_batch["J_A_per_mm2"]
                                                                                        df_batch["fill_margin"]      = df_batch["slot_fill_limit"] - df_batch["Slot_fill_ratio"]
                                                                                        df_batch["V_margin"]         = df_batch["V_LL_margin_V"]
                                                                                        df_batch["V_margin_pct"]     = df_batch["V_LL_margin_pct"]
                                                                                        # -------------------------------------------------
                                                                                        # [SAFETY NET] J constraint를 DataFrame 레벨에서도 재확인
                                                                                        # (GPU mask_all에서 이미 걸러지지만, 후처리/변형 과정에서
                                                                                        #  남는 경우를 방지)
                                                                                        # -------------------------------------------------
                                                                                        df_batch = df_batch[df_batch["J_A_per_mm2"] <= df_batch["J_max_A_per_mm2"]].copy()

                                                                                        # Power Mode@@
                                                                                        if POWER_MODE == "max_power":
                                                                                            fix_list = df_batch.apply(
                                                                                                lambda r: suggest_margin_fix(
                                                                                                    r,
                                                                                                    target_margin_pct=MARGIN_MIN_PCT if "MARGIN_MIN_PCT" in globals() else 0.0,
                                                                                                    max_turn_steps=0,          # 여기선 “추가 수정”이 아니라 “해당 row 자체의 max power”만 보고 싶으면 0
                                                                                                    max_parallel_steps=0,
                                                                                                    T_oper_C=120.0,
                                                                                                    coil_per_phase=int(globals().get("coils_per_phase", 4)),
                                                                                                    poles=int(globals().get("poles", 4)),
                                                                                                    beta_samples=121,
                                                                                                    logger=logging.getLogger("suggest_fix"),
                                                                                                ),
                                                                                                axis=1
                                                                                            )
                                                                                            fix_df = pd.DataFrame(list(fix_list))
                                                                                            ok = (fix_df.get("status", "") == "ok")
                                                                                            
                                                                                            # ---- (1) 계산 결과 컬럼 붙이기 ----
                                                                                            df_batch["P_kW_calculated"] = np.where(ok, fix_df.get("P_kW_calculated_new", np.nan), np.nan)
                                                                                            df_batch["T_Nm_calculated"] = np.where(ok, fix_df.get("T_Nm_calculated_new", np.nan), np.nan)
                                                                                            df_batch["I_rms_used"]      = np.where(ok, fix_df.get("I_rms_used_new", np.nan), np.nan)
                                                                                            df_batch["V_LL_req_V_new"]  = np.where(ok, fix_df.get("V_LL_req_V_new", np.nan), np.nan)
                                                                                            df_batch["V_margin_pct_new"]= np.where(ok, fix_df.get("V_LL_margin_pct_new", np.nan), np.nan)
    
                                                                                            # ---- (2) max_power의 "운전점"으로 핵심 컬럼 동기화 (중요) ----
                                                                                            # 전류/토크
                                                                                            df_batch["Current_rms_A"] = np.where(ok, df_batch["I_rms_used"], df_batch["Current_rms_A"])
                                                                                            df_batch["T_Nm"]          = np.where(ok, df_batch["T_Nm_calculated"], df_batch["T_Nm"])
    
                                                                                            # 전압요구량: 기존 Vreq_LL_rms(0A기준) 대신 계산된 V_LL_req_V_new로 덮어쓰기
                                                                                            df_batch["Vreq_LL_rms"] = np.where(ok, df_batch["V_LL_req_V_new"], df_batch["Vreq_LL_rms"])
    
                                                                                            # 동손/전류밀도: suggest_margin_fix가 제공하면 그 값을 반영 (없으면 최소한 Current 기반 재계산 권장)
                                                                                            if "P_cu_W_new_est" in fix_df.columns:
                                                                                                df_batch["Pcu_W"] = np.where(ok, fix_df["P_cu_W_new_est"], df_batch["Pcu_W"])
                                                                                                df_batch["P_cu_W"] = df_batch["Pcu_W"]
    
                                                                                            if "J_A_per_mm2_new" in fix_df.columns:
                                                                                                df_batch["J_A_per_mm2"] = np.where(ok, fix_df["J_A_per_mm2_new"], df_batch["J_A_per_mm2"])
    
                                                                                            # 여유전압/마진 재계산(덮어쓴 Vreq 기준)
                                                                                            df_batch["V_LL_max_V"]      = df_batch["Vavail_LL_rms"]
                                                                                            df_batch["V_LL_req_V"]      = df_batch["Vreq_LL_rms"]
                                                                                            df_batch["V_LL_margin_V"]   = df_batch["V_LL_max_V"] - df_batch["V_LL_req_V"]
                                                                                            df_batch["V_LL_margin_pct"] = df_batch["V_LL_margin_V"] / np.maximum(1e-9, df_batch["V_LL_max_V"])
                                                                                            df_batch["V_margin"]        = df_batch["V_LL_margin_V"]
                                                                                            df_batch["V_margin_pct"]    = df_batch["V_LL_margin_pct"]
    
                                                                                            # 기계출력도 계산된 토크 기준으로 갱신
                                                                                            df_batch["P_shaft_kW"] = (df_batch["T_Nm"].astype(float) * (2.0*math.pi*rpm/60.0)) / 1000.0
    
                                                                                            # ---- (3) power axis는 항상 숫자 ----
                                                                                            # max_power에서는 calculated가 곧 power축
                                                                                            df_batch["P_kW_eff"] = df_batch["P_kW_calculated"]
                                                                                            df_batch["P_kW_case"] = None
                                                                                            
                                                                                            # --- [NEW] total length bookkeeping ---
                                                                                            # L_total_m: compute_lengths_side_basis() 결과(3상 등가 길이)
                                                                                            # 실제 구리 총 길이(물리 길이)는 병렬수만큼 증가
                                                                                            df_batch["L_total_eq_m"] = df_batch["L_total_m"]
                                                                                            df_batch["L_total_copper_m"] = df_batch["L_total_m"] * df_batch["Parallels"]
 
                                                                                        if "Temp_C" in df_batch.columns:
                                                                                            df_batch = df_batch[
                                                                                                failure_probability(
                                                                                                    df_batch["Temp_C"],
                                                                                                    df_batch["J_A_per_mm2"]
                                                                                                ) <= 0.05
                                                                                            ].copy()
 
                                                                                        # 탈자 마진
                                                                                        Nphase_k = df_batch["N_turns_phase_series"].to_numpy()
                                                                                        I_k      = df_batch["Current_rms_A"].to_numpy()
                                                                                        Fpk      = 1.5*kw1*Nphase_k*np.sqrt(2.0)*I_k
                                                                                        dHg      = Fpk / g_eff
                                                                                        H_op     = (Br_T(T_max) + mu0*dHg) / (mu0*(mu_r_mag + t_m/g_eff))
                                                                                        margin   = 100.0*(Hcj_T(T_max) - np.abs(H_op))/Hcj_T(T_max)
                                                                                        df_batch["Demag_Margin_pct"] = margin
    
                                                                                        # TopK pre-trim (P_rev_kW 우선)
                                                                                        # 1) P_rev_kW 큰 것 우선(내림차순)
                                                                                        # 2) 동손(Pcu_W) 작은 것 우선
                                                                                        # 3) 전류밀도(J_A_per_mm2) 작은 것 우선
                                                                                        #
                                                                                        # 주의: P_rev_kW가 NaN인 row는 뒤로 밀림
                                                                                        if "P_rev_kW" in df_batch.columns:
                                                                                            df_batch["P_rev_kW"] = pd.to_numeric(df_batch["P_rev_kW"], errors="coerce")
                                                                                            df_batch = df_batch.sort_values(
                                                                                                by=["P_rev_kW", "Pcu_W", "J_A_per_mm2"],
                                                                                                ascending=[False, True, True],
                                                                                                na_position="last",
                                                                                                kind="mergesort",   # 안정 정렬(동률 시 재현성)
                                                                                            )
                                                                                            if TOPK is not None:
                                                                                                df_batch = df_batch.head(int(TOPK))
                                                                                        else:
                                                                                            # fallback: 기존 방식 유지
                                                                                            df_batch = topk_pretrim_df(df_batch, TOPK, cols=("Pcu_W", "J_A_per_mm2"))

                                                                                            # rpm quota (P_rev_kW가 케이스 목표(P_kW_target)를 만족하는 row를 우선 채택)
                                                                                            if RPM_QUOTA:
                                                                                                quota = RPM_QUOTA.get(int(rpm), None)
                                                                                                if quota is not None:
                                                                                                    remaining = int(quota) - int(kept_per_rpm[int(rpm)])
                                                                                                    if remaining <= 0:
                                                                                                        df_batch = df_batch.iloc[0:0]
                                                                                                    else:
                                                                                                        # (1) 목표 전력이 있는 케이스면, P_rev_kW로 우선순위 선발
                                                                                                        #     - P_kW_target가 None이면: 그냥 상단부터 remaining개
                                                                                                        if (P_kW_target is not None) and ("P_rev_kW" in df_batch.columns):
                                                                                                            tgt = float(P_kW_target)
                                                                                                            p = pd.to_numeric(df_batch["P_rev_kW"], errors="coerce")
                                                                                                            ok_mask = (p >= tgt)
                                                                                        
                                                                                                            df_ok  = df_batch.loc[ok_mask].copy()
                                                                                                            df_bad = df_batch.loc[~ok_mask].copy()
                                                                                        
                                                                                                            # 이미 TOPK에서 P_rev desc 정렬을 했으므로 그대로 head를 사용하면 됨
                                                                                                            take_ok = df_ok.head(remaining)
                                                                                                            rem2 = remaining - len(take_ok)
                                                                                                            if rem2 > 0:
                                                                                                                take_bad = df_bad.head(rem2)
                                                                                                                df_batch = pd.concat([take_ok, take_bad], axis=0, ignore_index=True)
                                                                                                            else:
                                                                                                                df_batch = take_ok
                                                                                                        else:
                                                                                                            # (2) 목표 전력이 없거나 P_rev_kW가 없으면 기존처럼 상단부터 컷
                                                                                                            if len(df_batch) > remaining:
                                                                                                                df_batch = df_batch.head(remaining)
    
                                                                                        if len(df_batch):
                                                                                            results.append(df_batch)
                                                                                            added = len(df_batch)
                                                                                            funnel["pass_all"] += added
                                                                                            ROWS_TOTAL += added
                                                                                            kept_per_rpm[int(rpm)] += added
    
                                                                                # 진행/조기종료
                                                                                progress_tick_tile(1)
                                                                                progress_update(tag=f"tile {tile_idx}/{int(tile_count)} +{added}",
                                                                                                force=(case_idx == 1 and tile_idx == 1))
    
                                                                                elapsed_min = (time.perf_counter() - PROF["start_wall"]) / 60.0
                                                                                if (LIMIT_ROWS is not None and ROWS_TOTAL >= LIMIT_ROWS) or \
                                                                                   (LIMIT_MIN  is not None and elapsed_min >= LIMIT_MIN):
                                                                                    print(f"\n[EARLY STOP] rows={ROWS_TOTAL:,}, elapsed={elapsed_min:.1f} min — saving current results...")
                                                                                    raise EarlyStop
                                                                                if RPM_QUOTA and _all_quotas_met():
                                                                                    print("\n[EARLY STOP] all rpm quotas met — saving current results...")
                                                                                    raise EarlyStop
 
                                                                                #  (핵심) 이 rpm의 quota를 채웠으면, 남은 조합을 더 돌지 말고 다음 rpm으로
                                                                                quota_this_rpm2 = RPM_QUOTA.get(int(rpm)) if RPM_QUOTA else None
                                                                                # kept_per_rpm 키는 int(rpm)로만 관리해야 함 (float 키 접근 버그 수정)
                                                                                if quota_this_rpm2 is not None and kept_per_rpm[int(rpm)] >= quota_this_rpm2:
                                                                                    print(f"[RPM-QUOTA] rpm={int(rpm)} reached {kept_per_rpm[int(rpm)]}/{quota_this_rpm2} -> next rpm")
                                                                                    raise _NextCase

            except _NextCase:
                continue # 다음 케이스(RPM)로 점프

        # 모든 루프 종료 후 GEO 보정 (마지막 case_idx를 case_total로 맞춤)
        GEO["case_idx"] = GEO.get("case_total", 0)
        PROG["last"] = 0.0  # 직전에 언제 찍었든 상관없이 다시 찍게 리셋
        progress_update(tag="all cases done (run_sweep exit)", force=True)
        # ===================================================================================
        # [NEW] Finalize results -> df_pass1/df_pass2 and RETURN
        # - funnel 카운트만 올라가고 저장 DF가 비는 문제 방지
        # - results 는 tile 루프에서 results.append(df_batch)로 누적된 DataFrame 리스트라고 가정
        # ===================================================================================
    except EarlyStop:
        pass # 조기 종료 시 정상 흐름으로 진행
    finally:
        # --- [3] 전역 변수 복구 (반드시 실행) ---
        for k, v in _G_BACKUP.items():
            globals()[k] = v
        # 마지막 진행 상황 출력
        GEO["case_idx"] = GEO.get("case_total", 0)
        progress_update(tag="run_sweep complete", force=True)

    # --- [4] 결과 마무리 (Finalize) ---
    try:
        if isinstance(results, list) and len(results) > 0:
            _dfs = [d for d in results if hasattr(d, "columns")]
            df_pass1 = pd.concat(_dfs, ignore_index=True) if len(_dfs) > 0 else pd.DataFrame()
        else:
            df_pass1 = pd.DataFrame()
    except Exception:
        df_pass1 = pd.DataFrame()

    df_pass2 = df_pass1.copy()

    # 전역 변수 동기화 및 반환, 다른 코드가 전역 df_pass1/df_pass2 를 참조하는 경우를 대비
    globals()["df_pass1"] = df_pass1
    globals()["df_pass2"] = df_pass2

    return df_pass1, df_pass2

# ============================================================
# 5) 사용 예시: main() 또는 run 전에 3줄로 붙이기
# ============================================================
def setup_rpm_adaptive_envelope_and_run(
    *,
    cases: list[dict],
    par_hard_max: int,
    save_to_excel: bool = True,
    save_to_parquet: bool = True,
    save_to_csvgz: bool = True,
    out_xlsx: str | None = None,
    out_parq: str | None = None,
    out_csvgz: str | None = None,
    safety_extra_par: int = 2,
    awg_span_down: int = 0,
    awg_span_up: int = 1,
    nslot_expand_lo: int = 0,
    nslot_expand_hi: int = 0,
    verbose: bool = True,
):
    """
    예시:
      - cases는 이미 만들어졌다고 가정(build_power_torque_cases 등)
      - PAR_HARD_MAX는 전역이 있다고 가정
    """
    # (1) rpm별 envelope 생성
    RPM_ENV = build_rpm_adaptive_envelopes(
        cases,
        safety_extra_par=safety_extra_par,
        par_hard_max=par_hard_max,
        awg_span_down=awg_span_down,
        awg_span_up=awg_span_up,
        nslot_expand_lo=nslot_expand_lo,
        nslot_expand_hi=nslot_expand_hi,
        verbose=verbose,
    )
    # (2) run_sweep 내부에서 케이스마다 apply_envelope_for_case 호출하도록
    #     아래 "run_sweep patch"를 반영한 뒤 run_sweep() 호출

    ret = run_sweep(RPM_ENV=RPM_ENV)
    df_pass1, df_pass2 = ret if isinstance(ret, tuple) and len(ret) == 2 else (globals().get("df_pass1"), globals().get("df_pass2"))

    return df_pass1, df_pass2

def load_hp_and_cases():
    hp = load_hp_from_yaml_and_globals()
    cases = build_power_torque_cases(hp)
    return hp, cases


def _bflow_baseline_guard(hp, cases):
    """
    Adaptive/full과 동일 조건으로 최소 1회 보장 sweep
    narrowing 전 baseline 확보
    """
    _apply_hp_to_globals(hp, cases)
    globals()["cases"] = cases

    # 2) 누적 변수 초기화 (bflow에서 case_total=0/0 방지)
    _reset_sweep_accumulators(case_total=len(cases) if cases is not None else 0)
    df_base = run_sweep()
    return df_base

# ---- (선택) 조합 수만 빠르게 계산하고 종료 ----
def run_full_pipeline(
    *,
    out_xlsx: str | None = None,
    out_parq: str | None = None,
    out_csvgz: str | None = None,
    save_to_excel: bool = True,
    save_to_parquet: bool = True,
    save_to_csvgz: bool = True,
    wc_cfg: dict | None = None,
    fast_target_pct: float = 0.05,
    exact_target_pct: float = 0.05,
    exact_topk: int | None = 200,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    
    C.turn_candidates_base = validate_and_fix_turn_ranges(
        turn_candidates_base_=getattr(C, "turn_candidates_base", None),
        NSLOT_USER_RANGE_=getattr(C, "NSLOT_USER_RANGE", None),
    )

    """
    전체 파이프라인

    1) (옵션) planned combos 분모 사전 계산
    2) run_sweep() 실행
    3) do_profile_summary_and_save()
       - ranked / worst-case ranked 생성 및 저장
    4) worst-case 기준 auto-fix 워크북 저장
    5) STATS / funnel / GPU 프로파일 요약
    """

    # ------------------------------------------------------------
    # 0) (옵션) 분모 사전 계산 → 진행률(%) 분모
    # ------------------------------------------------------------
    global USE_PRECOUNT_PLANNED
    if USE_PRECOUNT_PLANNED:
        total = init_planned_combos_denominator()
        if total <= 0:
            print(f"[INIT][WARN] planned combos predicted = {total} → fallback to incremental denominator")
            USE_PRECOUNT_PLANNED = False

    # ------------------------------------------------------------
    # 1) Sweep 실행
    # ------------------------------------------------------------
    try:
        print_gpu_banner()

        PROF["start_wall"]       = time.perf_counter()
        PROF["combos_evaluated"] = 0
        PROF["gpu_ms_mask"]      = 0.0
        PROF["gpu_ms_collect"]   = 0.0

        run_sweep()

    except EarlyStop:
        print("\n[EARLY STOP] run_sweep 조기 종료(시간/행수/쿼터 상한 도달)")

    finally:
        progress_newline()

        # ------------------------------------------------------------
        # 2) Worst-case 설정 로드
        # ------------------------------------------------------------
        if wc_cfg is None:
            wc_cfg = globals().get("WORST", None)

        df_ranked = None
        df_wc_ranked = None

        # ------------------------------------------------------------
        # 3) 결과 요약 + 저장
        # ------------------------------------------------------------
        try:
            _, df_ranked, df_wc_ranked = do_profile_summary_and_save(
                wc_cfg=wc_cfg,
                fast_target_pct=fast_target_pct,
                exact_target_pct=exact_target_pct,
                exact_topk=exact_topk,
            )
        except TypeError:
            # 구버전 호환 (반환값 없음)
            do_profile_summary_and_save()
            df_ranked = None
            df_wc_ranked = None

        # ------------------------------------------------------------
        # 4) Worst-case 우선 auto-fix 워크북 저장
        # ------------------------------------------------------------
        try:
            if df_ranked is not None and len(df_ranked) > 0:
                if (
                    df_wc_ranked is not None
                    and "WC_pass" in df_wc_ranked.columns
                    and df_wc_ranked["WC_pass"].any()
                ):
                    df_for_fix = df_wc_ranked[df_wc_ranked["WC_pass"]].copy()
                else:
                    df_for_fix = df_ranked

                save_rank_and_fixes_workbook(
                    df_for_fix,
                    K=50,
                    target_margin=0.10,
                )

        except Exception as e:
            print(f"[FIX_SAVE][WARN] {e}")

        # ------------------------------------------------------------
        # 5) STATS / funnel 요약
        # ------------------------------------------------------------
        rows_now = ROWS_TOTAL

        print("[STATS] -----------------------------")
        print(f"  tiles_len_empty   : {STATS.get('tiles_len_empty', 0):,}")
        print(f"  pairs_parJ_empty  : {STATS.get('pairs_parJ_empty', 0):,}")
        print(f"  fill_fails        : {STATS.get('fill_fails', 0):,}")
        print(f"  volt_fails        : {STATS.get('volt_fails', 0):,}")
        print(f"  pass_count        : {STATS.get('pass_count', 0):,}")
        print(f"  skip_empty_par    : {funnel.get('skip_empty_par', 0):,}")
        print(f"  skip_empty_turn   : {funnel.get('skip_empty_turn', 0):,}")
        print(f"  pass_all (kept)   : {funnel.get('pass_all', 0):,}")
        print("-------------------------------------")

        print_prof_summary()
        print(f"\n[Torch] Passed rows so far: {rows_now:,}  | batches={len(results)}")
        final_to_save = df_wc_ranked if df_wc_ranked is not None else df_ranked

        # =========================================================
        #            PASS-2 완료 후 자동 FEMM 생성
        # =========================================================
        print("[FEMM] PASS-2 최종 후보 → FEMM 자동 생성 시작")
        build_femm_for_top_designs(df_pass2, topk=1)
        print("[FEMM] 생성 완료")
      
        if final_to_save is not None and len(final_to_save) > 0:
            final_to_save.reset_index(drop=True, inplace=True)
            print(f"[Torch] Final saved rows: {len(final_to_save):,}")
        else:
            print("[Torch] No final rows to save.")
          
    return df_ranked, df_wc_ranked

def bflow_sweep_once_with_hp(
    hp: dict,
    cases,
    save_outputs: bool = False,
    out_xlsx: str | None = None,
    out_parq: str | None = None,
    out_csvgz: str | None = None,
    label: str = "pass1",
):
    """
    B안에서 "한 번의 sweep"을 수행하는 공통 함수.

    1) hp/cases 를 globals에 반영
    2) 누적 변수 초기화
    3) run_sweep() 실행 (EarlyStop 예외 처리 포함)
    4) results 를 concat 해서 df_candidates 반환
    5) save_outputs=True 면 do_profile_summary_and_save()로 저장

    반환:
        df_candidates (concat된 단일 DataFrame)
    """
    # 함수 시작 부분에 아래 줄을 추가하거나 확인하세요.
    global turn_candidates_base, awg_candidates, par_candidates

    # 1) 전역에 파라미터 적용
    _apply_hp_to_globals(hp, cases)

    # [BFLOW GUARD] par가 지나치게 큰 구간으로만 압축되면 fill 전멸 가능성이 급증
    # NOTE: list 재할당 금지(참조 깨짐 방지). slice로 갱신.
    try:
        if isinstance(par_candidates, list) and par_candidates:
            if min(par_candidates) > 20:
                print("[B-FLOW][GUARD] par over-compressed -> restore wider range")
                par_candidates[:] = list(range(PAR_HARD_MIN, min(PAR_HARD_MIN + 20, PAR_HARD_MAX + 1)))
    except Exception:
        pass
#    globals()["cases"] = cases

    # ============================================================
    #       [BFLOW STABLE GUARD] 최소 sweep 범위 보장
    # ============================================================
    if not awg_candidates:
        awg_candidates = list(AWG_TABLE.keys())[:3]
    if not par_candidates or len(par_candidates) < 5:
        par_candidates[:] = list(range(PAR_HARD_MIN, min(PAR_HARD_MIN + 15, PAR_HARD_MAX)))
        #  과도 압축 방지
    if min(par_candidates) > 20:
        print("[B-FLOW][GUARD] par over-compressed → restoring wider range")
        par_candidates[:] = list(range(PAR_HARD_MIN, PAR_HARD_MIN+20))
#        rebuild_awg_par_tensors(awg_candidates, par_candidates)
    if not turn_candidates_base or len(turn_candidates_base) < 5:
        lo = NSLOT_USER_RANGE[0]
        turn_candidates_base = list(range(lo, lo+10))

    rebuild_awg_par_tensors(awg_candidates, par_candidates)

    # 2) 누적 변수 초기화
    # 2) 누적 변수 초기화 (bflow에서 case_total=0/0 방지)
    _reset_sweep_accumulators(case_total=len(cases) if cases is not None else 0)
    # bflow는 run_sweep 진입 전에도 출력이 나올 수 있으니 여기서 한번 더 보정
    if isinstance(GEO, dict):
        GEO["case_total"] = int(len(cases) if cases is not None else 0)
        GEO["case_idx"] = 0

    print(f"\n[B-FLOW] {label}: run_sweep() 시작")
    t_start = time.perf_counter()
    try:
        run_sweep()
    except EarlyStop:
        print(f"[B-FLOW] {label}: EarlyStop 발생, 현재까지의 결과로 정리합니다.")
    # 방탄: 결과가 0인데 너무 오래 돌았으면 중단
    if (not results) and (time.perf_counter() - t_start > 100.0):
        print("[B-FLOW][GUARD] No results after 100s -> stop PASS.")
        return pd.DataFrame([{"Note": "No feasible combinations (timeout)."}])
#    except Exception as e:
#        print(f"[B-FLOW][ERROR] {label}: run_sweep 중 예외: {e}")
#        raise

    # 3) results → df_candidates
    if not results:
        df_candidates = pd.DataFrame([{"Note": "No feasible combinations."}])
    else:
        df_candidates = pd.concat(results, ignore_index=True)

    print(f"[B-FLOW] {label}: 후보 행수 = {len(df_candidates)}")

    # 4) 저장 옵션
    if save_outputs:
        if out_xlsx is not None:
            global OUT_XLSX, OUT_PARQ, OUT_CSVGZ
            OUT_XLSX = out_xlsx
            OUT_PARQ = out_parq if out_parq else out_xlsx.replace(".xlsx", ".parquet")
            OUT_CSVGZ= out_csvgz if out_csvgz else out_xlsx.replace(".xlsx", ".csv.gz")
        # do_profile_summary_and_save() 가 전역 OUT_* 경로를 읽는 구조라면,
        # 여기서 바로 호출
        do_profile_summary_and_save()

    return df_candidates

def run_bflow_full_two_pass(
    rpm_list,
    P_kW_list=None,
    T_Nm_list=None,
    motor_type="IPM",
    pass1_out_xlsx=None,
    pass2_out_xlsx=None,
    min_margin_pct=0.0,
    passrows_topk=1000,
    out_xlsx=None,
):
    global awg_candidates, par_candidates, turn_candidates_base, NSLOT_USER_RANGE
    """
    B안 전체 플로우:

    1) PASS-1:
       - auto_generate_inputs() → hp1, cases1
       - sweep 1회 → df_pass1 (필요하면 pass1_out_xlsx에 저장)

    2) PASS-1 결과에서 pass_rows 추출

    3) auto_adjust_by_pass(hp1, pass_rows) → hp2

    4) PASS-2:
       - hp2 + (원하면 같은 cases1 또는 새 cases2)로 sweep 1회
       - 최종 결과를 pass2_out_xlsx(또는 OUT_XLSX 기본값)에 저장

    반환:
        df_pass1, df_pass2
    """
    if POWER_MODE != "max_power":
        if (not P_kW_list) and (not T_Nm_list):
            raise ValueError(
                "[B-FLOW] load_cases 모드에서는 P_kW_list 또는 T_Nm_list 중 하나가 필요합니다."
            )
    if USE_PRECOUNT_PLANNED:
        total = init_planned_combos_denominator()
        print(f"[INIT] planned combos (PASS-1 기준) = {total:,}")

    # 이후 narrowing 적용

    # Baseline 보장은 PASS-1 이후 결과로 판단
    # ===========================================================
    #           PASS-1
    # ===========================================================
    hp1, cases1 = bflow_pass1_build_hp_and_cases(
        rpm_list=rpm_list,
        P_kW_list=P_kW_list,
        T_Nm_list=T_Nm_list,
        motor_type=motor_type,
        verbose=True,
    )

    if not cases1:
        raise RuntimeError("[B-FLOW] cases1 is empty (auto_generate_inputs failed). Check rpm_list / P_kW_list / POWER_MODE inputs.")
    # PASS-1 output path: 반드시 out_xlsx(권장) 기반으로 생성
    if pass1_out_xlsx is None:
        if not out_xlsx:
            raise RuntimeError("[B-FLOW] out_xlsx is required (or pass pass1_out_xlsx explicitly).")
        pass1_out_xlsx = out_xlsx.replace(".xlsx", "_pass1.xlsx")
        
    pass1_out_parq  = pass1_out_xlsx.replace(".xlsx", ".parquet")
    pass1_out_csvgz = pass1_out_xlsx.replace(".xlsx", ".csv.gz")

    print("[B-FLOW] PASS-1: run_sweep() start")

    # 2) 누적 변수 초기화 (bflow에서 case_total=0/0 방지)
    # reset은 반드시 cases1 기준
    _reset_sweep_accumulators(case_total=len(cases1))

    if isinstance(PROG, dict):
        PROG["case_total"] = int(len(cases1))
    if isinstance(GEO, dict):
        GEO["case_total"] = int(len(cases1))
        GEO["case_idx"]   = 0

    df_pass1 = bflow_sweep_once_with_hp(
        hp=hp1,
        cases=cases1,
        save_outputs=False,  # PASS-1에서는 일단 저장하지 않고, 나중에 do_profile_summary_and_save()로 한 번에 저장하는 방향 권장 (저장 방식이 전역 OUT_XLSX 의존적이어서)
        out_xlsx=pass1_out_xlsx,
        out_parq=pass1_out_parq,
        out_csvgz=pass1_out_csvgz,
        label="PASS-1",
    )
    print("[DBG] awg_candidates =", globals().get("awg_candidates"))
    print("[DBG] par_candidates =", globals().get("par_candidates"))
    print("[DBG] NSLOT_USER_RANGE =", globals().get("NSLOT_USER_RANGE"))
    print("[DBG] turn_candidates_base len =",
          len(globals().get("turn_candidates_base", [])))

    # --------------------
    # PASS-1 → pass_rows
    # --------------------
    pass_rows = bflow_select_pass_rows_for_autotune(
        df_pass1,
        min_margin_pct=min_margin_pct,
        topk=passrows_topk,
    )

    # ---------------------------------------------------------
    # PASS-1 전멸 시 FULL fallback
    # ---------------------------------------------------------
    if pass_rows is None or pass_rows.empty:
        print("[B-FLOW] PASS-1 전멸 → FULL fallback 실행")
        df_full = run_full_pipeline(
            out_xlsx=OUT_XLSX,
            save_to_excel=True,
            save_to_parquet=False,
            save_to_csvgz=False,
        )
        if isinstance(df_full, tuple):
            return df_full[0], df_full[1]
        else:
            return df_full, df_full

    # ---------------------------------------------------------
    # Narrowing 단계 (Bayesian → Statistical window)
    # ---------------------------------------------------------
    bayes_window = compute_bayesian_window(
        pass_rows,
        awg_candidates,
        par_candidates,
        turn_candidates_base,
    )

    stat_window = compute_esc_optimal_window(pass_rows)

    # Bayesian window 우선, 없으면 stat window
    final_window = bayes_window if bayes_window else stat_window

    if final_window:
        print(f"[B-FLOW] Narrowing window = {final_window}")
        # globals() 대신 sys.modules[__name__]을 사용하여 실제 모듈 객체를 전달
        apply_window_to_globals(final_window, sys.modules[__name__])

    # --------------------
    # auto_adjust_by_pass
    # --------------------
    hp2 = auto_adjust_by_pass(
        hp_raw=hp1,
        pass_rows=pass_rows,
        verbose=True,
    )

    if min(par_candidates) > 20:
        print("[B-FLOW][GUARD] par over-compressed → restoring wider range")
        par_candidates[:] = list(range(PAR_HARD_MIN, PAR_HARD_MIN+20))

    # --------------------
    # PASS-2
    #   - 보통 동일 rpm/P_kW/T_Nm grid를 다시 쓰므로 cases1 재사용
    #   - 필요 시 auto_generate_inputs(hp2)로 별도 cases2 만들어도 됨
    # --------------------
    # PASS-2 output path: 전역 OUT_XLSX 의존 제거 (bflow만 전역이 꼬이는 가장 큰 원인 중 하나)
    if pass2_out_xlsx is None:
        if not out_xlsx:
            # out_xlsx가 없으면 PASS-1과 같은 위치에 _pass2_final을 붙여 저장
            pass2_out_xlsx = pass1_out_xlsx.replace(".xlsx", "_pass2_final.xlsx")
        else:
            pass2_out_xlsx = out_xlsx.replace(".xlsx", "_pass2_final.xlsx")

    pass2_out_parq  = pass2_out_xlsx.replace(".xlsx", ".parquet")
    pass2_out_csvgz = pass2_out_xlsx.replace(".xlsx", ".csv.gz")

    print("[B-FLOW] PASS-2: run_sweep() start")

    # 2) 누적 변수 초기화 (bflow에서 case_total=0/0 방지)
    # PASS-2도 cases1 기준
    _reset_sweep_accumulators(case_total=len(cases1))

    if isinstance(PROG, dict):
        PROG["case_total"] = int(len(cases1))
    if isinstance(GEO, dict):
        GEO["case_total"]  = int(len(cases1))
        GEO["case_idx"]    = 0

    df_pass2 = bflow_sweep_once_with_hp(
        hp=hp2,
        cases=cases1,
        save_outputs=True,
        out_xlsx=pass2_out_xlsx,
        out_parq=pass2_out_parq,
        out_csvgz=pass2_out_csvgz,
        label="PASS-2",
    )

    print("[B-FLOW] 두 패스 완료.")

    if df_pass2 is not None and not df_pass2.empty:
        save_pass_patterns(df_pass2)

        # =========================================================
        #            PASS-2 완료 후 자동 FEMM 생성
        # =========================================================
        print("[FEMM] PASS-2 최종 후보 → FEMM 자동 생성 시작")
        build_femm_for_top_designs(df_pass2, topk=1)
        print("[FEMM] 생성 완료")

    print(f"  PASS-1 rows = {len(df_pass1)}")
    print(f"  PASS-2 rows = {len(df_pass2)} (최종 후보)")

    return df_pass1, df_pass2

def run_bflow_pass1_only(
    rpm_list,
    P_kW_list=None,
    T_Nm_list=None,
    motor_type="IPM",
    min_margin_pct=0.0,
    passrows_topk=1000,
    verbose=True
):
    """
    [B-FLOW STEP 1] 1차 고속 스윕 및 AI 학습용 데이터 생성
    """
    global awg_candidates, par_candidates, turn_candidates_base
    
    print("\n" + "="*60)
    print(f"[B-FLOW-P1] Starting Global Search (Pass 1)")
    print("="*60)

    # 1. 하이퍼파라미터 및 케이스 빌드
    hp1, cases1 = bflow_pass1_build_hp_and_cases(
        rpm_list=rpm_list,
        P_kW_list=P_kW_list,
        T_Nm_list=T_Nm_list,
        motor_type=motor_type,
        verbose=verbose,
    )

    if not cases1:
        raise RuntimeError("[B-FLOW-P1] cases1 is empty. Check inputs.")

    # 2. 프로그레스 초기화
    _reset_sweep_accumulators(case_total=len(cases1))

    # 3. 1차 스윕 실행 (정밀 FEMM 없이 물리 봉선도 위주)
    df_pass1 = bflow_sweep_once_with_hp(
        hp=hp1,
        cases=cases1,
        save_outputs=False, # AI 학습 전이므로 파일 저장 생략 가능
        label="PASS-1",
    )

    print(f"[B-FLOW-P1] Completed. Total Candidates: {len(df_pass1)}")
    return df_pass1

def run_bflow_pass2_with_feedback(
    df_pass1_filtered,
    out_xlsx,
    passrows_topk=1000,
    min_margin_pct=0.0,
    save_to_excel=True
):
    """
    [B-FLOW STEP 2] AI 피드백을 반영한 정밀 해석 및 최종 랭킹
    """
    global awg_candidates, par_candidates, turn_candidates_base
    
    print("\n" + "="*60)
    print(f"[B-FLOW-P2] Starting Refined Search (Pass 2) with AI Feedback")
    print(f"[B-FLOW-P2] Input Candidates: {len(df_pass1_filtered)}")
    print("="*60)

    # 1. AI가 선택한 후보군(df_pass1_filtered)에서 Narrowing Window 계산
    # Bayesian 및 통계적 기법으로 탐색 범위를 압축합니다.
    pass_rows = bflow_select_pass_rows_for_autotune(
        df_pass1_filtered,
        min_margin_pct=min_margin_pct,
        topk=passrows_topk,
    )

    if pass_rows is None or pass_rows.empty:
        print("[WARN] No candidates survived after AI filtering. Aborting Pass 2.")
        return df_pass1_filtered, None

    # 2. 탐색 창 압축 (Narrowing)
    bayes_window = compute_bayesian_window(
        pass_rows, awg_candidates, par_candidates, turn_candidates_base
    )
    final_window = bayes_window if bayes_window else compute_esc_optimal_window(pass_rows)

    if final_window:
        print(f"[B-FLOW-P2] Applying Narrowing Window: {final_window}")
        apply_window_to_globals(final_window, sys.modules[__name__])

    # 3. 하이퍼파라미터 재조정 (hp2 생성)
    # Pass 1의 설정에서 Pass 2 전용 설정(정밀 해석 등)으로 업데이트
    hp2 = auto_adjust_by_pass(hp_raw=None, pass_rows=pass_rows, verbose=True)

    # 4. Pass 2 정밀 스윕 실행
    # 이 단계에서 Ld, Lq 추출 및 실 부하 매칭(Scroll Matching) 연동 가능
    pass2_out_xlsx = out_xlsx.replace(".xlsx", "_pass2_final.xlsx")
    
    df_pass2 = bflow_sweep_once_with_hp(
        hp=hp2,
        cases=None, # hp2 내부의 정제된 케이스 사용 또는 Pass 1 케이스 재사용
        save_outputs=save_to_excel,
        out_xlsx=pass2_out_xlsx,
        label="PASS-2",
    )

    # 5. 최종 결과 처리 (FEMM 자동 생성 등)
    if df_pass2 is not None and not df_pass2.empty:
        print("[FEMM] Generating Final Optimal FEMM Models...")
        build_femm_for_top_designs(df_pass2, topk=1)

    print(f"[B-FLOW-P2] Completed. Final Candidates: {len(df_pass2)}")
    return df_pass1_filtered, df_pass2


def run_mode_bflow_pass1(args, out_paths):
    """
    [STEP 1] 전역 설계 공간 탐색 (Fast Filtering)
    AI 모델 학습을 위한 기초 데이터를 생성하고 1차 필터링된 후보군을 반환합니다.
    """
    from core import engine as eng
    
    # RPM 및 출력 리스트 방탄 처리
    p_list = getattr(args, "p_kw_list", getattr(args, "p_list", []))
    rpm_list = getattr(args, "rpm_list", [600, 1800, 3600])

    print(f"[BFLOW-P1] Starting Pass 1: RPMs={rpm_list}, P_kW={p_list}")

    # Pass 1은 정밀 해석(Ld/Lq) 없이 물리 엔진의 봉선도(Envelope) 체크만 수행
    # eng.run_bflow_pass1_internal은 기존 run_bflow_full_two_pass의 앞부분 로직입니다.
    df_pass1 = eng.run_bflow_pass1_only(
        rpm_list=rpm_list,
        P_kW_list=p_list,
        motor_type=args.motor_type,
        min_margin_pct=args.min_margin_pct,
        # Pass 1에서는 결과가 너무 많을 수 있으므로 적절히 Top-K 유지
        passrows_topk=args.passrows_topk * 5 
    )

    if df_pass1 is None or df_pass1.empty:
        print("[WARN] Pass 1 returned empty results.")
        return None, None

    # 중간 결과 저장 (AI 학습용 Raw Data)
    if not args.no_excel:
        df_pass1.to_excel(out_paths["OUT_XLSX"].replace(".xlsx", "_p1_raw.xlsx"))

    return df_pass1, None

def run_mode_bflow_pass2(args, df_pass1_filtered, out_paths):
    """
    [STEP 2] 정밀 설계 검증 (AI-Selected Candidates Only)
    필터링된 후보들에 대해 상세 해석을 수행하고 최종 리포트를 생성합니다.
    """
    from core import engine as eng
    
    if df_pass1_filtered is None or df_pass1_filtered.empty:
        print("[ERR] No candidates provided for Pass 2.")
        return None, None

    print(f"[BFLOW-P2] Starting Pass 2 for {len(df_pass1_filtered)} AI-selected candidates.")

    # 추출된 Ld/Lq 등을 반영하여 최종 정밀 랭킹 산출
    # eng.run_bflow_pass2_internal은 기존 로직의 뒷부분(Ranking & Save)입니다.
    ret = eng.run_bflow_pass2_with_feedback(
        df_pass1=df_pass1_filtered,
        out_xlsx=out_paths["OUT_XLSX"],
        save_to_excel=not args.no_excel,
        passrows_topk=args.passrows_topk
    )

    # _normalize_engine_return을 통해 (df_pass1, df_pass2) 튜플 반환
    return _normalize_engine_return(ret)

# ============================================================
# ✅ 전역 coils_per_phase를 실제로 잠글 때도 __main__ 안에서만 실행 권장
# ============================================================
# [5] SEED-RETRY 트리거 조건 강화
#   - (a) df_pass1이 비었거나
#   - (b) 3600rpm이 0개면 => seed 재시도
# ============================================================
def _need_seed_retry(df) -> bool:
    if df is None or df.empty:
        return True
    rpm_col = next((c for c in ["rpm", "rpm_case", "RPM", "rpm_check"] if c in df.columns), None)
    if rpm_col is None:
        return True
    try:
        s = df[rpm_col].astype(float).round().astype(int)
        return (s == 3600).sum() == 0
    except Exception:
        return True

def _apply_seed_relaxation_globals():
    """Seed 모드 완화: run_bflow_full_two_pass 내부가 전역변수를 참조한다는 가정 하에 여기서만 변경."""
    global awg_candidates, par_candidates, NSLOT_USER_RANGE, turn_candidates_base
    global slot_area_mm2_list, slot_fill_limit_list, MLT_scale_list, coil_span_slots_list
    # (필요 시) global alpha_end, C_end_mm

    awg_candidates = [16, 17, 18, 19, 20]
    par_candidates = list(range(2, 41))
    NSLOT_USER_RANGE = (6, 60)
    turn_candidates_base = list(range(6, 61))

    slot_area_mm2_list   = (130.0,)
    slot_fill_limit_list = (0.80, 0.85, 0.90)
    MLT_scale_list       = (0.85, 0.90, 0.95, 1.00)
    coil_span_slots_list = (4, 5)

    # 전역으로 실제 사용되는 경우만
    # global alpha_end, C_end_mm
    # alpha_end = 0.95
    # C_end_mm  = 5.0


def _save_final(df_pass1, df_pass2, out_dir):
    final_to_save = df_pass2 if (df_pass2 is not None and not df_pass2.empty) else df_pass1
    if final_to_save is None or final_to_save.empty:
        print("[FAIL] No feasible combinations found even after retry.")
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = os.path.join(out_dir, f"final_candidates_{ts}.xlsx")
    final_to_save.to_excel(out_xlsx, index=False)
    print(f"[SAVE] {len(final_to_save)} rows -> {out_xlsx}")
    return final_to_save


def recheck_candidates_at_rpms(df, rpms=(600, 1800, 3600), mode="same_P", margin_pct_min=0.025):
    """
    df: sweep 결과(행마다 설계/조건이 들어있는 df_pass1/df_pass2)
    rpms: 검증할 rpm 목록
    mode:
      - same_P: row의 P_kW_case(또는 유사 컬럼)를 유지하고 rpm만 바꿔 토크 재계산
      - same_T: row의 T_Nm를 유지하고 rpm만 바꿔 필요한 전력 재계산
    """

    REQUIRED_BASE = [
        "rpm",
        "AWG", "Parallels", "Turns_per_slot_side",
        "Kt_rms", "J_max_A_per_mm2",
        "slot_fill_limit", "Slot_fill_ratio",
        "Vavail_LL_rms", "Vreq_LL_rms",
    ]

    # same_P에서 power 컬럼은 Rev마다 다를 수 있으니 후보를 둔다
    REQUIRED_SAME_P_ANY = ["P_kW_eff", "P_kW_case", "P_check_kW", "P_kW"]

    def _pick_power_col(df_: pd.DataFrame) -> str | None:
        for c in REQUIRED_SAME_P_ANY:
            if c in df_.columns:
                return c
        return None

    REQUIRED_SAME_T = ["T_Nm"]

    # ---- df 방탄 ----
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print("[recheck] skip: df is None/empty")
        return pd.DataFrame(columns=(REQUIRED_BASE + ["rpm_check", "PASS_at_rpm", "Note"]))

    miss_base = [c for c in REQUIRED_BASE if c not in df.columns]
    if miss_base:
        print(f"[recheck] skip: missing base columns = {miss_base}")
        out = df.copy()
        out["rpm_check"] = np.nan
        out["PASS_at_rpm"] = False
        out["Note"] = f"recheck skipped: missing base columns {miss_base}"
        return out

    # ---- 모드별 추가 요구 컬럼 점검 + power 정규화 ----
    df0 = df
    if mode == "same_P":
        pcol = _pick_power_col(df0)
        if pcol is None:
            print(f"[recheck] skip: missing any of {REQUIRED_SAME_P_ANY} for same_P")
            out = df0.copy()
            out["rpm_check"] = np.nan
            out["PASS_at_rpm"] = False
            out["Note"] = f"recheck skipped: missing P column for same_P (need one of {REQUIRED_SAME_P_ANY})"
            return out

        # 정규화: 이후 계산은 무조건 P_kW_eff만 사용
        if pcol != "P_kW_eff":
            df0 = df0.copy()
            df0["P_kW_eff"] = pd.to_numeric(df0[pcol], errors="coerce")
        else:
            # 혹시 object면 숫자로
            if not np.issubdtype(df0["P_kW_eff"].dtype, np.number):
                df0 = df0.copy()
                df0["P_kW_eff"] = pd.to_numeric(df0["P_kW_eff"], errors="coerce")

    elif mode == "same_T":
        miss_t = [c for c in REQUIRED_SAME_T if c not in df0.columns]
        if miss_t:
            print(f"[recheck] skip: missing columns for same_T = {miss_t}")
            out = df0.copy()
            out["rpm_check"] = np.nan
            out["PASS_at_rpm"] = False
            out["Note"] = f"recheck skipped: missing columns {miss_t}"
            return out
    else:
        raise ValueError("mode must be 'same_P' or 'same_T'")

    out_rows = []
    for rpm2 in rpms:
        d = df0.copy()
        d["rpm_check"] = float(rpm2)

        # ---- POWER_MODE 방탄 ----
        if globals().get("POWER_MODE", None) == "max_power":
            print("[recheck] POWER_MODE=max_power: recheck skipped (TODO path).")
            d["PASS_at_rpm"] = False
            d["Note"] = "recheck skipped: POWER_MODE=max_power (TODO)"
            out_rows.append(d)
            continue

        # ---- 부하 재정의 ----
        if mode == "same_P":
            # P 고정 → T 재계산
            d["P_check_kW"] = pd.to_numeric(d["P_kW_eff"], errors="coerce").astype(float)
            d["T_check_Nm"] = 9550.0 * d["P_check_kW"] / float(rpm2)

        elif mode == "same_T":
            # T 고정 → P 재계산
            d["T_check_Nm"] = pd.to_numeric(d["T_Nm"], errors="coerce").astype(float)
            d["P_check_kW"] = d["T_check_Nm"] * float(rpm2) / 9550.0

        # ---- 전류(최소) ----
        d["I_check_A"] = pd.to_numeric(d["T_check_Nm"], errors="coerce").astype(float) / \
                         pd.to_numeric(d["Kt_rms"], errors="coerce").astype(float).clip(lower=1e-9)

        # ---- J 조건 ----
        d["awg_area_mm2"] = d["AWG"].astype(int).map(lambda a: float(AWG_TABLE[int(a)]["area"]))
        d["J_check"] = d["I_check_A"] / (d["Parallels"] * d["awg_area_mm2"]).clip(lower=1e-9)

        # ---- Fill 조건 ----
        d["fill_check"] = pd.to_numeric(d["Slot_fill_ratio"], errors="coerce").astype(float)

        # ---- 전압/EMF 조건(보수적 근사) ----
        base_rpm = pd.to_numeric(d["rpm"], errors="coerce").astype(float).replace(0, np.nan)
        scale_rpm = float(rpm2) / base_rpm

        # base 전류 추정: (가능하면 row의 토크 기반, 없으면 P_kW_eff 기반)
        T_base = pd.to_numeric(d.get("T_Nm", np.nan), errors="coerce").astype(float)
        Kt_base = pd.to_numeric(d["Kt_rms"], errors="coerce").astype(float).replace(0, np.nan)
        I_base = (T_base / Kt_base)

        if mode == "same_P":
            # same_P면 base 토크가 NaN일 수 있으니 P로 보강
            P_base = pd.to_numeric(d["P_kW_eff"], errors="coerce").astype(float)
            T_base_fromP = 9550.0 * P_base / base_rpm
            I_base = I_base.where(np.isfinite(I_base), T_base_fromP / Kt_base)

        I_check = pd.to_numeric(d["I_check_A"], errors="coerce").astype(float)
        scale_I = I_check / I_base.replace(0, np.nan)

        scale = np.maximum(scale_rpm, scale_I.fillna(scale_rpm))
        d["Vreq_check"] = pd.to_numeric(d["Vreq_LL_rms"], errors="coerce").astype(float) * scale
        d["Vavail_check"] = pd.to_numeric(d["Vavail_LL_rms"], errors="coerce").astype(float)

        d["V_margin_check"] = d["Vavail_check"] - d["Vreq_check"]
        d["V_margin_pct_check"] = d["V_margin_check"] / d["Vavail_check"].replace(0, np.nan)

        # ---- PASS 판정 ----
        pass_mask = (
            (d["J_check"] <= pd.to_numeric(d["J_max_A_per_mm2"], errors="coerce").astype(float)) &
            (d["fill_check"] <= pd.to_numeric(d["slot_fill_limit"], errors="coerce").astype(float)) &
            (d["Vreq_check"] <= d["Vavail_check"]) &
            (d["V_margin_pct_check"] >= float(margin_pct_min))
        )
        d["PASS_at_rpm"] = pass_mask.fillna(False)

        out_rows.append(d)

    return pd.concat(out_rows, ignore_index=True)

def apply_worstcase_pipeline(
    df_candidates: pd.DataFrame,
    wc: dict,
    fast_target_pct: float = 0.05,
    exact_target_pct: float = 0.05,
    exact_topk: int | None = 200,
    sort_for_exact: list[str] = None,
) -> pd.DataFrame:
    """
    1단계: worstcase_margin_ok_fast()로 전체 후보를 빠르게 스캔 → WC_fast_ok
    2단계: fast OK 중 상위 일부(exact_topk)에 대해 worstcase_margin_ok()로 정밀검증 → WC_exact_ok
    최종: WC_pass = WC_fast_ok & (exact 수행했으면 WC_exact_ok, 미수행이면 fast 결과 유지)
    + WC_checked로 exact 수행여부를 별도 표기

    Parameters
    ----------
    df_candidates : pd.DataFrame
        run_sweep()/do_profile_summary_and_save() 이후의 candidates / ranked DF
    wc : dict
        worst-case 스케일 파라미터 (예: WORST 전역)
        예: {"Vdc":325.0, "m_max":0.925, "Ke_scale":1.05, "R_scale":1.40, "L_scale":0.85}
    fast_target_pct : float
        fast 단계에서 요구하는 최소 전압 여유(%). ex) 0.05 → 5%
    exact_target_pct : float
        정확 β-sweep 단계에서 요구하는 최소 전압 여유(%).
    exact_topk : int | None
        fast OK 중 상위 몇 개에 대해 exact 체크할지. None이면 fast OK 전부에 대해 exact 수행.
    sort_for_exact : list[str] | None
        exact 검사 대상으로 뽑을 때 정렬 기준 컬럼들.
        None이면 기본으로 ["Pcu_W", "J_A_per_mm2", "Slot_fill_ratio"] 사용.
    """
    if df_candidates is None or df_candidates.empty:
        print("[WC-PIPE] 입력 DataFrame 이 비어 있습니다.")
        return df_candidates

    df = df_candidates.copy()

    # ------------------------------
    # 1) FAST 단계: 전체 후보 스캔
    # ------------------------------
    fast_flags = []
    for row in df.itertuples(index=False):
        try:
            ok_fast = worstcase_margin_ok_fast(row, wc, target_pct=fast_target_pct)
        except Exception:
            ok_fast = False
        fast_flags.append(bool(ok_fast))

    df["WC_fast_ok"] = fast_flags
    n_fast_ok = int(df["WC_fast_ok"].sum())
    print(f"[WC-PIPE] FAST OK = {n_fast_ok} / {len(df)}")

    if n_fast_ok == 0:
        df["WC_checked"]  = False
        df["WC_exact_ok"] = False
        df["WC_pass"]     = False
        df["WC_pass_conf"]= "none"
        return df

    # ------------------------------
    # 2) EXACT 단계: fast OK 중 상위 일부만 정밀검증
    # ------------------------------
    # fast OK 만 추출
    df_fast_ok = df[df["WC_fast_ok"]].copy()

    if sort_for_exact is None:
        sort_for_exact = ["Pcu_W", "J_A_per_mm2", "Slot_fill_ratio"]

    sort_cols = [c for c in sort_for_exact if c in df_fast_ok.columns]
    if sort_cols:
        df_fast_ok_sorted = df_fast_ok.sort_values(sort_cols, ascending=[True]*len(sort_cols))
    else:
        df_fast_ok_sorted = df_fast_ok

    if exact_topk is not None and len(df_fast_ok_sorted) > exact_topk:
        df_exact_targets = df_fast_ok_sorted.head(exact_topk)
    else:
        df_exact_targets = df_fast_ok_sorted

    print(f"[WC-PIPE] EXACT 대상 = {len(df_exact_targets)} (fast OK 중 상위 후보)")

    # --- exact 수행 여부 플래그 ---
    df["WC_checked"]   = False
    df.loc[df_exact_targets.index, "WC_checked"] = True

    # 결과 저장용 (기본값 False)
    df["WC_exact_ok"] = False

    for idx, row in df_exact_targets.iterrows():
        try:
            ok_exact = worstcase_margin_ok(row, wc, target_pct=exact_target_pct)
        except Exception:
            ok_exact = False
        df.at[idx, "WC_exact_ok"] = bool(ok_exact)

    # fast OK인데 exact 대상에서 제외된 행:
    #   → 일단 "fast 결과만 믿는다" or "불확실" 중 선택 가능.
    #   여기서는 보수적으로 fast만 믿고 pass 처리 (원하면 False로 바꾸셔도 됩니다).
    # 최종 pass 정책:
    #   - exact 수행한 행: fast_ok & exact_ok
    #   - exact 미수행 행: fast_ok (pass_conf="fast")
    df["WC_pass"] = df["WC_fast_ok"] & (~df["WC_checked"] | df["WC_exact_ok"])
    df["WC_pass_conf"] = np.where(df["WC_checked"], "exact", "fast")

    print(f"[WC-PIPE] WC_pass = {int(df['WC_pass'].sum())} / {len(df)}")

    return df

try:
    import yaml  # pip install pyyaml (없으면 아래 B안/C안 사용)
except Exception:
    yaml = None

# === [ADD] 하이퍼파라미터 수집기 ===
def load_hp_from_yaml_and_globals() -> dict:
    """
    1) CONFIG_PATH/HYPERPARAM_PATH가 있으면 YAML을 읽고
    2) 현재 globals()에 있는 *_list들을 덮어써 병합해서 hp dict을 만든다.
    """
    hp = {}

    # 1) YAML 경로 자동 감지 (있을 때만)
    cfg = globals().get("CONFIG_PATH") or globals().get("HYPERPARAM_PATH")
    if cfg and os.path.exists(cfg) and yaml is not None:
        with open(cfg, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        if not isinstance(y, dict):
            raise ValueError(f"YAML must be a dict, got {type(y)}")
        hp.update(y)

    # 2) 현재 파일 상단에 정의된 리스트들을 병합(있을 때만)
    keys = [
        "rpm_list","P_kW_list","T_Nm_list",
        "awg_candidates","par_candidates",
        "coil_span_slots_list","slot_area_mm2_list","slot_fill_limit_list",
        "Ke_scale_list","Ld_mH_list","Lq_mH_list",
        "slot_pitch_mm_list","MLT_scale_list",
        "Vdc_list","m_max_list",
        "Kt_rms_list","J_max_list",
        "stack_mm_list","end_factor_list",
    ]
    for k in keys:
        if k in globals():
            v = globals()[k]
            # ⚠️ P/T 리스트는 "빈 리스트"면 넣지 않아서 YAML 우선/자동계산 로직에 방해 안 되게
            if k in ("P_kW_list", "T_Nm_list") and (v is None or v == []):
                continue
            hp[k] = v
    # P_kW_list 또는 T_Nm_list 둘 중 하나는 존재해야 합니다(나중에 build_power_torque_cases에서 검증).
    return hp

# === [CHANGE] YAML/파라미터 로드 직후, cases 만들기 전에 추가 ===
hp = load_hp_from_yaml_and_globals()
# 아래 한 줄로 rpm/P/T 조합 cases 생성
cases = build_power_torque_cases(hp, pair_mode="product", torque_round=4)

def prescan_feasible_nslot_windows(cases, cfg, max_print=20):
    user_lo, user_hi = cfg.NSLOT_USER_RANGE
    tbase_lo, tbase_hi = int(cfg.turn_candidates_base[0]), int(cfg.turn_candidates_base[-1])

    cnt_total = 0
    cnt_ok = 0
    cnt_dead_user = 0
    cnt_dead_len = 0
    cnt_dead_emf = 0

    dead_emf_printed  = 0

    for case in cases:
        rpm = float(case["rpm"])
#        T_Nm = float(case["T_Nm"])

        for Ke_scale in Ke_scale_list:
            Ke_use = Ke_LL_rms_per_krpm_nom * Ke_scale

            for Vdc in Vdc_list:
                for m_max in m_max_list:
                    Vavail = m_max * Vdc / math.sqrt(3)
                    gate_pct = max(MARGIN_MIN_PCT, (MARGIN_MIN_V / Vavail) if MARGIN_MIN_V is not None else 0.0)

                    # EMF cap (nslot_emf_cap)
                    if Ke_use <= 0 or rpm <= 0:
                        nslot_emf_cap = 10**9
                    else:
                        relax_emf = SAFETY_RELAX
                        nphase_cap = math.floor(((1.0 - gate_pct) * Vavail / (Ke_use * (rpm/1000.0))) * (Nref_turn / relax_emf))
                        nslot_emf_cap = max(0, nphase_cap // max(1, coils_per_phase))

                    # 길이 cap은 geometry 따라 바뀌므로, 대표 geometry 1개만 찍고 싶으면 여기서 break 가능
                    for stack_mm in stack_mm_list:
                        for slot_pitch_scale in slot_pitch_mm_list:
                            slot_pitch_mm = slot_pitch_mm_nom * float(slot_pitch_scale)
                            for end_factor in end_factor_list:
                                for coil_span in coil_span_slots_list:
                                    MLT = estimate_mlt_mm(
                                        slot_pitch_mm=slot_pitch_mm,
                                        stack_mm=float(stack_mm),
                                        coil_span_slots=int(coil_span),
                                        N_slots=int(N_slots),
                                        D_use=float(D_use),
                                        alpha_end=float(end_factor),
                                        C_end_mm=float(globals().get("C_end_mm", 10.0)),
                                        span_is_inclusive=True,
                                    )

                                    for mlt_scale in MLT_scale_list:
                                        MLT_mm = MLT * mlt_scale
                                        if MLT_mm <= 0:
                                            continue

                                        denom = (m * coils_per_phase * MLT_mm * 1e-3)
                                        Nslot_len_min = math.ceil(L_total_min_m / denom)
                                        Nslot_len_max = math.floor(L_total_max_m / denom)

                                        base_low  = max(tbase_lo, user_lo, Nslot_len_min)
                                        base_high = min(tbase_hi, user_hi, Nslot_len_max, nslot_emf_cap)

                                        cnt_total += 1
                                        if base_low <= base_high:
                                            cnt_ok += 1
                                            if dead_emf_printed  < max_print:
                                                print(f"[OK] rpm={rpm:.0f} Ke={Ke_scale} Vdc={Vdc} m={m_max} "
                                                      f"geo(stack={stack_mm},span={coil_span})  nslot=[{base_low}..{base_high}] "
                                                      f"(len=[{Nslot_len_min}..{Nslot_len_max}], emf_cap={nslot_emf_cap})")
                                                dead_emf_printed  += 1
                                            else:
                                                # 병목 원인 식별
                                                low_sources = {
                                                    "tbase_lo": tbase_lo,
                                                    "user_lo": user_lo,
                                                    "len_min": Nslot_len_min,
                                                }
                                                high_sources = {
                                                    "tbase_hi": tbase_hi,
                                                    "user_hi": user_hi,
                                                    "len_max": Nslot_len_max,
                                                    "emf_cap": nslot_emf_cap,
                                                }
                                                low_key  = max(low_sources, key=low_sources.get)
                                                high_key = min(high_sources, key=high_sources.get)
                                            
                                                # dead 분류(더 정확)
                                                if low_sources[low_key] > high_sources[high_key]:
                                                    # 무엇이 low를 올렸고(high를 내렸고) 때문에 죽었는지
                                                    if high_key == "emf_cap":
                                                        cnt_dead_emf += 1
                                                        if dead_emf_printed < 5:
                                                            print(
                                                                f"[EMF-DEAD] rpm={rpm:.0f} Ke_scale={Ke_scale:.3f} Ke_use={Ke_use:.3f} "
                                                                f"Vdc={Vdc:.1f} m_max={m_max:.3f} Vavail={Vavail:.2f} gate_pct={gate_pct:.4f} "
                                                                f"nphase_cap={nphase_cap} nslot_emf_cap={nslot_emf_cap} "
                                                                f"len=[{Nslot_len_min}..{Nslot_len_max}] user=[{user_lo}..{user_hi}] "
                                                                f"tbase=[{tbase_lo}..{tbase_hi}] -> base=[{base_low}..{base_high}] "
                                                                f"(low_from={low_key}, high_from={high_key})"
                                                            )
                                                            dead_emf_printed += 1
                                                    elif high_key == "len_max" or low_key == "len_min":
                                                        cnt_dead_len += 1
                                                    else:
                                                        cnt_dead_user += 1

    print("\n[PRESCAN] ------------------")
    print(f"  total checks : {cnt_total:,}")
    print(f"  feasible     : {cnt_ok:,}")
    print(f"  dead(user)   : {cnt_dead_user:,}")
    print(f"  dead(length) : {cnt_dead_len:,}")
    print(f"  dead(emf)    : {cnt_dead_emf:,}")
    print("----------------------------")
    return cnt_ok

def validate_and_fix_turn_ranges(*, NSLOT_USER_RANGE_=None, turn_candidates_base_=None):
    global NSLOT_USER_RANGE, turn_candidates_base

    # optional override from caller (avoid param/global name collision)
    if NSLOT_USER_RANGE_ is not None:
        NSLOT_USER_RANGE = NSLOT_USER_RANGE_
    if turn_candidates_base_ is not None:
        turn_candidates_base = turn_candidates_base_

    if not turn_candidates_base:
        raise ValueError("turn_candidates_base is empty")

    t_lo, t_hi = int(turn_candidates_base[0]), int(turn_candidates_base[-1])
    u_lo, u_hi = NSLOT_USER_RANGE

    # 교집합 계산
    lo = max(t_lo, int(u_lo))
    hi = min(t_hi, int(u_hi))

    if lo > hi:
        print("[TURN-RANGE][FATAL] turn_candidates_base and NSLOT_USER_RANGE have no overlap.")
        print(f"  turn_candidates_base = ({t_lo}..{t_hi})")
        print(f"  NSLOT_USER_RANGE     = ({u_lo}..{u_hi})")
        # 자동 복구: 사용자 범위를 turn_candidates_base로 덮어씀
        NSLOT_USER_RANGE = (t_lo, t_hi)
        print(f"[TURN-RANGE][AUTO-FIX] NSLOT_USER_RANGE -> {NSLOT_USER_RANGE}")
    else:
        # 사용자 범위를 교집합으로 “정리”
        NSLOT_USER_RANGE = (lo, hi)
        print(f"[TURN-RANGE] OK, using NSLOT_USER_RANGE={NSLOT_USER_RANGE}")
 
    return turn_candidates_base

# ============================================================================
# Moved from configs.config (functions) to reduce config side-effects
# ============================================================================

def do_profile_summary_and_save(
    wc_cfg=None,
    fast_target_pct=0.05,
    exact_target_pct=0.05,
    exact_topk=200,
):
    """
    - results → df_candidates 통합
    - 파생 컬럼/랭킹 df_ranked 생성
    - (옵션) worst-case 파이프라인 → df_wc_ranked
    - Parquet / CSV.GZ / Excel 저장
    - df_candidates, df_ranked, df_wc_ranked 를 반환
    """
    # 출력 경로(디렉토리 보장)
    out_xlsx = os.path.abspath(OUT_XLSX)
    out_parq = os.path.abspath(OUT_PARQ)
    out_csvgz= os.path.abspath(OUT_CSVGZ)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    os.makedirs(os.path.dirname(out_parq), exist_ok=True)
    os.makedirs(os.path.dirname(out_csvgz), exist_ok=True)
    SAVE_TO_EXCEL   = True; SAVE_TO_PARQUET = True; SAVE_TO_CSVGZ   = True
    
    print(f"[SAVE] Targets:\n  XLSX : {out_xlsx}\n  PARQ : {out_parq}\n  CSVGZ: {out_csvgz}")

    # 후보 합치기
    if not results:
        df_candidates = pd.DataFrame([{"Note": "No feasible combinations."}])
    else:
        df_candidates = pd.concat(results, ignore_index=True)
        
    # --- Power axis normalize (place right before ranking) ---
    def _bucketize_power(pkW, step=0.25):
        try:
            if pkW is None or (isinstance(pkW, float) and np.isnan(pkW)):
                return None
            return round(step * round(float(pkW) / step), 3)
        except Exception:
            return None
    
    # 1) 항상 존재하는 power 축: P_kW_eff
    #    - max_power: P_kW_calculated 우선
    #    - load_cases: P_kW_case(있으면) 우선, 없으면 P_shaft_kW 사용
    if "P_kW_eff" not in df_candidates.columns:
        df_candidates["P_kW_eff"] = np.nan
    
    if POWER_MODE == "max_power":
        if "P_kW_calculated" in df_candidates.columns:
            df_candidates["P_kW_eff"] = df_candidates["P_kW_calculated"].astype(float, errors="ignore")
        else:
            # max_power인데 calculated가 없다면, 마지막 fallback으로 P_shaft_kW라도 사용
            if "P_shaft_kW" in df_candidates.columns:
                df_candidates["P_kW_eff"] = df_candidates["P_shaft_kW"].astype(float, errors="ignore")
    else:
        if "P_kW_case" in df_candidates.columns:
            # P_kW_case가 None/NaN이면 P_shaft_kW로 대체
            pk = pd.to_numeric(df_candidates["P_kW_case"], errors="coerce")
            if "P_shaft_kW" in df_candidates.columns:
                ps = pd.to_numeric(df_candidates["P_shaft_kW"], errors="coerce")
                df_candidates["P_kW_eff"] = pk.fillna(ps)
            else:
                df_candidates["P_kW_eff"] = pk
        else:
            if "P_shaft_kW" in df_candidates.columns:
                df_candidates["P_kW_eff"] = pd.to_numeric(df_candidates["P_shaft_kW"], errors="coerce")
    
    # 2) bucket은 무조건 P_kW_eff 기준 (항상 생성)
    df_candidates["P_kW_bucket"] = df_candidates["P_kW_eff"].apply(_bucketize_power)
    
    # 3) power error는 load_cases에서만 의미있게 (필요 시만)
    if "P_error_pct" in df_candidates.columns:
        if POWER_MODE != "max_power":
            pe = pd.to_numeric(df_candidates["P_error_pct"], errors="coerce")
            df_candidates["P_error_abs"] = pe.abs()
        else:
            df_candidates["P_error_abs"] = np.nan
            
    # ---- Heatmap 호환 별칭 보정 ----
    if "Vavail_LL_rms" in df_candidates.columns and "V_LL_max_V" not in df_candidates.columns:
        df_candidates["V_LL_max_V"] = df_candidates["Vavail_LL_rms"]
    
    if "Vreq_LL_rms" in df_candidates.columns and "V_LL_req_V" not in df_candidates.columns:
        df_candidates["V_LL_req_V"] = df_candidates["Vreq_LL_rms"]
    
    need_margin = {"V_LL_margin_V", "V_LL_margin_pct"} - set(df_candidates.columns)
    if need_margin and {"V_LL_max_V", "V_LL_req_V"} <= set(df_candidates.columns):
        vmax = pd.to_numeric(df_candidates["V_LL_max_V"], errors="coerce")
        vreq = pd.to_numeric(df_candidates["V_LL_req_V"], errors="coerce")
        df_candidates["V_LL_margin_V"] = vmax - vreq
    
        # 0 나누기 방탄: vmax가 0/NaN이면 margin_pct도 NaN 처리
        denom = vmax.where(vmax.abs() > 1e-9, np.nan)
        df_candidates["V_LL_margin_pct"] = df_candidates["V_LL_margin_V"] / denom
    
    if "P_cu_W" not in df_candidates.columns and "Pcu_W" in df_candidates.columns:
        df_candidates["P_cu_W"] = df_candidates["Pcu_W"]

    # ── 랭킹 결과 만들기 ──────────────────────────────────────────────
    df_ranked = None
    try:
        if "Note" not in df_candidates.columns:
            df_ranked = make_rank(df_candidates)
            print(f"[RANK] preset={RANK_PRESET}  rows={len(df_ranked)}  "
                  f"(topk={RANK_TOPK}, group_topk={RANK_GROUP_TOPK})")
    except Exception as e:
        print(f"[RANK][WARN] ranking failed: {e}")

    # ── (옵션) worst-case 파이프라인 ────────────────────────────────
    df_wc_ranked = None
    if df_ranked is not None and wc_cfg is not None and "Note" not in df_ranked.columns:
        try:
            df_wc_ranked = apply_worstcase_pipeline(
                df_candidates=df_ranked,
                wc=wc_cfg,
                fast_target_pct=fast_target_pct,
                exact_target_pct=exact_target_pct,
                exact_topk=exact_topk,
            )
        except Exception as e:
            print(f"[WC-PIPE][WARN] worst-case pipeline failed: {e}")
            df_wc_ranked = None

    # 메타
    meta_df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "N_slots": N_slots, "m": m, "p": p,
        "slots_per_phase": slots_per_phase, "coils_per_phase": coils_per_phase,
        "L_total_min_m": L_total_min_m, "L_total_max_m": L_total_max_m,
        "rpm_list": str(rpm_list), "Ke_scale_list": str(Ke_scale_list),
        "Ld_mH_list": str(Ld_mH_list), "Vdc_list": str(Vdc_list), "m_max_list": str(m_max_list),
        "T_Nm_list": str(T_Nm_list), "Kt_rms_list": str(globals().get("Kt_rms_list", None)),
        "J_max_list": str(J_max_list),
        "P_kW_list": str(P_kW_list),
        "POWER_MODE": str(globals().get("POWER_MODE","load_cases")),
        "P_MIN_KW": float(globals().get("P_MIN_KW", 0.0)),
        "stack_mm_list": str(stack_mm_list), "end_factor_list": str(end_factor_list),
        "slot_pitch_scale_list": str(slot_pitch_mm_list), "MLT_scale_list": str(MLT_scale_list),
        "coil_span_slots_list": str(coil_span_slots_list), "slot_area_mm2_list": str(slot_area_mm2_list),
        "slot_fill_limit_list": str(slot_fill_limit_list),
        "awg_candidates": str(awg_candidates), "par_candidates": str(par_candidates),
        "turn_candidates_base": f"{turn_candidates_base[0]}..{turn_candidates_base[-1]}",
        "Ke_nom": Ke_LL_rms_per_krpm_nom, "Nref_turn": Nref_turn,
        "rank_preset": RANK_PRESET, "rank_topk": RANK_TOPK,
        "rank_group_topk": RANK_GROUP_TOPK, "margin_min_pct": MARGIN_MIN_PCT, "margin_min_v": MARGIN_MIN_V
    }])

    # Parquet / CSV.GZ
    try:
        if SAVE_TO_PARQUET:
            df_candidates.to_parquet(out_parq, index=False, compression="zstd")
            print(f"[SAVE] Parquet -> {out_parq}")
    except Exception as e:
        print(f"[SAVE][WARN] Parquet save failed: {e}")

    try:
        if SAVE_TO_CSVGZ:
            df_candidates.to_csv(out_csvgz, index=False, compression="gzip")
            print(f"[SAVE] CSV.GZ  -> {out_csvgz}")
    except Exception as e:
        print(f"[SAVE][WARN] CSV.GZ save failed: {e}")
        
    # --- 실패 통계 시트 ---
    try:
        stats_df = pd.DataFrame([STATS])
    except Exception:
        stats_df = pd.DataFrame([{"note": "no stats"}])
        
    # Excel (항상 최소 1개 시트 쓰도록)
    if SAVE_TO_EXCEL:
        MAX_XLSX_ROWS = 1_048_576
        CHUNK_ROWS    = 1_000_000
        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
                # inputs
                meta_df.to_excel(w, sheet_name="inputs", index=False)

                # candidates
                n = len(df_candidates)
                if n == 0:
                    df_candidates.to_excel(w, sheet_name="candidates", index=False)
                    print("[SAVE] Excel: wrote empty candidates.")
                elif n <= MAX_XLSX_ROWS:
                    df_candidates.to_excel(w, sheet_name="candidates", index=False)
                    print(f"[SAVE] Excel: wrote candidates ({n} rows) into a single sheet.")
                else:
                    print(f"[SAVE] Excel: candidates has {n} rows, splitting into chunks...")
                    start = 0
                    part  = 1
                    while start < n:
                        end = min(start + CHUNK_ROWS, n)
                        df_chunk = df_candidates.iloc[start:end]
                        sheet_name = f"cand_{part}"
                        df_chunk.to_excel(w, sheet_name=sheet_name, index=False)
                        print(f"  - wrote rows [{start}:{end}) to sheet '{sheet_name}'")
                        start = end
                        part += 1
                                
                # 랭킹 시트
                if df_ranked is not None and len(df_ranked) > 0:
                    df_ranked.to_excel(w, sheet_name=f"rank_best_{RANK_PRESET}", index=False)

                # (옵션) worst-case 랭킹/패스 시트
                if df_wc_ranked is not None and len(df_wc_ranked) > 0:
                    df_wc_ranked.to_excel(w, sheet_name=f"rank_wc_{RANK_PRESET}", index=False)
                    df_wc_pass = df_wc_ranked[df_wc_ranked["WC_pass"]].copy()
                    if len(df_wc_pass) > 0:
                        df_wc_pass.to_excel(w, sheet_name=f"rank_wc_pass_{RANK_PRESET}", index=False)

                stats_df.to_excel(w, sheet_name="fail_stats", index=False)
                
            print(f"[SAVE] Excel workbook saved to {out_xlsx}")
        except Exception as e:
            print(f"[SAVE][WARN] Excel save failed: {e}")

    # 최종 DataFrame 들 반환
    return df_candidates, df_ranked, df_wc_ranked


def _infer_cases_for_prescan():
    # cases가 이미 build_power_torque_cases()로 만들어졌으면 그대로 사용
    # cases가 없으면:
    if POWER_MODE == "max_power" or (not P_kW_list and not T_Nm_list):
        return [{"rpm": int(rpm), "P_kW": None, "T_Nm": None} for rpm in rpm_list]
    # load_cases fallback
    tmp = []
    for rpm in rpm_list:
        for pkW in P_kW_list:
            tmp.append({"rpm": int(rpm), "P_kW": float(pkW), "T_Nm": kw_rpm_to_torque_nm(pkW, rpm)})
    return tmp

def autotune_par_candidates_for_revision(
    safety_extra = 2,
    auto_raise_hard_max = True,
    hard_max_cap: int = 100,
    keep_user_list_if_ok: bool = False,
):
    """
    전역 변수들을 읽어서:
      - 필요한 par_reco_max 계산
      - PAR_HARD_MAX가 부족하면 경고(옵션으로 자동 상향)
      - par_candidates를 추천 범위로 자동 재구성(옵션으로 사용자가 준 리스트 유지)
    """
    global PAR_HARD_MAX, par_candidates

    cases_local = _infer_cases_for_prescan()

    info = compute_required_par_bounds(
        cases_local=cases_local,
        awg_list=list(awg_candidates),
        jmax_list=list(J_max_list),
        kt_list=list(Kt_rms_list),
        par_hard_min=int(globals().get("PAR_HARD_MIN", 1)),
        par_hard_max=int(globals().get("PAR_HARD_MAX", PAR_HARD_MAX)),
        safety_extra=safety_extra,
    )

    print("\n[PAR-AUTOTUNE] ------------------------")
    wc = info["worst_case"]
    print(f"  worst_case : rpm={wc['rpm']:.0f}, P_kW={wc['P_kW']}, T={wc['T_Nm']:.3f} Nm")
    print(f"  worst_wire : AWG{info['awg_min']} area={info['A_min']:.4f} mm2  (best=AWG{info['awg_max']} area={info['A_max']:.4f})")
    print(f"  worst_J/Kt : J_min={info['J_min']}, Kt_min={info['Kt_min']}")
    print(f"  worst_Irms : I_worst={info['I_worst']:.3f} A")
    print(f"  par_need   : {info['par_need_worst']}  (reco_max={info['par_reco_max']} incl. safety_extra={safety_extra})")
    print(f"  PAR_HARD_MAX(now) = {PAR_HARD_MAX}")
    print("--------------------------------------")

    # 하드맥스가 부족하면
    if not info["hard_max_ok"]:
        msg = (
            f"[PAR-AUTOTUNE][WARN] PAR_HARD_MAX={PAR_HARD_MAX} < "
            f"required_par_max={info['par_reco_max']} "
            f"(par_need_worst={info['par_need_worst']}, "
            f"safety_extra={info['par_reco_max'] - info['par_need_worst']})"
        )
        print(msg)
        

        if auto_raise_hard_max:
            old_hard = PAR_HARD_MAX
            new_hard = min(max(old_hard, info["par_reco_max"]), int(hard_max_cap))
        
            if new_hard != old_hard:
                PAR_HARD_MAX = new_hard
                print(
                    f"[PAR-AUTOTUNE] PAR_HARD_MAX raised: "
                    f"{old_hard} → {PAR_HARD_MAX} (cap={hard_max_cap})"
                )
            else:
                print(
                    f"[PAR-AUTOTUNE] PAR_HARD_MAX unchanged: "
                    f"{PAR_HARD_MAX} (already sufficient or capped)"
                )

    # 추천 par_candidates 구성
    reco_min = int(globals().get("PAR_HARD_MIN", 1))
    reco_max = min(int(PAR_HARD_MAX), int(info["par_reco_max"]))
    reco_list = list(range(reco_min, reco_max + 1))

    # 사용자가 준 리스트가 “이미 충분”하면 유지 옵션
    if keep_user_list_if_ok and isinstance(par_candidates, list) and len(par_candidates) > 0:
        if max(par_candidates) >= reco_max and min(par_candidates) <= reco_min:
            print(f"[PAR-AUTOTUNE] keeping existing par_candidates (covers {reco_min}..{reco_max}).")
            return info

    par_candidates = reco_list
    print(f"[PAR-AUTOTUNE] par_candidates set to range({reco_min}, {reco_max}+1) → len={len(par_candidates)}")
    return info

def print_prof_summary():
    if not ENABLE_PROFILING:
        return
    cuda_sync()
    gpu_total_s = (PROF.get("gpu_ms_mask", 0.0) + PROF.get("gpu_ms_collect", 0.0)) / 1000.0
    wall_total  = (time.perf_counter() - PROF["start_wall"]) if PROF.get("start_wall") else None
    free_gb, total_gb = gpu_mem_info_gb()
    print("\n[PROF] ---- GPU Profiling Summary ----")
    print(f"Combos evaluated    : {PROF.get('combos_evaluated',0):,}")
    print(f"GPU time (mask)     : {PROF.get('gpu_ms_mask',0.0)/1000.0:.3f} s")
    print(f"GPU time (collect)  : {PROF.get('gpu_ms_collect',0.0)/1000.0:.3f} s")
    print(f"GPU time (total)    : {gpu_total_s:.3f} s")
    if wall_total is not None and wall_total > 0:
        print(f"Wall-clock (total)  : {wall_total:.3f} s")
        print(f"Throughput (wall)   : {PROF.get('combos_evaluated',0)/wall_total:,.0f} combos/s")
    if gpu_total_s > 0:
        print(f"Throughput (GPU)    : {PROF.get('combos_evaluated',0)/gpu_total_s:,.0f} combos/s")
    if free_gb is not None:
        print(f"GPU Mem (final)     : free={free_gb:.2f} GB / total={total_gb:.2f} GB")
    print("--------------------------------------")
    print(f"GPU Mem (peak)      : free={PROF.get('gpu_mem_free_peak', 'N/A'):.2f} GB / total={PROF.get('gpu_mem_total', 'N/A'):.2f} GB")
    print(f"Max combos attempted: {PROF.get('combos_evaluated', 0):,}")
    print("--------------------------------------\n")

