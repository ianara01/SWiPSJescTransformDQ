# -*- coding: utf-8 -*-
"""
Created on Tue Feb 08 09:42:02 2026

@author: user, SANG JIN PARK
configs.config.py

Single Source Of Truth (SSOT) for constants & default settings.

IMPORTANT
- This module must be SAFE to import (no side effects): no prints, no file IO,
  no torch allocator tweaks, no case generation, no sweeps.
- Computation functions belong in core.engine / core.physics / utils.utils.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math
import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore

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

# =============================================================================
# Ranking presets (ascending=True 기준)
# =============================================================================
_RANK_PRESETS = {
    "balanced": (
        ["Pcu_W", "J_A_per_mm2", "Slot_fill_ratio", "V_margin_penalty",
         "Parallels", "Turns_per_slot_side", "L_total_m"],
        [True, True, True, True, True, True, True],
    ),
    "efficiency": (
        ["Pcu_W", "J_A_per_mm2", "Slot_fill_ratio", "V_margin_penalty",
         "R_phase_ohm", "L_total_m"],
        [True, True, True, True, True, True],
    ),
    "safe": (
        ["J_A_per_mm2", "Slot_fill_ratio", "V_margin_penalty",
         "Pcu_W", "Demag_Margin_pct"],
        [True, True, True, True, False],
    ),
    "voltage": (
        ["V_margin_penalty", "Pcu_W", "J_A_per_mm2", "Slot_fill_ratio"],
        [True, True, True, True],
    ),
    "powerfit": (
        ["P_error_abs", "V_margin_penalty", "Pcu_W",
         "J_A_per_mm2", "Slot_fill_ratio"],
        [True, True, True, True, True],
    ),
    "safety_margin": (
        ["J_margin_calc_pct", "Slot_fill_ratio", "V_margin_penalty",
         "Demag_Margin_pct"],
        [False, True, True, False],
    ),
}

# ----------------------------------------------------------------------
# Output columns (DataFrame schema)
# ----------------------------------------------------------------------
CANDIDATE_COLS = [
    "rpm","Ke_scale","Ld_mH","Vdc","m_max","T_Nm","Kt_rms","J_max_A_per_mm2",
    "stack_mm","end_factor","coil_span_slots","slot_pitch_mm","MLT_mm",
    "slot_area_mm2","slot_fill_limit","AWG","Parallels",
    "Turns_per_slot_side","N_turns_phase_series","Current_rms_A",
    "J_A_per_mm2","Slot_fill_ratio","L_phase_m","L_total_m",
    "L_total_eq_m","L_total_copper_m",
    "R_phase_ohm","Pcu_W","Vreq_LL_rms","Vavail_LL_rms",
    "f_e_Hz","omega_e_radps",
    "P_shaft_kW","P_kW_case",
    "P_kW_calculated","T_Nm_calculated","I_rms_used",
    "V_margin_pct_new","V_LL_req_V_new",
    "P_kW_eff","P_kW_bucket",
    "P_error_pct",
]

# ----------------------------------------------------------------------
# “버튼처럼 바꾸는” 런타임 기본값
# ----------------------------------------------------------------------
RANK_PRESET      = "voltage"
RANK_TOPK        = 1000
RANK_GROUP_TOPK  = None
RANK_GROUP_COLS  = ["rpm","T_Nm","Vdc","Ke_scale","Ld_mH","m_max","P_kW_bucket"]

POWER_MODE = "max_power"   # "load_cases" | "max_power"
P_MIN_KW   = 0.0           # max_power 모드에서 필터로 사용

TOPK = 5000                # sweep 결과에서 우선 유지할 상위 후보(프리필터)

# ----------------------------------------------------------------------
# Performance toggles (torch 세부 초기화는 main에서)
# ----------------------------------------------------------------------

USE_PRECOUNT_PLANNED = False
ENABLE_PROFILING     = False

BACKEND = "Torch"                     # ← CuPy 파일이면 "cupy", Torch 파일이면 "torch"
PROGRESS_TILES_LABEL = f"tiles({BACKEND})"
LIVE_PROGRESS = True              # 끄고 싶으면 False
PROGRESS_EVERY_SEC   = 5.0
PROGRESS_LOG_PATH    = None

TILE_NSLOTS          = 8192
PAIR_TILE            = 8192

PROG = {
    "t0": time.perf_counter(),
    "last": 0.0,
    # [NEW] case / geometry 진행률용 메타
    "case_total": 0,
    "case_done": 0,
    "geo_done": 0,   # 총 몇 개의 geometry 조합을 밟았는지 카운트
}

# ----------------------------------------------------------------------
# Motor / winding defaults (Rev33 기본)
# ----------------------------------------------------------------------
N_slots = 24
m       = 3
p       = 2
slots_per_phase = N_slots // m
coils_per_phase = slots_per_phase // 2

Ke_LL_rms_per_krpm_nom = 20.0
Nref_turn              = 20

# Copper properties
rho_Cu_20C = 1.724e-8
alpha_Cu   = 0.00393

# Geometry
ID_stator = 77.0
OD_stator = 145.0
ID_rotor = 25.0
OD_rotor = 76.0
Stack_rotor = 55.0
ID_slot = 80.5
OD_slot = 122.5
D_use = (ID_slot + OD_slot) / 2.0
slot_pitch_mm_nom = (math.pi * D_use) / N_slots

# Operating sweep defaults
rpm_list   = [600, 1800, 3600]
P_kW_list  : list[float] = []
T_Nm_list  : list[float] = []
Kt_rms_list = [0.4, 0.5, 0.6, 0.7]

awg_candidates       = [16, 17, 18, 19]
par_candidates       = list(range(2, 61))
turn_candidates_base = list(range(10, 61))
NSLOT_SWEEP_MODE     = True
NSLOT_USER_RANGE     = (10, 60)

Ke_scale_list = [0.90, 0.95, 1.00]
J_max_list    = list(range(8, 12))

Ld_mH_list = [1.5, 2.0]
SAL_RATIO_LIST = [1.1, 1.3, 1.5, 1.7]
MOTOR_TYPE = "IPM"

# Lq list는 엔진에서 MOTOR_TYPE + SAL_RATIO_LIST로 생성 가능 (기본값만 제공)
Lq_mH_list = [1.95, 2.60, 3.00]

stack_mm_list        = [55.0, 56.0, 57.0]
end_factor_list      = [1.25, 1.35, 1.45]
coil_span_slots_list = [5]
slot_pitch_mm_list   = [1.0]   # scale list
MLT_scale_list       = [0.95, 1.00, 1.05]

slot_area_mm2_list   = [130.0]
slot_fill_limit_list = [0.70, 0.75, 0.80]

Vdc_list   = [380.0, 540.0]
m_max_list = [0.925, 0.975]

RPM_QUOTA = {600: 2000, 1800: 1025, 3600: 825}

LIMIT_ROWS = 4000
LIMIT_MIN  = 1000

# Voltage margin policy
MARGIN_TARGET  = 0.05
MARGIN_MIN_PCT = 0.03
MARGIN_MIN_V   = 3.0

# global knobs
SAFETY_RELAX = 1.00

PAR_HARD_MIN = 1
PAR_HARD_MAX = 60  # 실제 계산 기반 동적 상한은 core.physics.get_dynamic_constraints에서 계산(엔진에서 반영)

# 길이 윈도우(총 3상) — 최근 조정값 반영
L_total_min_m = 50.0
L_total_max_m = 130.0

# Magnet / gap (demag etc.)
mu0=4*math.pi*1e-7
mu_r_mag=1.05
Br_20=1.30
Hcj_20=1.60e6
alpha_Br=-0.0012
alpha_Hcj=-0.0060
g_eff = 0.5e-3 * 1.10
t_m   = 3.0e-3
kw1   = 0.933
T_max = 120.0

# Wire table
AWG_TABLE = {
    13: {"area": 2.6242},
    14: {"area": 2.0811},
    15: {"area": 1.6504},
    16: {"area": 1.3070},
    17: {"area": 1.0388},
    18: {"area": 0.8232},
    19: {"area": 0.6528},
    20: {"area": 0.5177},
    21: {"area": 0.4106},
    22: {"area": 0.3256},
    23: {"area": 0.2582},
    24: {"area": 0.2048},
}

# 2. 테이블의 키(AWG 번호)들을 리스트로 변환하여 할당
# dict.keys()를 리스트로 변환하면 [13, 14, ..., 24]가 됩니다.
awg_candidates = list(AWG_TABLE.keys())

# 3. 만약 정렬된 상태를 보장하고 싶다면 sorted()를 사용합니다.
awg_candidates = sorted(list(AWG_TABLE.keys()))

# 4. awg_area는 AWG_TABLE에서 area 값을 추출한 리스트입니다.
awg_area = [AWG_TABLE[awg]["area"] for awg in awg_candidates]

# Shared runtime counters (mutated by engine/progress; data only)
GEO: Dict[str, Any] = {"case_idx": 0, "case_total": 0, "geo_steps": 0, "tile_hits": 0}
PROF: Dict[str, Any] = {"combos_evaluated": 0, "gpu_ms_mask": 0.0, "gpu_ms_collect": 0.0, "start_wall": None, "combos_planned": 0}
PROG2: Dict[str, Any] = {"tiles_done": 0, "tiles_total": 0}

# 실행 시 계산
# Worst-case 세트
WORST = dict(Vdc=325.0, m_max=0.925, Ke_scale=1.05, R_scale=1.40, L_scale=0.85)
# ======================================================================
#  - rpm×P_kW -> T_Nm_list 자동생성
#  - IPM/SPM에 따른 Lq_mH_list 자동생성
#  - EMF + 길이 제약으로 NSLOT_USER_RANGE 자동 추천
#  - 토크/Jmax/AWG로 par_candidates 자동 범위 생성
# ======================================================================

# ----------------------------------------------------------------------
# Config(dataclass)
# ----------------------------------------------------------------------
@dataclass
class Config:
    """Runtime configuration container.

    - config.py에 남아있는 값들은 '기본값'이고,
      실제 실행(main)에서 cfg를 만들어 전달/수정한다.
    """
    out_dir: str = "./results"

    # runtime device/dtype are resolved in main at runtime (torch optional)
    device: Any = None
    dtype: Any = None
    itype: Any = None

    seed: int = 1234

    # toggles
    enable_profiling: bool = ENABLE_PROFILING
    use_precount_planned: bool = USE_PRECOUNT_PLANNED

    # rank
    rank_preset: str = RANK_PRESET
    rank_topk: int = RANK_TOPK
    rank_group_topk: Optional[int] = RANK_GROUP_TOPK
    rank_group_cols: list[str] = field(default_factory=lambda: list(RANK_GROUP_COLS))

    # power mode
    power_mode: str = POWER_MODE
    p_min_kw: float = P_MIN_KW

    extra: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any=None) -> Any:
        return self.extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.extra[key] = value


def build_default_cfg(out_dir: str = "./results") -> Config:
    return Config(out_dir=out_dir)
