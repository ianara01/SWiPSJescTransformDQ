# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:33:29 2026

@author: user, SANG JIN PARK

Winding table utilities for 24S4P motors.

This module exists primarily to break circular imports between:
- utils (generic helpers)
- engine (sweep / ranking)
- physics (electrical / geometric models)

Move any 'static table' type helpers here.
이 모듈은 main.py의 파이프라인과 femm_builder.py 사이의 데이터 인터페이스를 담당합니다.
"""
# core/winding_table.py
from __future__ import annotations

import os
import math
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import configs.config as C

# 6-step phase belt mapping (one common convention)
# sector: 0..5
# 0:A+  1:C-  2:B+  3:A-  4:C+  5:B-
# -*- coding: utf-8 -*-
"""
Winding table utilities for 24S4P Motors.
Break circular imports between utils, engine, and physics.
"""


# =============================================================================
#            1. 상수 및 기본 패턴 정의 (24S4P 전용 권선 설정 및 매핑)
# =============================================================================
# 전기각 60도 60도 섹터별 상(Phase) 및 극성(Polarity) 매핑: 0..5 섹터
_SECTOR_MAP = {
    0: ("A", "+"), 1: ("C", "-"), 2: ("B", "+"),
    3: ("A", "-"), 4: ("C", "+"), 5: ("B", "-"),
}

_TURN_DIR = {"+": "CW", "-": "CCW"}

# 24슬롯 4극 3상 기본 분포권 패턴
BASE_PATTERN = [
    ("A", "+"), ("C", "-"), ("B", "+"), 
    ("A", "-"), ("C", "+"), ("B", "-"),
]

@dataclass(frozen=True)
class WindingSpec24s4p:
    n_slots: int = 24
    n_poles: int = 4
    n_phases: int = 3
    coil_span_slots: int = 5
    double_layer: bool = True

# =============================================================================
#                       2. 내부 유틸리티 함수
# # [1] 전기적 위치 계산: 슬롯 번호(1-based)를 넣으면 전기각을 반환=============================================================================
def _slot_electrical_deg(slot_1based: int, n_slots: int, n_poles: int) -> float:
    p = n_poles / 2.0
    deg_per_slot = 360.0 * p / float(n_slots)
    return (slot_1based - 1) * deg_per_slot

# [2] 섹터 결정: 전기각을 넣으면 상(Phase)과 극성(Polarity)을 반환
def _phase_belt_for_deg(deg_e: float) -> Tuple[str, str]:
    # 30도 offset을 주어 섹터의 중심이 맞도록 조정
    x = (deg_e + 30.0) % 360.0
    sector = int(math.floor(x / 60.0)) % 6
    return _SECTOR_MAP[sector]

# [3] 슬롯 순환 처리: 24번 슬롯 + 5칸 이동 = 29번(X) -> 5번(O) 처리
def _wrap_slot(slot_1based: int, n_slots: int) -> int:
    """슬롯 번호가 범위를 벗어날 경우(ex: 25번 슬롯) 순환하도록 처리"""
    return ((slot_1based - 1) % n_slots) + 1

# =============================================================================
#                   3. 핵심 권선 테이블 생성 로직
# =============================================================================
def build_winding_table_24s4p(
    turns_per_slot_side: int,
    n_slots: int = 24,
    n_poles: int = 4,
    coil_span_slots: int = 5,
    double_layer: bool = True,
    parallels: int = 1
) -> pd.DataFrame:
    """
    내부 유틸리티 함수를 사용하여 24S4P 권선표 생성
    
    24S4P 모터의 슬롯별 상 배치표를 생성합니다.
    femm_builder.py에서 이 테이블을 읽어 블록 라벨을 할당합니다.
    """
    records = []
    coil_counters = {"A": 0, "B": 0, "C": 0} # 각 상별 코일 순번

    for s in range(1, n_slots + 1):
        # 함수 활용 1: 현재 슬롯의 전기적 위상 파악
        deg = _slot_electrical_deg(s, n_slots, n_poles)
        ph, pol = _phase_belt_for_deg(deg)
        
        # 새로운 코일 ID 생성 (물리적 연결 추적용)
        coil_counters[ph] += 1
        coil_id = f"{ph}{coil_counters[ph]}"

        # Upper Layer 추가
        records.append({
            "Slot": s,
            "Layer": "U",
            "Phase": ph,
            "Polarity": pol,
            "Coil_ID": coil_id,
            "Turns_per_slot_side": int(turns_per_slot_side),
            "Parallels": parallels
        })

        if double_layer:
            # 함수 활용 2: 코일이 돌아오는(Return) 슬롯을 안전하게 계산
            mate_slot = _wrap_slot(s + coil_span_slots, n_slots)
            # 극성은 시작점과 반대여야 폐회로가 형성됩니다.
            pol_l = "-" if pol == "+" else "+"
            
            # Lower Layer 추가
            records.append({
                "Slot": mate_slot,
                "Layer": "L",
                "Phase": ph,
                "Polarity": pol_l,
                "Coil_ID": coil_id,
                "Turns_per_slot_side": int(turns_per_slot_side),
                "Parallels": parallels
            })

    return pd.DataFrame(records).sort_values(by=["Slot", "Layer"]).reset_index(drop=True)


def build_winding_table_from_row(row: Any) -> pd.DataFrame:
    """
    engine sweep 결과(df_pass2의 row)를 입력받아 권선 테이블을 생성합니다.
    main.py와 femm_builder.py의 가교 역할을 합니다.
    """
    # Series나 Dict 형태 모두 대응
    data = row.to_dict() if hasattr(row, "to_dict") else row
    
    # 컬럼명 대응 (Turns_per_slot_side 또는 turns)
    turns = int(data.get("Turns_per_slot_side", data.get("turns", 10)))
    parallels = int(data.get("Parallels", 1))
    span = int(data.get("coil_span_slots", 5))

    return build_winding_table_24s4p(
        turns_per_slot_side=turns,
        parallels=parallels,
        coil_span_slots=span
    )

# =============================================================================
#        4. main.py 파이프라인 연동 함수 (main.py 모드 대응 - Mode: femm_gen)
# =============================================================================
def generate_fw_safe_winding_tables(df_pass2: pd.DataFrame, out_dir: str, fw_margin_min: float = 0.05):
    """
    main.py에서 호출: FW margin을 만족하는 설계안들에 대해 CSV 권선표를 일괄 생성합니다.
    """
    if df_pass2 is None or df_pass2.empty:
        print("[WINDING] No data to process.")
        return {}

    # 조건 필터링
    df_fw_safe = df_pass2[df_pass2["FW_margin"] >= fw_margin_min]
    winding_tables = {}

    out_wind = os.path.join(out_dir, "winding_tables")
    os.makedirs(out_wind, exist_ok=True)

    for _, row in df_fw_safe.iterrows():
        # 고유 키 생성 (AWG, 병렬수, 턴수)
        key = (int(row["AWG"]), int(row["Parallels"]), int(row["Turns_per_slot_side"]))
        
        wt_df = build_winding_table_from_row(row)
        winding_tables[key] = wt_df

        # CSV 저장
        fn = f"winding_24S4P_AWG{key[0]}_PAR{key[1]}_N{key[2]}.csv"
        wt_df.to_csv(os.path.join(out_wind, fn), index=False)

    print(f"[WINDING] Generated {len(winding_tables)} CSV tables in {out_wind}")
    return winding_tables

def generate_femm_files_from_windings(winding_tables: dict, out_dir: str, r_slot_mid_mm: float):
    """
    main.py에서 호출: 생성된 권선표를 바탕으로 실제 FEMM(.fem) 파일을 생성합니다.
    """
    if not winding_tables:
        return

    # 순환 참조 방지를 위해 함수 내에서 import
    from utils.femm_builderOld import build_fem_from_winding

    femm_out = os.path.join(out_dir, "femm")
    os.makedirs(femm_out, exist_ok=True)

    for key, wt in winding_tables.items():
        awg, par, nslot = key
        fem_name = f"motor_24S4P_AWG{awg}_PAR{par}_N{nslot}.fem"
        fem_path = os.path.join(femm_out, fem_name)

        # femm_builder.py의 build_fem_from_winding 호출
        build_fem_from_winding(
            winding_table=wt,
            out_fem_path=fem_path,
            r_slot_mid=r_slot_mid_mm
        )
    print(f"[FEMM] Generated {len(winding_tables)} .fem files in {femm_out}")
# End of core/winding_table.py