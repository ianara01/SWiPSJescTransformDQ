# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:47:26 2026

@author: USER
"""
import femm
import os
import re

from typing import Dict, Optional, Tuple


# ===================================================
# FEMM helpers
# ===================================================
_FEM_KEY_RE = re.compile(r"AWG(\d+)_PAR(\d+)_Nslot(\d+)", re.IGNORECASE)


def parse_key_from_fem_filename(fname: str) -> Optional[Tuple[int, int, int]]:
    """
    motor_24S4P_AWG17_PAR3_Nslot20.fem -> (17, 3, 20)
    """
    m = _FEM_KEY_RE.search(fname)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())

def batch_extract_ldlq_from_femm(femm_dir: str, I_test: float) -> Dict[tuple, Dict[str, float]]:
    """
    FEMM .fem 파일들을 순회하며 Ld/Lq 추출
    Returns: dict[key -> {Ld_mH, Lq_mH}]
    """
    from utils.femm_ldlq import extract_ld_lq_from_femm

    LdLq_DB = {}

    for fname in os.listdir(femm_dir):
        if not fname.endswith(".fem"):
            continue

        key = parse_key_from_fem_filename(fname)
        if key is None:
            continue

        fem_path = os.path.join(femm_dir, fname)
        print(f"[FEMM-LDLQ] extracting {fname}")

        Ld_H, Lq_H = extract_ld_lq_from_femm(
            fem_file=fem_path,
            I_test=I_test,
        )

        LdLq_DB[key] = {
            "Ld_mH": 1e3 * Ld_H,
            "Lq_mH": 1e3 * Lq_H,
        }

    return LdLq_DB

def generate_femm_files_from_windings(winding_tables: dict, out_dir: str, r_slot_mid_mm: float):
    """
    main.py에서 호출: 생성된 권선표를 바탕으로 실제 FEMM(.fem) 파일을 생성합니다.
    """
    if not winding_tables:
        return

    # 순환 참조 방지를 위해 함수 내에서 import
    from utils.femm_builder import build_fem_from_winding

    femm_out = os.path.join(out_dir, "femm")
    os.makedirs(femm_out, exist_ok=True)

    for key, wt in winding_tables.items():
        awg, par, nslot = key
        fem_name = f"motor_24S4P_AWG{awg}_PAR{par}_N{nslot}.fem"
        fem_path = os.path.join(femm_out, fem_name)

        print(f"[FEMM] Generating {fem_name}")

        # femm_builder.py의 build_fem_from_winding 호출
        build_fem_from_winding(
            winding_table=wt,
            file_path=fem_path,
            r_slot_mid=r_slot_mid_mm
        )
    print(f"[FEMM] Generated {len(winding_tables)} .fem files in {femm_out}")

# utils/femm_pipeline.py