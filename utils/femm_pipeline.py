# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:47:26 2026

@author: USER
"""
import os
import re

from typing import Dict, Optional, Tuple

from core.winding_table import build_winding_table_24s4p

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

def generate_fw_safe_winding_tables(df_pass2, out_dir, fw_margin_min=0.05):
    if df_pass2 is None or df_pass2.empty:
        return {}

    if "FW_margin" not in df_pass2.columns:
        return {}

    df_fw_safe = df_pass2[df_pass2["FW_margin"] >= fw_margin_min]
    winding_tables = {}

    for _, row in df_fw_safe.iterrows():
        key = (
            int(row["AWG"]),
            int(row["Parallels"]),
            int(row["Turns_per_slot_side"]),
        )

        winding_tables[key] = build_winding_table_24s4p(
            coils_per_phase=int(
                row["N_turns_phase_series"] // row["Turns_per_slot_side"]
            ),
            turns_per_slot_side=int(row["Turns_per_slot_side"]),
        )

    out_wind = os.path.join(out_dir, "winding_tables")
    os.makedirs(out_wind, exist_ok=True)

    for k, df_tbl in winding_tables.items():
        awg, par, nslot = k
        fn = f"winding_24S4P_AWG{awg}_PAR{par}_Nslot{nslot}.csv"
        df_tbl.to_csv(os.path.join(out_wind, fn), index=False)

    print(f"[WINDING] Generated {len(winding_tables)} FW-safe winding tables")
    return winding_tables

def generate_femm_files_from_windings(
    winding_tables,
    out_dir,
    r_slot_mid_mm,
):
    """
    FW-safe winding_tables -> FEMM .fem 자동 생성

    Parameters
    ----------
    winding_tables : dict
        key = (AWG, Parallels, Turns_per_slot_side)
        value = winding_table DataFrame
    out_dir : str
        main output directory
    r_slot_mid_mm : float
        슬롯 중심 반경 (mm)
    """

    if not winding_tables:
        print("[FEMM] No winding tables provided. Skip FEMM generation.")
        return

    from utils.femm_builder import build_fem_from_winding

    femm_out = os.path.join(out_dir, "femm")
    os.makedirs(femm_out, exist_ok=True)

    for key, wt in winding_tables.items():
        awg, par, nslot = key

        fem_name = f"motor_24S4P_AWG{awg}_PAR{par}_Nslot{nslot}.fem"
        fem_path = os.path.join(femm_out, fem_name)

        print(f"[FEMM] Generating {fem_name}")

        build_fem_from_winding(
            winding_table=wt,
            out_fem_path=fem_path,
            r_slot_mid=r_slot_mid_mm,
        )

    print(f"[FEMM] Generated {len(winding_tables)} FEMM files in {femm_out}")
# utils/femm_pipeline.py