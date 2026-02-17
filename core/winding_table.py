# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:33:29 2026

@author: user, SANG JIN PARK

Winding table utilities.

This module exists primarily to break circular imports between:
- utils (generic helpers)
- engine (sweep / ranking)
- physics (electrical / geometric models)

Move any 'static table' type helpers here.
"""

from __future__ import annotations

import os
import pandas as pd


def build_winding_table_24s4p(
    coils_per_phase: int,
    turns_per_slot_side: int,
    double_layer: bool = True,
) -> pd.DataFrame:
    """Build a simple 24-slot / 4-pole / 3-phase FSCW winding table.

    Notes
    - This is a *template* table generator; adapt the phase/polarity pattern to your actual winding.
    - `double_layer` is kept for API compatibility; the current table assumes a DL-style representation.

    Returns
    -------
    pd.DataFrame with:
      Slot, Phase, Polarity, Coil_ID, Turns_per_slot_side, Turn_dir
    """

    slots = list(range(1, 25))

    # Basic repeating phase pattern (example)
    phases_pattern = (["A", "C", "B"] * 8)[:24]
    polarity_pattern = (["+", "-"] * 12)[:24]

    rows = []
    coil_counter = {"A": 1, "B": 1, "C": 1}

    # crude coil id increment policy: split slots evenly among coils per phase
    slots_per_coil = max(1, 24 // max(1, (coils_per_phase * 3)))
    for idx, (slot, ph, pol) in enumerate(zip(slots, phases_pattern, polarity_pattern), start=1):
        coil_id = f"{ph}{coil_counter[ph]}"
        turn_dir = "CW" if pol == "+" else "CCW"

        rows.append(
            {
                "Slot": int(slot),
                "Phase": str(ph),
                "Polarity": str(pol),
                "Coil_ID": str(coil_id),
                "Turns_per_slot_side": int(turns_per_slot_side),
                "Turn_dir": str(turn_dir),
                "Double_layer": bool(double_layer),
            }
        )

        if (idx % slots_per_coil) == 0:
            coil_counter[ph] += 1

    return pd.DataFrame(rows)

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
